# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Request scheduling, admission control and live server state.

A single GPU serving several users is the case that goes wrong quietly. Without
admission control, the tenth concurrent request does not fail: it makes the other nine
slow, and eventually the whole box runs out of memory mid-generation. So this module
bounds three things explicitly and reports all of them:

* how many models may be resident at once (LRU eviction beyond that),
* how many requests may be in flight against one model,
* how deep the wait queue may grow before new requests are rejected.

Everything the status API and the console's server page show comes from here.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .catalog import Catalog, Model
from .config import Settings
from .engines import Delta, Engine, EngineError, GenerationRequest

log = logging.getLogger(__name__)

_request_seq = itertools.count(1)


@dataclass
class Session:
    """One logical client. Several requests may share a session."""

    id: str
    client: str
    user_agent: str
    api: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def touch(self, api: str) -> None:
        self.last_seen = time.time()
        self.api = api

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        return {
            "id": self.id,
            "client": self.client,
            "user_agent": self.user_agent,
            "api": self.api,
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "age_s": round(now - self.created_at, 1),
            "idle_s": round(now - self.last_seen, 1),
        }


@dataclass
class ActiveRequest:
    id: str
    model_id: str
    session_id: str
    api: str
    client: str
    state: str = "queued"  # queued | loading | prefill | decoding | done | error
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    first_token_at: float | None = None
    tokens: int = 0
    load_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        ttft = round(self.first_token_at - self.queued_at, 3) if self.first_token_at else None
        elapsed = now - self.queued_at
        tps = None
        if self.first_token_at and self.tokens:
            decode = now - self.first_token_at
            tps = round(self.tokens / decode, 1) if decode > 0 else None
        return {
            "id": self.id,
            "model": self.model_id,
            "session": self.session_id,
            "api": self.api,
            "client": self.client,
            "state": self.state,
            "wait_s": round((self.started_at or now) - self.queued_at, 2),
            "elapsed_s": round(elapsed, 2),
            "load_s": round(self.load_s, 3),
            "ttft_s": ttft,
            "tokens": self.tokens,
            "tps": tps,
        }


class QueueFull(EngineError):
    """Raised when the wait queue is at its configured depth."""


class Scheduler:
    def __init__(self, engine: Engine, catalog: Catalog, settings: Settings):
        self.engine = engine
        self.catalog = catalog
        self.settings = settings
        self.started_at = time.time()

        self._slots: dict[str, asyncio.Semaphore] = {}
        self._waiting: dict[str, int] = {}
        self._lru: deque[str] = deque()
        self._load_lock = asyncio.Lock()

        self.sessions: dict[str, Session] = {}
        self.active: dict[str, ActiveRequest] = {}

        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_errors = 0
        self.rejected = 0
        self.recent: deque[dict[str, Any]] = deque(maxlen=50)

    # --- sessions -------------------------------------------------------------

    def session(
        self, session_id: str | None, *, client: str, user_agent: str, api: str
    ) -> Session:
        sid = session_id or f"s-{uuid.uuid4().hex[:10]}"
        session = self.sessions.get(sid)
        if session is None:
            session = Session(id=sid, client=client, user_agent=user_agent, api=api)
            self.sessions[sid] = session
        session.touch(api)
        return session

    def prune_sessions(self, max_idle_s: float = 3600.0) -> None:
        cutoff = time.time() - max_idle_s
        for sid in [s.id for s in self.sessions.values() if s.last_seen < cutoff]:
            self.sessions.pop(sid, None)

    # --- admission ------------------------------------------------------------

    def _slot(self, model_id: str) -> asyncio.Semaphore:
        if model_id not in self._slots:
            self._slots[model_id] = asyncio.Semaphore(self.settings.max_concurrent_per_model)
        return self._slots[model_id]

    @property
    def queue_depth(self) -> int:
        return sum(self._waiting.values())

    async def _ensure_capacity(self, model: Model) -> None:
        """Evict least-recently-used models until this one may be resident."""
        async with self._load_lock:
            if model.id in self._lru:
                self._lru.remove(model.id)
            self._lru.append(model.id)
            while len(self._lru) > max(1, self.settings.max_loaded_models):
                victim = self._lru.popleft()
                if victim == model.id:
                    self._lru.append(victim)
                    break
                busy = any(r.model_id == victim for r in self.active.values())
                if busy:
                    # Never evict a model that is mid-generation; try again next time.
                    self._lru.append(victim)
                    break
                log.info("evicting %s to make room for %s", victim, model.id)
                with contextlib.suppress(Exception):
                    await self.engine.unload(victim)

    # --- generation -----------------------------------------------------------

    async def stream(
        self,
        request: GenerationRequest,
        *,
        session: Session,
        api: str,
        client: str,
    ) -> AsyncIterator[Delta]:
        """Admit, schedule and run a generation, tracking it for the status API."""
        model = request.model
        if self.queue_depth >= self.settings.max_queue_depth:
            self.rejected += 1
            raise QueueFull(
                f"server is at capacity ({self.queue_depth} requests queued). "
                "Retry shortly, or raise HIPEP_MAX_QUEUE."
            )

        req = ActiveRequest(
            id=f"r-{next(_request_seq):06d}",
            model_id=model.id,
            session_id=session.id,
            api=api,
            client=client,
        )
        self.active[req.id] = req
        self.total_requests += 1
        session.requests += 1

        slot = self._slot(model.id)
        self._waiting[model.id] = self._waiting.get(model.id, 0) + 1
        try:
            await slot.acquire()
        finally:
            self._waiting[model.id] = max(0, self._waiting.get(model.id, 1) - 1)

        req.started_at = time.time()
        try:
            req.state = "loading"
            await self._ensure_capacity(model)
            # Load here rather than leaving it to the engine's own lazy load, so the
            # cost is measured and reported. It is the single largest surprise in a
            # first request, and a client that cannot see it reads it as slow decode.
            load_started = time.perf_counter()
            await self.engine.load(model)
            req.load_s = time.perf_counter() - load_started
            req.state = "prefill"

            usage: dict[str, Any] | None = None
            async for delta in self.engine.stream(request):
                if delta.text:
                    if req.first_token_at is None:
                        req.first_token_at = time.time()
                        req.state = "decoding"
                    req.tokens += 1
                if delta.done:
                    if delta.usage is not None:
                        delta.usage["load_s"] = round(req.load_s, 4)
                    usage = delta.usage
                yield delta

            req.state = "done"
            if usage:
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_completion_tokens += usage.get("completion_tokens", 0)
                session.prompt_tokens += usage.get("prompt_tokens", 0)
                session.completion_tokens += usage.get("completion_tokens", 0)
                self.recent.appendleft(
                    {
                        "id": req.id,
                        "model": model.id,
                        "api": api,
                        "at": time.time(),
                        **usage,
                    }
                )
        except asyncio.CancelledError:
            req.state = "cancelled"
            raise
        except Exception:
            req.state = "error"
            self.total_errors += 1
            raise
        finally:
            slot.release()
            self.active.pop(req.id, None)

    # --- reporting ------------------------------------------------------------

    async def unload_idle(self) -> None:
        """Background sweep: drop models nobody has used for a while."""
        idle = self.settings.idle_unload_seconds
        if idle <= 0:
            return
        cutoff = time.time() - idle
        busy = {r.model_id for r in self.active.values()}
        last_used: dict[str, float] = {}
        for entry in self.recent:
            last_used.setdefault(entry["model"], entry["at"])
        for model_id in list(self.engine.loaded_models()):
            if model_id in busy:
                continue
            if last_used.get(model_id, 0) < cutoff:
                log.info("unloading idle model %s", model_id)
                with contextlib.suppress(Exception):
                    await self.engine.unload(model_id)
                if model_id in self._lru:
                    self._lru.remove(model_id)

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        self.prune_sessions()
        active = [r.to_dict() for r in self.active.values()]
        return {
            "uptime_s": round(now - self.started_at, 1),
            "engine": {
                "name": self.engine.name,
                "real": self.engine.real,
                "loaded": self.engine.loaded_models(),
            },
            "limits": {
                "max_loaded_models": self.settings.max_loaded_models,
                "max_concurrent_per_model": self.settings.max_concurrent_per_model,
                "max_queue_depth": self.settings.max_queue_depth,
                "idle_unload_seconds": self.settings.idle_unload_seconds,
            },
            "load": {
                "active_requests": len(active),
                "queued_requests": self.queue_depth,
                "sessions": len(self.sessions),
                "loaded_models": len(self.engine.loaded_models()),
            },
            "totals": {
                "requests": self.total_requests,
                "errors": self.total_errors,
                "rejected": self.rejected,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
            },
            "active": active,
            "sessions_detail": [s.to_dict() for s in self.sessions.values()],
            "recent": list(self.recent)[:20],
        }
