# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Long-running jobs: pull, port, quantize, benchmark.

Everything a user starts that takes more than a few seconds is a job. One abstraction
means one status stream, one log format, one artifact layout and one place to look when
something goes wrong — instead of a different progress mechanism per operation.

A job is a list of stages. Stages are resumable: a stage that has already produced its
output directory is skipped on re-run, so an import that fails at quantization does not
re-download 30 GB of weights.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


@dataclass
class Stage:
    name: str
    label: str
    state: JobState = JobState.PENDING
    progress: float = 0.0  # 0..1
    detail: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data["duration_s"] = (
            round((self.finished_at or time.time()) - self.started_at, 1)
            if self.started_at
            else None
        )
        return data


@dataclass
class LogLine:
    at: float
    level: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"at": self.at, "level": self.level, "text": self.text}


@dataclass
class Job:
    id: str
    kind: str
    title: str
    stages: list[Stage]
    params: dict[str, Any] = field(default_factory=dict)
    state: JobState = JobState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    logs: list[LogLine] = field(default_factory=list)

    @property
    def progress(self) -> float:
        if not self.stages:
            return 1.0 if self.state.terminal else 0.0
        return sum(s.progress for s in self.stages) / len(self.stages)

    @property
    def current_stage(self) -> Stage | None:
        return next((s for s in self.stages if s.state == JobState.RUNNING), None)

    def to_dict(self, *, include_logs: bool = True, log_tail: int = 200) -> dict[str, Any]:
        data = {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "state": self.state.value,
            "progress": round(self.progress, 4),
            "params": self.params,
            "stages": [s.to_dict() for s in self.stages],
            "current_stage": self.current_stage.name if self.current_stage else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round((self.finished_at or time.time()) - self.started_at, 1)
            if self.started_at
            else None,
            "error": self.error,
            "result": self.result,
        }
        if include_logs:
            data["logs"] = [line.to_dict() for line in self.logs[-log_tail:]]
        return data


class JobCancelled(Exception):
    """Raised inside a stage when the job has been cancelled."""


class JobContext:
    """What a stage function is handed: progress reporting, logging, cancellation."""

    def __init__(self, job: Job, manager: JobManager, workdir: Path):
        self.job = job
        self.workdir = workdir
        self._manager = manager
        self._stage: Stage | None = None

    # --- stage control --------------------------------------------------------

    def begin(self, stage: Stage) -> None:
        self._stage = stage
        stage.state = JobState.RUNNING
        stage.started_at = time.time()
        self.log(f"{stage.label} started")
        self._manager.publish(self.job)

    def progress(self, fraction: float, detail: str = "") -> None:
        if self._stage is None:
            return
        self._stage.progress = max(0.0, min(1.0, fraction))
        if detail:
            self._stage.detail = detail
        self._manager.publish(self.job)

    def log(self, text: str, level: str = "info") -> None:
        self.job.logs.append(LogLine(at=time.time(), level=level, text=text))
        # Keep memory bounded; the full log is on disk in the job directory.
        if len(self.job.logs) > 2000:
            del self.job.logs[:1000]
        log.info("[%s] %s", self.job.id, text)
        self._manager.publish(self.job)

    def check_cancelled(self) -> None:
        if self._manager.is_cancelled(self.job.id):
            raise JobCancelled(self.job.id)

    def result(self, **values: Any) -> None:
        self.job.result.update(values)
        self._manager.publish(self.job)


StageFn = Callable[[JobContext], Awaitable[None]]


class JobManager:
    """Runs jobs with bounded concurrency and fans status out to subscribers."""

    def __init__(self, jobs_dir: Path, max_concurrent: int = 1):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._global_subscribers: list[asyncio.Queue] = []
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent))

    # --- registry -------------------------------------------------------------

    def create(
        self, kind: str, title: str, stages: list[Stage], params: dict[str, Any]
    ) -> Job:
        job = Job(
            id=f"{kind}-{uuid.uuid4().hex[:8]}",
            kind=kind,
            title=title,
            stages=stages,
            params=params,
        )
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, *, kind: str | None = None, limit: int = 100) -> list[Job]:
        jobs = list(self._jobs.values())
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def workdir(self, job: Job) -> Path:
        path = self.jobs_dir / job.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --- execution ------------------------------------------------------------

    def start(self, job: Job, stage_fns: dict[str, StageFn]) -> Job:
        task = asyncio.create_task(self._run(job, stage_fns), name=f"job-{job.id}")
        self._tasks[job.id] = task
        return job

    async def _run(self, job: Job, stage_fns: dict[str, StageFn]) -> None:
        async with self._semaphore:
            job.state = JobState.RUNNING
            job.started_at = time.time()
            self.publish(job)
            ctx = JobContext(job, self, self.workdir(job))
            try:
                for stage in job.stages:
                    ctx.check_cancelled()
                    fn = stage_fns.get(stage.name)
                    if fn is None:
                        stage.state = JobState.SUCCEEDED
                        stage.progress = 1.0
                        stage.skipped = True
                        stage.detail = "no handler"
                        continue
                    ctx.begin(stage)
                    await fn(ctx)
                    stage.state = JobState.SUCCEEDED
                    stage.progress = 1.0
                    stage.finished_at = time.time()
                    ctx.log(f"{stage.label} finished in {stage.to_dict()['duration_s']}s")
                job.state = JobState.SUCCEEDED
            except JobCancelled:
                job.state = JobState.CANCELLED
                job.error = "cancelled by user"
                for stage in job.stages:
                    if stage.state == JobState.RUNNING:
                        stage.state = JobState.CANCELLED
                        stage.finished_at = time.time()
                ctx.log("job cancelled", level="warning")
            except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
                job.state = JobState.FAILED
                job.error = str(exc)
                for stage in job.stages:
                    if stage.state == JobState.RUNNING:
                        stage.state = JobState.FAILED
                        stage.finished_at = time.time()
                        stage.detail = str(exc)
                ctx.log(f"job failed: {exc}", level="error")
                ctx.log(traceback.format_exc(), level="debug")
            finally:
                job.finished_at = time.time()
                self._cancelled.discard(job.id)
                self._tasks.pop(job.id, None)
                self._persist(job)
                self.publish(job)
                self._close_subscribers(job.id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.state.terminal:
            return False
        self._cancelled.add(job_id)
        task = self._tasks.get(job_id)
        if task and job.state == JobState.PENDING:
            task.cancel()
        return True

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled

    def _persist(self, job: Job) -> None:
        path = self.workdir(job) / "job.json"
        with contextlib.suppress(OSError):
            path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

    # --- pub/sub --------------------------------------------------------------

    def publish(self, job: Job) -> None:
        payload = job.to_dict(log_tail=25)
        for queue in list(self._subscribers.get(job.id, [])):
            _offer(queue, payload)
        summary = job.to_dict(include_logs=False)
        for queue in list(self._global_subscribers):
            _offer(queue, summary)

    async def subscribe(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subscribers.setdefault(job_id, []).append(queue)
        job = self._jobs.get(job_id)
        if job:
            _offer(queue, job.to_dict(log_tail=200))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
                if item.get("state") in {"succeeded", "failed", "cancelled"}:
                    break
        finally:
            with contextlib.suppress(ValueError):
                self._subscribers.get(job_id, []).remove(queue)

    async def subscribe_all(self) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._global_subscribers.append(queue)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            with contextlib.suppress(ValueError):
                self._global_subscribers.remove(queue)

    def _close_subscribers(self, job_id: str) -> None:
        for queue in list(self._subscribers.pop(job_id, [])):
            _offer(queue, None)


def _offer(queue: asyncio.Queue, item: Any) -> None:
    """Publish without ever blocking the producer.

    A slow or abandoned subscriber must not stall a running job, so when its queue is
    full we drop the oldest update. Progress is a stream of snapshots, not a ledger:
    losing an intermediate frame is harmless because the next one supersedes it.
    """
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(item)
