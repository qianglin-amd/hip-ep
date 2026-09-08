# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Ollama-compatible API.

The server listens on 11434 by default, so an existing Ollama client works against
HIP EP with no configuration at all. That is the cheapest possible on-ramp: every tool
in the local-LLM ecosystem already speaks this protocol.

Responses are newline-delimited JSON rather than SSE, and durations are nanoseconds,
because that is what Ollama clients parse.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import __version__
from ..engines import ChatMessage, EngineError, GenerationRequest, ModelNotAvailable
from ..engines.tools import ToolCall
from ..scheduler import QueueFull
from .common import (
    build_params,
    client_address,
    get_catalog,
    get_scheduler,
    get_store,
    ndjson,
    open_session,
    parse_ollama_messages,
    parse_tools,
    reject_unsupported_images,
    resolve_model,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ollama"])

NS = 1_000_000_000


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _tag(model, size_bytes: int = 0) -> dict[str, Any]:
    families = [model.family] if model.family else []
    return {
        "name": model.id,
        "model": model.id,
        "modified_at": _iso(time.time()),
        "size": size_bytes,
        "digest": f"hipep:{model.id}",
        "details": {
            "parent_model": model.base_model or "",
            "format": "onnx",
            "family": model.family or "",
            "families": families,
            "parameter_size": model.parameters or "",
            "quantization_level": model.quant_label,
        },
    }


@router.get("/version")
async def version():
    # Ollama clients gate features on this. Report a modern baseline plus our own.
    return {"version": "0.5.0", "hipep_version": __version__}


@router.get("/tags")
async def tags(request: Request):
    """Locally available models. Ollama clients treat this as "what can I run now".

    We list every certified generative model, and mark which are actually downloaded,
    because the console's model manager and `ollama list` both want the full picture.
    """
    catalog = get_catalog(request)
    store = get_store(request)
    models = []
    for model in catalog.list(generative_only=True):
        local = store.read(model.id)
        entry = _tag(model, local.size_bytes if local else 0)
        entry["hipep"] = {
            "installed": local is not None,
            "ready": bool(local and local.ready),
            "status": model.status,
        }
        models.append(entry)
    return {"models": models}


@router.get("/ps")
async def ps(request: Request):
    """Models currently resident in memory."""
    scheduler = get_scheduler(request)
    catalog = get_catalog(request)
    store = get_store(request)
    running = []
    for model_id in scheduler.engine.loaded_models():
        model = catalog.get(model_id)
        if not model:
            continue
        local = store.read(model_id)
        entry = _tag(model, local.size_bytes if local else 0)
        entry["size_vram"] = int((model.min_vram_gb or 0) * 1024**3)
        entry["expires_at"] = _iso(time.time() + scheduler.settings.idle_unload_seconds)
        running.append(entry)
    return {"models": running}


@router.post("/show")
async def show(request: Request):
    body = await request.json()
    model = resolve_model(
        request, body.get("model") or body.get("name") or "", generative=False
    )
    store = get_store(request)
    local = store.read(model.id)
    return {
        "license": model.license or "",
        "modelfile": f"# {model.name}\nFROM {model.base_model or model.id}\n",
        "parameters": f"num_ctx {model.default_context}",
        "template": "",
        "details": _tag(model, local.size_bytes if local else 0)["details"],
        "model_info": {
            "general.architecture": model.family or "",
            "general.parameter_count": model.parameters or "",
            "general.quantization": model.quant_label,
            "context.length": model.max_context,
        },
        "capabilities": [k for k, v in (model.capabilities or {}).items() if v],
        "hipep": model.to_dict(),
    }


@router.post("/chat")
async def chat(request: Request):
    body = await request.json()
    model = resolve_model(request, body.get("model") or "")
    messages = parse_ollama_messages(body.get("messages") or [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    reject_unsupported_images(model, messages)

    scheduler = get_scheduler(request)
    params = build_params(
        body, default_max_tokens=request.app.state.settings.default_max_tokens, model=model
    )
    session = open_session(request, "ollama")
    tools, tool_choice = parse_tools(body)
    gen = GenerationRequest(
        model=model,
        messages=messages,
        params=params,
        session_id=session.id,
        tools=tools,
        tool_choice=tool_choice,
    )
    stream = body.get("stream", True)

    if stream:
        return StreamingResponse(
            _stream_ollama(
                scheduler, gen, session, model.id, client=client_address(request), chat=True
            ),
            media_type="application/x-ndjson",
        )

    parts, usage, calls, reason = await _collect(
        scheduler, gen, session, client_address(request)
    )
    message: dict[str, Any] = {"role": "assistant", "content": "".join(parts)}
    if calls:
        message["tool_calls"] = [call.to_ollama() for call in calls]

    return JSONResponse(
        {
            "model": model.id,
            "created_at": _iso(time.time()),
            "message": message,
            "done": True,
            "done_reason": reason,
            **_durations(usage),
        }
    )


@router.post("/generate")
async def generate(request: Request):
    body = await request.json()
    model = resolve_model(request, body.get("model") or "")
    images = body.get("images") or []
    messages: list[ChatMessage] = []
    if body.get("system"):
        messages.append(ChatMessage(role="system", content=str(body["system"])))
    from .common import decode_image

    messages.append(
        ChatMessage(
            role="user",
            content=str(body.get("prompt") or ""),
            images=[decode_image(v) for v in images],
        )
    )
    reject_unsupported_images(model, messages)

    scheduler = get_scheduler(request)
    params = build_params(
        body, default_max_tokens=request.app.state.settings.default_max_tokens, model=model
    )
    session = open_session(request, "ollama")
    gen = GenerationRequest(
        model=model, messages=messages, params=params, session_id=session.id
    )

    if body.get("stream", True):
        return StreamingResponse(
            _stream_ollama(
                scheduler, gen, session, model.id, client=client_address(request), chat=False
            ),
            media_type="application/x-ndjson",
        )

    # /api/generate has no tool surface, so the calls and reason are not used here.
    parts, usage, _, _ = await _collect(scheduler, gen, session, client_address(request))
    return JSONResponse(
        {
            "model": model.id,
            "created_at": _iso(time.time()),
            "response": "".join(parts),
            "done": True,
            "done_reason": "stop",
            **_durations(usage),
        }
    )


@router.post("/pull")
async def pull(request: Request):
    """Download a certified model, streaming Ollama-style progress frames."""
    body = await request.json()
    name = body.get("model") or body.get("name") or ""
    model = resolve_model(request, name, generative=False)
    pipeline = request.app.state.pipeline
    job = pipeline.start_pull(model)

    if not body.get("stream", True):
        return {"status": "started", "job": job.id}

    async def frames() -> AsyncIterator[str]:
        manager = request.app.state.jobs
        yield ndjson({"status": f"pulling {model.id}"})
        async for snapshot in manager.subscribe(job.id):
            stage = snapshot.get("current_stage") or "working"
            total = snapshot.get("result", {}).get("total_bytes")
            completed = snapshot.get("result", {}).get("completed_bytes")
            frame: dict[str, Any] = {"status": stage}
            if total:
                frame["total"] = total
                frame["completed"] = completed or 0
            yield ndjson(frame)
            if snapshot.get("state") == "succeeded":
                yield ndjson({"status": "success"})
            elif snapshot.get("state") in {"failed", "cancelled"}:
                yield ndjson({"error": snapshot.get("error") or "pull failed"})

    return StreamingResponse(frames(), media_type="application/x-ndjson")


@router.delete("/delete")
async def delete(request: Request):
    body = await request.json()
    model = resolve_model(
        request, body.get("model") or body.get("name") or "", generative=False
    )
    store = get_store(request)
    if not store.remove(model.id):
        raise HTTPException(status_code=404, detail=f"{model.id} is not installed")
    return {"status": "success"}


# --- helpers ------------------------------------------------------------------


def _durations(usage: dict[str, Any]) -> dict[str, Any]:
    """Translate our usage block into Ollama's nanosecond duration fields."""
    ttft = usage.get("ttft_s", 0.0) or 0.0
    decode = usage.get("decode_s", 0.0) or 0.0
    load = usage.get("load_s", 0.0) or 0.0
    total = usage.get("total_s", ttft + decode) or 0.0
    return {
        # The engine's total excludes the load it did not perform, but an Ollama
        # client expects total_duration to cover the whole request.
        "total_duration": int((total + load) * NS),
        "load_duration": int(load * NS),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "prompt_eval_duration": int(ttft * NS),
        "eval_count": usage.get("completion_tokens", 0),
        "eval_duration": int(decode * NS),
    }


async def _collect(
    scheduler, gen, session, client
) -> tuple[list[str], dict[str, Any], list[ToolCall], str]:
    parts: list[str] = []
    usage: dict[str, Any] = {}
    calls: list[ToolCall] = []
    reason = "stop"
    try:
        async for delta in scheduler.stream(gen, session=session, api="ollama", client=client):
            if delta.text:
                parts.append(delta.text)
            if delta.tool_calls:
                calls.extend(delta.tool_calls)
            if delta.done:
                usage = delta.usage or usage
                reason = delta.finish_reason or reason
    except QueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelNotAvailable as exc:
        # Ollama clients treat 404 on a chat as "pull it first", which is the fix.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EngineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return parts, usage, calls, reason


async def _stream_ollama(
    scheduler, gen, session, model_id: str, *, client: str, chat: bool
) -> AsyncIterator[str]:
    def payload(text: str) -> dict[str, Any]:
        base = {"model": model_id, "created_at": _iso(time.time()), "done": False}
        if chat:
            base["message"] = {"role": "assistant", "content": text}
        else:
            base["response"] = text
        return base

    try:
        async for delta in scheduler.stream(gen, session=session, api="ollama", client=client):
            if delta.text:
                yield ndjson(payload(delta.text))
            if delta.tool_calls and chat:
                frame = payload("")
                frame["message"]["tool_calls"] = [c.to_ollama() for c in delta.tool_calls]
                yield ndjson(frame)
            if delta.done:
                final = payload("")
                final.update({"done": True, "done_reason": delta.finish_reason or "stop"})
                final.update(_durations(delta.usage or {}))
                yield ndjson(final)
    except (QueueFull, EngineError) as exc:
        yield ndjson({"model": model_id, "error": str(exc), "done": True})
