# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""OpenAI-compatible API.

Implements the endpoints that the official OpenAI SDKs, LangChain, LlamaIndex,
Continue, and most chat front ends actually call. Compatibility is the point: an
application should be able to change base_url and nothing else.

Two deliberate extensions, both additive and ignorable:

* `usage` on chat responses carries `ttft_s`, `decode_s` and `tps` alongside the
  standard token counts, because a local inference server that cannot tell you its
  latency is not much use.
* streaming chunks carry the same `usage` object on the final frame.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..engines import ChatMessage, EngineError, GenerationRequest
from ..scheduler import QueueFull
from .common import (
    build_params,
    client_address,
    get_catalog,
    get_scheduler,
    open_session,
    parse_openai_messages,
    reject_unsupported_images,
    resolve_model,
    sse,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai"])


def _model_object(model, owned_by: str = "amd") -> dict[str, Any]:
    return {
        "id": model.id,
        "object": "model",
        "created": 0,
        "owned_by": owned_by,
        # Non-standard but harmless, and it saves clients a second round trip.
        "hipep": {
            "name": model.name,
            "tier": model.tier,
            "type": model.type,
            "quantization": model.quant_label,
            "context_length": model.max_context,
            "capabilities": model.capabilities,
            "status": model.status,
        },
    }


@router.get("/models")
async def list_models(
    request: Request, capability: str | None = None, tier: str | None = None
):
    catalog = get_catalog(request)
    models = catalog.list(tier=tier)
    if capability:
        models = [m for m in models if m.capabilities.get(capability)]
    return {"object": "list", "data": [_model_object(m) for m in models]}


@router.get("/models/{model_id:path}")
async def retrieve_model(request: Request, model_id: str):
    model = resolve_model(request, model_id, generative=False)
    return _model_object(model)


@router.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = resolve_model(request, body.get("model", ""))
    messages = parse_openai_messages(body.get("messages") or [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    reject_unsupported_images(model, messages)

    scheduler = get_scheduler(request)
    params = build_params(
        body, default_max_tokens=request.app.state.settings.default_max_tokens, model=model
    )
    session = open_session(request, "openai")
    gen = GenerationRequest(
        model=model, messages=messages, params=params, session_id=session.id
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    stream = bool(body.get("stream"))

    if stream:
        return StreamingResponse(
            _stream_chat(
                scheduler,
                gen,
                session,
                completion_id,
                created,
                model.id,
                client=client_address(request),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-HIPEP-Session": session.id,
            },
        )

    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    try:
        async for delta in scheduler.stream(
            gen, session=session, api="openai", client=client_address(request)
        ):
            if delta.text:
                text_parts.append(delta.text)
            if delta.done and delta.usage:
                usage = delta.usage
    except QueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EngineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model.id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(text_parts)},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        },
        headers={"X-HIPEP-Session": session.id},
    )


async def _stream_chat(
    scheduler,
    gen: GenerationRequest,
    session,
    completion_id: str,
    created: int,
    model_id: str,
    *,
    client: str,
) -> AsyncIterator[str]:
    def frame(delta: dict[str, Any], finish: str | None = None, usage=None) -> str:
        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage:
            payload["usage"] = usage
        return sse(payload)

    yield frame({"role": "assistant", "content": ""})
    try:
        async for delta in scheduler.stream(gen, session=session, api="openai", client=client):
            if delta.text:
                yield frame({"content": delta.text})
            if delta.done:
                yield frame({}, finish=delta.finish_reason or "stop", usage=delta.usage)
    except QueueFull as exc:
        yield sse({"error": {"message": str(exc), "type": "server_overloaded"}})
    except EngineError as exc:
        log.exception("stream failed")
        yield sse({"error": {"message": str(exc), "type": "engine_error"}})
    yield sse("[DONE]")


@router.post("/completions")
async def completions(request: Request):
    """Legacy text completion, mapped onto the chat path."""
    body = await request.json()
    model = resolve_model(request, body.get("model", ""))
    prompt = body.get("prompt") or ""
    if isinstance(prompt, list):
        prompt = "\n".join(str(p) for p in prompt)

    scheduler = get_scheduler(request)
    params = build_params(
        body, default_max_tokens=request.app.state.settings.default_max_tokens, model=model
    )
    session = open_session(request, "openai")
    gen = GenerationRequest(
        model=model,
        messages=[ChatMessage(role="user", content=str(prompt))],
        params=params,
        session_id=session.id,
    )

    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if body.get("stream"):

        async def gen_stream() -> AsyncIterator[str]:
            try:
                async for delta in scheduler.stream(
                    gen, session=session, api="openai", client=client_address(request)
                ):
                    if delta.text:
                        yield sse(
                            {
                                "id": completion_id,
                                "object": "text_completion",
                                "created": created,
                                "model": model.id,
                                "choices": [
                                    {"index": 0, "text": delta.text, "finish_reason": None}
                                ],
                            }
                        )
                    if delta.done:
                        yield sse(
                            {
                                "id": completion_id,
                                "object": "text_completion",
                                "created": created,
                                "model": model.id,
                                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                                "usage": delta.usage,
                            }
                        )
            except (QueueFull, EngineError) as exc:
                yield sse({"error": {"message": str(exc)}})
            yield sse("[DONE]")

        return StreamingResponse(gen_stream(), media_type="text/event-stream")

    parts: list[str] = []
    usage: dict[str, Any] = {}
    try:
        async for delta in scheduler.stream(
            gen, session=session, api="openai", client=client_address(request)
        ):
            if delta.text:
                parts.append(delta.text)
            if delta.done and delta.usage:
                usage = delta.usage
    except QueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EngineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model.id,
        "choices": [{"index": 0, "text": "".join(parts), "finish_reason": "stop"}],
        "usage": usage,
    }


@router.post("/embeddings")
async def embeddings(request: Request):
    """Not implemented, and explicit about it.

    Returning zeros here would let a retrieval pipeline build an index of meaningless
    vectors and fail silently much later, so this refuses instead.
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "embeddings are not implemented yet. No embedding model is certified for "
            "HIP EP; this endpoint is planned for P2. Track it in product/MILESTONES.md."
        ),
    )
