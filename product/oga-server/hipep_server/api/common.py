# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Shared request plumbing for the API routers."""

from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Any

from fastapi import HTTPException, Request

from ..catalog import Catalog, Model
from ..engines import ChatMessage, GenerationParams
from ..scheduler import Scheduler
from ..store import ModelStore

DATA_URI = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.*)$", re.DOTALL)

# Guard against a browser or a script pushing an unreasonable image through the JSON
# body. 24 MB decoded is far beyond any sensible input for the certified VLMs.
MAX_IMAGE_BYTES = 24 * 1024 * 1024


def get_catalog(request: Request) -> Catalog:
    return request.app.state.catalog


def get_scheduler(request: Request) -> Scheduler:
    return request.app.state.scheduler


def get_store(request: Request) -> ModelStore:
    return request.app.state.store


def client_address(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def open_session(request: Request, api: str):
    """Find or create the caller's session.

    Clients that send an explicit session header keep one session across requests;
    everyone else is bucketed by address and user agent, which is what makes the
    server status page show "three users" rather than "forty requests".
    """
    scheduler = get_scheduler(request)
    explicit = request.headers.get("x-hipep-session")
    client = client_address(request)
    user_agent = request.headers.get("user-agent", "unknown")
    session_id = explicit or f"s-{abs(hash((client, user_agent))) % 10**10:010d}"
    return scheduler.session(session_id, client=client, user_agent=user_agent, api=api)


def resolve_model(request: Request, model_id: str, *, generative: bool = True) -> Model:
    catalog = get_catalog(request)
    model = catalog.resolve(model_id)
    if model is None:
        known = ", ".join(m.id for m in catalog.list(generative_only=True)[:8])
        raise HTTPException(
            status_code=404,
            detail=(
                f"model '{model_id}' is not in the catalog. "
                f"Try one of: {known}. GET /v1/models lists them all."
            ),
        )
    if generative and not model.is_generative:
        raise HTTPException(
            status_code=400,
            detail=(
                f"model '{model.id}' is a {model.tier} model and does not support chat. "
                "Use /v1/audio/transcriptions or /v1/vision/infer instead."
            ),
        )
    return model


def decode_image(value: str) -> bytes:
    """Accept a data URI or a bare base64 payload and return raw image bytes."""
    match = DATA_URI.match(value.strip())
    payload = match.group("data") if match else value.strip()
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "image must be a base64 data URI or a base64 string. "
                "Remote image URLs are not fetched by the server."
            ),
        ) from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image is {len(data) // (1024 * 1024)} MB; the limit is "
            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB",
        )
    return data


def parse_openai_messages(raw: list[dict[str, Any]]) -> list[ChatMessage]:
    """Convert OpenAI chat messages, including multimodal content parts."""
    messages: list[ChatMessage] = []
    for item in raw:
        role = item.get("role", "user")
        content = item.get("content")
        if isinstance(content, str) or content is None:
            messages.append(ChatMessage(role=role, content=content or ""))
            continue
        text_parts: list[str] = []
        images: list[bytes] = []
        for part in content:
            if not isinstance(part, dict):
                text_parts.append(str(part))
                continue
            kind = part.get("type")
            if kind == "text":
                text_parts.append(part.get("text", ""))
            elif kind == "image_url":
                url = part.get("image_url", {})
                url = url.get("url") if isinstance(url, dict) else url
                if url:
                    images.append(decode_image(url))
            elif kind in {"input_image", "image"}:
                data = part.get("image") or part.get("data") or part.get("image_url")
                if data:
                    images.append(decode_image(data))
        messages.append(
            ChatMessage(
                role=role, content="\n".join(p for p in text_parts if p), images=images
            )
        )
    return messages


def parse_ollama_messages(raw: list[dict[str, Any]]) -> list[ChatMessage]:
    """Convert Ollama chat messages, whose images are a flat base64 list."""
    messages: list[ChatMessage] = []
    for item in raw:
        images = [decode_image(v) for v in (item.get("images") or [])]
        messages.append(
            ChatMessage(
                role=item.get("role", "user"),
                content=item.get("content", "") or "",
                images=images,
            )
        )
    return messages


def reject_unsupported_images(model: Model, messages: list[ChatMessage]) -> None:
    if any(m.images for m in messages) and not model.supports_vision:
        raise HTTPException(
            status_code=400,
            detail=(
                f"model '{model.id}' is text-only. "
                "Pick a vision model such as gemma3-vl:4b or qwen3.6-vl:35b-a3b; "
                "GET /v1/models?capability=vision lists them."
            ),
        )


def build_params(
    body: dict[str, Any], *, default_max_tokens: int, model: Model
) -> GenerationParams:
    options = body.get("options") or {}

    def pick(*names: str, default: Any = None) -> Any:
        for name in names:
            if body.get(name) is not None:
                return body[name]
            if options.get(name) is not None:
                return options[name]
        return default

    max_tokens = pick("max_tokens", "max_completion_tokens", "num_predict", default=None)
    if max_tokens is None:
        max_tokens = default_max_tokens
    max_tokens = max(1, min(int(max_tokens), model.max_context))

    stop = pick("stop", default=[]) or []
    if isinstance(stop, str):
        stop = [stop]

    return GenerationParams(
        max_tokens=max_tokens,
        temperature=float(pick("temperature", default=0.7)),
        top_p=float(pick("top_p", default=0.9)),
        top_k=int(pick("top_k", default=50)),
        repetition_penalty=float(pick("repetition_penalty", "repeat_penalty", default=1.0)),
        stop=[str(s) for s in stop],
        seed=pick("seed"),
        think=body.get("think"),
    )


def _dumps(data: Any) -> str:
    # `default` is a safety net, not a convenience: an unserializable value inside a
    # long-lived stream would otherwise abort the response mid-body, which a browser
    # reports only as a truncated connection.
    return json.dumps(data, separators=(",", ":"), default=str)


def sse(data: Any) -> str:
    """Format one Server-Sent Events frame."""
    if isinstance(data, str):
        return f"data: {data}\n\n"
    return f"data: {_dumps(data)}\n\n"


def ndjson(data: Any) -> str:
    """Format one newline-delimited JSON frame, which is what Ollama clients read."""
    return _dumps(data) + "\n"
