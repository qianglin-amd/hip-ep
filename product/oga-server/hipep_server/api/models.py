# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Model catalog, local store and porting endpoints.

This is the console's view of models, which is richer than either the OpenAI or the
Ollama model list: it joins the certified catalog with what is installed locally, with
the published benchmark numbers, and with whether the model fits on this machine.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from ..hardware import fits_on_device
from .common import get_catalog, get_store, resolve_model

router = APIRouter(prefix="/v1", tags=["models"])


def _enrich(request: Request, model) -> dict[str, Any]:
    store = get_store(request)
    host = request.app.state.host
    local = store.read(model.id)
    data = model.to_dict()
    data["local"] = local.to_dict() if local else None
    data["installed"] = local is not None
    data["fits"] = fits_on_device(model.min_vram_gb, host)
    return data


@router.get("/catalog")
async def catalog(request: Request, tier: str | None = None, q: str | None = None):
    """The full model zoo, joined with local state and device fit."""
    cat = get_catalog(request)
    models = cat.list(tier=tier)
    if q:
        needle = q.lower()
        models = [
            m
            for m in models
            if needle in m.id.lower()
            or needle in m.name.lower()
            or needle in (m.family or "").lower()
            or needle in (m.base_model or "").lower()
        ]
    return {
        "models": [_enrich(request, m) for m in models],
        "tiers": cat.tiers(),
        "benchmark": cat.benchmark_meta,
        "counts": {tier_: len(cat.list(tier=tier_)) for tier_ in cat.tiers()},
    }


@router.get("/catalog/{model_id:path}")
async def catalog_entry(request: Request, model_id: str):
    model = resolve_model(request, model_id, generative=False)
    return _enrich(request, model)


@router.get("/local")
async def local_models(request: Request):
    store = get_store(request)
    models = store.list()
    return {
        "models": [m.to_dict() for m in models],
        "total_bytes": sum(m.size_bytes for m in models),
        "root": str(store.root),
    }


@router.delete("/local/{model_id:path}")
async def delete_local(request: Request, model_id: str):
    store = get_store(request)
    if not store.remove(model_id):
        raise HTTPException(status_code=404, detail=f"{model_id} is not installed")
    return {"deleted": model_id}


@router.post("/pull")
async def pull(request: Request, body: dict[str, Any] = Body(...)):
    """Start a pull job for a certified model. Returns immediately with the job id."""
    model = resolve_model(request, body.get("model", ""), generative=False)
    job = request.app.state.pipeline.start_pull(model)
    return {"job": job.to_dict(include_logs=False)}


@router.post("/port")
async def port(request: Request, body: dict[str, Any] = Body(...)):
    """Start a bring-your-own-model job.

    Accepts a Hugging Face repo id, a URL or a local path, plus the quantization plan.
    Quark defaults apply when the caller does not override them: chunk_size 512 and
    logit pruning on.
    """
    source = (body.get("source") or "").strip()
    if not source:
        raise HTTPException(
            status_code=400,
            detail="source is required: a Hugging Face repo id, a URL, or a local path",
        )
    spec = {
        "source": source,
        "name": body.get("name"),
        "quantization": body.get("quantization", "int4-rtn-g128"),
        "chunk_size": int(body.get("chunk_size", 512)),
        "prune_logit": bool(body.get("prune_logit", True)),
        "calibration_samples": int(body.get("calibration_samples", 128)),
        "device": body.get("device", "auto"),
        "benchmark": bool(body.get("benchmark", True)),
    }
    job = request.app.state.pipeline.start_port(spec)
    return {"job": job.to_dict(include_logs=False)}


@router.get("/recipes")
async def recipes(request: Request):
    """Quantization recipes available for porting a model."""
    from ..recipes import list_recipes

    return {"recipes": list_recipes(request.app.state.settings.recipes_dir)}
