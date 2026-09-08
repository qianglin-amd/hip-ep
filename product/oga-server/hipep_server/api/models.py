# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Model catalog, local store and porting endpoints.

This is the console's view of models, which is richer than either the OpenAI or the
Ollama model list: it joins the certified catalog with what is installed locally, with
the published benchmark numbers, and with whether the model fits on this machine.
"""

from __future__ import annotations

import json
from pathlib import Path
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
    # An imported model exists only because it was installed, so deleting the files
    # must also retire the catalog entry. Certified entries are left alone.
    removed_entry = get_catalog(request).unregister(model_id)
    return {"deleted": model_id, "catalog_entry_removed": removed_entry}


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
    from ..recipes import RecipeError, find_recipe, recipe_architectures, recipe_plan

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

    # The console sends the chosen recipe's id as the scheme, so a recipe is identified
    # by whichever of the two fields names one. Reading its quantization block is what
    # makes the choice mean anything: parsing settings out of the id instead lost the
    # block size that several of these recipes exist to change.
    chosen = body.get("recipe") or spec["quantization"]
    recipe = find_recipe(request.app.state.settings.recipes_dir, str(chosen))
    if recipe:
        try:
            plan = recipe_plan(recipe)
        except RecipeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        spec["quantization"] = recipe["id"]
        spec["recipe"] = recipe["id"]
        spec["plan"] = plan
        spec["expects_architectures"] = recipe_architectures(recipe)
        if "chunk_size" in plan and "chunk_size" not in body:
            spec["chunk_size"] = plan["chunk_size"]
        if "prune_logit" not in body:
            spec["prune_logit"] = plan["prune_logit"]
    elif body.get("recipe"):
        raise HTTPException(status_code=404, detail=f"no recipe named {chosen!r}")
    job = request.app.state.pipeline.start_port(spec)
    return {"job": job.to_dict(include_logs=False)}


@router.get("/recipes")
async def recipes(request: Request):
    """Quantization recipes available for porting a model."""
    from ..recipes import list_recipes

    return {"recipes": list_recipes(request.app.state.settings.recipes_dir)}


@router.get("/architectures")
async def architectures(request: Request):
    """What is known about porting each model architecture."""
    from .. import architectures as kb

    recipes_dir = request.app.state.settings.recipes_dir
    return {"architectures": kb.architectures(recipes_dir), "rules": kb.rules()}


@router.post("/port/preflight")
async def port_preflight(request: Request, body: dict[str, Any] = Body(...)):
    """Say what porting a model would do, before fetching any weights.

    Reads the model's `config.json` and nothing else, which is a few kilobytes against
    the tens of gigabytes a port downloads. Answering "this will not work, and here is
    why" after that download is a far worse answer than answering it now.
    """
    from .. import architectures as kb

    source = (body.get("source") or "").strip()
    if not source:
        raise HTTPException(status_code=400, detail="source is required")

    try:
        config = await _read_model_config(source)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - reported to the caller as-is
        raise HTTPException(
            status_code=502, detail=f"could not read the model's config: {exc}"
        ) from exc

    verdict = kb.assess(config, request.app.state.settings.recipes_dir)
    verdict["source"] = source
    return verdict


async def _read_model_config(source: str) -> dict[str, Any]:
    """The `config.json` of a local directory or a Hugging Face repository."""
    import asyncio

    local = Path(source) / "config.json"
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))

    def fetch() -> dict[str, Any]:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import RepositoryNotFoundError

        try:
            path = hf_hub_download(source, "config.json")
        except RepositoryNotFoundError as exc:
            # Hugging Face answers 404 for a repository that does not exist and for one
            # that exists behind a licence the caller has not accepted, so this cannot
            # tell them apart and must not claim to.
            raise FileNotFoundError(
                f"could not read {source!r}: it does not exist, or it is gated and this "
                "machine is not authenticated. If it is gated, accept the licence on "
                "the model page and run `huggingface-cli login`."
            ) from exc
        return json.loads(Path(path).read_text(encoding="utf-8"))

    return await asyncio.to_thread(fetch)
