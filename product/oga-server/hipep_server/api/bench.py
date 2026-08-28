# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Benchmark endpoints.

`/v1/benchmarks/published` serves the certified results shipped in the repo.
`/v1/benchmarks/run` starts a measurement job on this machine, which the console then
plots against the published numbers so a user can see immediately whether their box
performs the way the certification host did.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from .common import get_catalog, resolve_model

router = APIRouter(prefix="/v1/benchmarks", tags=["benchmarks"])


@router.get("/published")
async def published(request: Request, tier: str | None = None):
    catalog = get_catalog(request)
    rows = []
    for model in catalog.list(tier=tier):
        if not model.benchmark:
            continue
        rows.append(
            {
                "model": model.id,
                "name": model.name,
                "tier": model.tier,
                "quantization": model.quant_label,
                "status": model.status,
                "certification": model.certification,
                **model.benchmark.to_dict(),
            }
        )
    return {"meta": catalog.benchmark_meta, "results": rows}


@router.get("/local")
async def local_results(request: Request):
    """Benchmark runs measured on this machine."""
    results_dir = request.app.state.settings.results_dir
    runs: list[dict[str, Any]] = []
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("*.json"), reverse=True):
            try:
                runs.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return {"runs": runs}


@router.post("/run")
async def run(request: Request, body: dict[str, Any] = Body(default={})):
    model = resolve_model(request, body.get("model", ""))
    contexts = body.get("contexts") or [128, 2048]
    if not isinstance(contexts, list) or not all(
        isinstance(c, int) and c > 0 for c in contexts
    ):
        raise HTTPException(
            status_code=400, detail="contexts must be a list of positive integers"
        )
    over = [c for c in contexts if c > model.max_context]
    if over:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{model.id} supports at most {model.max_context} tokens; requested {over}"
            ),
        )
    spec = {
        "contexts": contexts,
        "generate_tokens": int(body.get("generate_tokens", 64)),
        "repeats": int(body.get("repeats", 3)),
        "warmup": int(body.get("warmup", 1)),
    }
    job = request.app.state.pipeline.start_bench(model, spec)
    return {"job": job.to_dict(include_logs=False)}
