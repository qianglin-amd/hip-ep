# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Sharing a model artifact between machines.

A user who ports a model on one box wants it on another without repeating a
half-hour quantization. The port job hands out a share URL; these endpoints are what
makes that URL mean something:

* `GET /v1/artifacts/{id}` returns a manifest: the files, their sizes and their
  digests, plus the metadata needed to reconstruct the catalog entry.
* `GET /v1/artifacts/{id}/files/{path}` serves one file.
* `POST /v1/import` starts a job on *this* server that reads that manifest from a
  peer and installs the model here.

Per-file transfer rather than one archive is deliberate. A model directory is tens of
gigabytes, so the server must never hold the payload in memory, and an interrupted
transfer should resume at the file it stopped on instead of at zero.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse

from ..store import SIDECAR
from .common import get_catalog, get_store

router = APIRouter(prefix="/v1", tags=["artifacts"])

# Hashing 30 GB to answer a manifest request would take minutes, so digests are only
# computed for files small enough to be cheap. The importer verifies sizes always and
# digests where it has them, which catches truncation without the cost.
DIGEST_LIMIT_BYTES = 64 * 1024 * 1024


def _digest(path: Path) -> str | None:
    if path.stat().st_size > DIGEST_LIMIT_BYTES:
        return None
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _local_or_404(request: Request, model_id: str):
    local = get_store(request).read(model_id)
    if local is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{model_id}' is not installed on this server, so it cannot be shared",
        )
    return local


@router.get("/artifacts/{model_id:path}/files/{relative:path}")
async def artifact_file(request: Request, model_id: str, relative: str):
    local = _local_or_404(request, model_id)
    root = Path(local.path).resolve()
    target = (root / relative).resolve()
    # A client controls `relative`, so confirm the resolved path is still inside the
    # model directory before opening it.
    if not target.is_file() or root not in target.parents:
        raise HTTPException(status_code=404, detail=f"no file '{relative}' in {model_id}")
    return FileResponse(target, media_type="application/octet-stream", filename=target.name)


@router.get("/artifacts/{model_id:path}")
async def artifact_manifest(request: Request, model_id: str) -> dict[str, Any]:
    local = _local_or_404(request, model_id)
    root = Path(local.path).resolve()

    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == SIDECAR:
            # The sidecar describes the install rather than being part of it; the
            # importer writes its own from this manifest.
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
        )

    entry = get_catalog(request).get(local.id)
    return {
        "model": local.id,
        "origin": local.origin,
        "source": local.source,
        "recipe": local.recipe,
        "ready": local.ready,
        "extra": local.extra,
        "files": files,
        "total_bytes": sum(f["size_bytes"] for f in files),
        "catalog": entry.to_dict() if entry else None,
        "server": {"base_url": request.app.state.settings.base_url},
    }


@router.post("/import")
async def import_artifact(request: Request, body: dict[str, Any] = Body(...)):
    """Install a model from a peer oga-server. Returns a job to follow."""
    url = (body.get("url") or "").strip()
    model_id = (body.get("model") or "").strip()
    if not url:
        raise HTTPException(
            status_code=400,
            detail="url is required: the base address of the server holding the model",
        )
    if not model_id:
        raise HTTPException(status_code=400, detail="model is required")
    job = request.app.state.pipeline.start_import(
        url=url, model_id=model_id, name=(body.get("name") or "").strip() or None
    )
    return {"job": job.to_dict(include_logs=False)}
