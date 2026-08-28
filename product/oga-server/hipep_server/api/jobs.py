# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Job status endpoints.

Every long operation reports through the same three shapes: a list, a snapshot, and a
Server-Sent Events stream. The console uses the stream; scripts and CI poll the
snapshot; nothing needs a bespoke progress protocol.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .common import sse

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get("")
async def list_jobs(request: Request, kind: str | None = None, limit: int = 50):
    manager = request.app.state.jobs
    return {
        "jobs": [j.to_dict(include_logs=False) for j in manager.list(kind=kind, limit=limit)]
    }


@router.get("/events")
async def all_events(request: Request):
    """One stream carrying summaries of every job, for the console's activity bar."""
    manager = request.app.state.jobs

    async def stream() -> AsyncIterator[str]:
        for job in manager.list(limit=20):
            yield sse(job.to_dict(include_logs=False))
        async for update in manager.subscribe_all():
            if await request.is_disconnected():
                break
            yield sse(update)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}")
async def get_job(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job '{job_id}'")
    return job.to_dict()


@router.get("/{job_id}/events")
async def job_events(request: Request, job_id: str):
    manager = request.app.state.jobs
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"no job '{job_id}'")

    async def stream() -> AsyncIterator[str]:
        async for update in manager.subscribe(job_id):
            yield sse(update)
        yield sse("[DONE]")

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str):
    manager = request.app.state.jobs
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"no job '{job_id}'")
    if not manager.cancel(job_id):
        raise HTTPException(status_code=409, detail="job has already finished")
    return {"cancelled": job_id}
