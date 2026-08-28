# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Server status, health and hardware reporting.

The status payload answers the questions a user actually asks when a local inference
server misbehaves: what engine am I on and why, what is loaded, who else is using it,
how deep is the queue, and is this machine capable of the model I picked.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .. import __version__
from .common import sse

router = APIRouter(tags=["status"])


def _process_stats() -> dict:
    try:
        import psutil

        proc = psutil.Process()
        virtual = psutil.virtual_memory()
        return {
            "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
            "memory_used_gb": round((virtual.total - virtual.available) / 1024**3, 2),
            "memory_total_gb": round(virtual.total / 1024**3, 2),
            "process_memory_gb": round(proc.memory_info().rss / 1024**3, 2),
        }
    except Exception:  # noqa: BLE001 - stats are decoration, never a failure mode
        return {}


def _status_payload(request: Request) -> dict:
    app = request.app
    scheduler = app.state.scheduler
    store = app.state.store
    snapshot = scheduler.snapshot()
    snapshot.update(
        {
            "version": __version__,
            "engine_reason": app.state.engine_reason,
            "host": app.state.host.to_dict(),
            "process": _process_stats(),
            "storage": {
                "root": str(store.root),
                "models": len(store.list()),
                "bytes": store.total_bytes(),
            },
            "catalog": {
                "models": len(app.state.catalog),
                "release": app.state.catalog.benchmark_meta.get("release"),
                "run_date": app.state.catalog.benchmark_meta.get("run_date"),
            },
            "jobs": {
                "running": len(
                    [j for j in app.state.jobs.list() if j.state.value == "running"]
                ),
                "total": len(app.state.jobs.list(limit=10_000)),
            },
            "now": time.time(),
        }
    )
    return snapshot


@router.get("/v1/status")
async def status(request: Request):
    return _status_payload(request)


@router.get("/v1/status/events")
async def status_events(request: Request, interval: float = 1.0):
    """Poll-free status for the console's server page."""
    period = max(0.25, min(float(interval), 10.0))

    async def stream() -> AsyncIterator[str]:
        while True:
            if await request.is_disconnected():
                break
            yield sse(_status_payload(request))
            await asyncio.sleep(period)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health")
async def health(request: Request):
    """Liveness probe. Cheap on purpose: no store scan, no catalog walk."""
    return {
        "status": "ok",
        "version": __version__,
        "engine": request.app.state.scheduler.engine.name,
    }


@router.get("/v1/hardware")
async def hardware(request: Request):
    from ..hardware import detect_host

    # Re-detect rather than serving the cached startup value: a user plugging in an
    # eGPU or updating a driver should see the change without restarting the server.
    return detect_host().to_dict()
