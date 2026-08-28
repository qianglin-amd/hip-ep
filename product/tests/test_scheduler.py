# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Admission control.

A single GPU shared by several users is the case that fails quietly: without bounds the
extra requests do not error, they just make everyone slower until the box runs out of
memory mid-generation. These tests pin the three bounds and the accounting the status
page reports, because a limit nobody can observe is indistinguishable from no limit.
"""

from __future__ import annotations

import asyncio

import pytest
from hipep_server.catalog import Catalog
from hipep_server.config import Settings
from hipep_server.engines import ChatMessage, GenerationParams, GenerationRequest
from hipep_server.engines.replay import ReplayEngine
from hipep_server.scheduler import QueueFull, Scheduler


@pytest.fixture
def make_scheduler(catalog: Catalog, settings: Settings):
    def build(**overrides) -> Scheduler:
        for key, value in overrides.items():
            setattr(settings, key, value)
        return Scheduler(ReplayEngine(speed=0.0), catalog, settings)

    return build


def request_for(catalog: Catalog, model_id: str, *, max_tokens: int = 8):
    return GenerationRequest(
        model=catalog.get(model_id),
        messages=[ChatMessage(role="user", content="Hello")],
        params=GenerationParams(max_tokens=max_tokens),
    )


async def drain(scheduler: Scheduler, request: GenerationRequest) -> dict:
    """Consume a generation and return its usage block."""
    session = scheduler.session(None, client="test", user_agent="pytest", api="test")
    usage: dict = {}
    async for delta in scheduler.stream(request, session=session, api="test", client="test"):
        if delta.done and delta.usage:
            usage = delta.usage
    return usage


@pytest.mark.asyncio
async def test_usage_is_accounted_to_the_session_and_the_totals(
    make_scheduler, catalog: Catalog, chat_model: str
) -> None:
    scheduler = make_scheduler()
    usage = await drain(scheduler, request_for(catalog, chat_model))

    assert usage["completion_tokens"] > 0
    assert scheduler.total_requests == 1
    assert scheduler.total_completion_tokens == usage["completion_tokens"]
    assert scheduler.total_prompt_tokens == usage["prompt_tokens"]

    session = next(iter(scheduler.sessions.values()))
    assert session.requests == 1
    assert session.completion_tokens == usage["completion_tokens"]


@pytest.mark.asyncio
async def test_finished_requests_leave_the_active_set(
    make_scheduler, catalog: Catalog, chat_model: str
) -> None:
    scheduler = make_scheduler()
    await drain(scheduler, request_for(catalog, chat_model))
    assert scheduler.active == {}
    assert scheduler.queue_depth == 0
    # But it stays in the recent history the console renders.
    assert scheduler.recent[0]["model"] == chat_model


@pytest.mark.asyncio
async def test_a_full_queue_rejects_rather_than_degrading(
    make_scheduler, catalog: Catalog, chat_model: str
) -> None:
    """Rejecting is the honest outcome: accepting work the server cannot start turns a
    fast failure into a slow one for every other user."""
    scheduler = make_scheduler(max_concurrent_per_model=1, max_queue_depth=1)
    request = request_for(catalog, chat_model, max_tokens=64)

    # Hold the only slot open so the next arrivals have to queue.
    holder = scheduler._slot(chat_model)
    await holder.acquire()
    session = scheduler.session(None, client="test", user_agent="pytest", api="test")

    queued = asyncio.create_task(
        anext(
            scheduler.stream(request, session=session, api="test", client="test").__aiter__()
        )
    )
    await asyncio.sleep(0)  # let the task reach the queue

    with pytest.raises(QueueFull) as excinfo:
        await drain(scheduler, request)
    assert "capacity" in str(excinfo.value)
    assert scheduler.rejected == 1

    holder.release()
    queued.cancel()
    with pytest.raises((asyncio.CancelledError, StopAsyncIteration)):
        await queued


@pytest.mark.asyncio
async def test_concurrency_limit_serialises_work_on_one_model(
    make_scheduler, catalog: Catalog, chat_model: str
) -> None:
    scheduler = make_scheduler(max_concurrent_per_model=2, max_queue_depth=64)
    requests = [request_for(catalog, chat_model, max_tokens=16) for _ in range(6)]

    peak = 0

    async def run(request):
        nonlocal peak
        session = scheduler.session(None, client="test", user_agent="pytest", api="test")
        async for _ in scheduler.stream(request, session=session, api="test", client="test"):
            peak = max(peak, len(scheduler.active))

    await asyncio.gather(*(run(r) for r in requests))
    assert scheduler.total_requests == 6
    # `active` counts admitted-and-running requests, which the semaphore bounds.
    assert peak <= 2


@pytest.mark.asyncio
async def test_least_recently_used_model_is_evicted(make_scheduler, catalog: Catalog) -> None:
    scheduler = make_scheduler(max_loaded_models=2)
    ids = [m.id for m in catalog.list(tier="llm")[:3]]

    for model_id in ids:
        await drain(scheduler, request_for(catalog, model_id))

    resident = scheduler.engine.loaded_models()
    assert len(resident) == 2
    assert ids[0] not in resident, "the oldest model should have been evicted"
    assert ids[2] in resident


@pytest.mark.asyncio
async def test_load_cost_is_charged_once(
    make_scheduler, catalog: Catalog, chat_model: str
) -> None:
    scheduler = make_scheduler()
    scheduler.engine = ReplayEngine(speed=100.0)

    cold = await drain(scheduler, request_for(catalog, chat_model))
    warm = await drain(scheduler, request_for(catalog, chat_model))

    assert cold["load_s"] > 0
    assert warm["load_s"] == 0


@pytest.mark.asyncio
async def test_idle_sweep_releases_unused_models(
    make_scheduler, catalog: Catalog, chat_model: str
) -> None:
    scheduler = make_scheduler(idle_unload_seconds=0)
    await drain(scheduler, request_for(catalog, chat_model))
    assert scheduler.engine.loaded_models() == [chat_model]

    # idle_unload_seconds=0 disables the sweep entirely rather than unloading
    # everything immediately, which would make the setting a foot-gun.
    await scheduler.unload_idle()
    assert scheduler.engine.loaded_models() == [chat_model]

    scheduler.settings.idle_unload_seconds = 1
    scheduler.recent[0]["at"] -= 3600
    await scheduler.unload_idle()
    assert scheduler.engine.loaded_models() == []


@pytest.mark.asyncio
async def test_snapshot_reports_limits_and_totals(
    make_scheduler, catalog: Catalog, chat_model: str
) -> None:
    scheduler = make_scheduler()
    await drain(scheduler, request_for(catalog, chat_model))
    snapshot = scheduler.snapshot()

    assert snapshot["engine"]["name"] == "replay"
    assert snapshot["engine"]["real"] is False
    assert snapshot["limits"]["max_loaded_models"] == scheduler.settings.max_loaded_models
    assert snapshot["load"]["active_requests"] == 0
    assert snapshot["load"]["loaded_models"] == 1
    assert snapshot["totals"]["requests"] == 1
    assert snapshot["totals"]["errors"] == 0
    assert snapshot["sessions_detail"][0]["requests"] == 1
