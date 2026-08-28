# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Shared fixtures.

Every test here runs against the replay engine with its clock disabled, so the suite
needs no AMD GPU, no onnxruntime-genai and no network, and finishes in seconds on both
Windows and Linux. What is being tested is the product around the engine: the catalog,
the scheduler, the two API surfaces, the job pipeline and the documentation build.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

PRODUCT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRODUCT_DIR / "oga-server"))
sys.path.insert(0, str(PRODUCT_DIR / "toolkit"))

from hipep_server.app import create_app  # noqa: E402
from hipep_server.catalog import load_catalog  # noqa: E402
from hipep_server.config import Settings  # noqa: E402


@pytest.fixture(scope="session")
def catalog():
    """The shipped catalog, loaded once. Read-only in every test that uses it."""
    return load_catalog(
        PRODUCT_DIR / "modelzoo" / "catalog", PRODUCT_DIR / "modelzoo" / "benchmarks"
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "home",
        engine="replay",
        replay_speed=0.0,
        simulation_speed=200.0,
        # Small enough that a test can reach the limits without generating load.
        max_loaded_models=2,
        max_concurrent_per_model=2,
        max_queue_depth=4,
        max_concurrent_jobs=1,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A started server. The context manager is what runs the lifespan."""
    with TestClient(create_app(settings)) as started:
        yield started


@pytest.fixture
def paced_client(settings: Settings) -> Iterator[TestClient]:
    """A server whose replay clock still runs, for assertions about elapsed time.

    Compressed 100x so a certified multi-second timing costs tens of milliseconds while
    staying well clear of the ~15.6 ms Windows timer granularity.
    """
    settings.replay_speed = 100.0
    with TestClient(create_app(settings)) as started:
        yield started


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def live_server(tmp_path: Path) -> Iterator[Callable[..., tuple[str, httpx.Client]]]:
    """Starts real servers on real ports, and stops them at the end of the test.

    Needed wherever something crosses a socket rather than going through the ASGI app
    in process: the benchmark harness drives the server over HTTP, and the artifact
    importer builds its own client to reach a peer.
    """
    servers: list[tuple[uvicorn.Server, threading.Thread]] = []
    clients: list[httpx.Client] = []

    def start(name: str = "server", **overrides) -> tuple[str, httpx.Client]:
        port = _free_port()
        defaults = {
            "engine": "replay",
            "replay_speed": 0.0,
            "simulation_speed": 200.0,
        }
        settings = Settings(
            data_dir=tmp_path / name,
            host="127.0.0.1",
            port=port,
            **{**defaults, **overrides},
        )
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(settings), host="127.0.0.1", port=port, log_level="error"
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        servers.append((server, thread))

        deadline = time.monotonic() + 20
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started, f"{name} did not start on port {port}"

        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60.0)
        clients.append(client)
        return f"http://127.0.0.1:{port}", client

    try:
        yield start
    finally:
        for client in clients:
            client.close()
        for server, thread in servers:
            server.should_exit = True
            thread.join(timeout=10)


@pytest.fixture
def finished():
    """Returns a helper that polls a job until it reaches a terminal state.

    The server runs on its own thread under TestClient, so sleeping here is what lets
    it make progress.
    """

    def wait(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        snapshot: dict = {}
        while time.monotonic() < deadline:
            snapshot = client.get(f"/v1/jobs/{job_id}").json()
            if snapshot.get("state") in {"succeeded", "failed", "cancelled"}:
                return snapshot
            time.sleep(0.02)
        raise AssertionError(
            f"job {job_id} did not finish within {timeout}s; last state "
            f"{snapshot.get('state')} at stage {snapshot.get('current_stage')}"
        )

    return wait


# Resolved from the catalog rather than from a running server, so that a test can pick
# a model without deciding which server fixture it wants first.
@pytest.fixture
def chat_model(catalog) -> str:
    """A certified text model that has published benchmarks."""
    model = next(
        (m for m in catalog.list(tier="llm") if m.benchmark and m.benchmark.ttft_s), None
    )
    if model is None:
        pytest.skip("no benchmarked LLM in the catalog")
    return model.id


@pytest.fixture
def vision_model(catalog) -> str:
    models = catalog.list(tier="vlm")
    if not models:
        pytest.skip("no vision model in the catalog")
    return models[0].id


@pytest.fixture
def png_data_url() -> str:
    """A 1x1 PNG as a data URL, enough to exercise the image path."""
    pixel = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
        "DwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    return f"data:image/png;base64,{pixel}"
