# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Benchmark harness.

The harness compares a measurement against a published number, so a bias anywhere in
it is indistinguishable from a regression in the model. Two biases have bitten this
code before and are pinned here: a prompt builder and a token estimator that disagree
about what "2048 tokens" means, and end-to-end HTTP cost being charged to the engine.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from hipep_server.bench import build_prompt
from hipep_server.catalog import Catalog
from hipep_server.engines.base import estimate_tokens, image_prefill_tokens

CERTIFIED_CONTEXTS = [128, 2048, 16384]


@pytest.mark.parametrize("target", [16, 128, 512, 2048, 16384])
def test_the_prompt_builder_stays_inside_the_estimator_snap_window(target: int) -> None:
    """The estimator prices prose at 1.1 tokens per word; the harness fills its prompt
    with words that are one token each, so the estimate reads about 10% long. That gap
    is why the published-timing lookup snaps within 15% instead of interpolating - if
    the gap ever grew past the snap window, a benchmark at a labelled context would
    silently be priced at a different context."""
    estimated = estimate_tokens(build_prompt(target))
    ratio = estimated / target
    assert 1 / 1.15 < ratio < 1.15, (
        f"asked for {target} tokens, built a prompt the estimator reads as {estimated}"
    )


@pytest.mark.parametrize("context", CERTIFIED_CONTEXTS)
def test_a_benchmark_prompt_is_priced_at_the_context_it_asked_for(
    catalog: Catalog, chat_model: str, context: int
) -> None:
    """The property the snap window exists to guarantee, checked end to end: a prompt
    built for a certified context must select that context's published numbers."""
    benchmark = catalog.get(chat_model).benchmark
    estimated = estimate_tokens(build_prompt(context))
    assert benchmark.at(estimated) == (benchmark.ttft_s[context], benchmark.tps[context])


def test_prompt_length_is_monotonic() -> None:
    lengths = [len(build_prompt(n)) for n in (128, 512, 2048)]
    assert lengths == sorted(lengths)


def test_estimating_an_empty_prompt_is_zero() -> None:
    assert estimate_tokens("") == 0


def test_a_dense_script_is_estimated_by_characters() -> None:
    """CJK text has almost no whitespace, so a word count would read a long prompt as
    a handful of tokens."""
    dense = "这是一段没有空格的中文文本" * 20
    assert estimate_tokens(dense) > len(dense.split()) * 2


def test_images_dominate_the_prompt_budget() -> None:
    assert image_prefill_tokens(0) == 0
    assert image_prefill_tokens(1) > 1000
    assert image_prefill_tokens(3) == 3 * image_prefill_tokens(1)
    # A smaller resize is cheaper, but never free.
    assert 0 < image_prefill_tokens(1, 360) < image_prefill_tokens(1, 720)


# --- through the server -------------------------------------------------------
#
# The harness drives the server over HTTP by design, since it measures what an
# application experiences. That means a real listening port, not an in-process app.


@pytest.fixture
def bench_server(live_server):
    """A server whose replay clock still runs, so a measurement has something to
    measure. Compressed 100x to keep the suite fast while staying clear of the
    ~15.6 ms Windows timer granularity.

    Compression scales the measurement but not the published numbers it is compared
    against, so anything asserting on the delta needs `certified_bench_server`.
    """
    return live_server("bench", replay_speed=100.0)


@pytest.fixture
def certified_bench_server(live_server):
    """A server replaying at the certified pace, for comparisons against it."""
    return live_server("certified", replay_speed=1.0)


def run_bench(client, model: str, finished, **spec) -> dict:
    started = client.post(
        "/v1/benchmarks/run",
        json={
            "model": model,
            "contexts": [128, 2048],
            "generate_tokens": 16,
            "repeats": 1,
            "warmup": 1,
            **spec,
        },
    )
    assert started.status_code == 200, started.text
    snapshot = finished(client, started.json()["job"]["id"], timeout=120.0)
    assert snapshot["state"] == "succeeded", snapshot.get("error")
    return snapshot["result"]["summary"]


def test_benchmark_measures_every_requested_context(
    bench_server, chat_model: str, finished
) -> None:
    _, client = bench_server
    summary = run_bench(client, chat_model, finished)

    assert set(summary["contexts"]) == {"128", "2048"}
    for context, values in summary["contexts"].items():
        assert values["ttft_s"] > 0, context
        assert values["tps"] > 0, context
        assert values["runs"] == 1


def test_transport_cost_is_reported_beside_the_engine_number(
    bench_server, chat_model: str, finished
) -> None:
    """Comparing an end-to-end number against an engine-level certified number would
    report HTTP overhead as a model regression."""
    _, client = bench_server
    summary = run_bench(client, chat_model, finished)

    for values in summary["contexts"].values():
        assert values["e2e_ttft_s"] >= values["ttft_s"]
        assert values["transport_s"] >= 0
        assert values["transport_s"] < values["e2e_ttft_s"]


@pytest.mark.slow
def test_measurements_track_the_certified_numbers(
    certified_bench_server, chat_model: str, finished
) -> None:
    """The replay engine reproduces the published timings, so a benchmark of it should
    land on the published curve. A large delta here means the harness, the pacing or
    the context lookup has drifted rather than the model.

    This one runs at the certified pace, so it costs real seconds; that is the whole
    point of it.
    """
    _, client = certified_bench_server
    summary = run_bench(client, chat_model, finished, generate_tokens=24)

    assert summary["delta_vs_published"], "no comparison against the published numbers"
    for context, delta in summary["delta_vs_published"].items():
        assert abs(delta["ttft_pct"]) < 15, f"TTFT drifted at context {context}: {delta}"
        assert abs(delta["tps_pct"]) < 15, f"decode drifted at context {context}: {delta}"


def test_warmup_runs_are_excluded_from_the_result(
    bench_server, chat_model: str, finished
) -> None:
    """The first run pays the model load; folding it into the median would make every
    benchmark look like a regression."""
    _, client = bench_server
    summary = run_bench(client, chat_model, finished, contexts=[128], repeats=2, warmup=1)
    assert summary["contexts"]["128"]["runs"] == 2


def test_benchmarking_an_unknown_model_is_404(client: TestClient) -> None:
    assert client.post("/v1/benchmarks/run", json={"model": "nope:1b"}).status_code == 404


def test_a_context_beyond_the_model_maximum_is_refused(
    client: TestClient, chat_model: str
) -> None:
    response = client.post(
        "/v1/benchmarks/run", json={"model": chat_model, "contexts": [10**9]}
    )
    assert response.status_code == 400
    assert "supports at most" in response.json()["detail"]


def test_malformed_contexts_are_refused(client: TestClient, chat_model: str) -> None:
    response = client.post(
        "/v1/benchmarks/run", json={"model": chat_model, "contexts": [0, -1]}
    )
    assert response.status_code == 400


def test_published_results_are_served_for_the_docs(client: TestClient) -> None:
    body = client.get("/v1/benchmarks/published").json()
    assert body["meta"]["release"]
    assert body["results"]
    for row in body["results"]:
        assert row["model"] and row["quantization"]
