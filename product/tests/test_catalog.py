# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Catalog loading, name resolution and benchmark lookup.

The catalog is the product's source of truth: the replay engine takes its timings from
it, the console renders it, the docs are generated from it and the benchmark harness
compares against it. A silent regression here is visible everywhere at once, which is
why the shipped YAML is validated rather than a fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hipep_server.catalog import SHIPPED_TIERS, Benchmark, Catalog, Model, load_catalog

PRODUCT_DIR = Path(__file__).resolve().parent.parent


def test_catalog_loads_every_shipped_tier(catalog: Catalog) -> None:
    assert len(catalog) > 20
    assert set(catalog.tiers()) == set(SHIPPED_TIERS)


def test_unshipped_tiers_stay_out_of_the_default_catalog() -> None:
    """The catalog directory carries tiers still being brought up. They load only when
    asked for explicitly, so no caller can list them by forgetting to filter."""
    everything = load_catalog(
        PRODUCT_DIR / "modelzoo" / "catalog",
        PRODUCT_DIR / "modelzoo" / "benchmarks",
        shipped_only=False,
    )
    assert set(everything.tiers()) > set(SHIPPED_TIERS), "nothing is being held back"

    shipped = load_catalog(
        PRODUCT_DIR / "modelzoo" / "catalog", PRODUCT_DIR / "modelzoo" / "benchmarks"
    )
    assert len(shipped) < len(everything)
    for model in shipped.models:
        assert model.tier in SHIPPED_TIERS
        assert (model.certification or {}).get("status") != "fail"


def test_every_model_is_completely_described(catalog: Catalog) -> None:
    """A model card with holes in it is worse than no card: it still looks
    authoritative."""
    for model in catalog.models:
        for attribute in ("name", "tier", "type", "family", "license"):
            assert getattr(model, attribute), f"{model.id} is missing {attribute}"
        assert model.description.strip(), f"{model.id} has no description"
        assert model.quant_label, f"{model.id} has no quantization label"
        assert (model.artifact or {}).get("dashboard_name"), (
            f"{model.id} cannot be traced back to a dashboard entry"
        )
        assert model.recipe, f"{model.id} does not say how it was built"


def test_generative_models_declare_size_and_context(catalog: Catalog) -> None:
    """Parameter count and context length are the two numbers someone picking an LLM
    reads first. They are meaningless for a classical vision graph, which is why this
    is scoped to the generative tiers."""
    for model in catalog.list(generative_only=True):
        assert model.parameters, f"{model.id} does not state its parameter count"
        assert model.context.get("max"), f"{model.id} does not state a context length"
        assert 0 < model.default_context <= model.max_context


def test_model_ids_are_unique_and_tagged(catalog: Catalog) -> None:
    ids = [m.id for m in catalog.models]
    assert len(ids) == len(set(ids))
    # `family:size` is what the Ollama-compatible surface expects.
    assert all(":" in model_id for model_id in ids)


def test_generative_tiers_are_llm_and_vlm(catalog: Catalog) -> None:
    generative = {m.tier for m in catalog.list(generative_only=True)}
    assert generative == {"llm", "vlm"}


def test_vlm_models_declare_vision_capability(catalog: Catalog) -> None:
    for model in catalog.list(tier="vlm"):
        assert model.supports_vision, f"{model.id} is a VLM but does not claim vision"
        assert model.vision, f"{model.id} has no vision settings"


def test_llm_models_do_not_claim_vision(catalog: Catalog) -> None:
    for model in catalog.list(tier="llm"):
        assert not model.supports_vision, f"{model.id} is an LLM but claims vision"


def test_list_is_ordered_by_tier_then_size(catalog: Catalog) -> None:
    tiers = [m.tier for m in catalog.models]
    assert tiers == sorted(tiers, key=lambda t: list(SHIPPED_TIERS).index(t))


@pytest.mark.parametrize(
    "given",
    ["llama3.1:8b", "LLAMA3.1:8B", "llama3.1/8b"],
)
def test_resolve_accepts_the_forms_clients_send(catalog: Catalog, given: str) -> None:
    model = catalog.resolve(given)
    assert model is not None and model.id == "llama3.1:8b"


def test_resolve_a_bare_family_picks_the_smallest_variant(catalog: Catalog) -> None:
    """Asking for a family name should return something that fits on the machine, not
    the largest member of the family."""
    family = catalog.resolve("qwen2.5")
    assert family is not None
    siblings = [m for m in catalog.models if (m.family or "") == family.family]
    assert family.min_vram_gb == min(m.min_vram_gb for m in siblings)


def test_resolve_rejects_unknown_and_empty_names(catalog: Catalog) -> None:
    assert catalog.resolve("no-such-model:1b") is None
    assert catalog.resolve("") is None


def test_benchmarks_are_attached_with_release_metadata(catalog: Catalog) -> None:
    assert catalog.benchmark_meta.get("release")
    measured = [m for m in catalog.models if m.benchmark]
    assert measured, "no model has published benchmarks"
    for model in measured:
        assert model.benchmark.release
        assert model.benchmark.device.get("name")


def test_published_contexts_are_the_certified_points(catalog: Catalog) -> None:
    for model in catalog.list(generative_only=True):
        if not model.benchmark:
            continue
        contexts = model.benchmark.context_lengths
        assert contexts, f"{model.id} has a benchmark with no context series"
        assert all(c > 0 for c in contexts)
        assert contexts == sorted(contexts)


def test_weights_size_is_estimated_when_not_measured(catalog: Catalog) -> None:
    """Load-time and footprint estimates need a size for every model, including the
    entries whose artifact was never weighed."""
    for model in catalog.models:
        assert model.weights_gb > 0
        if model.min_vram_gb:
            assert model.weights_gb <= model.min_vram_gb

    unmeasured = Model(
        id="x:1b",
        name="X",
        tier="llm",
        type="text-to-text",
        footprint={"min_vram_gb": 20},
    )
    assert 10 < unmeasured.weights_gb < 20


def test_published_context_never_exceeds_the_declared_maximum(catalog: Catalog) -> None:
    for model in catalog.list(generative_only=True):
        if model.benchmark:
            for context in model.benchmark.context_lengths:
                assert context <= model.max_context, (
                    f"{model.id} is benchmarked at {context} but declares "
                    f"a {model.max_context} maximum"
                )


# --- Benchmark.at ------------------------------------------------------------


@pytest.fixture
def series() -> Benchmark:
    return Benchmark(
        ttft_s={128: 0.2, 2048: 1.5, 16384: 12.0},
        tps={128: 44.0, 2048: 39.0, 16384: 20.0},
    )


def test_measured_contexts_return_their_measurement(series: Benchmark) -> None:
    for context in (128, 2048, 16384):
        ttft, tps = series.at(context)
        assert ttft == series.ttft_s[context]
        assert tps == series.tps[context]


@pytest.mark.parametrize("context", [115, 140, 1800, 2300])
def test_a_near_miss_snaps_to_the_measurement(series: Benchmark, context: int) -> None:
    """A prompt length derived from a token estimate lands a few percent off the
    labelled context. Interpolating from there across an order of magnitude turns a 5%
    estimate error into a 30% TTFT error, so a close context uses the measurement."""
    nearest = min(series.ttft_s, key=lambda n: abs(n - context))
    ttft, tps = series.at(context)
    assert ttft == series.ttft_s[nearest]
    assert tps == series.tps[nearest]


def test_between_measurements_interpolates_monotonically(series: Benchmark) -> None:
    ttfts = [series.at(c)[0] for c in (300, 700, 1200)]
    assert ttfts == sorted(ttfts)
    assert ttfts[0] > 0.2 and ttfts[-1] < 1.5


def test_beyond_the_measured_range_clamps(series: Benchmark) -> None:
    assert series.at(1)[0] == 0.2
    assert series.at(10**6)[0] == 12.0


def test_a_scalar_only_benchmark_has_no_context_series() -> None:
    """Speech and vision results are scalars stored under a sentinel key. Reporting
    that as a measurement at a zero-token prompt would be nonsense."""
    scalar = Benchmark(ttft_s={0: None}, rtf=0.05)
    assert scalar.context_lengths == []
    assert scalar.at(128) == (None, None)


# --- registration ------------------------------------------------------------


def test_registering_an_imported_model_makes_it_resolvable(catalog: Catalog) -> None:
    imported = Model(
        id="mine:7b", name="Mine 7B", tier="llm", type="text-to-text", status="imported"
    )
    try:
        catalog.register(imported)
        assert catalog.resolve("mine:7b") is imported
        assert catalog.unregister("mine:7b") is True
        assert catalog.resolve("mine:7b") is None
    finally:
        catalog.unregister("mine:7b")


def test_certified_entries_cannot_be_unregistered(catalog: Catalog) -> None:
    """Deleting downloaded files must not remove the catalog entry, or the catalog
    would start depending on install state."""
    assert catalog.unregister("llama3.1:8b") is False
    assert catalog.resolve("llama3.1:8b") is not None
