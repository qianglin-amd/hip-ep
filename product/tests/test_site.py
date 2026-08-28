# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Documentation site.

Every factual claim on the site is generated from the model zoo, which is what stops a
model card from drifting away from the catalog it describes. The risk that creates is
the opposite one: renaming a family or retiring a model breaks pages nobody edited, in
a build with no compiler to catch it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PRODUCT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRODUCT_DIR / "site"))

pytest.importorskip("markdown", reason="the site generator needs python-markdown")

import build as site  # noqa: E402


@pytest.fixture(scope="module")
def site_dir(tmp_path_factory) -> Path:
    """Build the site once; every test in this module reads the same output."""
    out = tmp_path_factory.mktemp("site")
    assert site.build(out) > 0
    return out


def pages(site_dir: Path) -> dict[str, str]:
    return {
        p.relative_to(site_dir).as_posix(): p.read_text(encoding="utf-8")
        for p in site_dir.rglob("*.html")
    }


def test_the_expected_pages_are_written(site_dir: Path) -> None:
    written = pages(site_dir)
    for required in (
        "index.html",
        "docs/index.html",
        "docs/install-windows.html",
        "docs/install-linux.html",
        "docs/cli.html",
        "docs/server-mode.html",
        "docs/api.html",
        "docs/port-a-model.html",
        "docs/troubleshooting.html",
        "docs/models/index.html",
        "docs/benchmarks/index.html",
        "docs/recipes.html",
    ):
        assert required in written, f"{required} was not generated"


def test_every_family_gets_its_own_page(site_dir: Path, catalog) -> None:
    families = {m.family for m in catalog.models}
    written = pages(site_dir)
    generated = {
        name.removeprefix("docs/models/").removesuffix(".html")
        for name in written
        if name.startswith("docs/models/") and not name.endswith("index.html")
    }
    assert len(generated) == len(families)


def test_assets_are_copied_beside_the_pages(site_dir: Path) -> None:
    """The site has to open from disk and publish to any static host, so nothing may
    be fetched from a CDN."""
    assert (site_dir / "assets" / "site.css").is_file()
    for text in pages(site_dir).values():
        assert "cdn." not in text
        assert "https://fonts.googleapis" not in text


def test_no_internal_link_is_broken(site_dir: Path) -> None:
    assert site.check_links(site_dir) == []


def test_every_model_gets_a_card(site_dir: Path, catalog) -> None:
    written = " ".join(pages(site_dir).values())
    for model in catalog.models:
        assert model.id in written, f"{model.id} has no page anywhere on the site"


def test_a_family_page_carries_the_specification(site_dir: Path, catalog) -> None:
    """The model cards are the reason someone reads the site; a card missing its
    quantization or context is the one thing it cannot get away with."""
    model = catalog.get("llama3.1:8b")
    family = (site_dir / "docs" / "models" / "llama3-1.html").read_text(encoding="utf-8")

    assert model.id in family
    assert model.name in family
    assert model.quant_label in family
    assert model.license in family
    assert model.parameters in family
    assert "Context window" in family
    assert f"{model.max_context // 1024}K" in family


def test_benchmark_pages_quote_the_release(site_dir: Path, catalog) -> None:
    release = catalog.benchmark_meta["release"]
    benchmarks = {
        name: text
        for name, text in pages(site_dir).items()
        if name.startswith("docs/benchmarks/")
    }
    assert benchmarks
    for name, text in benchmarks.items():
        assert release in text, f"{name} does not say which release it reports"


def test_the_landing_page_states_what_the_product_is(site_dir: Path, catalog) -> None:
    landing = (site_dir / "index.html").read_text(encoding="utf-8")
    assert "HIP EP" in landing
    assert str(len(catalog)) in landing, "the model count is not shown"
    # A visitor should be able to start without reading further.
    assert "pip install" in landing or "hipep" in landing


def test_recipes_page_documents_the_quark_defaults(site_dir: Path) -> None:
    recipes = (site_dir / "docs" / "recipes.html").read_text(encoding="utf-8")
    assert "chunk_size" in recipes
    assert "512" in recipes
    assert "prune_logit" in recipes


def test_every_page_has_a_title_and_navigation(site_dir: Path) -> None:
    for name, text in pages(site_dir).items():
        assert re.search(r"<title>[^<]+</title>", text), f"{name} has no title"
        assert "<nav" in text, f"{name} has no navigation"


def test_generated_pages_do_not_leak_none(site_dir: Path) -> None:
    """A missing catalog field rendered straight into the page shows up as the literal
    text `None`, which looks like a real value to a reader."""
    for name, text in pages(site_dir).items():
        stripped = re.sub(r"<[^>]+>", " ", text)
        assert not re.search(r"\bNone\b", stripped), f"{name} renders a literal None"
        assert not re.search(r"\bnan\b", stripped), f"{name} renders a literal nan"
