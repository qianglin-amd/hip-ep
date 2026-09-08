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
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hipep_server.app import create_app
from hipep_server.catalog import load_catalog
from hipep_server.config import Settings

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


@pytest.fixture
def every_model(catalog):
    """The catalog including the tiers that have not shipped.

    The `catalog` fixture is already filtered to what ships, which is what the site
    documents. This unfiltered view is what lets a test check the boundary itself,
    rather than only checking that the two filtered things agree.
    """
    return load_catalog(
        PRODUCT_DIR / "modelzoo" / "catalog",
        PRODUCT_DIR / "modelzoo" / "benchmarks",
        shipped_only=False,
    )


@pytest.fixture
def served(site_dir: Path, tmp_path: Path) -> Iterator[TestClient]:
    """A running server with this build of the site mounted, as an install would have."""
    settings = Settings(data_dir=tmp_path / "home", engine="replay", site_dir=site_dir)
    with TestClient(create_app(settings)) as client:
        yield client


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


def test_the_site_never_advertises_an_install_that_does_not_exist(site_dir: Path) -> None:
    """The landing page and both install guides offered `pip install hipep-server`.

    Nothing is published under that name -- nor under `hipep-product`, which is what
    the package is actually called -- so the very first command a new user ran failed,
    on the page whose entire job is getting them to a first token. Until a release
    exists, the documented path has to be the one that works from a checkout.
    """
    unpublished = ("hipep-server", "hipep-product")
    for name, text in pages(site_dir).items():
        for package in unpublished:
            assert f"pip install {package}" not in text, (
                f"{name} tells the reader to install {package}, which is not published"
            )


def test_the_server_serves_the_built_site(served: TestClient) -> None:
    assert served.get("/docs/").status_code == 200
    assert served.get("/landing").status_code == 200
    assert served.get("/assets/site.css").status_code == 200


def test_the_bare_docs_path_reaches_the_docs(served: TestClient) -> None:
    """A mount answers at "/docs/" only. The console's own Docs link points at the bare
    "/docs", which is also what a reader types, so the redirect is load-bearing."""
    redirect = served.get("/docs", follow_redirects=False)
    assert redirect.status_code in (301, 302, 307, 308)
    assert redirect.headers["location"] == "/docs/"
    assert served.get("/docs", follow_redirects=True).status_code == 200


def test_the_console_links_to_something_that_exists(served: TestClient) -> None:
    """The console is hand-written, so its links to the generated site are the pair most
    able to drift apart."""
    console = served.get("/").text
    targets = set(re.findall(r'href="(/[^"#]*)"', console))
    assert "/docs" in targets, "the console no longer links to the docs"
    for target in targets:
        assert served.get(target, follow_redirects=True).status_code == 200, target


def test_the_docs_offer_the_console_only_where_one_is_running(site_dir: Path) -> None:
    """The same build is published to a static host and served by the server, so the
    link back to the console cannot be decided at build time. It ships hidden and is
    revealed only once a server answers, which is why every page must load the script
    that asks."""
    script = (site_dir / "assets" / "site.js").read_text(encoding="utf-8")
    assert "/v1/status" in script, "nothing checks whether a server is there"

    for name, text in pages(site_dir).items():
        assert 'class="to-studio"' in text, f"{name} cannot offer the console"
        assert re.search(r'<a class="to-studio"[^>]*\shidden', text), (
            f"{name} shows the console link before knowing a server exists"
        )
        assert "site.js" in text, f"{name} never runs the check that reveals the link"


def test_every_model_gets_a_card(site_dir: Path, catalog) -> None:
    written = " ".join(pages(site_dir).values())
    for model in catalog.models:
        assert model.id in written, f"{model.id} has no page anywhere on the site"


def test_the_site_publishes_only_the_shipped_tiers(
    site_dir: Path, catalog, every_model
) -> None:
    written = " ".join(pages(site_dir).values())
    shipped = {m.id for m in catalog.models}
    withheld = [m for m in every_model.models if m.id not in shipped]
    assert withheld, "this test is vacuous if the catalog has nothing to hold back"
    for model in withheld:
        assert model.id not in written, f"{model.id} is not shipped but the site lists it"


def test_the_site_does_not_publish_the_release_gate(site_dir: Path, catalog) -> None:
    """The per-suite pass/fail summary grades a release against the previous one, so a
    "fail" there is a few percent of regression in a model that works. It is an
    internal gate; on a public page it reads as a broken product."""
    written = " ".join(pages(site_dir).values()).lower()
    assert "pass rate" not in written
    assert "pass_rate" not in written
    assert "procyon" not in written

    # The suite names themselves are not searched for: "vision" is a substring of
    # "vision-language", so a page that correctly describes a VLM would fail such a
    # check. Which models reach the site is covered by the tier test above.
    assert catalog.benchmark_meta.get("summary"), "the gate should still exist internally"


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
