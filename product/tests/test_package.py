# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The delivery bundle.

A bundle is only checked properly once, by hand, on a machine that does not have this
repository. What these tests defend is the part that would otherwise be found by the
recipient: a virtualenv or a `__pycache__` from the packaging host, a missing licence, a
manifest whose model count disagrees with the catalog it ships, or an installer that is
absent from the one directory the recipient actually opens.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

PRODUCT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRODUCT_DIR))

import package  # noqa: E402

# --- what the bundle carries --------------------------------------------------


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    """Stage a bundle once; it takes a few seconds because it builds the docs."""
    out = tmp_path_factory.mktemp("dist")
    assert package.main(["--out", str(out)]) == 0
    staged = [p for p in out.iterdir() if p.is_dir()]
    assert len(staged) == 1, staged
    return staged[0]


@pytest.mark.parametrize(
    "entry",
    [
        "START-HERE.md",
        "MANIFEST.json",
        "SHA256SUMS",
        "LICENSE",
        "install.ps1",
        "install.sh",
        "install-port-toolchain.ps1",
        "install-port-toolchain.sh",
        "pyproject.toml",
        "oga-server/hipep_server/app.py",
        "toolkit/hipep_toolkit/cli.py",
        "modelzoo/catalog",
        "examples",
        "site/dist/index.html",
    ],
)
def test_the_bundle_carries_what_an_install_needs(bundle: Path, entry: str) -> None:
    assert (bundle / entry).exists(), f"{entry} is missing from the bundle"


def test_nothing_from_the_packaging_host_is_shipped(bundle: Path) -> None:
    """A virtualenv holds absolute paths from this machine and a `__pycache__` holds
    this interpreter's version, so either one fails somewhere other than where it was
    unpacked."""
    strays = [
        p.relative_to(bundle).as_posix()
        for p in bundle.rglob("*")
        if p.name in {"__pycache__", ".venv", ".pytest_cache", ".ruff_cache"}
        or p.suffix in {".pyc", ".pyo"}
        or p.name.endswith(".egg-info")
    ]
    assert strays == [], f"host artifacts in the bundle: {strays}"


def test_the_sweep_removes_bytecode_a_later_step_leaves_behind(tmp_path) -> None:
    """Building the docs and reading the catalog both import from the staged tree, so
    each can drop a `__pycache__` after the copy filtered bytecode out."""
    cache = tmp_path / "oga-server" / "hipep_server" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "catalog.cpython-314.pyc").write_bytes(b"\x00")
    (tmp_path / "stray.pyc").write_bytes(b"\x00")
    keep = tmp_path / "oga-server" / "hipep_server" / "catalog.py"
    keep.write_text("x = 1", encoding="utf-8")

    removed = package.sweep(tmp_path)

    assert not cache.exists()
    assert not (tmp_path / "stray.pyc").exists()
    assert keep.exists()
    assert len(removed) == 2


def test_the_review_deck_is_not_shipped(bundle: Path) -> None:
    """`plan/` is how the product is reviewed internally, not part of it."""
    assert not (bundle / "plan").exists()


def test_the_documentation_is_prebuilt(bundle: Path) -> None:
    """The bundle is opened by someone deciding whether to install anything, so the
    front page cannot require a build step first."""
    index = (bundle / "site" / "dist" / "index.html").read_text(encoding="utf-8")
    assert "HIP EP" in index
    assert (bundle / "site" / "dist" / "docs" / "recipes.html").exists()


# --- the manifest -------------------------------------------------------------


def test_the_manifest_model_count_is_the_count_that_will_be_served(bundle: Path) -> None:
    """Counted from the catalog YAML this would drift: the loader drops tiers that have
    not shipped and entries whose certification regressed."""
    sys.path.insert(0, str(bundle / "oga-server"))
    try:
        from hipep_server.catalog import load_catalog

        served = load_catalog(
            bundle / "modelzoo" / "catalog", bundle / "modelzoo" / "benchmarks"
        )
    finally:
        sys.path.pop(0)

    catalog = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))["contents"][
        "catalog"
    ]
    assert "error" not in catalog, catalog
    assert catalog["models"] == len(served.models)
    assert catalog["release"]


def test_the_manifest_lists_the_recipes_that_ship(bundle: Path) -> None:
    data = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    on_disk = sorted(p.name for p in (bundle / "toolkit/hipep_toolkit/recipes").glob("*.yaml"))
    assert data["contents"]["recipes"] == on_disk
    assert on_disk, "no recipes shipped"


def test_the_manifest_records_where_the_bundle_came_from(bundle: Path) -> None:
    """A bundle found on a share a year from now has to be traceable to a commit."""
    provenance = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))[
        "provenance"
    ]
    assert set(provenance) == {"commit", "describe", "branch"}


def test_the_manifest_is_json_serializable_end_to_end(bundle: Path) -> None:
    """YAML turns an unquoted date into `datetime.date`, which would raise here rather
    than in the catalog it came from."""
    json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))


# --- checksums ----------------------------------------------------------------


def test_every_file_is_checksummed(bundle: Path) -> None:
    listed = {
        line.split("  ", 1)[1]
        for line in (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if line
    }
    on_disk = {
        p.relative_to(bundle).as_posix()
        for p in bundle.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS"
    }
    assert listed == on_disk


def test_a_checksum_matches_its_file(bundle: Path) -> None:
    import hashlib

    digest, name = (
        (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines()[0].split("  ", 1)
    )
    assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest


# --- the wheels ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("onnxruntime_genai_directml-0.14.0-cp314-cp314-win_amd64.whl", "3.14"),
        ("onnxruntime_directml-1.27.0-cp311-cp311-win_amd64.whl", "3.11"),
        ("onnxruntime_ep_hip-0.4.0-py3-none-win_amd64.whl", None),
    ],
)
def test_the_python_version_is_read_from_the_wheel_name(name, expected) -> None:
    """The version decides which interpreter the virtualenv is built on, and the file
    name is the only place it is written down."""
    assert package.wheel_python([name]) == expected


def test_a_pure_python_wheel_does_not_hide_a_tagged_one() -> None:
    names = [
        "onnxruntime_ep_hip-0.4.0-py3-none-win_amd64.whl",
        "onnxruntime_directml-1.27.0-cp314-cp314-win_amd64.whl",
    ]
    assert package.wheel_python(names) == "3.14"


def test_wheels_are_copied_and_recorded(tmp_path) -> None:
    source = tmp_path / "wheels"
    source.mkdir()
    (source / "onnxruntime_directml-1.27.0-cp314-cp314-win_amd64.whl").write_bytes(b"x")
    stage = tmp_path / "stage"
    stage.mkdir()

    names = package.copy_wheels(stage, source)

    assert names == ["onnxruntime_directml-1.27.0-cp314-cp314-win_amd64.whl"]
    assert (stage / "wheels" / names[0]).exists()


def test_an_empty_wheel_directory_is_refused(tmp_path) -> None:
    (tmp_path / "wheels").mkdir()
    with pytest.raises(SystemExit):
        package.copy_wheels(tmp_path / "stage", tmp_path / "wheels")


# --- START-HERE ---------------------------------------------------------------


def test_start_here_names_the_python_the_wheels_need() -> None:
    """Wheels built for one Python and an interpreter chosen by habit is the failure
    that cost the most time to diagnose, so the version is stated up front."""
    text = package.start_here(
        {
            "version": "0.1.0",
            "gpu_wheels": {"included": True, "python": "3.14", "files": ["a.whl"]},
            "contents": {"catalog": {"models": 23, "release": "release_0.4"}, "recipes": []},
        }
    )
    assert "3.14" in text
    assert "-GpuWheels .\\wheels" in text


def test_start_here_explains_the_replay_engine_when_no_wheels_ship() -> None:
    """A bundle without wheels still runs everything but GPU generation, and a
    recipient who is not told that reads the replay timings as GPU results."""
    text = package.start_here(
        {
            "version": "0.1.0",
            "gpu_wheels": {"included": False, "python": None, "files": []},
            "contents": {"catalog": {"models": 23, "release": "release_0.4"}, "recipes": []},
        }
    )
    assert "replay" in text
    assert "-GpuWheels" in text


def test_start_here_reports_the_real_model_count() -> None:
    text = package.start_here(
        {
            "version": "0.1.0",
            "gpu_wheels": {"included": False, "python": None, "files": []},
            "contents": {
                "catalog": {"models": 23, "release": "release_0.4"},
                "recipes": ["a.yaml"],
            },
        }
    )
    assert "23 certified models (release_0.4)" in text
    assert "1 recipes" in text


def test_start_here_survives_an_unreadable_catalog() -> None:
    """The manifest records a catalog failure rather than raising, so the page it feeds
    has to render without the count."""
    text = package.start_here(
        {
            "version": "0.1.0",
            "gpu_wheels": {"included": False, "python": None, "files": []},
            "contents": {"catalog": {"error": "boom"}, "recipes": []},
        }
    )
    assert "certified model registry" in text


# --- the archive --------------------------------------------------------------


def test_the_zip_unpacks_into_one_directory(tmp_path) -> None:
    """A zip that scatters its contents into the current directory is the archive
    equivalent of a broken install."""
    assert package.main(["--out", str(tmp_path), "--no-docs"]) == 0
    archive = next(tmp_path.glob("*.zip"))
    roots = {Path(n).parts[0] for n in zipfile.ZipFile(archive).namelist()}
    assert roots == {archive.stem}


def test_staging_twice_does_not_accumulate(tmp_path) -> None:
    """The staging directory is reused between builds, and a stale file left in it ends
    up shipped and checksummed as though it belonged."""
    assert package.main(["--out", str(tmp_path), "--no-docs", "--no-zip"]) == 0
    staged = next(p for p in tmp_path.iterdir() if p.is_dir())
    stray = staged / "left-over.txt"
    stray.write_text("stale", encoding="utf-8")

    assert package.main(["--out", str(tmp_path), "--no-docs", "--no-zip"]) == 0

    assert not stray.exists()
