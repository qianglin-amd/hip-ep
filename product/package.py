# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Assemble the HIP EP delivery bundle.

The bundle is what someone who has never seen this repository receives: the server, the
console, the toolkit, the model registry, the recipes, the samples, the install scripts
and the documentation site, already built, with a manifest recording where it came from.

Two rules shape what goes in. Everything needed to reach a working chat is included, so
the bundle does not assume a checkout of this repository. Nothing that was produced on
the packaging host is included, because a virtualenv, a model cache or a compiled
extension built here is at best useless elsewhere and at worst misleading -- an
artifact that only runs on the machine that made it is the thing this bundle exists to
avoid.

The GPU wheels are optional and passed by path. They are built per ROCm version, per
GPU architecture and per Python version, so a bundle carrying one set of them is only
valid for one kind of machine; the manifest records which.

    python product/package.py --out dist/
    python product/package.py --out dist/ --gpu-wheels ../wheels
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PRODUCT_DIR = Path(__file__).resolve().parent
REPO_DIR = PRODUCT_DIR.parent

# Paths are relative to product/. Directories are copied whole, minus EXCLUDE below.
CONTENTS = (
    "README.md",
    "MILESTONES.md",
    "pyproject.toml",
    "requirements.txt",
    "serve.py",
    "install.ps1",
    "install.sh",
    "install-port-toolchain.ps1",
    "install-port-toolchain.sh",
    "oga-server",
    "toolkit",
    "modelzoo",
    "examples",
    "site",
    "docker",
    "tests",
)

# Build output, caches and virtualenvs. A virtualenv records absolute paths from this
# host and a __pycache__ records this interpreter's version, so both break quietly
# somewhere else rather than failing where they were unpacked.
EXCLUDE = (
    "__pycache__",
    "*.pyc",
    "*.egg-info",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "site/dist",
    # The plan deck and its slide renders are review material, not product.
    "plan",
    # Tests for the packaging tool, which is not shipped: assembling a bundle is
    # something this repository does, not something its recipient does. Left in, it
    # breaks collection for the whole suite the recipient runs to check their install.
    "tests/test_package.py",
)

# Copied from the repository root rather than product/, because a bundle without a
# licence is not deliverable.
ROOT_CONTENTS = ("LICENSE",)


def _excluded(rel: str, name: str) -> bool:
    posix = rel.replace("\\", "/")
    return any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(posix, pat) for pat in EXCLUDE)


def _copy(src: Path, dst: Path, root: Path) -> None:
    rel = str(src.relative_to(root))
    if _excluded(rel, src.name):
        return
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in sorted(src.iterdir()):
            _copy(child, dst / child.name, root)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _git(*args: str) -> str:
    """Return a git value, or "unknown" outside a checkout.

    The bundle is also built from an exported tree, where there is no git to ask.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _version() -> str:
    sys.path.insert(0, str(PRODUCT_DIR / "toolkit"))
    try:
        from hipep_toolkit import __version__

        return __version__
    except Exception:
        return "0.0.0"
    finally:
        sys.path.pop(0)


def build_site(stage: Path) -> bool:
    """Build the documentation into the staged copy.

    Shipping the built site matters more than it looks: the bundle is opened by someone
    deciding whether to install anything at all, and asking them to run a build first
    puts a Python dependency in front of the front page.
    """
    script = stage / "site" / "build.py"
    if not script.exists():
        return False
    out = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=stage / "site",
        # Keeps the generator from leaving this interpreter's bytecode in the bundle.
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stdout + out.stderr)
        return False
    return (stage / "site" / "dist" / "index.html").exists()


def copy_wheels(stage: Path, wheels: Path) -> list[str]:
    names = sorted(p.name for p in wheels.glob("*.whl"))
    if not names:
        raise SystemExit(f"no .whl files in {wheels}")
    target = stage / "wheels"
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(wheels / name, target / name)
    return names


def wheel_python(names: list[str]) -> str | None:
    """The Python version the wheels were built for, from their file names."""
    for name in names:
        for part in name.split("-"):
            if part.startswith("cp3") and part[3:].isdigit():
                return f"3.{part[3:]}"
    return None


def manifest(stage: Path, wheels: list[str]) -> dict:
    """Record what this bundle is and where it came from.

    The provenance fields exist so a bundle found on a share a year from now can be
    traced to a commit, and so a wheel set can be matched to the machine it suits.
    """
    # Asked of the server's own loader rather than counted from the YAML, so the number
    # in the manifest is the number the bundle will actually serve: the loader drops
    # tiers that have not shipped and entries whose certification regressed.
    catalog: dict = {}
    sys.path.insert(0, str(stage / "oga-server"))
    # Importing from the staged tree would otherwise write __pycache__ into it, after
    # the copy had already filtered bytecode out, and ship this interpreter's .pyc files
    # to a machine that may not run this interpreter.
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from hipep_server.catalog import load_catalog

        loaded = load_catalog(
            stage / "modelzoo" / "catalog", stage / "modelzoo" / "benchmarks"
        )
        catalog = {
            "release": (loaded.benchmark_meta or {}).get("release"),
            "models": len(loaded.models),
            "tiers": sorted({m.tier for m in loaded.models}),
        }
    except Exception as exc:
        catalog = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        sys.dont_write_bytecode = bytecode
        sys.path.pop(0)

    recipes = sorted(
        p.name for p in (stage / "toolkit" / "hipep_toolkit" / "recipes").glob("*.yaml")
    )

    return {
        "name": "hip-ep-product",
        "version": _version(),
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "commit": _git("rev-parse", "HEAD"),
            "describe": _git("describe", "--tags", "--always", "--dirty"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        },
        "built_on": {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
        },
        "contents": {
            "catalog": catalog,
            "recipes": recipes,
            "docs_prebuilt": (stage / "site" / "dist" / "index.html").exists(),
            "examples": sorted(
                p.name for p in (stage / "examples").iterdir() if not p.name.startswith(".")
            )
            if (stage / "examples").is_dir()
            else [],
        },
        "gpu_wheels": {
            "included": bool(wheels),
            "python": wheel_python(wheels),
            "files": wheels,
        },
        "requires": {
            "python": ">=3.10 for the server; the GPU wheels pin their own version",
            "porting_toolchain": "3.11-3.13 (AMD Quark does not publish for 3.14)",
            "gpu": "AMD GPU with a current Adrenalin or ROCm install; optional",
        },
    }


def sweep(stage: Path) -> list[str]:
    """Remove bytecode left in the staged tree after the copy filtered it out.

    Building the docs and reading the catalog both import from the stage, so anything
    that runs there can drop a `__pycache__` compiled by the packaging host's
    interpreter. Suppressing that at each call site is easy to forget on the next step
    added here, and a sweep before the manifest and the checksums cannot be.
    """
    removed = []
    for path in sorted(stage.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            removed.append(path.relative_to(stage).as_posix())
            shutil.rmtree(path)
    for path in sorted(stage.rglob("*.py[co]")):
        removed.append(path.relative_to(stage).as_posix())
        path.unlink()
    return removed


def checksums(stage: Path) -> str:
    """A line per file, so a partial copy is detectable without unpacking twice."""
    lines = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(stage).as_posix()}")
    return "\n".join(lines) + "\n"


def zip_bundle(stage: Path, out: Path, name: str) -> Path:
    archive = out / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                zf.write(path, Path(name) / path.relative_to(stage))
    return archive


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO_DIR / "dist", help="where to write")
    ap.add_argument("--gpu-wheels", type=Path, help="directory of AMD GPU wheels to include")
    ap.add_argument("--no-docs", action="store_true", help="skip building the docs site")
    ap.add_argument("--no-zip", action="store_true", help="leave the staged tree, do not zip")
    args = ap.parse_args(argv)

    version = _version()
    name = f"hip-ep-product-{version}"
    out = args.out.resolve()
    stage = out / name

    print(f"==> staging {name}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for entry in CONTENTS:
        src = PRODUCT_DIR / entry
        if not src.exists():
            print(f"  skip  {entry} (missing)")
            continue
        _copy(src, stage / entry, PRODUCT_DIR)
    for entry in ROOT_CONTENTS:
        src = REPO_DIR / entry
        if src.exists():
            shutil.copy2(src, stage / entry)

    if args.no_docs:
        print("  skip  documentation site")
    else:
        print("==> building the documentation site")
        if not build_site(stage):
            raise SystemExit("building the documentation site failed")

    wheels: list[str] = []
    if args.gpu_wheels:
        wheels = copy_wheels(stage, args.gpu_wheels.resolve())
        print(f"==> including {len(wheels)} GPU wheels (Python {wheel_python(wheels)})")

    sweep(stage)

    data = manifest(stage, wheels)
    (stage / "MANIFEST.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (stage / "START-HERE.md").write_text(start_here(data), encoding="utf-8")
    (stage / "SHA256SUMS").write_text(checksums(stage), encoding="utf-8")

    files = sum(1 for p in stage.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print(f"==> {files} files, {size / 1e6:.1f} MB")

    if args.no_zip:
        print(f"\nstaged at {stage}")
        return 0

    archive = zip_bundle(stage, out, name)
    print(f"\n{archive}  ({archive.stat().st_size / 1e6:.1f} MB)")
    return 0


def start_here(data: dict) -> str:
    """The first file someone opens. It answers only: what do I type?"""
    wheels = data["gpu_wheels"]
    if wheels["included"]:
        gpu_note = (
            f"The AMD GPU wheels are in `wheels/`, built for Python "
            f"**{wheels['python']}**. The installer reads that from the file names and "
            f"builds the virtualenv on a matching interpreter, so Python "
            f"{wheels['python']} needs to be installed."
        )
        gpu_windows = ".\\install.ps1 -GpuWheels .\\wheels"
        gpu_linux = "./install.sh --gpu-wheels ./wheels"
    else:
        gpu_note = (
            "This bundle does not carry the AMD GPU wheels, because each set is built "
            "for one ROCm version, one GPU architecture and one Python version. Without "
            "them the server runs on the replay engine: the console, the APIs and the "
            "docs all work, and generation replays certified timings instead of using "
            "the GPU. Add them with `-GpuWheels <dir>` when you have the set for your "
            "machine."
        )
        gpu_windows = ".\\install.ps1"
        gpu_linux = "./install.sh"

    catalog = data["contents"]["catalog"]
    catalog_line = (
        f"{catalog['models']} certified models ({catalog['release']})"
        if catalog.get("models")
        else "the certified model registry"
    )

    return f"""<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP EP {data["version"]}

An OpenAI- and Ollama-compatible server that runs quantized ONNX models on AMD GPUs,
with a console, a model registry, a porting toolkit and the client samples.

## Install

Windows:

```powershell
{gpu_windows}
```

Linux:

```bash
{gpu_linux}
```

The installer creates a virtualenv, installs the server and the `hipep` command, and
finishes by running `hipep doctor`, which reports each precondition separately so a
failure names a layer rather than a symptom.

{gpu_note}

## Use it

```bash
hipep doctor          # what this machine can actually do
hipep serve           # http://127.0.0.1:11434
hipep models          # {catalog_line}
hipep pull <model>    # fetch one
hipep run <model> "hello"
```

Open <http://127.0.0.1:11434> for HIP EP Studio, or point any OpenAI or Ollama client
at the same port:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \\
  -H 'content-type: application/json' \\
  -d '{{"model": "<model>", "messages": [{{"role": "user", "content": "hello"}}]}}'
```

## Port your own model

Porting exports and quantizes a Hugging Face checkpoint into a HIP EP artifact. It
needs a second virtualenv, because AMD Quark does not publish for Python 3.14 and the
toolchain pulls in PyTorch:

```powershell
.\\install-port-toolchain.ps1     # Windows
./install-port-toolchain.sh       # Linux
```

Then, from the **Port a model** page in the console or from the CLI:

```bash
hipep port Qwen/Qwen2.5-0.5B-Instruct --recipe int4-rtn-g128
```

The console checks a model before anything is downloaded: it reads the checkpoint's
`config.json` alone and reports the architecture, the recipe it suggests, and the
quantization block size it will actually use, which is not always the one requested --
a block that does not divide the hidden size produces a corrupt artifact, so it gets
reduced. {len(data["contents"]["recipes"])} recipes ship in
`toolkit/hipep_toolkit/recipes/`.

## Documentation

`site/dist/index.html` is the built site: install guides, the API reference, the model
cards, the published benchmarks and the recipe collection.

## What is in here

| Path | What it is |
|---|---|
| `oga-server/` | The server and HIP EP Studio |
| `toolkit/` | The `hipep` CLI: fetch, export, quantize, convert, validate, benchmark |
| `modelzoo/` | Certified model registry, model cards, published benchmark results |
| `examples/` | C++, Python and C# client samples |
| `site/` | Documentation, prose and generator; `site/dist/` is prebuilt |
| `docker/` | Container image and compose file for serving on Linux |
| `tests/` | The test suite, which needs no GPU |
| `MANIFEST.json` | Version, commit and wheel provenance for this bundle |
| `SHA256SUMS` | A checksum per file |

## Without an AMD GPU

Everything except GPU generation works. The server falls back to a replay engine that
reproduces certified timings, so the console, the APIs, the docs, the samples and the
tests all run on any machine. `hipep doctor` says which engine is live, and
`/v1/status` reports it to clients.
"""


if __name__ == "__main__":
    raise SystemExit(main())
