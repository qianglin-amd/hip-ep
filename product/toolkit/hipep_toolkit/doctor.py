# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Preflight checks for a HIP EP install.

The first question a new user asks after a run goes wrong is which layer failed, and
the honest answer usually involves the driver, the wheels, the GPU architecture and the
model store in some combination. This walks those in dependency order and reports each
one separately, so "it doesn't work" turns into a single named precondition.

Checks return data rather than printing, so the CLI renders them and the test suite can
assert on them.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path

OK = "ok"
WARN = "warn"
FAIL = "fail"

MIN_PYTHON = (3, 10)

# The first interpreter no AMD Quark release supports. Serving has no such ceiling, so
# this bounds the porting check alone.
QUARK_MAX_PYTHON = (3, 14)

# Free space below this cannot hold one mid-sized certified model plus the working copy
# a conversion makes of it.
LOW_DISK_GB = 20.0


@dataclass
class Check:
    """One precondition, its verdict, and what to do when it is not met."""

    name: str
    state: str
    detail: str
    remedy: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, state: str, detail: str, remedy: str = "") -> Check:
        check = Check(name=name, state=state, detail=detail, remedy=remedy)
        self.checks.append(check)
        return check

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.state == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.state == WARN]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [
                {"name": c.name, "state": c.state, "detail": c.detail, "remedy": c.remedy}
                for c in self.checks
            ],
        }


def _installed(module: str) -> bool:
    """Whether a module can be imported without importing it.

    Importing to test costs seconds for torch and can abort the process outright when a
    GPU wheel meets a driver it does not match, which is exactly the broken install this
    command has to survive in order to report on.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _check_python(report: Report) -> None:
    version = platform.python_version()
    if sys.version_info >= MIN_PYTHON:
        report.add("python", OK, f"{version} at {sys.executable}")
    else:
        need = ".".join(str(p) for p in MIN_PYTHON)
        report.add(
            "python",
            FAIL,
            f"{version} is older than {need}",
            f"install Python {need} or newer and recreate the environment",
        )


def _check_packages(report: Report) -> None:
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "httpx": "httpx",
        "yaml": "PyYAML",
        "PIL": "Pillow",
        "psutil": "psutil",
    }
    missing = sorted({dist for mod, dist in required.items() if not _installed(mod)})
    if missing:
        report.add(
            "server dependencies",
            FAIL,
            f"missing: {', '.join(missing)}",
            "pip install -e product",
        )
    else:
        report.add("server dependencies", OK, f"{len(required)} packages present")

    if _installed("hipep_server"):
        report.add("server package", OK, "hipep_server importable")
    else:
        report.add(
            "server package",
            FAIL,
            "hipep_server is not importable",
            "pip install -e product",
        )


def _check_gpu(report: Report) -> None:
    try:
        from hipep_server.hardware import detect_host
    except ImportError:
        report.add(
            "gpu",
            WARN,
            "cannot detect without the server package",
            "pip install -e product, then re-run",
        )
        return

    host = detect_host()
    if not host.gpus:
        report.add(
            "gpu",
            WARN,
            "no AMD GPU detected",
            "install the AMD driver (Windows) or ROCm and join the render and video "
            "groups (Linux); the server still runs on the replay engine",
        )
        return

    for gpu in host.gpus:
        memory = f"{gpu.memory_gb:.1f} GB" if gpu.memory_gb else "unknown memory"
        state = OK if gpu.profile != "unknown" else WARN
        remedy = (
            ""
            if state == OK
            else "this architecture has no device profile yet; model fit is a guess"
        )
        report.add(
            f"gpu {gpu.index}",
            state,
            f"{gpu.name} ({gpu.gfx_arch}), {memory}, profile {gpu.profile}",
            remedy,
        )


def _check_engine(report: Report) -> None:
    """Report the engine the server would actually choose, using its own logic."""
    try:
        from hipep_server.engines.oga import oga_available
    except ImportError:
        report.add("engine", WARN, "cannot determine without the server package")
        return

    requested = os.environ.get("HIPEP_ENGINE", "auto").lower()
    available, why = oga_available()
    if available:
        report.add("engine", OK, f"onnxruntime-genai available (HIPEP_ENGINE={requested})")
    elif requested == "oga":
        report.add(
            "engine",
            FAIL,
            f"HIPEP_ENGINE=oga was requested but {why}",
            "install onnxruntime-genai for your GPU, or unset HIPEP_ENGINE",
        )
    else:
        report.add(
            "engine",
            WARN,
            f"{why}; the server will replay published timings instead of inferring",
            "install onnxruntime-genai to run real models",
        )


def _check_quantization(report: Report) -> None:
    """Porting dependencies, which most users never need.

    A missing toolchain is not a broken install: the server serves certified models
    without it, and the pipeline says so rather than failing halfway through a port.
    """
    needed = {"quark": "amd-quark", "torch": "torch", "transformers": "transformers"}
    missing = sorted({dist for mod, dist in needed.items() if not _installed(mod)})
    if not missing:
        report.add("porting toolchain", OK, "Quark, torch and transformers present")
        return

    # Every AMD Quark release so far caps the interpreter below 3.14. On a newer one
    # the install resolves to an unrelated 0.1.0 placeholder that ships no modules, so
    # it reports success and this check still finds nothing -- and the remedy below
    # would send the reader round that loop again without ever explaining it.
    if sys.version_info >= QUARK_MAX_PYTHON:
        major, minor = QUARK_MAX_PYTHON
        report.add(
            "porting toolchain",
            WARN,
            f"amd-quark has no release for Python {platform.python_version()}; "
            "ports will run in simulation",
            f"port from a Python 3.11-{major}.{minor - 1} environment; serving is unaffected",
        )
        return

    report.add(
        "porting toolchain",
        WARN,
        f"missing: {', '.join(missing)}; ports will run in simulation",
        'pip install -e "product[quant]"',
    )


def _check_store(report: Report) -> None:
    try:
        from hipep_server.config import Settings
    except ImportError:
        return

    settings = Settings()
    root = settings.data_dir
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    try:
        root.mkdir(parents=True, exist_ok=True)
        marker = root / ".hipep-write-test"
        marker.write_text("")
        marker.unlink()
    except OSError as exc:
        report.add(
            "model store",
            FAIL,
            f"{root} is not writable ({exc.strerror or exc})",
            "point HIPEP_HOME at a writable directory",
        )
        return

    free_gb = shutil.disk_usage(probe).free / (1024**3)
    state = OK if free_gb >= LOW_DISK_GB else WARN
    remedy = (
        ""
        if state == OK
        else "a certified model is 3-30 GB and a port needs room for a working copy"
    )
    report.add("model store", state, f"{root}, {free_gb:.0f} GB free", remedy)


def _check_catalog(report: Report) -> None:
    try:
        from hipep_server.catalog import load_catalog
        from hipep_server.config import Settings
    except ImportError:
        return

    settings = Settings()
    try:
        catalog = load_catalog(settings.catalog_dir, settings.benchmarks_dir)
    except Exception as exc:  # noqa: BLE001 - any failure here is one message to the user
        report.add(
            "catalog",
            FAIL,
            f"failed to load from {settings.catalog_dir}: {exc}",
            "reinstall the product package, or check HIPEP_HOME",
        )
        return

    count = len(catalog)
    if not count:
        report.add(
            "catalog",
            FAIL,
            f"no models found in {settings.catalog_dir}",
            "reinstall the product package",
        )
        return
    release = (catalog.benchmark_meta or {}).get("release") or "no published numbers"
    report.add("catalog", OK, f"{count} certified models, {release}")


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def _check_server(report: Report, url: str | None) -> None:
    """Whether a server is already answering, and whether the port is otherwise taken.

    A port that is busy but not answering as oga-server is the difference between "you
    already started it" and "something else owns 11434", and those need opposite fixes.
    """
    from .client import DEFAULT_URL, Client, ServerUnreachable

    target = url or os.environ.get("HIPEP_URL") or DEFAULT_URL
    try:
        with Client(target) as client:
            data = client.status()
    except ServerUnreachable:
        host, port = _split_hostport(target)
        if host and _port_in_use(host, port):
            report.add(
                "server",
                WARN,
                f"port {port} on {host} is in use but is not answering as oga-server",
                "stop whatever holds the port, or start with --port on another one",
            )
        else:
            report.add("server", OK, f"not running at {target}; the port is free")
        return
    except Exception as exc:  # noqa: BLE001
        report.add("server", WARN, f"{target} answered but not as expected: {exc}")
        return

    engine = (data.get("engine") or {}).get("name", "unknown")
    load = data.get("load") or {}
    report.add(
        "server",
        OK,
        f"running at {target} on the {engine} engine, "
        f"{load.get('active_requests', 0)} requests in flight",
    )


def _split_hostport(url: str) -> tuple[str | None, int]:
    from urllib.parse import urlsplit

    parts = urlsplit(url if "//" in url else f"http://{url}")
    return parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)


def _check_docs(report: Report) -> None:
    try:
        from hipep_server.config import Settings
    except ImportError:
        return
    dist = Settings().site_dir
    if (dist / "index.html").is_file():
        report.add("documentation", OK, f"built at {dist}")
    else:
        report.add(
            "documentation",
            WARN,
            "the site has not been built, so /docs will 404",
            "python product/site/build.py",
        )


def run_checks(url: str | None = None, *, include_server: bool = True) -> Report:
    """Run every check in dependency order.

    Order matters for readability: a missing package explains the GPU check that follows
    it, so a user reads down the list and stops at the first thing that is actually
    theirs to fix.
    """
    report = Report()
    _check_python(report)
    _check_packages(report)
    _check_gpu(report)
    _check_engine(report)
    _check_quantization(report)
    _check_store(report)
    _check_catalog(report)
    _check_docs(report)
    if include_server:
        _check_server(report, url)
    return report


def platform_summary() -> str:
    machine = platform.machine()
    if sys.platform.startswith("win"):
        return f"Windows {platform.release()} ({machine})"
    if sys.platform.startswith("linux"):
        name = ""
        release = Path("/etc/os-release")
        if release.is_file():
            for line in release.read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    name = line.split("=", 1)[1].strip().strip('"')
                    break
        return f"{name or platform.system()} ({machine})"
    return f"{platform.system()} {platform.release()} ({machine})"
