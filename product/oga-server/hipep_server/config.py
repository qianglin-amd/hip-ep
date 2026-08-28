# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Server configuration and on-disk layout.

Every path the server touches is derived here so that a deployment can be relocated
with one environment variable, and so tests can point the whole server at a temporary
directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SERVER_DIR = PACKAGE_DIR.parent
PRODUCT_DIR = SERVER_DIR.parent
REPO_DIR = PRODUCT_DIR.parent

# 11434 is Ollama's port. Being drop-in on the default port is the single biggest
# compatibility win available to us: existing Ollama clients need zero configuration.
DEFAULT_PORT = 11434


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    """Resolved server settings.

    Constructed once at startup and stored on `app.state.settings`.
    """

    host: str = field(default_factory=lambda: os.environ.get("HIPEP_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("HIPEP_PORT", DEFAULT_PORT))

    # Where downloaded and converted models live. Kept outside the repo by default so
    # a `git clean` never deletes tens of gigabytes of weights.
    data_dir: Path = field(
        default_factory=lambda: _env_path("HIPEP_HOME", Path.home() / ".hipep")
    )

    # Read-only product data shipped with the repo.
    catalog_dir: Path = field(default_factory=lambda: PRODUCT_DIR / "modelzoo" / "catalog")
    benchmarks_dir: Path = field(
        default_factory=lambda: PRODUCT_DIR / "modelzoo" / "benchmarks"
    )
    # Recipes live inside the hipep_toolkit package so that they ship in its wheel and
    # a pip install of the toolkit alone can still port a model.
    recipes_dir: Path = field(
        default_factory=lambda: PRODUCT_DIR / "toolkit" / "hipep_toolkit" / "recipes"
    )
    web_dir: Path = field(default_factory=lambda: PACKAGE_DIR / "web")
    # The generated documentation site, not its sources. Absent until someone runs
    # product/site/build.py, which the server treats as optional.
    site_dir: Path = field(default_factory=lambda: PRODUCT_DIR / "site" / "dist")

    # Engine selection. "auto" picks OGA when onnxruntime_genai and an AMD GPU are both
    # present, and the replay engine otherwise.
    engine: str = field(default_factory=lambda: os.environ.get("HIPEP_ENGINE", "auto"))

    # Replay-clock compression. 1.0 replays the certified timings in real time, which is
    # what a demo wants; 0 removes the pacing entirely so a test suite is not forced to
    # spend certified seconds waiting for placeholder text.
    replay_speed: float = field(default_factory=lambda: _env_float("HIPEP_REPLAY_SPEED", 1.0))

    # Scheduler limits. max_loaded_models is an admission-control bound, not a queue
    # bound: exceeding it evicts the least recently used model rather than failing.
    max_loaded_models: int = field(default_factory=lambda: _env_int("HIPEP_MAX_LOADED", 2))
    max_concurrent_per_model: int = field(
        default_factory=lambda: _env_int("HIPEP_MAX_CONCURRENT", 4)
    )
    max_queue_depth: int = field(default_factory=lambda: _env_int("HIPEP_MAX_QUEUE", 64))
    idle_unload_seconds: int = field(
        default_factory=lambda: _env_int("HIPEP_IDLE_UNLOAD", 900)
    )

    # Generation defaults, overridable per request.
    default_max_tokens: int = field(default_factory=lambda: _env_int("HIPEP_MAX_TOKENS", 512))

    # Allow cross-origin browser clients. On by default because the console is often
    # opened from a different host than the one serving the models.
    cors: bool = field(default_factory=lambda: _env_bool("HIPEP_CORS", True))

    # Job pipeline concurrency. Conversion and quantization are memory hungry; running
    # more than one at a time on a single box reliably ends in an OOM.
    max_concurrent_jobs: int = field(default_factory=lambda: _env_int("HIPEP_MAX_JOBS", 1))

    # Clock compression for simulated pipeline stages, which exist only on a host with
    # no export toolchain. 1.0 runs the script at demo pace; a test suite raises it so
    # the stage sequencing can be verified without waiting out the pretend work.
    simulation_speed: float = field(default_factory=lambda: _env_float("HIPEP_SIM_SPEED", 1.0))

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{host}:{self.port}"

    def ensure_dirs(self) -> None:
        for path in (self.models_dir, self.jobs_dir, self.cache_dir, self.results_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
