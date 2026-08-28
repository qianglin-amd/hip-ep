# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""HTTP routers."""

from . import artifacts, bench, jobs, models, ollama, openai, status

ROUTERS = [
    openai.router,
    ollama.router,
    models.router,
    artifacts.router,
    jobs.router,
    bench.router,
    status.router,
]

__all__ = [
    "ROUTERS",
    "artifacts",
    "bench",
    "jobs",
    "models",
    "ollama",
    "openai",
    "status",
]
