# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Engine selection."""

from __future__ import annotations

import logging

from ..config import Settings
from ..hardware import HostInfo
from .base import (
    ChatMessage,
    Delta,
    Engine,
    EngineError,
    GenerationParams,
    GenerationRequest,
    Usage,
    estimate_tokens,
)
from .oga import OgaEngine, oga_available
from .replay import ReplayEngine

log = logging.getLogger(__name__)

__all__ = [
    "ChatMessage",
    "Delta",
    "Engine",
    "EngineError",
    "GenerationParams",
    "GenerationRequest",
    "OgaEngine",
    "ReplayEngine",
    "Usage",
    "estimate_tokens",
    "select_engine",
]


def select_engine(settings: Settings, host: HostInfo) -> tuple[Engine, str]:
    """Pick an engine and explain the choice.

    The explanation is surfaced in /v1/status and in the console so a user who expected
    GPU inference and got the replay engine is told exactly which precondition failed,
    rather than being left to wonder why everything is suspiciously smooth.
    """
    choice = (settings.engine or "auto").lower()

    if choice == "replay":
        return ReplayEngine(), "replay engine requested via HIPEP_ENGINE"

    if choice == "oga":
        ok, why = oga_available()
        if not ok:
            raise EngineError(f"HIPEP_ENGINE=oga but {why}")
        return OgaEngine(models_dir=settings.models_dir), "OGA engine requested via HIPEP_ENGINE"

    # auto
    ok, why = oga_available()
    if not ok:
        return ReplayEngine(), f"{why}; falling back to the replay engine"
    if not host.has_amd_gpu:
        return ReplayEngine(), "no AMD GPU detected; falling back to the replay engine"
    gpu = host.primary_gpu
    log.info("selected OGA engine on %s (%s)", gpu.name, gpu.gfx_arch)
    return (
        OgaEngine(models_dir=settings.models_dir),
        f"onnxruntime-genai on {gpu.name} ({gpu.gfx_arch})",
    )
