# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Engine abstraction.

An engine turns a chat request into a stream of text deltas and reports the timing the
rest of the product cares about: time to first token, decode rate and token counts.

Two implementations ship:

* `OgaEngine`   — onnxruntime-genai driving HIP EP on an AMD GPU.
* `ReplayEngine` — no GPU required; reproduces the published certified timings.

The replay engine is not a toy. It is what lets the console, the docs, the API tests
and the demo work on any machine, and it keeps the whole product developable while
hardware is scarce. It never claims to be a real model response.
"""

from __future__ import annotations

import abc
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..catalog import Model


@dataclass
class ChatMessage:
    role: str
    content: str
    images: list[bytes] = field(default_factory=list)

    @property
    def has_images(self) -> bool:
        return bool(self.images)


@dataclass
class GenerationParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    think: bool | None = None


@dataclass
class GenerationRequest:
    model: Model
    messages: list[ChatMessage]
    params: GenerationParams = field(default_factory=GenerationParams)
    session_id: str | None = None

    @property
    def images(self) -> list[bytes]:
        return [img for m in self.messages for img in m.images]

    @property
    def prompt_text(self) -> str:
        return "\n".join(m.content for m in self.messages)


@dataclass
class Delta:
    """One streamed piece of a response."""

    text: str = ""
    done: bool = False
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float = 0.0
    decode_s: float = 0.0
    total_s: float = 0.0

    @property
    def tps(self) -> float:
        return self.completion_tokens / self.decode_s if self.decode_s > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "ttft_s": round(self.ttft_s, 4),
            "decode_s": round(self.decode_s, 4),
            "total_s": round(self.total_s, 4),
            "tps": round(self.tps, 2),
        }


class EngineError(RuntimeError):
    """Raised when an engine cannot serve a request."""


class ModelNotAvailable(EngineError):
    """The model is known but its weights are not on this machine.

    Separate from a general engine failure because the caller can fix it and a
    generic 500 tells an OpenAI client to retry a request that will never succeed.
    """


class Engine(abc.ABC):
    """Base class for inference engines."""

    #: Short identifier surfaced in /v1/status and in the console badge.
    name: str = "base"
    #: Whether responses come from real model weights on real hardware.
    real: bool = False

    @abc.abstractmethod
    async def load(self, model: Model) -> None:
        """Make `model` ready to serve. Must be idempotent."""

    @abc.abstractmethod
    async def unload(self, model_id: str) -> None:
        """Release a model's resources. Must tolerate an unloaded model."""

    @abc.abstractmethod
    def loaded_models(self) -> list[str]:
        """Ids currently resident."""

    @abc.abstractmethod
    async def stream(self, request: GenerationRequest) -> AsyncIterator[Delta]:
        """Yield response deltas, ending with one where `done` is True."""

    async def describe(self) -> dict[str, Any]:
        return {"engine": self.name, "real": self.real, "loaded": self.loaded_models()}


def estimate_tokens(text: str) -> int:
    """Rough token count used for prompt accounting and timing selection.

    Real tokenization happens inside the engine. This estimate only has to be good
    enough to choose a point on the published TTFT curve and to report a plausible
    prompt size, so a heuristic is sufficient and, unlike loading a tokenizer, costs
    nothing and never fails.

    It does have to be roughly unbiased, though: the replay engine picks its TTFT from
    this number, and an estimate that runs 25% long makes a benchmark at a labelled
    context look like a regression against the certified curve for that context.
    """
    if not text:
        return 0

    chars = len(text)
    words = len(text.split())
    # A space-delimited script averages a little over one BPE token per word. A dense
    # script (CJK, and text with no spaces at all) has few "words" and needs the
    # character estimate instead, since BPE splits it far more finely.
    if words and chars / words <= 12:
        return max(1, round(words * 1.1))
    return max(1, chars // 2)


def image_prefill_tokens(count: int, resize_height: int = 720) -> int:
    """Approximate tokens contributed by images in a VLM prompt.

    A 720p image lands near 1100 visual tokens through the patch-14 encoders used by
    the certified VLMs. Used for prompt accounting and to pick the right point on the
    TTFT curve; the real count comes from the processor when running on hardware.
    """
    per_image = max(256, int((resize_height / 720) ** 2 * 1100))
    return count * per_image


class Stopwatch:
    """Measures TTFT and decode time around a stream."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.first_token: float | None = None
        self.end: float | None = None

    def mark_first_token(self) -> None:
        self.mark_first_token_at(time.perf_counter())

    def mark_first_token_at(self, when: float) -> None:
        """Record when the first token existed, which may predate reading it.

        An engine that generates on a worker thread can run ahead of whoever is
        draining its queue. Timing the read instead of the generation charges that
        backlog to TTFT and collapses the decode window, which inflates tokens/second
        to values the hardware never reached.
        """
        if self.first_token is None:
            self.first_token = when

    def finish(self) -> None:
        self.end = time.perf_counter()

    def usage(self, prompt_tokens: int, completion_tokens: int) -> Usage:
        end = self.end if self.end is not None else time.perf_counter()
        ttft = (self.first_token - self.start) if self.first_token else (end - self.start)
        decode = (end - self.first_token) if self.first_token else 0.0
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft,
            decode_s=decode,
            total_s=end - self.start,
        )
