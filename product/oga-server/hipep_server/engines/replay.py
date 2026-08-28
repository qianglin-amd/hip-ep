# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Replay engine — certified timings without a GPU.

This engine streams a synthetic response paced to the model's published TTFT and TPS
for the prompt length it was given. It exists so that the console, the API surface, the
documentation examples, the client samples and the test suite all work on a machine
with no AMD GPU and no ONNX Runtime GenAI install.

It is explicit about what it is. Every response is labelled, `/v1/status` reports
`engine: replay`, and the console shows a replay badge. We do not want anyone mistaking
a paced placeholder for model output.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import textwrap
import time
from collections.abc import AsyncIterator

from ..catalog import Model
from .base import (
    Delta,
    Engine,
    GenerationRequest,
    Stopwatch,
    estimate_tokens,
    image_prefill_tokens,
)

# Model-load cost scales with weight size; this is the observed rate for reading a
# converted artifact from warm page cache on the certification host.
_LOAD_SECONDS_PER_GB = 0.35
_MIN_LOAD_SECONDS = 0.4

# Shortest sleep we will ask the event loop for while pacing decode. Windows timer
# granularity is ~15.6 ms and asyncio runs any timer that falls due within one clock
# resolution immediately, so a shorter request does not sleep at all. Tokens are
# coalesced into batches until each sleep clears this bar.
#
# The same granularity also makes each sleep overshoot slightly, so a replayed decode
# rate lands a few percent below the published rate rather than exactly on it. That is
# the accuracy floor of a userspace clock and it errs on the slow side, which is the
# safe direction for a number a user might compare against certified results.
_MIN_PACING_SLEEP = 0.03


class ReplayEngine(Engine):
    name = "replay"
    real = False

    def __init__(self, *, speed: float = 1.0) -> None:
        # `speed` compresses the replay clock. Tests run at speed=0 (instant); the
        # console and demos run at 1.0 so the pacing matches the published numbers.
        self._speed = speed
        self._loaded: dict[str, Model] = {}

    # --- lifecycle ------------------------------------------------------------

    async def load(self, model: Model) -> None:
        if model.id in self._loaded:
            return
        await self._sleep(max(_MIN_LOAD_SECONDS, model.weights_gb * _LOAD_SECONDS_PER_GB))
        self._loaded[model.id] = model

    async def unload(self, model_id: str) -> None:
        self._loaded.pop(model_id, None)

    def loaded_models(self) -> list[str]:
        return list(self._loaded)

    # --- generation -----------------------------------------------------------

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Delta]:
        model = request.model
        await self.load(model)

        prompt_tokens = estimate_tokens(request.prompt_text)
        images = request.images
        if images:
            resize = int((model.vision or {}).get("default_resize_height") or 720)
            prompt_tokens += image_prefill_tokens(len(images), resize)

        ttft, tps = self._timing(model, prompt_tokens)
        text = self._compose(request, prompt_tokens, ttft, tps)
        pieces = _tokenize_for_streaming(text)
        pieces = pieces[: max(1, request.params.max_tokens)]

        watch = Stopwatch()
        await self._sleep(ttft)
        watch.mark_first_token()

        # Pace against an absolute schedule rather than sleeping once per token, so a
        # sleep that returns early cannot compound into a response that streams faster
        # than the rate we are replaying.
        interval = self._scaled(1.0 / tps) if tps > 0 else 0.0
        per_sleep = max(1, math.ceil(_MIN_PACING_SLEEP / interval)) if interval > 0 else 1
        decode_start = time.perf_counter()

        emitted = 0
        for index, piece in enumerate(pieces, start=1):
            yield Delta(text=piece)
            emitted += 1
            if interval <= 0:
                continue
            if index % per_sleep and index != len(pieces):
                continue
            remaining = decode_start + index * interval - time.perf_counter()
            if remaining > 0:
                await asyncio.sleep(remaining)

        watch.finish()
        usage = watch.usage(prompt_tokens, emitted)
        yield Delta(done=True, finish_reason="stop", usage=usage.to_dict())

    # --- internals ------------------------------------------------------------

    def _scaled(self, seconds: float) -> float:
        """Replay-clock seconds converted to wall-clock seconds."""
        if self._speed <= 0:
            return 0.0
        return seconds / self._speed

    async def _sleep(self, seconds: float) -> None:
        wall = self._scaled(seconds)
        if wall <= 0:
            return
        await asyncio.sleep(wall)

    def _timing(self, model: Model, prompt_tokens: int) -> tuple[float, float]:
        """Published TTFT and TPS at this prompt length, with sane fallbacks."""
        if model.benchmark:
            ttft, tps = model.benchmark.at(max(1, prompt_tokens))
            if ttft is not None and tps is not None and tps > 0:
                return float(ttft), float(tps)
        # No published numbers: fall back to a rough size-based model so a
        # newly imported model still streams at a believable rate.
        weights_gb = model.weights_gb
        return 0.2 + weights_gb * 0.02, max(4.0, 320.0 / max(1.0, weights_gb))

    def _compose(
        self, request: GenerationRequest, prompt_tokens: int, ttft: float, tps: float
    ) -> str:
        model = request.model
        user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        images = request.images
        bench = model.benchmark
        release = bench.release if bench else "unpublished"

        head = (
            f"**Replay engine** — this response is generated by `oga-server` without "
            f"running model weights, so the text below is a placeholder. The *timing* "
            f"is real: it reproduces the certified {release} measurement for "
            f"**{model.name}** on {(bench.device or {}).get('name', 'the certification host') if bench else 'the certification host'}."
        )

        facts = textwrap.dedent(
            f"""
            | | |
            |---|---|
            | Model | `{model.id}` — {model.name} |
            | Quantization | {model.quant_label} |
            | Prompt | ~{prompt_tokens} tokens{f" including {len(images)} image(s)" if images else ""} |
            | Replayed TTFT | {ttft:.2f} s |
            | Replayed decode | {tps:.1f} tok/s |
            """
        ).strip()

        if images:
            body = (
                f"You attached {len(images)} image(s). On a GPU host the vision encoder "
                f"runs in fp16 while the text tower stays at {model.quant_label}, which "
                "is why the image contributes most of the time to first token and almost "
                "nothing to the decode rate. Install the GPU stack to get a real "
                "description of the image."
            )
        elif not user.strip():
            body = (
                "Send a message to see the streaming path, or open the Benchmark tab to "
                "replay the full certified matrix."
            )
        else:
            body = _synthetic_answer(user, model)

        tail = (
            "To get real responses, install ONNX Runtime GenAI and the HIP EP wheel on a "
            "machine with an AMD GPU and restart with `HIPEP_ENGINE=oga`. "
            "See the Windows or Linux quick start in the docs."
        )
        return f"{head}\n\n{facts}\n\n{body}\n\n{tail}"


def _synthetic_answer(prompt: str, model: Model) -> str:
    """Deterministic placeholder body, varied by prompt so streaming looks alive.

    Seeded from the prompt so the same question always replays identically, which makes
    console screenshots and doc examples reproducible.
    """
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    shape = rng.choice(["prose", "list", "code"])
    topic = " ".join(prompt.split()[:12]) or "your question"

    if shape == "list":
        return textwrap.dedent(
            f"""
            Regarding *{topic}* — a real run of {model.name} would answer here. The
            replay engine instead demonstrates the parts of the response pipeline the
            console has to handle:

            1. Streamed deltas arriving at the certified decode rate.
            2. Markdown rendering, including tables, lists and emphasis.
            3. Live time-to-first-token and tokens-per-second counters.
            4. Cancellation — press stop and the stream ends immediately.
            """
        ).strip()
    if shape == "code":
        return textwrap.dedent(
            f"""
            Regarding *{topic}* — a real run of {model.name} would answer here. In the
            meantime, this is how you would call this exact model from Python:

            ```python
            from openai import OpenAI

            client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="not-needed")
            stream = client.chat.completions.create(
                model="{model.id}",
                messages=[{{"role": "user", "content": "Hello"}}],
                stream=True,
            )
            for chunk in stream:
                print(chunk.choices[0].delta.content or "", end="")
            ```

            The same endpoint speaks the Ollama protocol, so `ollama run {model.id}`
            against this port works too.
            """
        ).strip()
    return textwrap.dedent(
        f"""
        Regarding *{topic}* — a real run of {model.name} would answer here. The replay
        engine exists so the rest of the product does not have to wait for hardware:
        the API surface, the console, the benchmark harness and the documentation
        examples all behave exactly as they will on a GPU host, at exactly the speed
        the certification run measured. When you point the server at a real AMD GPU,
        only this text changes.
        """
    ).strip()


def _tokenize_for_streaming(text: str) -> list[str]:
    """Split into token-sized pieces that reassemble to the original text exactly."""
    pieces: list[str] = []
    for line in text.splitlines(keepends=True):
        buf = ""
        for ch in line:
            buf += ch
            if ch == " " and len(buf) >= 3:
                pieces.append(buf)
                buf = ""
        if buf:
            pieces.append(buf)
    return pieces or [text]
