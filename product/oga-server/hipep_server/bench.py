# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Benchmark harness.

Measurements are taken over the server's own HTTP API rather than by calling the engine
directly. That is a deliberate constraint: it means every number we publish came from
the same path a user's application takes, including request parsing, scheduling and
streaming overhead. A number produced by a private fast path is not a number a user can
reproduce, and is therefore not worth publishing.

Metrics match the internal dashboard so results are directly comparable:

  TTFT  wall seconds from request submission to the first content token
  TPS   generated tokens divided by the decode window, excluding TTFT
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Any

import httpx

from .catalog import Model
from .config import Settings
from .jobs import JobContext

# Prompt filler chosen to tokenize predictably: common short English words average
# close to one token each, so repeating them lands within a few percent of the
# requested prompt length across every tokenizer the certified models use.
_FILLER = (
    "the quick brown fox jumps over the lazy dog while a bright red bird sings near "
    "an old stone wall in the cool morning air of a quiet green valley "
)


def build_prompt(target_tokens: int) -> str:
    """A prompt of approximately `target_tokens` tokens."""
    words = _FILLER.split()
    needed = max(1, int(target_tokens / 1.05))
    out = [words[i % len(words)] for i in range(needed)]
    return " ".join(out)


class BenchRunner:
    def __init__(self, settings: Settings, model: Model, params: dict[str, Any]):
        self.settings = settings
        self.model = model
        self.contexts: list[int] = [int(c) for c in params["contexts"]]
        self.generate_tokens: int = int(params["generate_tokens"])
        self.repeats: int = int(params["repeats"])
        # Not `self.warmup`: that would shadow the warmup() stage method.
        self.warmup_runs: int = int(params["warmup"])
        self.samples: dict[int, list[dict[str, float]]] = {c: [] for c in self.contexts}
        self.started_at = time.time()

    @property
    def url(self) -> str:
        return f"{self.settings.base_url}/v1/chat/completions"

    async def _one(self, client: httpx.AsyncClient, context: int) -> dict[str, float]:
        """Run one streamed generation and time it from the client side."""
        body = {
            "model": self.model.id,
            "messages": [{"role": "user", "content": build_prompt(context)}],
            "max_tokens": self.generate_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        start = time.perf_counter()
        first: float | None = None
        tokens = 0
        usage: dict[str, Any] = {}

        async with client.stream("POST", self.url, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    raise RuntimeError(chunk["error"].get("message", "generation failed"))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                text = delta.get("content")
                if text:
                    if first is None:
                        first = time.perf_counter()
                    tokens += 1

        end = time.perf_counter()
        ttft = (first - start) if first else (end - start)
        decode = (end - first) if first else 0.0
        return {
            "ttft_s": ttft,
            "decode_s": decode,
            "tokens": float(tokens),
            "tps": tokens / decode if decode > 0 else 0.0,
            # Server-side view, for comparison against the client-side numbers above.
            "server_ttft_s": float(usage.get("ttft_s") or 0.0),
            "server_tps": float(usage.get("tps") or 0.0),
            "prompt_tokens": float(usage.get("prompt_tokens") or 0.0),
        }

    async def prepare(self, ctx: JobContext) -> None:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(f"{self.settings.base_url}/v1/status")
            response.raise_for_status()
            status = response.json()
        engine = status.get("engine", {})
        ctx.log(f"engine: {engine.get('name')} (real={engine.get('real')})")
        if not engine.get("real"):
            ctx.log(
                "the server is on the replay engine, so these results reproduce the "
                "published certified timings rather than measuring this machine",
                level="warning",
            )
        ctx.result(engine=engine, device=status.get("host", {}).get("gpus", []))
        ctx.progress(1.0)

    async def warmup(self, ctx: JobContext) -> None:
        if self.warmup_runs <= 0:
            ctx.log("warmup disabled")
            ctx.progress(1.0)
            return
        timeout = httpx.Timeout(600.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            total = self.warmup_runs * len(self.contexts)
            done = 0
            for context in self.contexts:
                for _ in range(self.warmup_runs):
                    ctx.check_cancelled()
                    result = await self._one(client, context)
                    done += 1
                    ctx.progress(
                        done / total,
                        f"context {context}: {result['ttft_s']:.2f}s TTFT (discarded)",
                    )
        ctx.log("warmup complete; first-run compile cost excluded from results")

    async def measure(self, ctx: JobContext) -> None:
        timeout = httpx.Timeout(600.0, connect=10.0)
        total = self.repeats * len(self.contexts)
        done = 0
        async with httpx.AsyncClient(timeout=timeout) as client:
            for context in self.contexts:
                for rep in range(self.repeats):
                    ctx.check_cancelled()
                    result = await self._one(client, context)
                    self.samples[context].append(result)
                    done += 1
                    ctx.log(
                        f"context {context:>6} rep {rep + 1}/{self.repeats}: "
                        f"TTFT {result['ttft_s']:.3f}s, "
                        f"{result['tps']:.1f} tok/s over {int(result['tokens'])} tokens"
                    )
                    ctx.progress(done / total, f"context {context}, rep {rep + 1}")

    async def summarize(self, ctx: JobContext) -> dict[str, Any]:
        contexts: dict[str, Any] = {}
        for context, runs in self.samples.items():
            if not runs:
                continue
            ttfts = [r["ttft_s"] for r in runs]
            tpss = [r["tps"] for r in runs if r["tps"] > 0]
            contexts[str(context)] = {
                "ttft_s": round(statistics.median(ttfts), 4),
                "ttft_min_s": round(min(ttfts), 4),
                "ttft_max_s": round(max(ttfts), 4),
                "tps": round(statistics.median(tpss), 2) if tpss else 0.0,
                "tps_min": round(min(tpss), 2) if tpss else 0.0,
                "tps_max": round(max(tpss), 2) if tpss else 0.0,
                "runs": len(runs),
                "prompt_tokens": int(statistics.median(r["prompt_tokens"] for r in runs)),
            }

        published = {}
        if self.model.benchmark:
            bench = self.model.benchmark
            for context in self.contexts:
                ttft, tps = bench.at(context)
                published[str(context)] = {"ttft_s": ttft, "tps": tps}

        deltas = {}
        for key, measured in contexts.items():
            ref = published.get(key) or {}
            if ref.get("ttft_s") and ref.get("tps"):
                deltas[key] = {
                    "ttft_pct": round(
                        (measured["ttft_s"] - ref["ttft_s"]) / ref["ttft_s"] * 100, 1
                    ),
                    "tps_pct": round(
                        (measured["tps"] - ref["tps"]) / ref["tps"] * 100, 1
                    ),
                }

        summary = {
            "model": self.model.id,
            "model_name": self.model.name,
            "quantization": self.model.quant_label,
            "generate_tokens": self.generate_tokens,
            "repeats": self.repeats,
            "warmup": self.warmup_runs,
            "started_at": self.started_at,
            "finished_at": time.time(),
            "contexts": contexts,
            "published": published,
            "delta_vs_published": deltas,
            "harness": "oga-server /v1/chat/completions (streamed)",
        }
        for key, values in contexts.items():
            line = f"context {key:>6}: TTFT {values['ttft_s']:.3f}s, {values['tps']:.1f} tok/s"
            if key in deltas:
                line += (
                    f"  (vs published: TTFT {deltas[key]['ttft_pct']:+.1f}%, "
                    f"TPS {deltas[key]['tps_pct']:+.1f}%)"
                )
            ctx.log(line)
        ctx.progress(1.0)
        return summary
