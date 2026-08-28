# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The certified model catalog.

Loads the YAML tier files under modelzoo/catalog/, merges each tier's `defaults` into
its entries, and attaches the published benchmark results from modelzoo/benchmarks/.

Adding a certified model is a data change: a catalog entry plus a benchmark run. No
code in this module knows the name of any individual model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TIER_ORDER = {"llm": 0, "vlm": 1, "speech": 2, "vision": 3}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge `override` onto a copy of `base`, recursing into nested dicts.

    Tier defaults are shallow-looking but nest one level (quantization, artifact,
    certification), so a plain dict.update would silently drop sibling keys.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class Benchmark:
    """Published results for one model on the certification host."""

    ttft_s: dict[int, float | None] = field(default_factory=dict)
    tps: dict[int, float | None] = field(default_factory=dict)
    rtf: float | None = None
    first_ms: float | None = None
    avg_ms: float | None = None
    qps: float | None = None
    previous: dict[str, Any] = field(default_factory=dict)
    release: str | None = None
    run_date: str | None = None
    device: dict[str, Any] = field(default_factory=dict)

    @property
    def context_lengths(self) -> list[int]:
        return sorted(self.ttft_s)

    def at(self, context: int) -> tuple[float | None, float | None]:
        """Interpolate TTFT and TPS for an arbitrary prompt length.

        Published numbers exist at 128 / 2048 / 16384 only. The replay engine and the
        console both need a value at whatever length the user actually typed, so we
        interpolate linearly in log-context space, which tracks the measured curves
        much better than linear-in-tokens does.
        """
        lengths = self.context_lengths
        if not lengths:
            return None, None

        def pick(series: dict[int, float | None]) -> float | None:
            known = [(n, series[n]) for n in lengths if series.get(n) is not None]
            if not known:
                return None
            if context <= known[0][0]:
                return known[0][1]
            if context >= known[-1][0]:
                return known[-1][1]
            for (x0, y0), (x1, y1) in zip(known, known[1:]):
                if x0 <= context <= x1:
                    import math

                    span = math.log(x1) - math.log(x0)
                    t = 0.0 if span == 0 else (math.log(context) - math.log(x0)) / span
                    return y0 + t * (y1 - y0)
            return known[-1][1]

        return pick(self.ttft_s), pick(self.tps)

    def to_dict(self) -> dict:
        return {
            "ttft_s": {str(k): v for k, v in self.ttft_s.items()},
            "tps": {str(k): v for k, v in self.tps.items()},
            "rtf": self.rtf,
            "first_ms": self.first_ms,
            "avg_ms": self.avg_ms,
            "qps": self.qps,
            "previous": self.previous,
            "release": self.release,
            "run_date": self.run_date,
            "device": self.device,
        }


@dataclass
class Model:
    id: str
    name: str
    tier: str
    type: str
    family: str | None = None
    base_model: str | None = None
    license: str | None = None
    parameters: str | None = None
    active_parameters: str | None = None
    description: str = ""
    task: str | None = None
    quantization: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    vision: dict[str, Any] = field(default_factory=dict)
    footprint: dict[str, Any] = field(default_factory=dict)
    artifact: dict[str, Any] = field(default_factory=dict)
    recipe: str | None = None
    certification: dict[str, Any] = field(default_factory=dict)
    status: str = "certified"
    benchmark: Benchmark | None = None

    @property
    def supports_vision(self) -> bool:
        return bool(self.capabilities.get("vision"))

    @property
    def supports_tools(self) -> bool:
        return bool(self.capabilities.get("tools"))

    @property
    def supports_thinking(self) -> bool:
        return bool(self.capabilities.get("thinking"))

    @property
    def is_generative(self) -> bool:
        return self.tier in {"llm", "vlm"}

    @property
    def default_context(self) -> int:
        return int(self.context.get("default") or self.context.get("max") or 4096)

    @property
    def max_context(self) -> int:
        return int(self.context.get("max") or self.default_context)

    @property
    def min_vram_gb(self) -> float | None:
        value = self.footprint.get("min_vram_gb")
        return float(value) if value is not None else None

    @property
    def quant_label(self) -> str:
        """Short human label such as "INT4 AWQ g128" for lists and cards."""
        q = self.quantization or {}
        scheme = (q.get("scheme") or "").upper()
        if scheme in {"", "NONE"}:
            return str(q.get("weight_dtype", "fp16")).upper()
        dtype = str(q.get("weight_dtype", "int4")).upper()
        parts = [dtype, scheme]
        if q.get("group_size") == "per-group":
            parts.append("per-group")
        elif q.get("group_size"):
            parts.append(f"g{q['group_size']}")
        elif q.get("block_size"):
            parts.append(f"b{q['block_size']}")
        return " ".join(parts)

    def to_dict(self, *, include_benchmark: bool = True) -> dict:
        data = {
            "id": self.id,
            "name": self.name,
            "tier": self.tier,
            "type": self.type,
            "family": self.family,
            "base_model": self.base_model,
            "license": self.license,
            "parameters": self.parameters,
            "active_parameters": self.active_parameters,
            "description": self.description.strip(),
            "task": self.task,
            "quantization": self.quantization,
            "quant_label": self.quant_label,
            "context": self.context,
            "capabilities": self.capabilities,
            "vision": self.vision,
            "footprint": self.footprint,
            "artifact": self.artifact,
            "recipe": self.recipe,
            "certification": self.certification,
            "status": self.status,
        }
        if include_benchmark:
            data["benchmark"] = self.benchmark.to_dict() if self.benchmark else None
        return data


class Catalog:
    """In-memory view of the certified model registry."""

    def __init__(self, models: list[Model], benchmark_meta: dict[str, Any]):
        self._models = {m.id: m for m in models}
        self.benchmark_meta = benchmark_meta

    # --- lookup ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        return self.resolve(model_id) is not None

    def get(self, model_id: str) -> Model | None:
        return self._models.get(model_id)

    def resolve(self, model_id: str) -> Model | None:
        """Resolve a user-supplied name to a catalog entry.

        Accepts the canonical id ("qwen2.5:14b"), a bare family name ("qwen2.5", which
        picks the smallest certified variant), an OpenAI-style name with the tag
        separated by a slash, and the raw dashboard artifact name.
        """
        if not model_id:
            return None
        needle = model_id.strip()
        if needle in self._models:
            return self._models[needle]

        lowered = needle.lower()
        for candidate in (lowered, lowered.replace("/", ":"), lowered.replace("_", "-")):
            if candidate in self._models:
                return self._models[candidate]

        for model in self._models.values():
            if (model.artifact or {}).get("dashboard_name", "").lower() == lowered:
                return model
            if (model.base_model or "").lower() == lowered:
                return model

        # Bare family name: prefer the smallest certified variant so that asking for
        # "qwen2.5" gets something that fits rather than something that does not.
        matches = [m for m in self._models.values() if (m.family or "").lower() == lowered]
        if not matches:
            matches = [m for m in self._models.values() if m.id.split(":", 1)[0] == lowered]
        if matches:
            # Prefer the base variant over a specialisation, so "qwen2.5" resolves to
            # qwen2.5:14b rather than qwen2.5-coder:14b at equal size.
            matches.sort(
                key=lambda m: (
                    m.min_vram_gb or 1e9,
                    0 if m.id.split(":", 1)[0] == lowered else 1,
                    m.id,
                )
            )
            return matches[0]
        return None

    def list(
        self,
        *,
        tier: str | None = None,
        generative_only: bool = False,
        include_failing: bool = True,
    ) -> list[Model]:
        models = list(self._models.values())
        if tier:
            models = [m for m in models if m.tier == tier]
        if generative_only:
            models = [m for m in models if m.is_generative]
        if not include_failing:
            models = [m for m in models if m.status != "failing"]
        models.sort(key=lambda m: (TIER_ORDER.get(m.tier, 9), m.min_vram_gb or 0, m.id))
        return models

    def tiers(self) -> list[str]:
        seen = {m.tier for m in self._models.values()}
        return sorted(seen, key=lambda t: TIER_ORDER.get(t, 9))


def _load_tier(path: Path) -> list[Model]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tier = raw.get("tier") or path.stem
    defaults = raw.get("defaults") or {}
    models: list[Model] = []
    for entry in raw.get("models") or []:
        merged = _deep_merge(defaults, entry)
        merged.setdefault("tier", tier)
        known = {f for f in Model.__dataclass_fields__ if f != "benchmark"}
        kwargs = {k: v for k, v in merged.items() if k in known}
        kwargs.setdefault("type", "text-to-text")
        kwargs.setdefault("name", kwargs.get("id", "unknown"))
        models.append(Model(**kwargs))
    return models


def _latest_benchmark_file(benchmarks_dir: Path) -> Path | None:
    files = sorted(benchmarks_dir.glob("*.yaml"))
    return files[-1] if files else None


def load_catalog(catalog_dir: Path, benchmarks_dir: Path) -> Catalog:
    models: list[Model] = []
    for path in sorted(catalog_dir.glob("*.yaml")):
        models.extend(_load_tier(path))

    meta: dict[str, Any] = {}
    bench_path = _latest_benchmark_file(benchmarks_dir)
    if bench_path:
        raw = yaml.safe_load(bench_path.read_text(encoding="utf-8")) or {}
        meta = {
            "release": raw.get("release"),
            "previous_release": raw.get("previous_release"),
            "run_date": raw.get("run_date"),
            "source": raw.get("source"),
            "harness": raw.get("harness"),
            "device": raw.get("device", {}),
            "summary": raw.get("summary", {}),
            "procyon_v2": raw.get("procyon_v2", {}),
        }
        by_id: dict[str, Benchmark] = {}
        for model_id, values in (raw.get("generative") or {}).items():
            by_id[model_id] = Benchmark(
                ttft_s={int(k): v for k, v in (values.get("ttft_s") or {}).items()},
                tps={int(k): v for k, v in (values.get("tps") or {}).items()},
                previous=values.get("previous") or {},
            )
        for model_id, values in (raw.get("speech") or {}).items():
            by_id[model_id] = Benchmark(
                rtf=values.get("rtf"),
                tps={0: values.get("tps")},
                ttft_s={0: values.get("ttft_s")},
                previous=values.get("previous") or {},
            )
        for model_id, values in (raw.get("vision") or {}).items():
            by_id[model_id] = Benchmark(
                first_ms=values.get("first_ms"),
                avg_ms=values.get("avg_ms"),
                qps=values.get("qps"),
                previous=values.get("previous") or {},
            )
        for model in models:
            bench = by_id.get(model.id)
            if bench:
                bench.release = meta.get("release")
                bench.run_date = meta.get("run_date")
                bench.device = meta.get("device", {})
                model.benchmark = bench

    return Catalog(models, meta)
