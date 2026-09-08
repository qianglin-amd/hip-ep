# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Build the HIP EP product site: landing page, documentation, model cards, benchmarks.

Everything factual is generated from the model zoo, so a page cannot drift from the
catalog it describes. Prose lives in `content/*.md`; model specifications, quantization
schemas, footprints, published benchmark tables and recipe descriptions are read from
`../modelzoo/` and the toolkit's recipe collection at build time.

The catalog is loaded through oga-server's own loader rather than a second YAML reader,
so the site and the server always agree on what a model is.

    python build.py                 # writes dist/
    python build.py --serve         # writes dist/ and serves it on :8080

Output is plain HTML and one stylesheet: no build step for the reader, no JavaScript
required to read a page, and it can be published to any static host or opened from disk.
"""

from __future__ import annotations

import argparse
import functools
import html
import http.server
import re
import shutil
import socketserver
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SITE_DIR = Path(__file__).resolve().parent
PRODUCT_DIR = SITE_DIR.parent
sys.path.insert(0, str(PRODUCT_DIR / "oga-server"))

try:
    import markdown
except ImportError:  # pragma: no cover - the message is the point
    sys.exit(
        "python-markdown is required to build the site.\n"
        "Run: pip install -e 'product[site]'   (or: pip install markdown)"
    )

from hipep_server.catalog import Catalog, Model, load_catalog  # noqa: E402

CATALOG_DIR = PRODUCT_DIR / "modelzoo" / "catalog"
BENCHMARK_DIR = PRODUCT_DIR / "modelzoo" / "benchmarks"
RECIPE_DIR = PRODUCT_DIR / "toolkit" / "hipep_toolkit" / "recipes"
CONTENT_DIR = SITE_DIR / "content"
ASSET_DIR = SITE_DIR / "assets"

TIER_LABEL = {"llm": "LLM", "vlm": "Vision language", "speech": "Speech", "vision": "Vision"}
CONTEXTS = (128, 2048, 16384)

# The same build output is published to a static host and served by the local server at
# /docs. Only in the second case does a link to the console mean anything, so the link
# ships hidden and site.js reveals it after confirming a server answers at this origin.
STUDIO_LINK = '<a class="to-studio" href="/" hidden>Studio</a>'

DEFAULT_PORT = 11434


# --------------------------------------------------------------------------- helpers


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.{digits}f}"


_ACRONYMS = {"vl": "VL", "vlm": "VLM", "moe": "MoE", "oss": "OSS", "r1": "R1", "gpt": "GPT"}


def family_label(family: str) -> str:
    """Display name for a family id: `qwen3.8-vl` -> `Qwen3.8 VL`."""
    words = []
    for word in family.split("-"):
        words.append(_ACRONYMS.get(word, word[:1].upper() + word[1:]))
    return " ".join(words)


def context_label(tokens: int) -> str:
    if tokens >= 1024 and tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    return f"{tokens:,}"


def delta_badge(
    current: float | None, previous: float | None, *, lower_is_better: bool
) -> str:
    """Release-over-release change, coloured so green always means better."""
    if not current or not previous:
        return ""
    pct = (current - previous) / previous * 100
    if abs(pct) < 0.5:
        return '<span class="delta flat">±0%</span>'
    better = pct < 0 if lower_is_better else pct > 0
    return (
        f'<span class="delta {"up" if better else "down"}">'
        f"{'+' if pct > 0 else ''}{pct:.1f}%</span>"
    )


# ------------------------------------------------------------------------ site model


@dataclass
class Page:
    """One output page. `path` is relative to dist/ and doubles as the nav key."""

    path: str
    title: str
    subtitle: str = ""
    section: str = ""
    body: str = ""
    order: int = 0

    @property
    def depth(self) -> int:
        return self.path.count("/")


@dataclass
class NavItem:
    label: str
    href: str
    children: list[NavItem]


def read_markdown(path: Path) -> tuple[dict[str, Any], str]:
    """Split optional YAML front matter from a content file."""
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        return {}, raw
    _, front, rest = raw.split("---", 2)
    return yaml.safe_load(front) or {}, rest.lstrip("\n")


def render_markdown(source: str) -> str:
    converter = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
        output_format="html5",
    )
    return converter.convert(source)


# ------------------------------------------------------------------- generated pages


def quantization_rows(model: Model) -> list[tuple[str, str]]:
    """The quantization schema, spelled out. This is the detail users ask for first."""
    quant = model.quantization or {}
    labels = [
        ("scheme", "Algorithm", {"awq": "AWQ (activation-aware)", "rtn": "Round to nearest"}),
        ("weight_dtype", "Weight type", {}),
        ("activation_dtype", "Activation type", {}),
        ("symmetry", "Symmetry", {}),
        ("group_size", "Group size", {}),
        ("block_size", "Block size", {}),
        ("kv_cache_dtype", "KV cache", {}),
        ("vision_dtype", "Vision encoder", {}),
        ("chunk_size", "Quark chunk size", {}),
        ("prune_logit", "Pruned LM head", {}),
        ("calibration_dataset", "Calibration data", {}),
    ]
    rows = []
    for key, label, pretty in labels:
        if key not in quant or quant[key] is None:
            continue
        value = quant[key]
        if isinstance(value, bool):
            value = "yes" if value else "no"
        rows.append((label, pretty.get(value, str(value))))
    return rows


def model_card(model: Model, *, release: str, device: dict[str, Any]) -> str:
    """A model card: what it is, how it was quantized, what it needs, how fast it is."""
    bench = model.benchmark
    footprint = model.footprint or {}
    context = model.context or {}
    caps = model.capabilities or {}

    facts = [
        ("Model id", f"<code>{esc(model.id)}</code>"),
        ("Base model", esc(model.base_model or "—")),
        ("Parameters", esc(model.parameters or "—")),
        ("Quantization", esc(model.quant_label)),
        ("Context window", context_label(context["max"]) if context.get("max") else "—"),
        ("Weights on disk", f"{number(footprint.get('weights_gb'), 1)} GB"),
        ("Minimum VRAM", f"{footprint.get('min_vram_gb', '—')} GB"),
        ("License", esc(model.license or "—")),
    ]
    capability_tags = [
        ("vision", "image input"),
        ("tools", "tool calling"),
        ("thinking", "reasoning mode"),
        ("audio", "audio input"),
    ]
    tags = [label for key, label in capability_tags if caps.get(key)]

    source = (model.artifact or {}).get("source") or {}
    repo = source.get("repo")
    dashboard = (model.artifact or {}).get("dashboard_name")

    parts = [f'<section class="card model-card" id="{esc(slug(model.id))}">']
    parts.append(
        f'<header class="model-head"><div><h3>{esc(model.name)}</h3>'
        f'<p class="muted">{esc(model.description or "")}</p></div>'
        f'<div class="tags">{"".join(f"<span class=tag>{esc(t)}</span>" for t in tags)}'
        f'<span class="tag tier-{esc(model.tier)}">{esc(TIER_LABEL.get(model.tier, model.tier))}'
        f"</span></div></header>"
    )

    parts.append('<div class="kv-grid">')
    for label, value in facts:
        parts.append(f'<div class="k">{esc(label)}</div><div class="v">{value}</div>')
    parts.append("</div>")

    parts.append("<h4>Run it</h4>")
    parts.append(
        "<pre><code># download the certified artifact\n"
        f"hipep pull {esc(model.id)}\n\n"
        "# chat from the console, or from any OpenAI client\n"
        f"curl http://127.0.0.1:{DEFAULT_PORT}/v1/chat/completions \\\n"
        '  -H "Content-Type: application/json" \\\n'
        f'  -d \'{{"model": "{esc(model.id)}", '
        '"messages": [{"role": "user", "content": "Hello"}]}\'</code></pre>'
    )

    rows = quantization_rows(model)
    if rows:
        parts.append("<h4>Quantization</h4>")
        parts.append('<table class="data"><tbody>')
        for label, value in rows:
            parts.append(f"<tr><th>{esc(label)}</th><td>{esc(value)}</td></tr>")
        parts.append("</tbody></table>")

    if bench and bench.context_lengths:
        parts.append(f"<h4>Measured performance — {esc(release)}</h4>")
        parts.append(
            f'<p class="muted small">{esc(device.get("name", ""))}, '
            f"{esc(device.get('gpu', ''))} ({esc(device.get('gfx_arch', ''))}), "
            f"{esc(device.get('os', ''))}.</p>"
        )
        parts.append(
            '<table class="data"><thead><tr><th>Prompt</th>'
            "<th class=right>TTFT</th><th class=right>vs previous</th>"
            "<th class=right>Tokens/s</th><th class=right>vs previous</th></tr></thead><tbody>"
        )
        previous = bench.previous or {}
        for ctx in bench.context_lengths:
            ttft = bench.ttft_s.get(ctx)
            tps = bench.tps.get(ctx)
            prev_ttft = (previous.get("ttft_s") or {}).get(ctx)
            prev_tps = (previous.get("tps") or {}).get(ctx)
            parts.append(
                f"<tr><td>{context_label(ctx)} tokens</td>"
                f"<td class=right>{number(ttft, 2)} s</td>"
                f"<td class=right>{delta_badge(ttft, prev_ttft, lower_is_better=True)}</td>"
                f"<td class=right>{number(tps, 1)}</td>"
                f"<td class=right>{delta_badge(tps, prev_tps, lower_is_better=False)}</td></tr>"
            )
        parts.append("</tbody></table>")
    elif bench and bench.rtf is not None:
        parts.append(f"<h4>Measured performance — {esc(release)}</h4>")
        parts.append(
            f"<p>Real-time factor <strong>{number(bench.rtf, 3)}</strong>: "
            f"wall seconds per second of audio, lower is better.</p>"
        )
    elif bench and bench.avg_ms is not None:
        parts.append(f"<h4>Measured performance — {esc(release)}</h4>")
        parts.append(
            f'<table class="data"><tbody>'
            f"<tr><th>First inference</th><td>{number(bench.first_ms, 1)} ms</td></tr>"
            f"<tr><th>Steady state</th><td>{number(bench.avg_ms, 2)} ms</td></tr>"
            f"<tr><th>Throughput</th><td>{number(bench.qps, 1)} inferences/s</td></tr>"
            f"</tbody></table>"
        )
    else:
        parts.append(
            '<p class="notice">No published measurement in this release. '
            "Run one locally from the Benchmarks page and compare.</p>"
        )

    provenance = []
    if repo:
        provenance.append(
            f'Artifact: <a href="https://huggingface.co/{esc(repo)}">{esc(repo)}</a>'
        )
    if model.recipe:
        provenance.append(f"Recipe: <code>{esc(model.recipe)}</code>")
    if dashboard:
        provenance.append(f"Build: <code>{esc(dashboard)}</code>")
    # Stated as a fact rather than as a pass/fail verdict. The site only publishes
    # models that met the bar, so the vocabulary of a test result has nothing left
    # to distinguish and only invites the question of what failed.
    if (model.certification or {}).get("status") == "pass":
        provenance.append(f"Certified in {esc(release)}")
    if provenance:
        parts.append('<p class="muted small provenance">' + " · ".join(provenance) + "</p>")

    parts.append("</section>")
    return "\n".join(parts)


def family_pages(catalog: Catalog, release: str, device: dict[str, Any]) -> list[Page]:
    """One page per model family, in the tier order the catalog defines."""
    pages: list[Page] = []
    grouped: dict[str, list[Model]] = {}
    for model in catalog.models:
        grouped.setdefault(model.family or model.id.split(":")[0], []).append(model)

    for order, (family, models) in enumerate(grouped.items()):
        models = sorted(models, key=lambda m: m.id)
        title = family_label(family)
        tiers = sorted({TIER_LABEL.get(m.tier, m.tier) for m in models})
        count = f"{len(models)} certified {'model' if len(models) == 1 else 'models'}"
        body = [
            f'<p class="lede">{esc(count)} in this family, {esc(" and ".join(tiers))}.</p>'
        ]
        body.extend(model_card(m, release=release, device=device) for m in models)
        pages.append(
            Page(
                path=f"docs/models/{slug(family)}.html",
                title=title,
                subtitle=count,
                section="Models",
                body="\n".join(body),
                order=order + 1,
            )
        )
    return pages


def models_overview(catalog: Catalog, release: str, device: dict[str, Any]) -> Page:
    tiers: dict[str, list[Model]] = {}
    for model in catalog.models:
        tiers.setdefault(model.tier, []).append(model)

    body = [
        f'<p class="lede">{len(catalog.models)} certified models, measured on '
        f"{esc(device.get('name', 'the certification host'))} for {esc(release)}. "
        "Every entry below is reproducible: the artifact, the recipe that produced it and "
        "the measurement that certified it are all named.</p>",
        '<div class="stat-row">',
    ]
    for tier, models in tiers.items():
        body.append(
            f'<div class="stat"><div class="n">{len(models)}</div>'
            f'<div class="l">{esc(TIER_LABEL.get(tier, tier))}</div></div>'
        )
    body.append("</div>")

    for tier, models in tiers.items():
        body.append(f'<h2 id="{esc(tier)}">{esc(TIER_LABEL.get(tier, tier))}</h2>')
        body.append(
            '<table class="data zoo"><thead><tr><th>Model</th><th>Quantization</th>'
            "<th class=right>Params</th><th class=right>Context</th>"
            "<th class=right>VRAM</th><th class=right>TTFT</th>"
            "<th class=right>Tokens/s</th></tr></thead><tbody>"
        )
        for model in sorted(models, key=lambda m: m.id):
            bench = model.benchmark
            ttft = bench.ttft_s.get(2048) if bench else None
            tps = bench.tps.get(2048) if bench else None
            family = slug(model.family or model.id.split(":")[0])
            context = (model.context or {}).get("max")
            body.append(
                f'<tr><td><a href="{family}.html#{esc(slug(model.id))}">'
                f"{esc(model.name)}</a><div class=faint><code>{esc(model.id)}</code></div></td>"
                f"<td>{esc(model.quant_label)}</td>"
                f"<td class=right>{esc(model.parameters or '—')}</td>"
                f"<td class=right>{context_label(context) if context else '—'}</td>"
                f"<td class=right>{(model.footprint or {}).get('min_vram_gb', '—')} GB</td>"
                f"<td class=right>{number(ttft, 2) if ttft else '—'}"
                f"{' s' if ttft else ''}</td>"
                f"<td class=right>{number(tps, 1) if tps else '—'}</td></tr>"
            )
        body.append("</tbody></table>")

    body.append(
        '<p class="muted small">TTFT and tokens/s are shown at a 2,048-token prompt. '
        "Per-context numbers, including 16K, are on each family page and in "
        '<a href="../benchmarks/index.html">Benchmarks</a>.</p>'
    )
    return Page(
        path="docs/models/index.html",
        title="Models",
        subtitle="The certified model zoo",
        section="Models",
        body="\n".join(body),
        order=0,
    )


def benchmark_pages(
    catalog: Catalog, bench_meta: dict[str, Any], device: dict[str, Any]
) -> list[Page]:
    release = bench_meta.get("release", "unreleased")
    previous_release = bench_meta.get("previous_release", "the previous release")
    generative = [m for m in catalog.models if m.benchmark and m.benchmark.context_lengths]

    overview = [
        f'<p class="lede">Published results for {esc(release)}, measured '
        f"{esc(bench_meta.get('run_date', ''))} on "
        f"{esc(device.get('name', ''))}. Deltas compare against {esc(previous_release)}.</p>",
        '<div class="kv-grid">',
        f'<div class="k">Device</div><div class="v">{esc(device.get("name", "—"))}</div>',
        f'<div class="k">GPU</div><div class="v">{esc(device.get("gpu", "—"))} '
        f"({esc(device.get('gfx_arch', '—'))})</div>",
        f'<div class="k">Memory</div><div class="v">{esc(device.get("memory_gb", "—"))} GB</div>',
        f'<div class="k">Operating system</div><div class="v">{esc(device.get("os", "—"))}</div>',
        f'<div class="k">Harness</div><div class="v">'
        f"{esc(bench_meta.get('published_harness') or bench_meta.get('harness', '—'))}</div>",
        "</div>",
        "<h2>How these numbers are produced</h2>",
        "<p>TTFT is the wall time from submitting a request to the first generated token. "
        "Tokens per second is measured over steady-state decode and excludes TTFT. Both are "
        "taken at 128, 2,048 and 16,384-token prompts, because a model that looks fast at "
        "128 tokens can be a different model at 16K.</p>",
        "<p>A local run made from the Benchmarks page in HIP EP Studio uses the same "
        "definitions and the same HTTP path, so your machine's numbers are directly "
        "comparable with the table below. The harness reports the engine measurement and "
        "the end-to-end measurement separately: certified numbers are engine measurements, "
        "and folding HTTP and scheduling cost into them would report transport as a model "
        "regression.</p>",
    ]

    # The per-suite pass/fail summary is not published. It is an internal release gate
    # measured against the previous release, so its "fail" rows are small performance
    # regressions in models that work; printed on a public page they read as defects.

    overview.append("<h2>All generative models</h2>")
    overview.append(
        '<table class="data"><thead><tr><th rowspan=2>Model</th>'
        + "".join(
            f'<th colspan=2 class="center">{context_label(c)} tokens</th>' for c in CONTEXTS
        )
        + "</tr><tr>"
        + "".join("<th class=right>TTFT</th><th class=right>tok/s</th>" for _ in CONTEXTS)
        + "</tr></thead><tbody>"
    )
    for model in sorted(generative, key=lambda m: (m.tier, m.id)):
        bench = model.benchmark
        family = slug(model.family or model.id.split(":")[0])
        cells = []
        for ctx in CONTEXTS:
            ttft = bench.ttft_s.get(ctx)
            tps = bench.tps.get(ctx)
            cells.append(f"<td class=right>{number(ttft, 2) if ttft else '—'}</td>")
            cells.append(f"<td class=right>{number(tps, 1) if tps else '—'}</td>")
        overview.append(
            f'<tr><td><a href="{family}.html#{esc(slug(model.id))}">{esc(model.name)}</a></td>'
            + "".join(cells)
            + "</tr>"
        )
    overview.append("</tbody></table>")

    pages = [
        Page(
            path="docs/benchmarks/index.html",
            title="Benchmarks",
            subtitle=f"Published results, {release}",
            section="Benchmarks",
            body="\n".join(overview),
            order=0,
        )
    ]

    grouped: dict[str, list[Model]] = {}
    for model in generative:
        grouped.setdefault(model.family or model.id.split(":")[0], []).append(model)

    for order, (family, models) in enumerate(grouped.items()):
        title = family_label(family)
        body = [
            f'<p class="lede">{esc(release)} results for {esc(title)}, with the change '
            f"against {esc(previous_release)}.</p>"
        ]
        for model in sorted(models, key=lambda m: m.id):
            bench = model.benchmark
            previous = bench.previous or {}
            body.append(f'<h2 id="{esc(slug(model.id))}">{esc(model.name)}</h2>')
            body.append(
                f'<p class="muted small"><code>{esc(model.id)}</code> · '
                f"{esc(model.quant_label)} · {esc(model.parameters or '')}</p>"
            )
            body.append(
                '<table class="data"><thead><tr><th>Prompt</th>'
                f"<th class=right>TTFT</th><th class=right>{esc(previous_release)}</th>"
                "<th class=right>Change</th>"
                f"<th class=right>Tokens/s</th><th class=right>{esc(previous_release)}</th>"
                "<th class=right>Change</th></tr></thead><tbody>"
            )
            for ctx in bench.context_lengths:
                ttft, tps = bench.ttft_s.get(ctx), bench.tps.get(ctx)
                prev_ttft = (previous.get("ttft_s") or {}).get(ctx)
                prev_tps = (previous.get("tps") or {}).get(ctx)
                body.append(
                    f"<tr><td>{context_label(ctx)}</td>"
                    f"<td class=right>{number(ttft, 2)} s</td>"
                    f"<td class=right class=faint>{number(prev_ttft, 2)} s</td>"
                    f"<td class=right>{delta_badge(ttft, prev_ttft, lower_is_better=True)}</td>"
                    f"<td class=right>{number(tps, 1)}</td>"
                    f"<td class=right class=faint>{number(prev_tps, 1)}</td>"
                    f"<td class=right>{delta_badge(tps, prev_tps, lower_is_better=False)}</td>"
                    "</tr>"
                )
            body.append("</tbody></table>")
            body.append(
                f'<p class="muted small">Model card: '
                f'<a href="../models/{slug(family)}.html#{esc(slug(model.id))}">'
                f"{esc(model.name)}</a></p>"
            )
        pages.append(
            Page(
                path=f"docs/benchmarks/{slug(family)}.html",
                title=title,
                subtitle="Per-context results",
                section="Benchmarks",
                body="\n".join(body),
                order=order + 1,
            )
        )
    return pages


def recipes_page() -> Page:
    """The recipe collection, rendered from the YAML the porting pipeline actually reads."""
    recipes = []
    for path in sorted(RECIPE_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["_file"] = path.name
        recipes.append(data)

    body = [
        f'<p class="lede">{len(recipes)} recipes ship with HIP EP. A recipe is the '
        "complete description of how a family of models is exported and quantized: the "
        "port pipeline reads exactly these files, so a published model and your own port "
        "of the same architecture are produced the same way.</p>",
        "<p>Pick one on the <strong>Port a model</strong> page in HIP EP Studio, pass "
        "<code>--recipe</code> to <code>hipep port</code>, or set "
        "<code>quantization</code> on <code>POST /v1/port</code>. If you are unsure, "
        "<code>int4-rtn-g128</code> needs no calibration data and works on most decoder "
        "architectures.</p>",
        '<table class="data"><thead><tr><th>Recipe</th><th>Quantization</th>'
        "<th>Recommended for</th></tr></thead><tbody>",
    ]
    for recipe in recipes:
        quant = recipe.get("quantization") or {}
        summary = " · ".join(
            str(v)
            for v in [
                quant.get("scheme"),
                quant.get("weight_dtype"),
                f"group {quant['group_size']}" if quant.get("group_size") else None,
                quant.get("symmetry"),
            ]
            if v
        )
        body.append(
            f'<tr><td><a href="#{esc(slug(recipe.get("id", recipe["_file"])))}">'
            f"{esc(recipe.get('id', recipe['_file']))}</a></td>"
            f"<td>{esc(summary or 'none (fp16)')}</td>"
            f"<td>{esc(recipe.get('recommended_for', '—'))}</td></tr>"
        )
    body.append("</tbody></table>")

    for recipe in recipes:
        rid = recipe.get("id", recipe["_file"])
        body.append(f'<section class="card" id="{esc(slug(rid))}">')
        body.append(f"<h3>{esc(recipe.get('name', rid))} <code>{esc(rid)}</code></h3>")
        body.append(f'<p class="muted">{esc(recipe.get("description", ""))}</p>')
        body.append('<div class="kv-grid">')
        for key in ("recommended_for", "applies_to", "produces"):
            if recipe.get(key):
                value = recipe[key]
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                body.append(
                    f'<div class="k">{esc(key.replace("_", " "))}</div>'
                    f'<div class="v">{esc(value)}</div>'
                )
        body.append("</div>")
        for section in ("quantization", "calibration", "export", "build", "validation"):
            if not recipe.get(section):
                continue
            body.append(f"<h4>{esc(section)}</h4>")
            body.append(
                "<pre><code>"
                + esc(yaml.safe_dump(recipe[section], sort_keys=False).strip())
                + "</code></pre>"
            )
        if recipe.get("notes"):
            body.append(f'<p class="notice">{esc(recipe["notes"])}</p>')
        body.append(f'<p class="muted small">File: <code>{esc(recipe["_file"])}</code></p>')
        body.append("</section>")

    return Page(
        path="docs/recipes.html",
        title="Export recipes",
        subtitle="Bring your own model",
        section="Instructions",
        body="\n".join(body),
        order=50,
    )


# ------------------------------------------------------------------------- templating


SECTION_ORDER = ("Instructions", "Models", "Benchmarks")


def nav_tree(pages: Iterable[Page]) -> list[NavItem]:
    """Sidebar: flat entries first, then one group per section, ordered as authored.

    Sections follow SECTION_ORDER rather than the alphabet, because a reader works
    through instructions before browsing models and benchmarks.
    """

    def key(page: Page) -> tuple:
        section_rank = (
            SECTION_ORDER.index(page.section)
            if page.section in SECTION_ORDER
            else len(SECTION_ORDER)
        )
        return (page.section != "", section_rank, page.section, page.order, page.title)

    ordered = sorted(pages, key=key)
    roots: list[NavItem] = []
    groups: dict[str, NavItem] = {}
    for page in ordered:
        if page.path == "index.html":
            continue
        item = NavItem(page.title, page.path, [])
        if not page.section:
            roots.append(item)
            continue
        group = groups.get(page.section)
        if group is None:
            group = NavItem(page.section, "", [])
            groups[page.section] = group
            roots.append(group)
        group.children.append(item)
    return roots


def relative(from_path: str, to_path: str) -> str:
    depth = from_path.count("/")
    return "../" * depth + to_path


def render_nav(items: list[NavItem], current: str, page_path: str) -> str:
    out = ['<nav class="sidebar-nav">']
    for item in items:
        if item.children:
            open_attr = " open" if any(c.href == current for c in item.children) else ""
            out.append(f"<details{open_attr}><summary>{esc(item.label)}</summary><ul>")
            for child in item.children:
                active = ' class="active"' if child.href == current else ""
                out.append(
                    f'<li><a href="{esc(relative(page_path, child.href))}"{active}>'
                    f"{esc(child.label)}</a></li>"
                )
            out.append("</ul></details>")
        else:
            active = ' class="active"' if item.href == current else ""
            out.append(
                f'<a href="{esc(relative(page_path, item.href))}"{active}>{esc(item.label)}</a>'
            )
    out.append("</nav>")
    return "\n".join(out)


def shell(page: Page, nav: list[NavItem], *, release: str) -> str:
    up = "../" * page.depth
    return f"""<!DOCTYPE html>
<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.

Generated by product/site/build.py. Edit the content or the model zoo, not this file.
-->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(page.title)} · HIP EP</title>
<meta name="description" content="{esc(page.subtitle or "HIP EP documentation")}">
<link rel="stylesheet" href="{up}assets/site.css">
</head>
<body class="docs">
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <a class="brand" href="{up}index.html">
    <span class="mark">HIP<span class="accent">EP</span></span>
    <span class="brand-sub">for AMD GPUs</span>
  </a>
  <nav class="topnav">
    <a href="{up}docs/index.html">Docs</a>
    <a href="{up}docs/models/index.html">Models</a>
    <a href="{up}docs/benchmarks/index.html">Benchmarks</a>
    <a href="{up}docs/install-windows.html">Install</a>
    {STUDIO_LINK}
  </nav>
  <span class="release">{esc(release)}</span>
</header>
<div class="layout">
  <aside class="sidebar">
    <input class="filter" type="search" placeholder="Filter pages" aria-label="Filter pages">
    {render_nav(nav, page.path, page.path)}
  </aside>
  <main id="main">
    <div class="page-head">
      <h1>{esc(page.title)}</h1>
      {f'<p class="subtitle">{esc(page.subtitle)}</p>' if page.subtitle else ""}
    </div>
    <article class="prose">
{page.body}
    </article>
    <footer class="page-foot">
      <p>Certified numbers on this page come from {esc(release)} of the HIP EP model zoo.
      Documentation and model cards are generated from
      <code>product/modelzoo/</code>.</p>
    </footer>
  </main>
</div>
<script src="{up}assets/site.js" defer></script>
</body>
</html>
"""


def landing(catalog: Catalog, bench_meta: dict[str, Any], device: dict[str, Any]) -> str:
    release = bench_meta.get("release", "unreleased")
    llms = [m for m in catalog.models if m.tier == "llm"]
    vlms = [m for m in catalog.models if m.tier == "vlm"]
    fastest = sorted(
        (m for m in llms if m.benchmark and m.benchmark.tps.get(2048)),
        key=lambda m: -m.benchmark.tps[2048],
    )[:5]

    highlight_rows = "\n".join(
        f'<tr><td><a href="docs/models/{slug(m.family or m.id)}.html#{slug(m.id)}">'
        f"{esc(m.name)}</a></td>"
        f"<td>{esc(m.quant_label)}</td>"
        f"<td class=right>{number(m.benchmark.ttft_s.get(2048), 2)} s</td>"
        f"<td class=right>{number(m.benchmark.tps.get(2048), 1)}</td></tr>"
        for m in fastest
    )

    return f"""<!DOCTYPE html>
<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.

Generated by product/site/build.py.
-->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HIP EP · Run LLMs and VLMs on your AMD GPU</title>
<meta name="description" content="HIP EP is an ONNX Runtime execution provider for AMD
GPUs, with a local OpenAI- and Ollama-compatible server, a certified model zoo and
published benchmarks.">
<link rel="stylesheet" href="assets/site.css">
</head>
<body class="home">
<header class="topbar">
  <a class="brand" href="index.html">
    <span class="mark">HIP<span class="accent">EP</span></span>
    <span class="brand-sub">for AMD GPUs</span>
  </a>
  <nav class="topnav">
    <a href="docs/index.html">Docs</a>
    <a href="docs/models/index.html">Models</a>
    <a href="docs/benchmarks/index.html">Benchmarks</a>
    <a href="docs/install-windows.html">Install</a>
    {STUDIO_LINK}
  </nav>
  <span class="release">{esc(release)}</span>
</header>

<section class="hero">
  <div class="hero-copy">
    <h1>Your AMD GPU is a local AI machine.<br><span class="accent">Use all of it.</span></h1>
    <p class="lede">HIP EP runs language and vision-language models on AMD GPUs through
    ONNX Runtime, with a server that speaks the OpenAI and Ollama APIs, a zoo of
    {len(catalog.models)} certified models and benchmarks you can reproduce on your own
    machine in a minute.</p>
    <div class="cta">
      <a class="btn primary" href="docs/install-windows.html">Install on Windows</a>
      <a class="btn" href="docs/install-linux.html">Install on Linux</a>
      <a class="btn ghost" href="docs/models/index.html">Browse {len(catalog.models)} models</a>
    </div>
    <p class="muted small">Windows and Linux · Ryzen AI Max+ (Strix Halo), Ryzen AI,
    Radeon RX and Instinct · no cloud, no account, no telemetry</p>
  </div>
  <div class="hero-card">
    <div class="term">
      <div class="term-head"><span></span><span></span><span></span></div>
<pre><code>$ .\\product\\install.ps1
$ hipep serve
  engine   hip-ep on gfx1151
  catalog  {len(catalog.models)} certified models
  listen   http://127.0.0.1:{DEFAULT_PORT}

$ hipep pull llama3.1:8b
$ open http://127.0.0.1:{DEFAULT_PORT}</code></pre>
    </div>
  </div>
</section>

<section class="band">
  <div class="stat-row wide">
    <div class="stat"><div class="n">{len(llms)}</div><div class="l">certified LLMs</div></div>
    <div class="stat"><div class="n">{len(vlms)}</div><div class="l">vision language
      models</div></div>
    <div class="stat"><div class="n">2</div><div class="l">API surfaces: OpenAI and
      Ollama</div></div>
    <div class="stat"><div class="n">{esc(release)}</div><div class="l">published
      benchmarks</div></div>
  </div>
</section>

<section class="features">
  <h2>What you get</h2>
  <div class="cards">
    <article class="card">
      <h3>One server, every client</h3>
      <p>oga-server answers the OpenAI and Ollama APIs on the same port, so Open WebUI,
      Continue, LangChain and anything else that takes a base URL works unchanged.
      Streaming responses carry time to first token and tokens per second in the usage
      block.</p>
      <a href="docs/api.html">API and client usage →</a>
    </article>
    <article class="card">
      <h3>A console, not a curl loop</h3>
      <p>HIP EP Studio ships with the server: chat, image and text with a VLM, the model
      zoo, benchmarks and live server load for every connected user. No build step and no
      internet access required.</p>
      <a href="docs/server-mode.html">Server mode →</a>
    </article>
    <article class="card">
      <h3>Certified models, published numbers</h3>
      <p>Every model in the zoo names the artifact, the recipe that produced it and the
      measurement that certified it, at 128, 2K and 16K prompts, with the change against
      the previous release.</p>
      <a href="docs/benchmarks/index.html">Benchmarks →</a>
    </article>
    <article class="card">
      <h3>Bring your own model</h3>
      <p>Point the port page at a Hugging Face repo. HIP EP fetches it, quantizes it with
      Quark, converts it, validates it, benchmarks it, and registers it in your zoo — and
      it is usable the moment the job finishes.</p>
      <a href="docs/port-a-model.html">Port a model →</a>
    </article>
    <article class="card">
      <h3>Samples you can paste</h3>
      <p>Python, C++ and C#/.NET clients that stream and report their own TTFT, plus curl
      and Ollama recipes. The C++ sample builds with CMake and no package manager on both
      platforms.</p>
      <a href="docs/api.html#examples">Client examples →</a>
    </article>
    <article class="card">
      <h3>Develop before the hardware arrives</h3>
      <p>The replay engine streams placeholder text at each model's published TTFT and
      decode rate, so the whole product works with no GPU. Replayed responses are labelled
      everywhere they appear.</p>
      <a href="docs/index.html#replay">How replay works →</a>
    </article>
  </div>
</section>

<section class="band highlight">
  <h2>Fastest certified models at a 2,048-token prompt</h2>
  <p class="muted">Measured on {esc(device.get("name", "the certification host"))},
  {esc(device.get("gpu", ""))} ({esc(device.get("gfx_arch", ""))}), {esc(release)}.</p>
  <table class="data">
    <thead><tr><th>Model</th><th>Quantization</th><th class=right>TTFT</th>
      <th class=right>Tokens/s</th></tr></thead>
    <tbody>
{highlight_rows}
    </tbody>
  </table>
  <p><a class="btn" href="docs/benchmarks/index.html">All published results</a></p>
</section>

<section class="quickstart">
  <h2>Sixty seconds to your first token</h2>
  <div class="two">
    <div>
      <h3>Windows</h3>
<pre><code>git clone &lt;this repository&gt;
.\\product\\install.ps1
hipep serve
hipep pull llama3.1:8b</code></pre>
      <p class="muted small">Requires an AMD GPU with a current Adrenalin driver.
      <a href="docs/install-windows.html">Full instructions →</a></p>
    </div>
    <div>
      <h3>Linux</h3>
<pre><code>git clone &lt;this repository&gt;
./product/install.sh
hipep serve
hipep pull llama3.1:8b</code></pre>
      <p class="muted small">Requires ROCm on a supported kernel.
      <a href="docs/install-linux.html">Full instructions →</a></p>
    </div>
  </div>
  <p>Then open <code>http://127.0.0.1:{DEFAULT_PORT}</code> for the console, or point any
  OpenAI client at <code>http://127.0.0.1:{DEFAULT_PORT}/v1</code>.</p>
</section>

<footer class="site-foot">
  <div>
    <strong>HIP EP</strong> · ONNX Runtime execution provider for AMD GPUs
    <p class="muted small">Certified numbers from {esc(release)}, measured
    {esc(bench_meta.get("run_date", ""))}. Documentation, model cards and benchmark tables
    are generated from the model zoo in this repository.</p>
  </div>
  <nav>
    <a href="docs/index.html">Overview</a>
    <a href="docs/install-windows.html">Windows</a>
    <a href="docs/install-linux.html">Linux</a>
    <a href="docs/models/index.html">Models</a>
    <a href="docs/benchmarks/index.html">Benchmarks</a>
    <a href="docs/recipes.html">Recipes</a>
    <a href="docs/troubleshooting.html">Troubleshooting</a>
  </nav>
</footer>
<script src="assets/site.js" defer></script>
</body>
</html>
"""


# ------------------------------------------------------------------------------ build


def content_pages() -> list[Page]:
    pages = []
    for path in sorted(CONTENT_DIR.glob("*.md")):
        front, source = read_markdown(path)
        pages.append(
            Page(
                path=front.get("path") or f"docs/{path.stem}.html",
                title=front.get("title", path.stem),
                subtitle=front.get("subtitle", ""),
                section=front.get("section", ""),
                body=render_markdown(source),
                order=int(front.get("order", 0)),
            )
        )
    return pages


_LINK = re.compile(r'(?:href|src)="([^"]+)"')


def check_links(out_dir: Path) -> list[str]:
    """Verify every internal link and fragment resolves to something written.

    Most of this site's navigation is generated from the model zoo, so a renamed
    family or a retired model silently breaks links in pages nobody edited. Checking
    at build time is the only point where that is cheap to catch.
    """
    pages = {p.relative_to(out_dir).as_posix(): p for p in out_dir.rglob("*.html")}
    anchors: dict[str, set[str]] = {}
    for name, path in pages.items():
        text = path.read_text(encoding="utf-8")
        anchors[name] = set(re.findall(r'id="([^"]+)"', text))

    problems: list[str] = []
    for name, path in pages.items():
        base = Path(name).parent
        for target in _LINK.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#", "//")):
                continue
            url, _, fragment = target.partition("#")
            if not url:
                continue
            resolved = (base / url).as_posix()
            # posixpath does not normalise "a/../b" on its own.
            parts: list[str] = []
            for piece in resolved.split("/"):
                if piece == "..":
                    if parts:
                        parts.pop()
                elif piece not in ("", "."):
                    parts.append(piece)
            resolved = "/".join(parts)
            if resolved in pages:
                if fragment and fragment not in anchors[resolved]:
                    problems.append(f"{name}: '{target}' has no anchor '{fragment}'")
                continue
            if not (out_dir / resolved).exists():
                problems.append(f"{name}: '{target}' does not exist")
    return sorted(problems)


def build(out_dir: Path) -> int:
    # load_catalog already returns only what has shipped, so the site inherits that
    # boundary rather than defining a second, drifting version of it.
    catalog = load_catalog(CATALOG_DIR, BENCHMARK_DIR)
    bench_file = sorted(BENCHMARK_DIR.glob("*.yaml"))[-1]
    bench_meta = yaml.safe_load(bench_file.read_text(encoding="utf-8")) or {}
    device = bench_meta.get("device") or {}
    release = bench_meta.get("release", "unreleased")

    pages = content_pages()
    pages.append(models_overview(catalog, release, device))
    pages.extend(family_pages(catalog, release, device))
    pages.extend(benchmark_pages(catalog, bench_meta, device))
    pages.append(recipes_page())

    nav = nav_tree(pages)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "assets").mkdir(parents=True)

    for asset in ASSET_DIR.iterdir():
        if asset.is_file():
            shutil.copy2(asset, out_dir / "assets" / asset.name)

    (out_dir / "index.html").write_text(landing(catalog, bench_meta, device), encoding="utf-8")

    for page in pages:
        target = out_dir / page.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(shell(page, nav, release=release), encoding="utf-8")

    return len(pages) + 1


def serve(out_dir: Path, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    with socketserver.TCPServer(("127.0.0.1", port), handler) as server:
        print(f"serving {out_dir} on http://127.0.0.1:{port} (Ctrl-C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=SITE_DIR / "dist")
    ap.add_argument("--serve", action="store_true", help="serve the output after building")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    written = build(args.out)
    print(f"wrote {written} pages to {args.out}")

    problems = check_links(args.out)
    if problems:
        print(f"\n{len(problems)} broken internal link(s):")
        for problem in problems:
            print(f"  {problem}")
        return 1

    if args.serve:
        serve(args.out, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
