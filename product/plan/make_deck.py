#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Generate the HIP EP product review deck from the product plan.

The narrative source of truth is PRODUCT_PLAN.md; this script renders the review
deck so the two never drift apart by hand:

    python product/plan/make_deck.py [--out path.pptx]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --- design system -------------------------------------------------------------

INK = RGBColor(0x10, 0x14, 0x18)
INK_SOFT = RGBColor(0x53, 0x5C, 0x68)
ACCENT = RGBColor(0xE8, 0x11, 0x2D)  # AMD red
TEAL = RGBColor(0x00, 0xA3, 0xB5)
AMBER = RGBColor(0xE8, 0x8A, 0x00)
GREEN = RGBColor(0x18, 0x92, 0x5E)
PANEL = RGBColor(0xF3, 0xF5, 0xF7)
PANEL_EDGE = RGBColor(0xDD, 0xE2, 0xE8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Segoe UI"
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.62)
BODY_TOP = Inches(1.62)
CONTENT_W = SLIDE_W - 2 * MARGIN


# --- primitives ----------------------------------------------------------------


def _set_text(frame, runs, *, align=PP_ALIGN.LEFT, space_after=Pt(4), line=None):
    """Fill a text frame from a list of (text, size, bold, color) tuples."""
    frame.word_wrap = True
    for i, (text, size, bold, color) in enumerate(runs):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.alignment = align
        para.space_after = space_after
        if line:
            para.line_spacing = line
        run = para.add_run()
        run.text = text
        run.font.size = size
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = FONT
    return frame


def textbox(slide, left, top, width, height, runs, **kw):
    box = slide.shapes.add_textbox(left, top, width, height)
    _set_text(box.text_frame, runs, **kw)
    return box


def panel(slide, left, top, width, height, *, fill=PANEL, edge=PANEL_EDGE, radius=0.04):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.adjustments[0] = radius
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if edge is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = edge
        shape.line.width = Pt(1)
    shape.shadow.inherit = False
    shape.text_frame.word_wrap = True
    return shape


def bar(slide, left, top, width, height, color):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def new_slide(prs, title, kicker=None, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bar(slide, Emu(0), Emu(0), SLIDE_W, Inches(0.075), ACCENT)
    top = Inches(0.5)
    if kicker:
        textbox(
            slide,
            MARGIN,
            top,
            CONTENT_W,
            Inches(0.26),
            [(kicker.upper(), Pt(10.5), True, ACCENT)],
            space_after=Pt(0),
        )
        top = Inches(0.78)
    textbox(
        slide,
        MARGIN,
        top,
        CONTENT_W,
        Inches(0.5),
        [(title, Pt(28), True, INK)],
        space_after=Pt(0),
    )
    if subtitle:
        textbox(
            slide,
            MARGIN,
            Inches(1.3),
            CONTENT_W,
            Inches(0.32),
            [(subtitle, Pt(12.5), False, INK_SOFT)],
            space_after=Pt(0),
        )
    return slide


def table(
    slide,
    left,
    top,
    width,
    rows,
    col_ratios,
    *,
    header=True,
    row_h=Inches(0.34),
    font=Pt(10.5),
    zebra=True,
    colors=None,
):
    """Lightweight table built from shapes so we control the look precisely."""
    total = sum(col_ratios)
    widths = [Emu(int(width * r / total)) for r in col_ratios]
    y = top
    for r, row in enumerate(rows):
        h = Inches(0.4) if (header and r == 0) else row_h
        if header and r == 0:
            bar(slide, left, y, width, h, INK)
        elif zebra and r % 2 == 0:
            bar(slide, left, y, width, h, PANEL)
        x = left
        for c, cell in enumerate(row):
            fg = WHITE if (header and r == 0) else INK
            if colors and (r, c) in colors:
                fg = colors[(r, c)]
            tb = slide.shapes.add_textbox(x + Inches(0.09), y, widths[c] - Inches(0.12), h)
            tf = tb.text_frame
            tf.margin_top = tf.margin_bottom = Emu(0)
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            _set_text(
                tf,
                [(str(cell), font, bool(header and r == 0), fg)],
                space_after=Pt(0),
            )
            x += widths[c]
        y += h
    return y


# --- slides --------------------------------------------------------------------


def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bar(slide, Emu(0), Emu(0), SLIDE_W, SLIDE_H, INK)
    bar(slide, Emu(0), Emu(0), Inches(0.16), SLIDE_H, ACCENT)
    textbox(
        slide,
        Inches(1.0),
        Inches(1.9),
        Inches(11),
        Inches(0.4),
        [("AMD · ROCm · ONNX RUNTIME", Pt(12), True, ACCENT)],
        space_after=Pt(0),
    )
    textbox(
        slide,
        Inches(1.0),
        Inches(2.4),
        Inches(11.4),
        Inches(1.5),
        [
            ("HIP EP as a Product", Pt(48), True, WHITE),
            (
                "Server, console, model zoo, benchmarks and docs for AMD GPUs",
                Pt(19),
                False,
                RGBColor(0xB9, 0xC2, 0xCC),
            ),
        ],
        space_after=Pt(12),
    )
    bar(slide, Inches(1.0), Inches(4.55), Inches(2.2), Inches(0.03), ACCENT)
    textbox(
        slide,
        Inches(1.0),
        Inches(4.85),
        Inches(11),
        Inches(1.2),
        [
            ("Plan for review  ·  HIPEP-PROD-2026H2", Pt(13), True, WHITE),
            (
                "2 September 2026  ·  3 phases → GA 31 October 2026  ·  9.5 FTE peak",
                Pt(12),
                False,
                RGBColor(0x8E, 0x99, 0xA5),
            ),
            (
                "Primary dev + demo host: Strix Halo, Radeon 8060S (gfx1151), 48 GB",
                Pt(12),
                False,
                RGBColor(0x8E, 0x99, 0xA5),
            ),
        ],
        space_after=Pt(5),
    )


def slide_tldr(prs):
    slide = new_slide(
        prs, "The engine is strong. The product is missing.", kicker="Executive summary"
    )
    cards = [
        (
            "Today",
            ACCENT,
            "HIP EP is an ONNX Runtime Execution Provider with a real MLIR compiler and "
            "competitive kernels. Consuming it means C++ CLIs, pytest files and an "
            "internal dashboard. There is no server, no UI, no public model list and no "
            "published numbers.",
        ),
        (
            "The gap",
            AMBER,
            "Ollama, LM Studio and FastFlowLM do not beat us on the engine. They beat us "
            "on the layer above it: one install, one server, a model list, a chat box and "
            "a benchmark page. That layer is what users actually touch.",
        ),
        (
            "The plan",
            GREEN,
            "Three phases to GA on 31 October 2026. Ship oga-server (OpenAI + "
            "Ollama compatible), HIP EP Studio, a certified model zoo with cards, a "
            "bring-your-own-model pipeline with Quark quantization, one benchmark harness "
            "and a public docs site, on Windows and Linux. Foundations are already complete.",
        ),
    ]
    w = Inches(3.94)
    for i, (head, color, body) in enumerate(cards):
        x = MARGIN + Emu(int(i * (w + Inches(0.28))))
        p = panel(slide, x, BODY_TOP, w, Inches(3.2))
        bar(slide, x, BODY_TOP, w, Inches(0.055), color)
        tf = p.text_frame
        tf.margin_left = tf.margin_right = Inches(0.24)
        tf.margin_top = Inches(0.3)
        _set_text(
            tf,
            [(head, Pt(17), True, INK), (body, Pt(12), False, INK_SOFT)],
            space_after=Pt(10),
            line=1.28,
        )
    textbox(
        slide,
        MARGIN,
        Inches(5.2),
        CONTENT_W,
        Inches(1.4),
        [
            ("The ask", Pt(15), True, INK),
            (
                "Approve the roadmap and resourcing, and commit a Linux GPU host by P3. "
                "Everything else is already in motion: the branch, the product tree, the "
                "server and this deck are generated from the plan in-repo.",
                Pt(12.5),
                False,
                INK_SOFT,
            ),
        ],
        space_after=Pt(6),
        line=1.25,
    )


def slide_problem(prs):
    slide = new_slide(
        prs,
        "What users ask for, and what they get",
        kicker="Problem",
        subtitle="Every row below is a support conversation we have today.",
    )
    rows = [
        ["A user says…", "What exists today", "Cost"],
        [
            '"Give me a server I can point my app at"',
            "No server. ORT/OGA C++ CLIs and pytest files.",
            "Blocked",
        ],
        [
            '"Which models actually work?"',
            "An internal dashboard behind the corporate network.",
            "Days",
        ],
        [
            '"Run this Hugging Face model"',
            "Manual export, quantize, split and merge expedition.",
            "~2 days",
        ],
        [
            '"How fast is it?"',
            "Numbers in an internal tracker, not in public docs.",
            "Unanswerable",
        ],
        ['"Show me code for my app"', "hip-onnx-runner -m model.onnx", "Blocked"],
        ['"Let me try it in a browser"', "Nothing.", "Blocked"],
    ]
    colors = {(r, 2): ACCENT for r in range(1, len(rows))}
    end = table(
        slide,
        MARGIN,
        BODY_TOP,
        CONTENT_W,
        rows,
        [4.2, 6.4, 1.7],
        row_h=Inches(0.42),
        colors=colors,
    )
    p = panel(
        slide,
        MARGIN,
        end + Inches(0.35),
        CONTENT_W,
        Inches(0.95),
        fill=RGBColor(0xFD, 0xF1, 0xF3),
        edge=RGBColor(0xF6, 0xCF, 0xD5),
    )
    tf = p.text_frame
    tf.margin_left = Inches(0.28)
    tf.margin_top = Inches(0.18)
    _set_text(
        tf,
        [
            (
                "The pattern: we optimised the bottom of the stack and never built the top. "
                "Users cannot reach our performance because they cannot reach our engine.",
                Pt(13),
                False,
                INK,
            )
        ],
        space_after=Pt(0),
        line=1.3,
    )


def slide_vision(prs):
    slide = new_slide(prs, "Vision", kicker="Where we are going")
    p = panel(slide, MARGIN, BODY_TOP, CONTENT_W, Inches(1.35), fill=INK, edge=None)
    tf = p.text_frame
    tf.margin_left = Inches(0.4)
    tf.margin_top = Inches(0.3)
    _set_text(
        tf,
        [
            ("Ship HIP EP as a complete local-AI product for AMD GPUs:", Pt(20), True, WHITE),
            (
                "one server, one console, one model zoo, one benchmark — on Windows and Linux.",
                Pt(20),
                True,
                ACCENT,
            ),
        ],
        space_after=Pt(4),
    )
    goals = [
        ("10 minutes", "Cold machine to a streaming token in the browser.", TEAL),
        ("60 minutes", "Hugging Face URL to a quantized, benchmarked local model.", GREEN),
        ("16 models", "Certified LLM and VLM models with published cards and numbers.", AMBER),
        ("2 platforms", "Windows and Linux, across four AMD device profiles.", ACCENT),
    ]
    w = Inches(2.92)
    for i, (big, sub, color) in enumerate(goals):
        x = MARGIN + Emu(int(i * (w + Inches(0.22))))
        pp = panel(slide, x, Inches(3.45), w, Inches(1.95))
        bar(slide, x, Inches(3.45), Inches(0.05), Inches(1.95), color)
        tf = pp.text_frame
        tf.margin_left = Inches(0.26)
        tf.margin_top = Inches(0.26)
        _set_text(
            tf,
            [(big, Pt(23), True, color), (sub, Pt(11.5), False, INK_SOFT)],
            space_after=Pt(8),
            line=1.25,
        )
    textbox(
        slide,
        MARGIN,
        Inches(5.75),
        CONTENT_W,
        Inches(0.8),
        [
            (
                "Measured at GA. Every number above is a release gate, not an aspiration.",
                Pt(12),
                False,
                INK_SOFT,
            )
        ],
        space_after=Pt(0),
    )


def slide_surfaces(prs):
    slide = new_slide(
        prs,
        "Seven product surfaces",
        kicker="Scope",
        subtitle="Each surface has an owner, an exit criterion and a phase.",
    )
    items = [
        (
            "1",
            "oga-server",
            "OpenAI- and Ollama-compatible HTTP server over ONNX Runtime GenAI + HIP EP.",
            ACCENT,
        ),
        (
            "2",
            "HIP EP Studio",
            "Zero-install web console: chat, VLM, model manager, benchmarks, live server status.",
            TEAL,
        ),
        (
            "3",
            "Model Zoo",
            "Certified model registry, model cards, pull from known sources.",
            GREEN,
        ),
        (
            "4",
            "hipep CLI",
            "Export, quantize with Quark, convert, validate, benchmark, publish.",
            AMBER,
        ),
        ("5", "Recipes", "Per-architecture YAML so users can bring their own models.", TEAL),
        (
            "6",
            "Site + docs",
            "Public landing page, quick starts, model cards, benchmark numbers.",
            ACCENT,
        ),
        (
            "7",
            "SDK samples",
            "C++, Python and WinML clients users paste into their project.",
            GREEN,
        ),
    ]
    w = Inches(3.94)
    h = Inches(1.52)
    for i, (num, name, desc, color) in enumerate(items):
        col, row = i % 3, i // 3
        x = MARGIN + Emu(int(col * (w + Inches(0.28))))
        y = BODY_TOP + Emu(int(row * (h + Inches(0.24))))
        p = panel(slide, x, y, w, h)
        bar(slide, x, y, Inches(0.05), h, color)
        tf = p.text_frame
        tf.margin_left = Inches(0.26)
        tf.margin_top = Inches(0.2)
        _set_text(
            tf,
            [(f"{num}   {name}", Pt(15), True, INK), (desc, Pt(11), False, INK_SOFT)],
            space_after=Pt(6),
            line=1.24,
        )


def slide_architecture(prs):
    slide = new_slide(
        prs,
        "Architecture",
        kicker="How it fits together",
        subtitle="The server is the product boundary: the console, the benchmarks and the samples all go through it.",
    )
    y = BODY_TOP + Inches(0.1)

    # clients
    clients = ["Browser (Studio)", "OpenAI SDK", "Ollama clients", "curl / CI"]
    cw = Inches(2.92)
    for i, c in enumerate(clients):
        x = MARGIN + Emu(int(i * (cw + Inches(0.22))))
        p = panel(slide, x, y, cw, Inches(0.52), fill=WHITE, edge=PANEL_EDGE)
        tf = p.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        _set_text(tf, [(c, Pt(11.5), True, INK)], align=PP_ALIGN.CENTER, space_after=Pt(0))

    textbox(
        slide,
        MARGIN,
        y + Inches(0.58),
        CONTENT_W,
        Inches(0.26),
        [("HTTP / SSE", Pt(9.5), True, INK_SOFT)],
        align=PP_ALIGN.CENTER,
        space_after=Pt(0),
    )

    # server
    sy = y + Inches(0.9)
    panel(slide, MARGIN, sy, CONTENT_W, Inches(1.72), fill=INK, edge=None)
    textbox(
        slide,
        MARGIN + Inches(0.26),
        sy + Inches(0.14),
        Inches(4),
        Inches(0.3),
        [("oga-server", Pt(14), True, WHITE)],
        space_after=Pt(0),
    )
    apis = ["OpenAI API", "Ollama API", "Jobs API", "Bench API", "Status API"]
    gap = Inches(0.12)
    aw = Emu(int((CONTENT_W - Inches(0.52) - 4 * gap) / 5))
    for i, a in enumerate(apis):
        x = MARGIN + Inches(0.26) + Emu(int(i * (aw + gap)))
        p = panel(
            slide,
            x,
            sy + Inches(0.5),
            aw,
            Inches(0.44),
            fill=RGBColor(0x25, 0x2C, 0x35),
            edge=None,
        )
        tf = p.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        _set_text(tf, [(a, Pt(10.5), True, WHITE)], align=PP_ALIGN.CENTER, space_after=Pt(0))
    textbox(
        slide,
        MARGIN + Inches(0.26),
        sy + Inches(1.06),
        CONTENT_W - Inches(0.5),
        Inches(0.6),
        [
            (
                "Scheduler: per-model workers · queue · admission control · sessions",
                Pt(10.5),
                False,
                RGBColor(0xB9, 0xC2, 0xCC),
            ),
            (
                "Registry (certified models) · Store (local models) · Job runner",
                Pt(10.5),
                False,
                RGBColor(0xB9, 0xC2, 0xCC),
            ),
        ],
        space_after=Pt(1),
    )

    # two backends
    by = sy + Inches(1.95)
    bwidth = Emu(int((CONTENT_W - Inches(0.28)) / 2))
    left_p = panel(slide, MARGIN, by, bwidth, Inches(1.02), fill=WHITE)
    bar(slide, MARGIN, by, Inches(0.05), Inches(1.02), TEAL)
    _set_text(
        left_p.text_frame,
        [
            ("OGA engine", Pt(13), True, INK),
            (
                "onnxruntime_genai + AMDGPUExecutionProvider (profile=hip). Mock engine for GPU-free hosts.",
                Pt(10.5),
                False,
                INK_SOFT,
            ),
        ],
        space_after=Pt(4),
        line=1.2,
    )
    left_p.text_frame.margin_left = Inches(0.24)
    left_p.text_frame.margin_top = Inches(0.14)

    right_x = MARGIN + bwidth + Inches(0.28)
    right_p = panel(slide, right_x, by, bwidth, Inches(1.02), fill=WHITE)
    bar(slide, right_x, by, Inches(0.05), Inches(1.02), AMBER)
    _set_text(
        right_p.text_frame,
        [
            ("hipep toolkit", Pt(13), True, INK),
            (
                "fetch → export → quantize (Quark) → convert → validate → benchmark → publish",
                Pt(10.5),
                False,
                INK_SOFT,
            ),
        ],
        space_after=Pt(4),
        line=1.2,
    )
    right_p.text_frame.margin_left = Inches(0.24)
    right_p.text_frame.margin_top = Inches(0.14)

    # engine
    ey = by + Inches(1.2)
    panel(
        slide, MARGIN, ey, CONTENT_W, Inches(0.78), fill=RGBColor(0x2A, 0x1B, 0x1E), edge=None
    )
    tf = panel and slide.shapes[-1].text_frame
    tf.margin_left = Inches(0.26)
    tf.margin_top = Inches(0.1)
    _set_text(
        tf,
        [
            (
                "HIP EP  ·  hipgpu / amdgpu-ep  ·  ONNX → MLIR HIP dialect → LLVM → HIP",
                Pt(12.5),
                True,
                WHITE,
            ),
            (
                "Strix · Krackan · Strix Halo · Radeon dGPU · Instinct",
                Pt(10.5),
                False,
                RGBColor(0xC9, 0x9B, 0xA2),
            ),
        ],
        space_after=Pt(2),
    )


def slide_models(prs):
    slide = new_slide(
        prs,
        "Certified model scope",
        kicker="Model zoo",
        subtitle="Seeded from the internal ROCm HIP EP dashboard, release_0.4, run 2026-08-27. LLM and VLM lead.",
    )
    rows = [
        ["Tier", "Count", "Models", "Milestone"],
        [
            "LLM",
            "8",
            "Llama-3.1-8B · Mistral-7B-v0.3 · Qwen2.5-14B · Qwen2.5-Coder-14B · Phi-4-14B · gpt-oss-20b · gpt-oss-120b · DeepSeek-R1-Distill-70B",
            "P1",
        ],
        [
            "VLM",
            "8",
            "Qwen3.5-9B · Qwen3.5-35B-A3B · Qwen3.6-27B · Qwen3.6-35B-A3B · Qwen3.8-27B · Gemma-3-4B · Gemma-4-12B · Gemma-4-26B-A4B",
            "P1",
        ],
        [
            "Speech",
            "7",
            "Whisper tiny / base / small / medium / large-v3 / large-v3-turbo",
            "P2",
        ],
        [
            "Vision",
            "14",
            "ResNet50 · MobileNetV2 · DETR · Swin / SwinV2 · YOLOv5 · SAM 2.1 · BLIP · ConvNeXt · ESRGAN",
            "P2",
        ],
    ]
    colors = {(1, 0): ACCENT, (2, 0): TEAL, (3, 0): INK_SOFT, (4, 0): INK_SOFT}
    end = table(
        slide,
        MARGIN,
        BODY_TOP,
        CONTENT_W,
        rows,
        [1.3, 1.0, 9.0, 1.4],
        row_h=Inches(0.78),
        colors=colors,
    )
    p = panel(slide, MARGIN, end + Inches(0.3), CONTENT_W, Inches(1.3))
    tf = p.text_frame
    tf.margin_left = Inches(0.28)
    tf.margin_top = Inches(0.2)
    _set_text(
        tf,
        [
            ("Every certified model ships a card", Pt(14), True, INK),
            (
                "Base model · quantization scheme and group size · context length · VRAM footprint · "
                "TTFT and TPS at 128 / 2048 / 16384 · accuracy (PPL, MMLU, TinyGSM8K) · the exact "
                "recipe that reproduces it · the one-line command that runs it.",
                Pt(11.5),
                False,
                INK_SOFT,
            ),
        ],
        space_after=Pt(6),
        line=1.28,
    )


def slide_roadmap(prs):
    slide = new_slide(
        prs,
        "Roadmap",
        kicker="Three phases to GA",
        subtitle="GA 1.0 on 31 October 2026. Each phase ends with a git tag and a MILESTONES.md entry so any checkpoint is reviewable and re-testable.",
    )
    ms = [
        ("M0", "Foundations", "Aug 28 – Sep 1", 0, 1.0, INK_SOFT),
        ("P1", "Serving on hardware", "Sep 2 – Sep 30", 1.0, 4.0, ACCENT),
        ("P2", "Pipeline & cert", "Oct 1 – Oct 16", 5.0, 2.5, TEAL),
        ("P3", "Platform, docs & GA", "Oct 17 – Oct 31", 7.5, 2.5, GREEN),
    ]
    label_w = Inches(2.15)
    date_w = Inches(1.85)
    track_x = MARGIN + label_w + date_w
    track_w = CONTENT_W - label_w - date_w
    unit = Emu(int(track_w / 10.3))
    y = BODY_TOP + Inches(0.35)
    row_h = Inches(0.95)

    for i, (mid, name, window, start, span, color) in enumerate(ms):
        ry = y + Emu(int(i * row_h))
        textbox(
            slide,
            MARGIN,
            ry + Inches(0.08),
            label_w - Inches(0.12),
            Inches(0.4),
            [(f"{mid}  {name}", Pt(12), True, INK)],
            space_after=Pt(0),
        )
        textbox(
            slide,
            MARGIN + label_w,
            ry + Inches(0.1),
            date_w - Inches(0.12),
            Inches(0.4),
            [(window, Pt(10.5), False, INK_SOFT)],
            space_after=Pt(0),
        )
        bar(
            slide,
            track_x + Emu(int(start * unit)),
            ry + Inches(0.1),
            Emu(int(span * unit)),
            Inches(0.34),
            color,
        )

    gate_y = y + Emu(int(len(ms) * row_h)) + Inches(0.25)
    gates = [
        ("P1 gate", "The certified LLM and VLM set streams through HIP EP on real hardware."),
        ("P2 gate", "A user brings a Hugging Face URL and gets a benchmarked local model."),
        (
            "P3 gate",
            "Linux host validated — the largest schedule dependency, now in the final two weeks.",
        ),
    ]
    w = Inches(3.94)
    for i, (h, d) in enumerate(gates):
        x = MARGIN + Emu(int(i * (w + Inches(0.28))))
        p = panel(slide, x, gate_y, w, Inches(0.9), fill=WHITE)
        bar(slide, x, gate_y, Inches(0.05), Inches(0.9), INK)
        tf = p.text_frame
        tf.margin_left = Inches(0.22)
        tf.margin_top = Inches(0.12)
        _set_text(
            tf,
            [(h, Pt(11.5), True, INK), (d, Pt(10), False, INK_SOFT)],
            space_after=Pt(3),
            line=1.2,
        )


def slide_milestones(prs):
    slide = new_slide(
        prs,
        "Phase exit criteria",
        kicker="Definition of done",
        subtitle="A phase is done when its exit criteria are demonstrable on hardware, not when its code is merged.",
    )
    rows = [
        ["ID", "Phase", "Exit criteria", "Tag"],
        [
            "M0",
            "Foundations — complete",
            "Product tree, model registry, oga-server, console, CLI, client samples, docs site, installers and test suite running locally on Strix Halo.",
            "hipep-product-m0",
        ],
        [
            "P1",
            "Serving on hardware",
            "Real OGA engine streaming through HIP EP for the certified LLM and VLM set. OpenAI + Ollama parity. VLM image path. Console chat with live TTFT/TPS. Multi-user scheduling under load.",
            "hipep-product-p1",
        ],
        [
            "P2",
            "Pipeline & certification",
            "HF URL → export → Quark quantize → validate → publish from the console. Bench harness runs through oga-server; nightly certification matrix; published numbers match the internal dashboard within tolerance.",
            "hipep-product-p2",
        ],
        [
            "P3",
            "Platform, docs & GA 1.0",
            "Linux server + container, Windows installer, four device profiles. Model cards and C++/Python/WinML samples. Security review, opt-in telemetry, perf hardening, support rotation. Public launch.",
            "hipep-product-v1.0",
        ],
    ]
    colors = {(r, 0): ACCENT for r in range(1, len(rows))}
    table(
        slide,
        MARGIN,
        BODY_TOP,
        CONTENT_W,
        rows,
        [0.7, 2.4, 7.7, 2.0],
        row_h=Inches(0.9),
        font=Pt(10),
        colors=colors,
    )


def slide_resources(prs):
    slide = new_slide(
        prs,
        "Resourcing",
        kicker="Team",
        subtitle="Peak 9.5 FTE. The compressed schedule front-loads the team: no phase is quiet enough to absorb a slip.  ● primary load   ○ partial load",
    )
    rows = [
        ["Role", "FTE", "M0", "P1", "P2", "P3", "Owns"],
        ["Tech lead / architect", "1.0", "●", "●", "●", "●", "Architecture, reviews, roadmap"],
        ["Server & API engineer", "2.0", "●", "●", "●", "●", "oga-server, scheduler, APIs"],
        ["Front-end engineer", "1.5", "○", "●", "●", "●", "HIP EP Studio, landing page"],
        [
            "Model enablement engineer",
            "2.0",
            "○",
            "●",
            "●",
            "●",
            "Export, quantization, recipes",
        ],
        [
            "Performance engineer",
            "1.0",
            "○",
            "●",
            "●",
            "●",
            "Bench harness, tuning, regressions",
        ],
        ["Docs / DevRel", "1.0", "○", "○", "●", "●", "Docs site, model cards, samples"],
        ["QA / release engineer", "1.0", "○", "●", "●", "●", "CI, certification, packaging"],
        ["Total FTE", "9.5", "4.0", "8.5", "9.5", "9.5", ""],
    ]
    total_row = len(rows) - 1
    colors = {}
    for r in range(1, len(rows)):
        colors[(r, 1)] = ACCENT
        for c in range(2, 6):
            colors[(r, c)] = ACCENT if rows[r][c] == "●" else INK_SOFT
    for c in range(len(rows[0])):
        colors[(total_row, c)] = INK
    table(
        slide,
        MARGIN,
        BODY_TOP,
        CONTENT_W,
        rows,
        [3.2, 0.8, 0.9, 0.9, 0.9, 0.9, 5.2],
        row_h=Inches(0.5),
        font=Pt(11),
        colors=colors,
    )


def slide_hardware(prs):
    slide = new_slide(
        prs,
        "Hardware and infrastructure",
        kicker="Dependencies",
        subtitle="One item on this list is a hard schedule dependency and we do not have it yet.",
    )
    rows = [
        ["Item", "Qty", "Purpose", "Status", "Needed by"],
        [
            "Strix Halo — Radeon 8060S, gfx1151, 48 GB",
            "1",
            "Primary dev + demo server",
            "Available now",
            "M0",
        ],
        ["Strix / Krackan laptops", "2", "Device-profile validation", "Request", "P1"],
        [
            "Radeon dGPU workstation (gfx1100 / gfx1201)",
            "1",
            "Discrete GPU coverage",
            "Request",
            "P1",
        ],
        ["Linux GPU server", "1", "Hosted Linux oga-server", "NOT AVAILABLE", "P3"],
        ["Nightly CI runner with GPU", "1", "Certification matrix", "Request", "P2"],
        ["Public docs hosting", "1", "Landing page + docs site", "Request", "P3"],
    ]
    colors = {(1, 3): GREEN, (4, 3): ACCENT, (4, 0): ACCENT}
    end = table(
        slide,
        MARGIN,
        BODY_TOP,
        CONTENT_W,
        rows,
        [5.0, 0.8, 3.6, 1.9, 1.2],
        row_h=Inches(0.44),
        colors=colors,
    )
    p = panel(
        slide,
        MARGIN,
        end + Inches(0.35),
        CONTENT_W,
        Inches(1.25),
        fill=RGBColor(0xFD, 0xF1, 0xF3),
        edge=RGBColor(0xF6, 0xCF, 0xD5),
    )
    tf = p.text_frame
    tf.margin_left = Inches(0.28)
    tf.margin_top = Inches(0.18)
    _set_text(
        tf,
        [
            (
                "Linux: we build the path now, we validate it when we have hardware",
                Pt(14),
                True,
                ACCENT,
            ),
            (
                "The Linux server, container image, install script and device profiles are written and "
                "exercised in the existing docker/ build flow and are complete today. We do not claim Linux "
                "support in the docs until it runs on a real Linux GPU host. That claim is gated at P3.",
                Pt(11.5),
                False,
                INK,
            ),
        ],
        space_after=Pt(6),
        line=1.28,
    )


def slide_risks(prs):
    slide = new_slide(prs, "Risks and mitigations", kicker="Risk register")
    rows = [
        ["#", "Risk", "Impact", "Likelihood", "Mitigation"],
        [
            "R1",
            "No Linux GPU server available",
            "High",
            "High",
            "Build and test in docker/ now; gate the Linux claim on real hardware at P3",
        ],
        [
            "R2",
            "OGA + HIP EP wheel availability lags",
            "High",
            "Medium",
            "Mock engine keeps the product testable; pin a known-good wheel set; ship a preflight doctor",
        ],
        [
            "R3",
            "Export / quantize is slow and fragile",
            "High",
            "High",
            "Data-driven resumable recipes; every stage caches; failures produce a diagnosis, not a stack trace",
        ],
        [
            "R4",
            "Published numbers drift from the dashboard",
            "Medium",
            "Medium",
            "One harness through the server, same metric definitions, run metadata published with every number",
        ],
        [
            "R5",
            "Concurrency exhausts GPU memory",
            "Medium",
            "Medium",
            "Admission control, per-model worker cap, visible queue depth, memory headroom checks",
        ],
        [
            "R6",
            "Scope creep into a general model hub",
            "High",
            "Medium",
            "Certified tiers fixed per phase; BYO models are user-owned, never certified",
        ],
    ]
    sev = {"High": ACCENT, "Medium": AMBER, "Low": GREEN}
    colors = {(r, 0): ACCENT for r in range(1, len(rows))}
    for r in range(1, len(rows)):
        colors[(r, 2)] = sev[rows[r][2]]
        colors[(r, 3)] = sev[rows[r][3]]
    table(
        slide,
        MARGIN,
        BODY_TOP,
        CONTENT_W,
        rows,
        [0.5, 3.5, 1.0, 1.2, 6.6],
        row_h=Inches(0.62),
        font=Pt(10),
        colors=colors,
    )


def slide_metrics(prs):
    slide = new_slide(
        prs,
        "Success metrics",
        kicker="How we grade this",
        subtitle="These are release gates for GA, measured on the certification host.",
    )
    rows = [
        ["Metric", "Baseline today", "GA target"],
        ["Cold machine → first streamed token", "n/a — no server exists", "under 10 minutes"],
        [
            "Hugging Face URL → benchmarked local model",
            "about 2 days, manual",
            "under 60 minutes, unattended",
        ],
        ["Certified LLM + VLM models published with cards", "0", "16"],
        [
            "API compatibility",
            "none",
            "OpenAI chat / completions / models  +  Ollama tags / chat / generate / pull / show",
        ],
        [
            "Platforms and device profiles",
            "Windows dev builds",
            "Windows + Linux, 4 device profiles",
        ],
        ["Published benchmark coverage", "0", "16 models × 3 context lengths"],
    ]
    colors = {(r, 2): GREEN for r in range(1, len(rows))}
    colors.update({(r, 1): INK_SOFT for r in range(1, len(rows))})
    end = table(
        slide,
        MARGIN,
        BODY_TOP,
        CONTENT_W,
        rows,
        [4.4, 3.3, 6.4],
        row_h=Inches(0.52),
        colors=colors,
    )
    textbox(
        slide,
        MARGIN,
        end + Inches(0.4),
        CONTENT_W,
        Inches(1.2),
        [
            ("Out of scope for 1.0", Pt(14), True, INK),
            (
                "Training and fine-tuning · multi-node serving · a hosted public inference service · "
                "redistribution of third-party weights · Windows ARM · video understanding in the VLM path.",
                Pt(11.5),
                False,
                INK_SOFT,
            ),
        ],
        space_after=Pt(6),
        line=1.28,
    )


def slide_ask(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bar(slide, Emu(0), Emu(0), SLIDE_W, SLIDE_H, INK)
    bar(slide, Emu(0), Emu(0), Inches(0.16), SLIDE_H, ACCENT)
    textbox(
        slide,
        Inches(1.0),
        Inches(0.85),
        Inches(11),
        Inches(0.4),
        [("DECISIONS REQUESTED", Pt(12), True, ACCENT)],
        space_after=Pt(0),
    )
    textbox(
        slide,
        Inches(1.0),
        Inches(1.3),
        Inches(11),
        Inches(0.6),
        [("What we need from this review", Pt(34), True, WHITE)],
        space_after=Pt(0),
    )

    asks = [
        (
            "Approve roadmap and resourcing",
            "Three phases, GA on 31 October 2026, 9.5 FTE at peak.",
        ),
        (
            "Commit a Linux GPU host",
            "Required by P3. This is the largest single schedule risk in the plan.",
        ),
        (
            "Decide on public benchmark numbers",
            "Publish absolute numbers at GA, or relative-only against the internal dashboard?",
        ),
        (
            "Confirm the server name",
            "Ship as oga-server, or give the server a product name before the docs are written?",
        ),
        (
            "Decide on model redistribution",
            "Do we host converted and quantized weights, or only the recipes that reproduce them?",
        ),
    ]
    y = Inches(2.25)
    for i, (h, d) in enumerate(asks):
        yy = y + Emu(int(i * Inches(0.87)))
        num = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(1.0), yy, Inches(0.42), Inches(0.42)
        )
        num.fill.solid()
        num.fill.fore_color.rgb = ACCENT
        num.line.fill.background()
        num.shadow.inherit = False
        tf = num.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        _set_text(
            tf, [(str(i + 1), Pt(13), True, WHITE)], align=PP_ALIGN.CENTER, space_after=Pt(0)
        )
        textbox(
            slide,
            Inches(1.65),
            yy - Inches(0.03),
            Inches(10.6),
            Inches(0.75),
            [(h, Pt(15), True, WHITE), (d, Pt(11.5), False, RGBColor(0x9E, 0xA9, 0xB5))],
            space_after=Pt(2),
            line=1.15,
        )

    bar(
        slide,
        Inches(1.0),
        Inches(6.72),
        Inches(11.3),
        Inches(0.02),
        RGBColor(0x33, 0x3B, 0x45),
    )
    textbox(
        slide,
        Inches(1.0),
        Inches(6.85),
        Inches(11.3),
        Inches(0.4),
        [
            (
                "Work already in flight on branch feat/hipep-product — product tree, model registry, "
                "server skeleton, console and this deck are generated from product/plan/PRODUCT_PLAN.md",
                Pt(10.5),
                False,
                RGBColor(0x74, 0x7F, 0x8B),
            )
        ],
        space_after=Pt(0),
    )


BUILDERS = [
    slide_title,
    slide_tldr,
    slide_problem,
    slide_vision,
    slide_surfaces,
    slide_architecture,
    slide_models,
    slide_roadmap,
    slide_milestones,
    slide_resources,
    slide_hardware,
    slide_risks,
    slide_metrics,
    slide_ask,
]


def build(out: Path) -> Path:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    for builder in BUILDERS:
        builder(prs)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out


def main() -> None:
    default = Path(__file__).with_name("HIP-EP-Product-Plan.pptx")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=default)
    args = ap.parse_args()
    path = build(args.out)
    print(f"wrote {path} ({len(BUILDERS)} slides)")


if __name__ == "__main__":
    main()
