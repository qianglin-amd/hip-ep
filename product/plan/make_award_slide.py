#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Generate the one-slide HIP EP award slide for the BU all-hands.

This is a standalone recognition slide, not part of the product review deck in
make_deck.py. It borrows that deck's design system so the two look related when
they appear in the same meeting:

    python product/plan/make_award_slide.py [--out path.pptx]

The proof points are the release_0.4 certification run against release_0.3, on
the Strix Halo reference host. Keep them in sync with
product/modelzoo/benchmarks/release_0.4.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from make_deck import ACCENT, AMBER, INK, SLIDE_H, SLIDE_W, WHITE, bar, panel, textbox
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

MUTED = RGBColor(0x8E, 0x99, 0xA5)
BADGE_FILL = RGBColor(0x26, 0x1D, 0x0A)
BADGE_EDGE = RGBColor(0x6B, 0x4C, 0x12)
TILE_FILL = RGBColor(0x1A, 0x1F, 0x26)
TILE_EDGE = RGBColor(0x2E, 0x36, 0x40)

AWARD = "AIG BU EXCELLENCE AWARD"
SUBJECT = "HIP EP"
# The all-hands audience is BU-wide, so the slide has to say what HIP EP is before
# it can say why it was recognized.
WHAT_IT_IS = (
    "AMD's execution provider for ONNX Runtime: it compiles ONNX models through a custom "
    "MLIR compiler and runs them on AMD GPUs \u2014 Ryzen AI systems and Radeon graphics."
)
CITATION = "Recognized across the AIG Business Unit for making AMD GPU inference a product."

# Each tile is (headline figure, caption). Figures come from the release_0.4 run.
# Captions wrap on their own; an embedded newline renders as a stray glyph in
# PowerPoint even though the PNG preview absorbs it.
PROOF = [
    ("16", "models certified on Strix Halo, across the LLM and VLM tiers"),
    ("up to 4\u00d7", "faster time to first token, release 0.4 against release 0.3"),
    ("8 / 8", "of the VLM tier passing in the latest certification run"),
]

FOOTER = "Awarded to the HIP EP project  \u00b7  AIG Business Unit  \u00b7  2026"


def build_slide(prs) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bar(slide, Emu(0), Emu(0), SLIDE_W, SLIDE_H, INK)
    bar(slide, Emu(0), Emu(0), Inches(0.16), SLIDE_H, ACCENT)

    left = Inches(1.0)
    width = Inches(11.4)

    # Award badge. Sized to the text rather than the column so it reads as a pill.
    badge_w = Inches(4.35)
    badge_h = Inches(0.5)
    panel(slide, left, Inches(0.72), badge_w, badge_h, fill=BADGE_FILL, edge=BADGE_EDGE)
    textbox(
        slide,
        left + Inches(0.3),
        Inches(0.82),
        badge_w - Inches(0.4),
        Inches(0.36),
        [(AWARD, Pt(14), True, AMBER)],
        space_after=Pt(0),
    )

    textbox(
        slide,
        left,
        Inches(1.24),
        width,
        Inches(1.25),
        [(SUBJECT, Pt(78), True, WHITE)],
        space_after=Pt(0),
    )
    textbox(
        slide,
        left,
        Inches(2.66),
        Inches(11.0),
        Inches(0.95),
        [(WHAT_IT_IS, Pt(18.5), False, RGBColor(0xB9, 0xC2, 0xCC))],
        space_after=Pt(0),
        line=1.3,
    )
    # Amber ties the recognition line back to the badge, so the eye reads
    # award, subject, definition, reason.
    textbox(
        slide,
        left,
        Inches(3.58),
        width,
        Inches(0.5),
        [(CITATION, Pt(17), True, AMBER)],
        space_after=Pt(0),
        line=1.25,
    )

    bar(slide, left, Inches(4.18), Inches(2.2), Inches(0.03), ACCENT)

    # Proof tiles: the award is the headline, these are why it was earned.
    tile_y = Inches(4.52)
    tile_h = Inches(1.68)
    gap = Inches(0.3)
    tile_w = Emu(int((width - 2 * gap) / 3))
    for i, (figure, caption) in enumerate(PROOF):
        x = left + Emu(int(i * (tile_w + gap)))
        panel(slide, x, tile_y, tile_w, tile_h, fill=TILE_FILL, edge=TILE_EDGE)
        bar(slide, x, tile_y, Inches(0.05), tile_h, AMBER)
        textbox(
            slide,
            x + Inches(0.32),
            tile_y + Inches(0.2),
            tile_w - Inches(0.5),
            Inches(1.2),
            [
                (figure, Pt(34), True, WHITE),
                (caption, Pt(11.5), False, MUTED),
            ],
            space_after=Pt(4),
            line=1.2,
        )

    textbox(
        slide,
        left,
        Inches(6.58),
        width,
        Inches(0.4),
        [(FOOTER, Pt(12), False, MUTED)],
        space_after=Pt(0),
    )


def build(out: Path) -> Path:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    build_slide(prs)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out


def main() -> None:
    default = Path(__file__).with_name("HIP-EP-Award.pptx")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=default)
    args = ap.parse_args()
    path = build(args.out)
    print(f"wrote {path} (1 slide)")


if __name__ == "__main__":
    main()
