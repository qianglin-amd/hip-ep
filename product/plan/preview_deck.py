#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Render an approximate PNG preview of a .pptx for layout review.

PowerPoint and LibreOffice are not installed on every dev or CI host, so this
draws the shape geometry and text directly with Pillow. It is a layout check,
not a faithful renderer: use it to catch overflow, collisions and clipping.

    python product/plan/preview_deck.py [deck.pptx] [--outdir preview]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Emu

SCALE = 110  # pixels per inch
EMU_PER_INCH = 914400

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts")]
FONT_CANDIDATES = {
    False: ["segoeui.ttf", "DejaVuSans.ttf", "arial.ttf"],
    True: ["segoeuib.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"],
}
_font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def get_font(size_pt: float, bold: bool) -> ImageFont.FreeTypeFont:
    px = max(7, int(size_pt * SCALE / 72))
    key = (px, bold)
    if key not in _font_cache:
        for name in FONT_CANDIDATES[bold]:
            for d in FONT_DIRS:
                p = d / name
                if p.exists():
                    _font_cache[key] = ImageFont.truetype(str(p), px)
                    break
            if key in _font_cache:
                break
        else:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def px(emu) -> int:
    return int(Emu(int(emu)).inches * SCALE)


def rgb(color, default=(0, 0, 0)):
    try:
        if color and color.type is not None and color.rgb is not None:
            return tuple(color.rgb)
    except (AttributeError, TypeError, ValueError):
        pass
    return default


def shape_fill(shape):
    try:
        if shape.fill.type is not None and str(shape.fill.type).startswith("SOLID"):
            return tuple(shape.fill.fore_color.rgb)
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def shape_line(shape):
    try:
        if shape.line.fill.type is not None and str(shape.line.fill.type).startswith("SOLID"):
            return tuple(shape.line.color.rgb)
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_shape(draw, shape, problems, slide_px):
    x0, y0 = px(shape.left), px(shape.top)
    x1, y1 = x0 + px(shape.width), y0 + px(shape.height)

    fill = shape_fill(shape)
    line = shape_line(shape)
    if fill or line:
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=line, width=1)

    if not shape.has_text_frame:
        return
    tf = shape.text_frame
    pad_l = px(tf.margin_left) if tf.margin_left is not None else px(91440)
    pad_r = px(tf.margin_right) if tf.margin_right is not None else px(91440)
    pad_t = px(tf.margin_top) if tf.margin_top is not None else px(45720)
    max_w = max(10, (x1 - x0) - pad_l - pad_r)

    # dark fill implies light text unless the run says otherwise
    dark_bg = bool(fill) and (sum(fill) / 3) < 110
    y = y0 + pad_t
    for para in tf.paragraphs:
        for run in para.runs:
            if not run.text:
                continue
            size = run.font.size.pt if run.font.size else 12
            bold = bool(run.font.bold)
            font = get_font(size, bold)
            color = rgb(run.font.color, (255, 255, 255) if dark_bg else (16, 20, 24))
            for ln in wrap(draw, run.text, font, max_w):
                w = draw.textlength(ln, font=font)
                lx = x0 + pad_l
                if para.alignment is not None and "CENTER" in str(para.alignment):
                    lx = x0 + (x1 - x0 - w) / 2
                draw.text((lx, y), ln, font=font, fill=color)
                y += int(size * SCALE / 72 * 1.22)
            if para.space_after:
                y += px(para.space_after)
    if y > y1 + 6 and shape_fill(shape):
        problems.append(f"text overflows shape bottom by {y - y1}px at ({x0},{y0})")
    if x1 > slide_px[0] + 2 or y1 > slide_px[1] + 2 or shape.left < 0 or shape.top < 0:
        problems.append(f"shape outside slide: ({x0},{y0})-({x1},{y1})")


def render(deck: Path, outdir: Path) -> list[Path]:
    prs = Presentation(deck)
    W, H = px(prs.slide_width), px(prs.slide_height)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, slide in enumerate(prs.slides, 1):
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        problems: list[str] = []
        for shape in slide.shapes:
            draw_shape(draw, shape, problems, (W, H))
        p = outdir / f"slide-{i:02d}.png"
        img.save(p)
        paths.append(p)
        if problems:
            print(f"  slide {i}: " + "; ".join(dict.fromkeys(problems)))
    return paths


def contact_sheet(paths: list[Path], out: Path, cols: int = 3) -> Path:
    thumbs = [Image.open(p) for p in paths]
    tw, th = thumbs[0].width // 2, thumbs[0].height // 2
    rows = (len(thumbs) + cols - 1) // cols
    gap = 14
    sheet = Image.new(
        "RGB", (cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap), (222, 226, 232)
    )
    for i, t in enumerate(thumbs):
        c, r = i % cols, i // cols
        sheet.paste(
            t.resize((tw, th), Image.LANCZOS),
            (gap + c * (tw + gap), gap + r * (th + gap)),
        )
    sheet.save(out)
    return out


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("deck", nargs="?", type=Path, default=here / "HIP-EP-Product-Plan.pptx")
    ap.add_argument("--outdir", type=Path, default=here / "preview")
    args = ap.parse_args()
    paths = render(args.deck, args.outdir)
    sheet = contact_sheet(paths, args.outdir / "contact-sheet.png")
    print(f"rendered {len(paths)} slides -> {args.outdir}")
    print(f"contact sheet -> {sheet}")


if __name__ == "__main__":
    main()
