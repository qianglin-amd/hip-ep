# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Terminal output: tables, progress bars, colour and byte/duration formatting.

Output stays inside ASCII. A Windows console still defaults to a code page that cannot
encode a box-drawing character or an em dash, and a UnicodeEncodeError while reporting
progress is a worse failure than a plain-looking table.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Iterable, Sequence

_ANSI = re.compile(r"\033\[[0-9;]*m")

_TRUTHY = {"1", "true", "yes", "on"}


def _colour_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("HIPEP_COLOR", "").strip().lower() in _TRUTHY:
        return True
    return sys.stdout.isatty()


COLOUR = _colour_enabled()

_CODES = {
    "dim": "2",
    "bold": "1",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "cyan": "36",
}


def paint(text: str, *styles: str) -> str:
    if not COLOUR or not styles:
        return text
    codes = ";".join(_CODES[s] for s in styles if s in _CODES)
    return f"\033[{codes}m{text}\033[0m" if codes else text


def visible(text: str) -> int:
    """Screen columns a string occupies, ignoring colour escape sequences."""
    return len(_ANSI.sub("", text))


def width(default: int = 100) -> int:
    return shutil.get_terminal_size((default, 24)).columns


def gigabytes(value: float | int | None) -> str:
    if not value:
        return "-"
    return f"{float(value):.1f} GB"


def bytes_human(value: int | float | None) -> str:
    if value is None:
        return "-"
    if not value:
        return "0 B"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def ago(timestamp: float | None, now: float) -> str:
    if not timestamp:
        return "-"
    return f"{duration(now - timestamp)} ago"


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    right: Sequence[int] = (),
    indent: str = "",
) -> str:
    """Two-space-separated columns sized to their content.

    Columns are not truncated. A model id or a Hugging Face repo is something users
    copy out of this output, and a table that silently shortens it is worse than one
    that wraps in a narrow terminal.
    """
    body = [[str(cell) for cell in row] for row in rows]
    if not body:
        return ""
    # Cells may already carry colour, whose escape sequences occupy no screen columns.
    # Padding on len() would then indent every coloured column by several characters.
    widths = [visible(h) for h in headers]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], visible(cell))

    right_set = set(right)

    def render(cells: Sequence[str], *, header: bool) -> str:
        parts = []
        for index, cell in enumerate(cells):
            pad = " " * max(0, widths[index] - visible(cell))
            parts.append(pad + cell if index in right_set else cell + pad)
        line = indent + "  ".join(parts).rstrip()
        return paint(line, "bold") if header else line

    lines = [render(headers, header=True)]
    lines.append(indent + paint("  ".join("-" * w for w in widths), "dim"))
    lines.extend(render(row, header=False) for row in body)
    return "\n".join(lines)


def keyvalues(pairs: Iterable[tuple[str, str]], *, indent: str = "  ") -> str:
    pairs = [(k, v) for k, v in pairs if v not in (None, "")]
    if not pairs:
        return ""
    pad = max(len(k) for k, _ in pairs)
    return "\n".join(f"{indent}{paint(k.ljust(pad), 'dim')}  {v}" for k, v in pairs)


def bar(fraction: float, size: int = 24) -> str:
    filled = max(0, min(size, round(fraction * size)))
    return "[" + "#" * filled + "." * (size - filled) + "]"


class Line:
    """A single rewritable terminal line, used for job and download progress.

    Falls back to one line per update when stdout is not a terminal, so piping the
    output of a long job into a log file produces something readable rather than a
    file full of carriage returns.
    """

    def __init__(self, stream=None):
        self._stream = stream or sys.stdout
        self._tty = self._stream.isatty()
        self._last = ""

    def update(self, text: str) -> None:
        if not self._tty:
            if text != self._last:
                self._stream.write(text + "\n")
                self._stream.flush()
                self._last = text
            return
        limit = max(20, width() - 1)
        if "\033" not in text:
            # Truncating a coloured string could cut an escape sequence in half and
            # leave the terminal in that colour, so only plain text is shortened.
            text = text[:limit]
        pad = max(0, visible(self._last) - visible(text))
        self._stream.write("\r" + text + " " * pad)
        self._stream.flush()
        self._last = text

    def done(self, text: str | None = None) -> None:
        if text is not None:
            self.update(text)
        if self._tty and self._last:
            self._stream.write("\n")
            self._stream.flush()
        self._last = ""


STATE_STYLE = {
    "succeeded": ("green", "ok"),
    "running": ("cyan", "running"),
    "pending": ("dim", "pending"),
    "failed": ("red", "failed"),
    "cancelled": ("yellow", "cancelled"),
}


def state_label(state: str) -> str:
    style, label = STATE_STYLE.get(state, ("dim", state))
    return paint(label, style)
