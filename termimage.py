"""Bild im Terminal darstellen.

Protokoll-Reihenfolge: Kitty > iTerm2 > Sixel (img2sixel) > Halfblock-ANSI.
Halfblock laeuft ueberall, wo Truecolor unterstuetzt wird.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import sys


def protocol() -> str:
    forced = os.environ.get("AIGAME_IMAGE_PROTOCOL")
    if forced:
        return forced
    term = os.environ.get("TERM", "")
    if os.environ.get("KITTY_WINDOW_ID") or "kitty" in term or "ghostty" in term.lower():
        return "kitty"
    if os.environ.get("TERM_PROGRAM") in ("iTerm.app", "WezTerm"):
        return "iterm"
    if shutil.which("img2sixel"):
        return "sixel"
    return "halfblock"


def fit(image, max_cols: int, max_rows: int) -> tuple[int, int]:
    """Zellmasse berechnen. Eine Zelle gilt als 1:2 (breit:hoch)."""
    aspect = image.width / image.height
    cols = max_cols
    rows = max(1, round(cols / aspect / 2))
    if rows > max_rows:
        rows = max_rows
        cols = max(1, round(rows * 2 * aspect))
    return cols, rows


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _kitty(image, cols: int, rows: int) -> None:
    data = base64.standard_b64encode(_png(image))
    chunks = [data[i:i + 4096] for i in range(0, len(data), 4096)]
    for i, chunk in enumerate(chunks):
        more = 1 if i < len(chunks) - 1 else 0
        ctrl = (f"a=T,f=100,c={cols},r={rows},m={more}" if i == 0 else f"m={more}")
        sys.stdout.write(f"\x1b_G{ctrl};{chunk.decode()}\x1b\\")
    sys.stdout.write(f"\x1b[{rows}B\r")
    sys.stdout.flush()


def _iterm(image, cols: int, rows: int) -> None:
    b64 = base64.standard_b64encode(_png(image)).decode()
    sys.stdout.write(
        f"\x1b]1337;File=inline=1;width={cols};height={rows};"
        f"preserveAspectRatio=1:{b64}\x07\n"
    )
    sys.stdout.flush()


def _sixel(image, cols: int, rows: int) -> None:
    try:
        proc = subprocess.run(
            ["img2sixel", "-w", str(cols * 8)],
            input=_png(image), capture_output=True, timeout=20,
        )
        sys.stdout.write(proc.stdout.decode("utf-8", "ignore"))
        sys.stdout.flush()
    except (OSError, subprocess.SubprocessError):
        _halfblock(image, cols, rows)


def _halfblock(image, cols: int, rows: int, indent: str = "") -> None:
    img = image.convert("RGB").resize((cols, rows * 2))
    px = img.load()
    lines = []
    for y in range(rows):
        parts = [indent]
        for x in range(cols):
            r1, g1, b1 = px[x, y * 2]
            r2, g2, b2 = px[x, y * 2 + 1]
            parts.append(
                f"\x1b[38;2;{r1};{g1};{b1};48;2;{r2};{g2};{b2}m\u2580"
            )
        lines.append("".join(parts) + "\x1b[0m")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def show(image, max_cols: int, max_rows: int, margin: int = 0) -> None:
    """Bild an der aktuellen Cursorposition ausgeben, links um `margin` eingerueckt."""
    cols, rows = fit(image, max_cols, max_rows)
    proto = protocol()
    indent = " " * margin
    if proto == "halfblock":
        _halfblock(image, cols, rows, indent)
        return
    if indent:
        sys.stdout.write(indent)
    {"kitty": _kitty, "iterm": _iterm, "sixel": _sixel}[proto](image, cols, rows)
