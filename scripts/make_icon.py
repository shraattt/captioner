"""Generate captioner.ico, the app icon (window, shortcuts, web favicon).

Two things the previous generator got wrong, both fixed here:

* It shipped a single 256x256 frame, which Windows blurs down to the 16 and 32
  pixel sizes it actually draws most of the time. Every size is now drawn by
  Qt at its own resolution and embedded, so the small frames are crisp.
* It was green while the app is violet. The palette is read from
  ui/dashboard/theme.py, so the icon follows the theme instead of drifting.

The drawing is the same mark at every size (a rounded tile over a caption
strip), but detail is dropped as the size falls, because art that reads at
256px is mush at 16px:

    16-24  violet tile, two dark text lines, nothing else
    32     dark tile, a caption strip with one line
    36-47  dark tile, a caption strip with two lines
    48+    dark tile, waveform above a two-line caption strip

    python scripts/make_icon.py             # write captioner.ico
    python scripts/make_icon.py --preview   # also write a size sheet to inspect
"""

import argparse
import importlib.util
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

try:  # Pillow assembles the .ico; it is a development dependency only
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - main() explains how to install it
    Image = ImageDraw = None

OUT = ROOT / "captioner.ico"
PREVIEW = ROOT / "data" / "tmp" / "icon_preview.png"

# Every frame Windows asks for; 20 and 40 are used by Explorer at some DPIs.
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
TINY_MAX = 24  # at or below: the simplified mark
WAVE_MIN = 48  # from here up: the waveform is worth drawing
TWO_LINE_MIN = 36  # from here up: the strip fits two text lines

# Design grid: every coordinate below is a 0-256 value scaled to the real size.
GRID = 256.0
TILE = (4, 4, 248, 248)
TILE_RADIUS = 56


def _theme():
    """Load ui/dashboard/theme.py without importing the whole ui package."""
    path = ROOT / "ui" / "dashboard" / "theme.py"
    spec = importlib.util.spec_from_file_location("_captioner_icon_theme", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_THEME = _theme()
# Fallbacks mirror the theme, so the icon still builds if a name is renamed.
ACCENT_BRIGHT = QColor(getattr(_THEME, "_ACCENT_BRIGHT", "#a78bfa"))
ACCENT_DIM = QColor(getattr(_THEME, "_ACCENT_DIM", "#7c4fe0"))
ACCENT_TEXT = QColor(getattr(_THEME, "_ACCENT_TEXT", "#0e0a18"))
TILE_TOP = QColor("#1f1f2b")
TILE_BOTTOM = QColor(getattr(_THEME, "_BG", "#0b0b10"))
TILE_EDGE = QColor("#332d4d")


def _rect(size: int, x: float, y: float, w: float, h: float) -> QRectF:
    """A rectangle given in 256-grid coordinates, scaled to ``size``."""
    s = size / GRID
    return QRectF(x * s, y * s, w * s, h * s)


def _px(value: float, size: int) -> float:
    return value * size / GRID


def _brush(rect: QRectF, colors: tuple[QColor, ...]) -> QBrush:
    """A solid color, or a top-to-bottom gradient when given two."""
    if len(colors) == 1:
        return QBrush(colors[0])
    gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    gradient.setColorAt(0.0, colors[0])
    gradient.setColorAt(1.0, colors[1])
    return QBrush(gradient)


def _fill(p: QPainter, rect: QRectF, radius: float, *colors: QColor) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_brush(rect, colors))
    p.drawRoundedRect(rect, radius, radius)


def _draw_tile(p: QPainter, size: int, violet: bool) -> None:
    """The rounded square behind the mark: solid violet, or dark and edged."""
    x, y, w, h = TILE
    tile = _rect(size, x, y, w, h)
    radius = _px(TILE_RADIUS, size)
    if violet:
        _fill(p, tile, radius, ACCENT_BRIGHT, ACCENT_DIM)
        return
    pen = QPen(TILE_EDGE, max(1.0, _px(4, size)))
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(_brush(tile, (TILE_TOP, TILE_BOTTOM)))
    p.drawRoundedRect(tile, radius, radius)


def _draw_tiny(p: QPainter, size: int) -> None:
    """16-24px: a violet tile with two thick lines. Nothing else survives."""
    _draw_tile(p, size, violet=True)
    _fill(p, _rect(size, 60, 80, 136, 32), _px(16, size), ACCENT_TEXT)
    _fill(p, _rect(size, 60, 144, 92, 32), _px(16, size), ACCENT_TEXT)


def _caption_strip(p: QPainter, size: int, top: float, height: float,
                   lines: int) -> None:
    """A violet rounded strip carrying dark text lines, like a subtitle bar."""
    width = 168.0
    left = (GRID - width) / 2
    _fill(p, _rect(size, left, top, width, height), _px(20, size),
          ACCENT_BRIGHT, ACCENT_DIM)
    line_h = height * (0.24 if lines == 1 else 0.19)
    if lines == 1:
        spans = ((0.76, 0.38),)
    else:
        spans = ((0.76, 0.24), (0.48, 0.57))
    for fraction, offset in spans:
        _fill(
            p,
            _rect(size, left + width * 0.12, top + height * offset,
                  width * fraction, line_h),
            _px(line_h / 2, size),
            ACCENT_TEXT,
        )


def _draw_waveform(p: QPainter, size: int) -> None:
    """Five bars when there is room, three when there is not.

    At 48px a five-bar waveform leaves the outer bars about four pixels wide,
    which the rounded caps wash out; three thicker bars read far better.
    """
    if size >= WAVE_MIN * 4 // 3:
        heights, width, pitch = (26, 48, 74, 48, 26), 24.0, 36.0
    else:
        heights, width, pitch = (34, 74, 34), 26.0, 40.0
    baseline = 116.0
    x = (GRID - (len(heights) * width + (len(heights) - 1) * (pitch - width))) / 2
    for height in heights:
        _fill(p, _rect(size, x, baseline - height, width, height), _px(width * 0.38, size),
              ACCENT_BRIGHT)
        x += pitch


def _draw_detailed(p: QPainter, size: int) -> None:
    """32px and up: the tile, a caption strip, and a waveform once it fits."""
    _draw_tile(p, size, violet=False)
    if size >= WAVE_MIN:
        _draw_waveform(p, size)
    top, height = (132.0, 72.0) if size >= WAVE_MIN else (98.0, 104.0)
    _caption_strip(p, size, top, height, 2 if size >= TWO_LINE_MIN else 1)


def draw_icon(size: int) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    try:
        if size <= TINY_MAX:
            _draw_tiny(p, size)
        else:
            _draw_detailed(p, size)
    finally:
        p.end()
    return pm


def _pillow_frame(size: int):
    """Render one size and hand it to Pillow as an RGBA image."""
    png = QByteArray()
    buffer = QBuffer(png)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not draw_icon(size).save(buffer, "PNG"):
        raise RuntimeError(f"Qt could not encode the {size}px frame")
    buffer.close()
    return Image.open(io.BytesIO(bytes(png))).convert("RGBA")


def build(path: Path) -> tuple[list[int], int]:
    """Write a multi-size .ico; return its frame sizes and total bytes."""
    frames = {size: _pillow_frame(size) for size in SIZES}
    largest = max(SIZES)
    frames[largest].save(
        path,
        format="ICO",
        sizes=[(size, size) for size in SIZES],
        append_images=[frames[size] for size in SIZES if size != largest],
    )
    with Image.open(path) as written:
        embedded = sorted(size for size, _ in written.info["sizes"])
    if embedded != list(SIZES):
        raise RuntimeError(f"icon wrote {embedded}, expected {list(SIZES)}")
    return embedded, path.stat().st_size


def write_preview(path: Path) -> None:
    """A contact sheet: every size on dark and light, plus blown-up small ones."""
    pad = 24
    row = max(SIZES) + pad * 2
    width = sum(size + pad for size in SIZES) + pad
    sheet = Image.new("RGBA", (width, row * 2 + 280), (255, 255, 255, 255))
    dark = Image.new("RGBA", (width, row), (11, 11, 16, 255))
    light = Image.new("RGBA", (width, row), (242, 242, 246, 255))

    x = pad
    for size in SIZES:
        frame = _pillow_frame(size)
        y = pad + (max(SIZES) - size) // 2
        dark.alpha_composite(frame, (x, y))
        light.alpha_composite(frame, (x, y))
        ImageDraw.Draw(dark).text((x, row - pad - 2), str(size), fill=(150, 150, 160))
        x += size + pad
    sheet.paste(dark, (0, 0))
    sheet.paste(light, (0, row))

    zoom, y = 8, row * 2 + 30
    x = pad
    for size in (16, 24, 32, 48):
        big = _pillow_frame(size).resize((size * zoom, size * zoom), Image.Resampling.NEAREST)
        sheet.alpha_composite(big, (x, y))
        x += size * zoom + pad
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"preview sheet: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="also write data/tmp/icon_preview.png")
    args = parser.parse_args(argv)

    if Image is None:
        print("Pillow is required to assemble the .ico: "
              "python -m pip install -r requirements-dev.txt")
        return 1

    app = QApplication.instance() or QApplication([])  # noqa: F841 - must stay alive
    sizes, total = build(OUT)
    print(f"icon written: {OUT}")
    print(f"  frames: {', '.join(str(size) for size in sizes)}")
    print(f"  total: {total / 1024:.1f} KiB")
    if args.preview:
        write_preview(PREVIEW)
    return 0


if __name__ == "__main__":
    sys.exit(main())
