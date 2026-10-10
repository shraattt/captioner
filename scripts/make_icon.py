"""Generate captioner.ico — the app icon used by the desktop shortcut.

Drawn with Qt (no extra deps): a rounded dark tile with a green caption bar
and a waveform, matching the dashboard's black-and-green theme. Safe to
re-run; overwrites the icon. Qt's ico writer embeds one 256x256 image,
which Windows scales down cleanly.

    QT_QPA_PLATFORM=offscreen python scripts/make_icon.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "captioner.ico")

ACCENT = QColor("#1ed760")
BG = QColor("#0d0d0d")


def draw_icon(size: int) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 256.0  # design at 256, scale everything

    # rounded dark tile
    p.setBrush(BG)
    p.setPen(QPen(QColor("#1a3d26"), 4 * s))
    p.drawRoundedRect(QRectF(8 * s, 8 * s, 240 * s, 240 * s), 48 * s, 48 * s)

    # green caption bar (bottom), like a subtitle strip
    p.setBrush(ACCENT)
    p.setPen(Qt.PenStyle.NoPen)
    bar = QRectF(40 * s, 158 * s, 176 * s, 44 * s)
    path = QPainterPath()
    path.addRoundedRect(bar, 12 * s, 12 * s)
    p.drawPath(path)

    # "caption" dots inside the bar (dark)
    p.setBrush(BG)
    for x, w in ((58, 34), (100, 58), (166, 30)):
        p.drawRoundedRect(QRectF(x * s, 170 * s, w * s, 20 * s), 10 * s, 10 * s)

    # waveform above the bar
    p.setBrush(Qt.BrushStyle.NoBrush)
    pen = QPen(ACCENT, 10 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    heights = (44, 78, 108, 78, 44, 92, 60, 92, 44)
    x = 58
    for h in heights:
        top = 134 - h
        p.drawLine(QPointF(x * s, (top) * s), QPointF(x * s, 134 * s))
        x += 20
    p.end()
    return pm


def main() -> int:
    _app = QApplication.instance() or QApplication([])  # keep it alive
    icon = QIcon(draw_icon(256))
    ok = icon.pixmap(256, 256).save(OUT, "ICO")
    if not ok:
        print(f"could not write {OUT} (Qt ICO writer unavailable?)")
        return 1
    print(f"icon written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
