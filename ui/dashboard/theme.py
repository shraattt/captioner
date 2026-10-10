"""The violet-glass theme: palette, status colors, stylesheet, application.

Split out of the window because the stylesheet is the largest literal in the
UI: keeping it here keeps the widget code readable, and the colors can be
imported (and tested) on their own.
"""

from PySide6.QtGui import (
    QColor,
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication,
)

_STATUS_COLORS_LIGHT = {
    "queued": "#b9bdc7",
    "running": "#d6c9ff",
    "done": "#c9f0dd",
    "error": "#f2b8c6",
    "canceled": "#8b8b95",
}
#Visual direction: "violet glass"
# Deep near-black with a blue undertone (never flat #000), layered panel
# gradients that read as translucent sheets, one violet accent used sparingly
# (primary action, selection, progress, active states), soft greys and
# off-whites everywhere else. Destructive actions are muted rose that only
# surfaces on hover. Radii are generous (10-14px); worker/status badges are
# pills. Type: Segoe UI for everything, monospace reserved for timestamps.
_ACCENT = "#8b5cf6"  # violet 500 - the one accent
_ACCENT_BRIGHT = "#a78bfa"  # hover
_ACCENT_DIM = "#7c4fe0"  # pressed
_ACCENT_TEXT = "#0e0a18"  # text on accent
_ACCENT_SOFT = "#241d38"  # selection highlight (violet-tinted)
_BG = "#0b0b10"
_SURFACE = "rgba(255, 255, 255, 0.035)"  # glassy sheet over the dark bg
_SURFACE_SOLID = "#15151d"
_PANEL = "#111118"
_BORDER = "rgba(255, 255, 255, 0.08)"  # thin light border
_BORDER_HOVER = "rgba(167, 139, 250, 0.55)"
_TEXT = "#ececf1"
_TEXT_DIM = "#9b9ba6"
_TEXT_FAINT = "#6d6d78"
_DANGER = "#c96f85"  # muted rose - destructive, quiet until hover
_DANGER_HOVER = "#e08a9e"

# Status pill colors: (text, pill background). Accent only for "running";
# done is neutral, error is muted rose, no loud red anywhere.
_STATUS_PILLS = {
    "queued": ("#b9bdc7", "#23242e"),
    "running": ("#d6c9ff", "#31284a"),
    "done": ("#c9f0dd", "#1d2b25"),
    "error": ("#f2b8c6", "#38222c"),
    "canceled": ("#8b8b95", "#1e1e26"),
}
# Kept for the palette (light theme never really used; dark is forced).
_STATUS_COLORS_DARK = {
    "queued": "#b9bdc7",
    "running": "#d6c9ff",
    "done": "#c9f0dd",
    "error": "#f2b8c6",
    "canceled": "#8b8b95",
}
_QSS = f"""
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", Arial, "Helvetica Neue", sans-serif; }}
QWidget {{ color: {_TEXT}; font-size: 13px; }}
QMainWindow, QDialog {{ background-color: {_BG}; }}

QPushButton {{ background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                 stop:0 #7c5cd6, stop:1 {_ACCENT_DIM});
               color: {_ACCENT_TEXT}; font-weight: 600;
               border: none; border-radius: 10px; padding: 8px 18px; }}
QPushButton:hover {{ background-color: {_ACCENT_BRIGHT}; }}
QPushButton:pressed {{ background-color: {_ACCENT_DIM}; }}
QPushButton:disabled {{ background-color: #1d1b28; color: #5a5670; }}
QPushButton[danger="true"] {{ background-color: transparent; color: #b98a96;
               border: 1px solid rgba(201, 111, 133, 0.25); }}
QPushButton[danger="true"]:hover {{ background-color: rgba(201, 111, 133, 0.12);
               color: {_DANGER_HOVER}; border-color: rgba(224, 138, 158, 0.5); }}
QPushButton[secondary="true"] {{ background-color: {_SURFACE}; color: {_TEXT};
               border: 1px solid {_BORDER}; font-weight: 600; }}
QPushButton[secondary="true"]:hover {{ border-color: {_BORDER_HOVER};
               color: {_ACCENT_BRIGHT}; }}
QPushButton[accentOutline="true"] {{ background-color: rgba(139, 92, 246, 0.14);
               color: {_ACCENT_BRIGHT}; border: 1px solid rgba(139, 92, 246, 0.45); }}
QPushButton[accentOutline="true"]:hover {{ background-color: rgba(139, 92, 246, 0.24);
               border-color: {_ACCENT_BRIGHT}; }}

QComboBox {{ background: {_SURFACE}; border: 1px solid {_BORDER};
             border-radius: 9px; padding: 6px 10px; }}
QComboBox:focus {{ border-color: {_BORDER_HOVER}; }}
QComboBox QAbstractItemView {{ background: {_SURFACE_SOLID}; border: 1px solid {_BORDER};
             border-radius: 9px; selection-background-color: {_ACCENT_SOFT};
             selection-color: {_TEXT}; outline: none; }}

QPlainTextEdit, QTextEdit, QLineEdit {{ background: {_PANEL};
    border: 1px solid {_BORDER}; border-radius: 12px; padding: 6px;
    selection-background-color: {_ACCENT}; selection-color: {_ACCENT_TEXT}; }}
QPlainTextEdit:focus {{ border-color: {_BORDER_HOVER}; }}

#SummaryCard {{ background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
               stop:0 rgba(255,255,255,0.045), stop:1 rgba(255,255,255,0.015));
               border: 1px solid {_BORDER}; border-radius: 14px; }}
#WorkerPill {{ background-color: #1e1e28; border: 1px solid {_BORDER};
               border-radius: 11px; padding: 3px 12px; color: {_TEXT_DIM}; }}
#WorkerPill[running="true"] {{ background-color: #31284a; color: #d6c9ff;
               border-color: rgba(139, 92, 246, 0.4); }}
#EditStatePill {{ background-color: rgba(139, 92, 246, 0.18); color: #d6c9ff;
               border: 1px solid rgba(139, 92, 246, 0.4);
               border-radius: 10px; padding: 2px 10px; }}

QTableWidget {{ background: transparent; alternate-background-color: rgba(255,255,255,0.012);
                border: none; gridline-color: transparent;
                selection-background-color: {_ACCENT_SOFT}; selection-color: {_TEXT}; }}
QTableWidget::item {{ border: none; padding: 4px 6px; }}
QTableWidget::item:selected {{ background-color: {_ACCENT_SOFT}; border-radius: 8px; }}
QHeaderView {{ background: transparent; qproperty-defaultSectionSize: 40px; }}
QHeaderView::section {{ background: transparent; color: {_TEXT_FAINT}; padding: 8px;
               border: none; border-bottom: 1px solid {_BORDER};
               font-size: 11px; font-weight: 600; }}
QTableCornerButton::section {{ background: transparent; border: none; }}

QProgressBar {{ background: rgba(255,255,255,0.05); border: none;
               border-radius: 7px; text-align: center; color: {_TEXT_DIM}; }}
QProgressBar::chunk {{ background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
               stop:0 {_ACCENT_DIM}, stop:1 {_ACCENT_BRIGHT}); border-radius: 7px; }}

QSlider::groove:horizontal {{ height: 4px; background: rgba(255,255,255,0.10);
               border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {_ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 13px; height: 13px; margin: -5px 0;
               border-radius: 7px; background: {_TEXT}; }}
QSlider::handle:horizontal:hover {{ background: {_ACCENT_BRIGHT}; }}
QSlider::handle:horizontal:disabled {{ background: #3a3a44; }}

QCheckBox {{ spacing: 8px; color: {_TEXT_DIM}; }}
QCheckBox::indicator {{ width: 17px; height: 17px; border: 1px solid rgba(255,255,255,0.16);
                        border-radius: 5px; background: rgba(255,255,255,0.04); }}
QCheckBox::indicator:hover {{ border-color: {_BORDER_HOVER}; }}
QCheckBox::indicator:checked {{ background: {_ACCENT}; border-color: {_ACCENT}; }}

QMenuBar {{ background: {_BG}; color: {_TEXT_DIM}; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: 6px; }}
QMenuBar::item:selected {{ background: rgba(255,255,255,0.06); color: {_TEXT}; }}
QMenu {{ background: {_SURFACE_SOLID}; border: 1px solid {_BORDER}; border-radius: 12px;
         padding: 6px; }}
QMenu::item {{ padding: 7px 26px 7px 14px; border-radius: 7px; }}
QMenu::item:selected {{ background: {_ACCENT_SOFT}; color: {_TEXT}; }}
QMenu::separator {{ height: 1px; background: {_BORDER}; margin: 5px 8px; }}
QToolTip {{ background: {_SURFACE_SOLID}; color: {_TEXT}; border: 1px solid {_BORDER_HOVER};
            border-radius: 8px; padding: 6px 8px; }}
QStatusBar {{ background: transparent; color: {_TEXT_FAINT}; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background: transparent; height: 8px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: rgba(255,255,255,0.10); border-radius: 5px;
               min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {_ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: rgba(255,255,255,0.10); border-radius: 5px;
               min-width: 30px; }}
QScrollBar::handle:horizontal:hover {{ background: {_ACCENT}; }}
"""


def _apply_theme() -> None:
    """Force the violet-glass dark theme regardless of the OS light/dark setting."""
    app = QApplication.instance()
    if app is None:
        return
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(_BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(_TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(_PANEL))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#141420"))
    palette.setColor(QPalette.ColorRole.Text, QColor(_TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(_SURFACE_SOLID))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(_TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(_ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(_ACCENT_TEXT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(_SURFACE_SOLID))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(_TEXT))
    palette.setColor(QPalette.ColorRole.Link, QColor(_ACCENT_BRIGHT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#5d5d68"))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.Text,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#565662"))
    app.setPalette(palette)
