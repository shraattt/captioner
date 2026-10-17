"""Themes: palettes, status colors, stylesheet generation, application.

Three themes share one structure, so a layout rule is written once and lands in
all of them:

* **dark** - the original "violet glass" look: deep near-black with a blue
  undertone, layered translucent sheets, one violet accent.
* **light** - for bright rooms and daytime use, built from the same shape so
  nothing moves when you switch, only the values change.
* **high contrast** - opaque surfaces, near-white text on black, saturated
  status pills and heavier borders. For low vision, and for anyone who finds
  the glassy translucency hard to read. No translucency anywhere: every panel
  is a flat colour with a solid border.

"System" follows the OS light/dark preference (Windows' "Choose your default
app mode"). Whether a theme is dark is decided here and exported through
:func:`is_dark`, so the window does not have to guess from its own palette.

Text size is a preference, not a constant: every size in the sheet is derived
from one base number, so scaling the UI (Ctrl+= / Ctrl+- / Ctrl+0) grows the
whole window rather than one panel. :func:`px` is what widget code uses for
sizes it has to set inline.
"""

from dataclasses import dataclass, field

from PySide6.QtCore import Qt  # noqa: F401  (kept for callers importing Qt here)
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication,
)

THEME_SYSTEM = "system"
THEME_DARK = "dark"
THEME_LIGHT = "light"
THEME_CONTRAST = "contrast"
THEMES = (THEME_SYSTEM, THEME_DARK, THEME_LIGHT, THEME_CONTRAST)
# What the picker shows: the key is what is stored, the value is what is read.
THEME_NAMES = {
    THEME_SYSTEM: "Follow Windows (light or dark)",
    THEME_DARK: "Dark",
    THEME_LIGHT: "Light",
    THEME_CONTRAST: "High contrast",
}

# Text scale bounds: below 0.85 the 11px labels stop being readable, above 1.6
# the fixed-height bars and pills start to clip.
MIN_TEXT_SCALE = 0.85
MAX_TEXT_SCALE = 1.6
DEFAULT_TEXT_SCALE = 1.0
BASE_FONT_PX = 13


def clamp_scale(value) -> float:
    """A usable text scale, whatever was stored in the settings file."""
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TEXT_SCALE
    if scale != scale:  # NaN
        return DEFAULT_TEXT_SCALE
    return max(MIN_TEXT_SCALE, min(MAX_TEXT_SCALE, round(scale, 3)))


@dataclass(frozen=True)
class Palette:
    """Every colour the sheet uses, per theme.

    ``accent`` is the brand violet: fills, rings, progress, and the colour the
    app icon is drawn from. ``button`` is the tint used behind button *text*,
    kept separate on purpose: button labels measured only 4.6:1 on the original
    #8b5cf6 fill (AA is 4.5:1, so it passed by a hair), and tuning that tint is
    a UI decision that should not repaint the icon.
    """

    name: str
    dark: bool
    bg: str
    surface: str
    surface_solid: str
    panel: str
    border: str
    border_hover: str
    text: str
    text_dim: str
    text_faint: str
    accent: str
    accent_bright: str
    accent_dim: str
    accent_soft: str
    button: str
    button_text: str
    button_hover: str
    button_pressed: str
    danger: str
    danger_hover: str
    scroll: str
    radius: int
    # Status pill colours: status -> (text, pill background).
    pills: dict[str, tuple[str, str]] = field(default_factory=dict)


DARK = Palette(
    name=THEME_DARK,
    dark=True,
    # Deep near-black with a blue undertone (never flat #000), layered panel
    # gradients that read as translucent sheets, one violet accent used
    # sparingly (primary action, selection, progress, active states), soft
    # greys and off-whites everywhere else. Radii are generous (10-14px);
    # worker/status badges are pills.
    bg="#0b0b10",
    surface="rgba(255, 255, 255, 0.035)",  # glassy sheet over the dark bg
    surface_solid="#15151d",
    panel="#111118",
    border="rgba(255, 255, 255, 0.08)",
    border_hover="rgba(167, 139, 250, 0.55)",
    text="#ececf1",
    text_dim="#9b9ba6",
    # Not #6d6d78: that read as elegant and measured 3.8:1 on this background,
    # below the 4.5:1 normal text needs. It is used for hints, timestamps, the
    # status bar and table headers, all of which are text.
    text_faint="#8b8b95",  # 5.8:1
    accent="#8b5cf6",  # violet 500 - the one accent (the app icon's colour)
    accent_bright="#a78bfa",  # hover, focus ring
    accent_dim="#7c4fe0",  # pressed
    accent_soft="#241d38",  # selection highlight (violet-tinted)
    button="#a78bfa",  # 6.8:1 against the text below
    button_text="#16121f",
    button_hover="#c4b5fd",  # 10.1:1
    button_pressed="#9d7ff5",  # 6.1:1
    danger="#c96f85",  # muted rose - destructive, quiet until hover
    danger_hover="#e08a9e",
    scroll="rgba(255, 255, 255, 0.10)",
    radius=10,
    pills={
        "queued": ("#b9bdc7", "#23242e"),
        "running": ("#d6c9ff", "#31284a"),
        "done": ("#c9f0dd", "#1d2b25"),
        "error": ("#f2b8c6", "#38222c"),
        "canceled": ("#8b8b95", "#1e1e26"),
    },
)

LIGHT = Palette(
    name=THEME_LIGHT,
    dark=False,
    bg="#f4f4f8",
    surface="#ffffff",
    surface_solid="#ffffff",
    panel="#fbfbfd",
    border="rgba(18, 18, 30, 0.14)",
    border_hover="rgba(109, 40, 217, 0.65)",
    text="#1b1b23",
    text_dim="#4d4d5a",
    text_faint="#5b5b68",  # 6.1:1 (#6b6b7a was a marginal 4.8:1)
    accent="#6d28d9",  # darker violet: the dark theme's accent fails on white
    accent_bright="#7c3aed",
    accent_dim="#5b21b6",
    accent_soft="#ece4ff",
    button="#6d28d9",  # 7.1:1 with white text
    button_text="#ffffff",
    button_hover="#7c3aed",  # 5.7:1
    button_pressed="#5b21b6",  # 9.1:1
    danger="#b8446a",
    danger_hover="#9c3358",
    scroll="rgba(18, 18, 30, 0.20)",
    radius=10,
    pills={
        "queued": ("#3f3f4a", "#e8e8ef"),
        "running": ("#3b2a7a", "#e6e0ff"),
        "done": ("#14532d", "#dcf5e6"),
        "error": ("#7a1030", "#fbe1e8"),
        "canceled": ("#4a4a55", "#e6e6ec"),
    },
)

# High contrast keeps the brand's violet but makes everything opaque: the pill
# backgrounds and text are chosen for legibility first, and the focus ring is
# part of the theme rather than a hover afterthought.
CONTRAST = Palette(
    name=THEME_CONTRAST,
    dark=True,
    bg="#000000",
    surface="#121212",
    surface_solid="#121212",
    panel="#050505",
    border="rgba(255, 255, 255, 0.75)",
    border_hover="#ffe066",
    text="#ffffff",
    text_dim="#f0f0f0",
    text_faint="#d6d6d6",
    accent="#c4b5fd",
    accent_bright="#ffe066",  # focus/hover reads as yellow against violet
    accent_dim="#8b5cf6",
    accent_soft="#3b2f6b",
    button="#c4b5fd",  # 11.2:1 with black text
    button_text="#000000",
    button_hover="#ffe066",  # 16.8:1
    button_pressed="#a78bfa",  # 8.6:1
    danger="#ffb3c1",
    danger_hover="#ffd9e0",
    scroll="rgba(255, 255, 255, 0.55)",
    radius=6,
    pills={
        "queued": ("#ffffff", "#1c1c1c"),
        "running": ("#ffffff", "#43307a"),
        "done": ("#b7f7d0", "#0c2a1c"),
        "error": ("#ffc9d4", "#4a1020"),
        "canceled": ("#f0f0f0", "#1c1c1c"),
    },
)

PALETTES = {DARK.name: DARK, LIGHT.name: LIGHT, CONTRAST.name: CONTRAST}

# Current theme and text scale, so widgets can ask for a scaled size without
# threading both values through every constructor.
_current = [DARK, DEFAULT_TEXT_SCALE]


def status_colors(palette: Palette | None = None) -> dict[str, str]:
    """Status -> text colour, for the table's status column."""
    pills = (palette or _current[0]).pills
    return {key: value[0] for key, value in pills.items()}


def px(size: float) -> str:
    """``"15px"`` - ``size`` scaled by the user's text-size preference."""
    return f"{max(1, round(size * _current[1]))}px"


def text_faint() -> str:
    """The muted label colour of the current theme."""
    return _current[0].text_faint


def faint_style(extra: str = "") -> str:
    """Inline stylesheet text for a muted label (theme- and scale-aware)."""
    return f"color: {_current[0].text_faint}; {extra}".strip()


def current_palette() -> Palette:
    return _current[0]


def current_text_scale() -> float:
    return _current[1]


def is_dark(theme: str | None = None) -> bool:
    """True when the theme (or the resolved current one) is dark."""
    if theme is None:
        return _current[0].dark
    return resolve(theme).dark


def system_theme() -> str:
    """What Windows reports: ``"dark"``, ``"light"``, or ``"dark"`` if unsure."""
    hints = QGuiApplication.styleHints() if QGuiApplication.instance() else None
    scheme = getattr(hints, "colorScheme", None)
    try:
        value = scheme() if callable(scheme) else None
    except Exception:  # pragma: no cover - depends on the platform plugin
        value = None
    return "light" if value == Qt.ColorScheme.Light else "dark"


def resolve(theme: str | None) -> Palette:
    """The palette for a theme key; ``"system"`` follows the OS."""
    key = (theme or THEME_DARK).strip().lower()
    if key == THEME_SYSTEM:
        key = system_theme()
    return PALETTES.get(key, DARK)


def _qss(p: Palette, scale: float = DEFAULT_TEXT_SCALE) -> str:
    base = max(9, round(BASE_FONT_PX * scale))
    small = max(8, round(11 * scale))
    radius = p.radius
    focus = f"2px solid {p.accent_bright}"
    return f"""
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", Arial, "Helvetica Neue", sans-serif; }}
QWidget {{ color: {p.text}; font-size: {base}px; }}
QMainWindow, QDialog {{ background-color: {p.bg}; }}

QPushButton {{ background-color: {p.button};
               color: {p.button_text}; font-weight: 600;
               border: 2px solid transparent; border-radius: {radius}px; padding: 6px 16px; }}
QPushButton:hover {{ background-color: {p.button_hover}; }}
QPushButton:pressed {{ background-color: {p.button_pressed}; }}
QPushButton:disabled {{ background-color: {p.bg}; color: {p.text_faint};
               border-color: {p.border}; }}
QPushButton:focus {{ border: {focus}; }}
QPushButton[danger="true"] {{ background-color: transparent; color: {p.danger};
               border: 1px solid {p.danger}; }}
QPushButton[danger="true"]:hover {{ background-color: {p.danger_hover};
               color: {p.bg}; border-color: {p.danger_hover}; }}
QPushButton[secondary="true"] {{ background-color: {p.surface}; color: {p.text};
               border: 1px solid {p.border}; font-weight: 600; }}
QPushButton[secondary="true"]:hover {{ border-color: {p.border_hover};
               color: {p.accent_bright}; }}
QPushButton[accentOutline="true"] {{ background-color: {p.surface};
               color: {p.accent_bright}; border: 1px solid {p.accent}; }}
QPushButton[accentOutline="true"]:hover {{ background-color: {p.accent_soft};
               border-color: {p.accent_bright}; }}

QComboBox {{ background: {p.surface}; border: 1px solid {p.border};
             border-radius: {max(6, radius - 1)}px; padding: 5px 9px; }}
QComboBox:focus {{ border: {focus}; }}
QComboBox QAbstractItemView {{ background: {p.surface_solid}; border: 1px solid {p.border};
             border-radius: {max(6, radius - 1)}px; selection-background-color: {p.button};
             selection-color: {p.button_text}; outline: none; }}

QPlainTextEdit, QTextEdit, QLineEdit {{ background: {p.panel};
    border: 1px solid {p.border}; border-radius: {max(6, radius + 2)}px; padding: 6px;
    selection-background-color: {p.button}; selection-color: {p.button_text}; }}
QPlainTextEdit:focus, QTextEdit:focus, QLineEdit:focus {{ border: {focus}; }}

#SummaryCard {{ background-color: {p.surface};
               border: 1px solid {p.border}; border-radius: {radius + 4}px; }}
#WorkerPill {{ background-color: {p.surface}; border: 1px solid {p.border};
               border-radius: {max(6, radius + 1)}px; padding: 3px 12px; color: {p.text_dim}; }}
#WorkerPill[running="true"] {{ background-color: {p.accent_soft}; color: {p.accent_bright};
               border-color: {p.accent}; }}
#EditStatePill {{ background-color: {p.accent_soft}; color: {p.accent_bright};
               border: 1px solid {p.accent};
               border-radius: {radius}px; padding: 2px 10px; }}

QTableWidget {{ background: transparent; alternate-background-color: {p.surface};
                border: 1px solid transparent; gridline-color: transparent;
                selection-background-color: {p.accent_soft}; selection-color: {p.text}; }}
QTableWidget:focus {{ border: {focus}; }}
QTableWidget::item {{ border: none; padding: 4px 6px; }}
QTableWidget::item:selected {{ background-color: {p.accent_soft}; border-radius: 8px; }}
QHeaderView {{ background: transparent; qproperty-defaultSectionSize: 40px; }}
QHeaderView::section {{ background: transparent; color: {p.text_faint}; padding: 8px;
               border: none; border-bottom: 1px solid {p.border};
               font-size: {small}px; font-weight: 600; }}
QTableCornerButton::section {{ background: transparent; border: none; }}

QProgressBar {{ background: {p.surface}; border: 1px solid {p.border};
               border-radius: 7px; text-align: center; color: {p.text_dim}; }}
QProgressBar::chunk {{ background-color: {p.accent}; border-radius: 7px; }}
QProgressBar[quiet="true"]::chunk {{ background-color: {p.accent_bright}; }}

QSlider::groove:horizontal {{ height: 4px; background: {p.border};
               border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 13px; height: 13px; margin: -5px 0;
               border-radius: 7px; background: {p.text}; }}
QSlider::handle:horizontal:hover, QSlider::handle:horizontal:focus {{
               background: {p.accent_bright}; }}
QSlider::handle:horizontal:disabled {{ background: {p.text_faint}; }}
QSlider:focus {{ border: 1px solid {p.accent_bright}; border-radius: 4px; }}

QCheckBox {{ spacing: 8px; color: {p.text_dim}; }}
QCheckBox::indicator {{ width: 17px; height: 17px; border: 1px solid {p.border};
                        border-radius: 5px; background: {p.surface}; }}
QCheckBox::indicator:hover {{ border-color: {p.border_hover}; }}
QCheckBox::indicator:checked {{ background: {p.accent}; border-color: {p.accent}; }}
QCheckBox:focus {{ color: {p.text}; }}
QCheckBox:focus::indicator {{ border: {focus}; }}

QMenuBar {{ background: {p.bg}; color: {p.text_dim}; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: 6px; }}
QMenuBar::item:selected, QMenuBar::item:focus {{ background: {p.accent_soft};
               color: {p.text}; }}
QMenu {{ background: {p.surface_solid}; border: 1px solid {p.border};
         border-radius: {radius + 2}px; padding: 6px; }}
QMenu::item {{ padding: 7px 26px 7px 14px; border-radius: 7px; }}
QMenu::item:selected {{ background: {p.accent_soft}; color: {p.text}; }}
QMenu::separator {{ height: 1px; background: {p.border}; margin: 5px 8px; }}
QToolTip {{ background: {p.surface_solid}; color: {p.text}; border: 1px solid {p.border_hover};
            border-radius: 8px; padding: 6px 8px; }}
QStatusBar {{ background: transparent; color: {p.text_faint}; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background: transparent; height: 8px; }}
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p.scroll}; border-radius: 5px;
               min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {p.accent}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {p.scroll}; border-radius: 5px;
               min-width: 30px; }}
QScrollBar::handle:horizontal:hover {{ background: {p.accent}; }}

QGroupBox {{ border: 1px solid {p.border}; border-radius: {radius}px;
             margin-top: 14px; padding: 10px 10px 6px 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
             color: {p.text_dim}; font-weight: 600; }}
QLabel[role="heading"] {{ font-size: {small + 3}px; font-weight: 700; color: {p.text}; }}
QDialogButtonBox QPushButton {{ min-width: 84px; }}
"""


# The default sheet, kept as a module constant for the widgets that want one
# without going through apply_theme() first.
_QSS = _qss(DARK, DEFAULT_TEXT_SCALE)


def qss(theme: str = THEME_DARK, text_scale: float = DEFAULT_TEXT_SCALE) -> str:
    """The stylesheet for a theme and text scale (does not apply anything)."""
    return _qss(resolve(theme), clamp_scale(text_scale))


def apply_theme(theme: str = THEME_DARK,
                text_scale: float = DEFAULT_TEXT_SCALE) -> str:
    """Apply a theme to the running app and return its stylesheet.

    The palette is set application-wide (so native dialogs, menus and tooltips
    match) and the stylesheet is returned for the window to install; text sizes
    in it are scaled by ``text_scale``. Callers that only want the ring made of
    colours can ignore the return value.
    """
    palette = resolve(theme)
    scale = clamp_scale(text_scale)
    _current[0] = palette
    _current[1] = scale
    app = QApplication.instance()
    if app is None:
        return _qss(palette, scale)
    qt_palette = QPalette()
    qt_palette.setColor(QPalette.ColorRole.Window, QColor(palette.bg))
    qt_palette.setColor(QPalette.ColorRole.WindowText, QColor(palette.text))
    qt_palette.setColor(QPalette.ColorRole.Base, QColor(palette.panel))
    qt_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(palette.surface_solid))
    qt_palette.setColor(QPalette.ColorRole.Text, QColor(palette.text))
    qt_palette.setColor(QPalette.ColorRole.Button, QColor(palette.surface_solid))
    qt_palette.setColor(QPalette.ColorRole.ButtonText, QColor(palette.text))
    qt_palette.setColor(QPalette.ColorRole.Highlight, QColor(palette.button))
    qt_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(palette.button_text))
    qt_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(palette.surface_solid))
    qt_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(palette.text))
    qt_palette.setColor(QPalette.ColorRole.Link, QColor(palette.accent_bright))
    qt_palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(palette.text_faint))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.Text,
    ):
        qt_palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(palette.text_faint))
    app.setPalette(qt_palette)
    return _qss(palette, scale)


def _apply_theme(theme: str = THEME_DARK, text_scale: float = DEFAULT_TEXT_SCALE) -> None:
    """Backwards-compatible entry point used when the app starts."""
    apply_theme(theme, text_scale)
