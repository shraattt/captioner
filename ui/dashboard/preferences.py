"""Remembered preferences: appearance, accessibility, and queue defaults.

Everything the user changes and would be annoyed to set again lives here,
behind one object rather than scattered ``QSettings`` reads:

* **appearance** - theme (dark / light / high contrast / follow Windows),
  text size, and the live overlay's caption size; * **accessibility** - quiet mode (no beep, no flashing taskbar entry);

The live overlay's appearance (opacity, caption size, bar width) stays with the
overlay, in its own settings: it is edited from the caption bar's own menu, next
to the thing it changes.
* **queue defaults** - the model, spoken language, translation target and the
  option checkboxes the next file should be queued with, plus whether the
  worker starts with the window.

Values are read once at startup with defensive conversions: a settings file is
user-editable text, so a hand-edited "maybe" must not crash the window. Writing
is wholesale, immediate, and never raises - a preference that cannot be saved
is not worth losing a session over.
"""

import logging
from dataclasses import dataclass, field

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from .constants import MODEL_CHOICES
from .theme import (
    DEFAULT_TEXT_SCALE,
    MAX_TEXT_SCALE,
    MIN_TEXT_SCALE,
    THEME_CONTRAST,
    THEME_DARK,
    THEME_NAMES,
    THEMES,
    clamp_scale,
    faint_style,
    px,
)

logger = logging.getLogger(__name__)

GROUP = "preferences"


def _as_bool(value, default: bool) -> bool:
    """Coerce a stored setting to a bool (settings files are hand-editable)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _as_text(value, default: str = "") -> str:
    return default if value is None else str(value)


def scale_label(scale: float) -> str:
    """``1.25`` -> ``"125%"`` - what the text-size control shows."""
    return f"{round(scale * 100)}%"


@dataclass
class Preferences:
    """Live preference values, backed by ``QSettings`` when one is available.

    ``settings is None`` (the dashboard's test mode) keeps everything in memory:
    defaults in, no writes out, so a test run never touches the real registry.
    """

    settings: QSettings | None = None
    theme: str = THEME_DARK
    text_scale: float = DEFAULT_TEXT_SCALE
    quiet: bool = False
    model: str = "medium"
    language: str = ""            # "" = detect the spoken language
    translate_to: str = ""        # "" = do not translate
    diarize: bool = False
    align: bool = False
    watch: bool = False
    notify: bool = True
    autostart: bool = False
    _loaded: dict = field(default_factory=dict, repr=False)

    # -- load / save
    @classmethod
    def load(cls, settings: QSettings | None, model_default: str = "medium") -> "Preferences":
        prefs = cls(settings=settings, model=model_default)
        if settings is None:
            prefs.snapshot()
            return prefs
        value = settings.value
        prefs.theme = _as_text(value(f"{GROUP}/theme", THEME_DARK)).strip().lower()
        if prefs.theme not in THEMES:
            prefs.theme = THEME_DARK
        prefs.text_scale = clamp_scale(value(f"{GROUP}/text_scale", DEFAULT_TEXT_SCALE))
        prefs.quiet = _as_bool(value(f"{GROUP}/quiet"), False)
        prefs.model = _as_text(value(f"{GROUP}/model", model_default), model_default)
        if prefs.model not in MODEL_CHOICES:
            prefs.model = model_default if model_default in MODEL_CHOICES else "medium"
        prefs.language = _as_text(value(f"{GROUP}/language", ""))
        prefs.translate_to = _as_text(value(f"{GROUP}/translate_to", ""))
        prefs.diarize = _as_bool(value(f"{GROUP}/diarize"), False)
        prefs.align = _as_bool(value(f"{GROUP}/align"), False)
        prefs.watch = _as_bool(value(f"{GROUP}/watch"), False)
        prefs.notify = _as_bool(value(f"{GROUP}/notify"), True)
        prefs.autostart = _as_bool(value(f"{GROUP}/autostart"), False)
        prefs.snapshot()
        return prefs

    def snapshot(self) -> None:
        """Remember the current values, for :meth:`restore`."""
        self._loaded = self.as_dict()

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in (
            "theme", "text_scale", "quiet", "model", "language", "translate_to",
            "diarize", "align", "watch", "notify", "autostart",
        )}

    def restore(self, values: dict | None = None) -> None:
        """Put back the values from the last :meth:`snapshot` (or ``values``)."""
        source = self._loaded if values is None else values
        for name, value in source.items():
            setattr(self, name, value)

    def save(self) -> bool:
        """Write every value. Returns False when there is nowhere to write."""
        if self.settings is None:
            return False
        try:
            for name, value in self.as_dict().items():
                self.settings.setValue(f"{GROUP}/{name}", value)
            self.settings.sync()
        except Exception as exc:  # noqa: BLE001 - a lost preference is not fatal
            logger.warning("could not save preferences: %s", exc)
            return False
        self.snapshot()
        return True

    # -- appearance helpers, used by the shortcuts and the dialog
    def scale_text(self, delta: float) -> str:
        """Nudge the text scale by ``delta``; returns the new scale as text."""
        self.text_scale = clamp_scale(self.text_scale + delta)
        return scale_label(self.text_scale)

    def reset_text_scale(self) -> None:
        self.text_scale = DEFAULT_TEXT_SCALE

    def toggle_contrast(self) -> str:
        """Flip between high contrast and dark; returns the resulting name."""
        self.theme = THEME_DARK if self.theme == THEME_CONTRAST else THEME_CONTRAST
        return THEME_NAMES.get(self.theme, self.theme)


class PreferencesDialog(QDialog):
    """Appearance and accessibility, previewed live while you change it.

    The dialog edits the caller's :class:`Preferences` in place so the window
    can repaint on every change; Cancel puts the values from when it opened back
    and repaints, and Save writes them to disk.
    """

    def __init__(self, prefs: Preferences, parent=None):
        super().__init__(parent)
        self.prefs = prefs
        self._original = prefs.as_dict()
        self.setWindowTitle("Preferences")
        self.setAccessibleName("Preferences")
        self.setMinimumWidth(430)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 14)
        outer.setSpacing(12)

        #appearance
        look = QGroupBox("&Appearance")
        look_form = QFormLayout(look)
        look_form.setSpacing(10)

        self.theme_combo = QComboBox()
        for key in THEMES:
            self.theme_combo.addItem(THEME_NAMES[key], key)
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(prefs.theme)))
        self.theme_combo.setAccessibleName("Theme")
        self.theme_combo.setToolTip(
            "High contrast makes every surface opaque, brightens the text and\n"
            "adds a yellow focus ring - for low vision, or a bright room."
        )
        self.theme_combo.currentIndexChanged.connect(self.preview)
        look_form.addRow("&Theme", self.theme_combo)

        size_row = QHBoxLayout()
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(round(MIN_TEXT_SCALE * 100), round(MAX_TEXT_SCALE * 100))
        self.size_slider.setSingleStep(5)
        self.size_slider.setPageStep(10)
        self.size_slider.setValue(round(prefs.text_scale * 100))
        self.size_slider.setAccessibleName("Text size percentage")
        self.size_slider.setToolTip("Scales every label, menu and button (also Ctrl+= and Ctrl+-)")
        self.size_label = QLabel(scale_label(prefs.text_scale))
        self.size_label.setMinimumWidth(52)
        self.size_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.size_slider.valueChanged.connect(self.preview)
        size_row.addWidget(self.size_slider, stretch=1)
        size_row.addWidget(self.size_label)
        reset_size = QPushButton("Reset")
        reset_size.setProperty("secondary", True)
        reset_size.setAccessibleName("Reset text size to 100 percent")
        reset_size.setToolTip("Back to 100%")
        reset_size.clicked.connect(
            lambda: self.size_slider.setValue(round(DEFAULT_TEXT_SCALE * 100))
        )
        size_row.addWidget(reset_size)
        look_form.addRow("&Text size", size_row)
        outer.addWidget(look)

        #behaviour
        behaviour = QGroupBox("&Notifications and the queue")
        behaviour_col = QVBoxLayout(behaviour)
        behaviour_col.setSpacing(8)
        self.quiet_check = QCheckBox("&Quiet mode: no beep, no flashing taskbar")
        self.quiet_check.setChecked(prefs.quiet)
        self.quiet_check.setToolTip(
            "The queue still finishes, you are just not interrupted about it;\n"
            "the status bar reports it as usual."
        )
        self.quiet_check.toggled.connect(self.preview)
        behaviour_col.addWidget(self.quiet_check)
        self.autostart_check = QCheckBox("&Start the worker when this window opens")
        self.autostart_check.setChecked(prefs.autostart)
        self.autostart_check.setToolTip("Begin processing the queue as soon as the app starts")
        self.autostart_check.toggled.connect(self.preview)
        behaviour_col.addWidget(self.autostart_check)
        hint = QLabel(
            "Model, spoken language and the per-file options live in ⚙ Options; "
            "they are remembered too."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(faint_style(f"font-size: {px(11)};"))
        behaviour_col.addWidget(hint)
        outer.addWidget(behaviour)

        self.status = QLabel(self.summary())
        self.status.setWordWrap(True)
        self.status.setStyleSheet(faint_style(f"font-size: {px(11)};"))
        outer.addWidget(self.status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Save")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        outer.addWidget(buttons)

    # -- values
    def preview(self) -> None:
        """Copy the widgets into the live preferences; the parent repaints."""
        self.prefs.theme = self.theme_combo.currentData() or THEME_DARK
        self.prefs.text_scale = clamp_scale(self.size_slider.value() / 100.0)
        self.prefs.quiet = self.quiet_check.isChecked()
        self.prefs.autostart = self.autostart_check.isChecked()
        self.size_label.setText(scale_label(self.prefs.text_scale))
        self.status.setText(self.summary())
        parent = self.parent()
        if hasattr(parent, "apply_preferences"):
            parent.apply_preferences()

    def summary(self) -> str:
        """One line describing what the current settings do."""
        parts = [
            THEME_NAMES.get(self.prefs.theme, self.prefs.theme),
            f"{scale_label(self.prefs.text_scale)} text",
        ]
        if self.prefs.quiet:
            parts.append("quiet mode")
        if self.prefs.autostart:
            parts.append("worker starts with the window")
        return " · ".join(parts)

    def accept(self) -> None:
        self.preview()
        self.prefs.save()
        super().accept()

    def reject(self) -> None:
        self.prefs.restore(self._original)
        parent = self.parent()
        if hasattr(parent, "apply_preferences"):
            parent.apply_preferences()
        super().reject()
