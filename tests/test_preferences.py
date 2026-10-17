"""Tests for rememberable preferences, themes and the accessibility wiring.

These run the real window offscreen, with preferences backed by a temporary INI
file instead of the user's registry, so "it is remembered" is checked the way a
user experiences it: start the app twice, see the same choices.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from captioner import config, db
from ui.dashboard import theme
from ui.dashboard.preferences import Preferences, PreferencesDialog, scale_label


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def settings(tmp_path):
    """A real QSettings, in a temporary file the test owns."""
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    yield store
    store.clear()
    store.sync()


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "prefs.db")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    connection = db.connect()
    yield connection
    connection.close()


#the store

def test_defaults_without_a_settings_file():
    prefs = Preferences.load(None)
    assert prefs.theme == theme.THEME_DARK
    assert prefs.text_scale == theme.DEFAULT_TEXT_SCALE
    assert prefs.quiet is False
    assert prefs.language == "" and prefs.translate_to == ""
    assert prefs.notify is True
    assert prefs.save() is False  # nothing to write to, and no crash


def test_values_survive_a_reload(settings):
    prefs = Preferences.load(settings)
    prefs.theme = theme.THEME_CONTRAST
    prefs.text_scale = 1.3
    prefs.quiet = True
    prefs.language = "hi"
    prefs.translate_to = "bn"
    prefs.align = True
    prefs.autostart = True
    assert prefs.save() is True

    reloaded = Preferences.load(settings)
    assert reloaded.theme == theme.THEME_CONTRAST
    assert reloaded.text_scale == pytest.approx(1.3)
    assert reloaded.quiet and reloaded.align and reloaded.autostart
    assert (reloaded.language, reloaded.translate_to) == ("hi", "bn")


def test_a_hand_edited_file_cannot_break_the_window(settings):
    settings.setValue("preferences/theme", "neon")
    settings.setValue("preferences/text_scale", "huge")
    settings.setValue("preferences/quiet", "maybe")
    settings.setValue("preferences/model", "colossus")
    settings.sync()

    prefs = Preferences.load(settings, model_default="medium")

    assert prefs.theme == theme.THEME_DARK
    assert prefs.text_scale == theme.DEFAULT_TEXT_SCALE
    assert prefs.quiet is False
    assert prefs.model == "medium"


def test_the_text_scale_has_limits():
    assert theme.clamp_scale(99) == theme.MAX_TEXT_SCALE
    assert theme.clamp_scale(0.1) == theme.MIN_TEXT_SCALE
    assert theme.clamp_scale("nonsense") == theme.DEFAULT_TEXT_SCALE
    assert theme.clamp_scale(float("nan")) == theme.DEFAULT_TEXT_SCALE

    prefs = Preferences(None, text_scale=theme.MAX_TEXT_SCALE)
    assert prefs.scale_text(0.5) == scale_label(theme.MAX_TEXT_SCALE)
    assert prefs.scale_text(-0.1) == scale_label(theme.MAX_TEXT_SCALE - 0.1)
    prefs.reset_text_scale()
    assert prefs.text_scale == theme.DEFAULT_TEXT_SCALE


def test_contrast_toggles_between_high_contrast_and_dark():
    prefs = Preferences(None)
    assert prefs.toggle_contrast() == theme.THEME_NAMES[theme.THEME_CONTRAST]
    assert prefs.theme == theme.THEME_CONTRAST
    prefs.toggle_contrast()
    assert prefs.theme == theme.THEME_DARK


def test_a_dialog_can_be_cancelled(make_win, settings):
    # The dialog is parented to a real window so it is destroyed with it; a
    # parentless top-level dialog outliving the test is a Qt crash on exit.
    win = make_win(settings)
    prefs = win.prefs
    original = prefs.text_scale
    dialog = PreferencesDialog(prefs, win)
    dialog.size_slider.setValue(round(theme.MAX_TEXT_SCALE * 100))
    assert prefs.text_scale == pytest.approx(theme.MAX_TEXT_SCALE)  # previewed live
    dialog.reject()
    assert prefs.text_scale == pytest.approx(original)  # restored
    # Cancel stores nothing new: the file still says the original, not the
    # maximum that was previewed a moment ago.
    stored = float(settings.value("preferences/text_scale", theme.DEFAULT_TEXT_SCALE))
    assert stored == pytest.approx(original)
    assert settings.value("preferences/theme") is None  # the dialog wrote nothing


#themes

def test_every_theme_builds_a_sheet():
    for name in (theme.THEME_DARK, theme.THEME_LIGHT, theme.THEME_CONTRAST):
        sheet = theme.qss(name, 1.0)
        assert "QPushButton" in sheet and "QTableWidget" in sheet
        assert theme.resolve(name).name == name


def test_system_follows_the_platform_but_never_fails():
    assert theme.resolve(theme.THEME_SYSTEM).name in (
        theme.THEME_DARK, theme.THEME_LIGHT
    )
    assert theme.resolve("nonsense").name == theme.THEME_DARK


def test_text_scale_changes_the_sheet_and_px():
    small = theme.qss(theme.THEME_DARK, 1.0)
    large = theme.qss(theme.THEME_DARK, 1.5)
    assert "font-size: 13px" in small
    assert "font-size: 20px" in large  # 13 * 1.5, rounded
    assert small != large


def test_high_contrast_is_opaque_and_focus_is_visible_in_every_theme():
    sheet = theme.qss(theme.THEME_CONTRAST, 1.0)
    assert "rgba(255, 255, 255, 0.08)" not in sheet  # no translucent borders
    for name in (theme.THEME_DARK, theme.THEME_LIGHT, theme.THEME_CONTRAST):
        text = theme.qss(name, 1.0)
        assert "QPushButton:focus" in text
        assert "QComboBox:focus" in text
        assert "QPlainTextEdit:focus" in text


def _luminance(color: str) -> float:
    """Relative luminance of ``#rrggbb`` (WCAG 2.1, sRGB)."""
    value = color.lstrip("#")
    channels = []
    for index in (0, 2, 4):
        raw = int(value[index:index + 2], 16) / 255
        channels.append(raw / 12.92 if raw <= 0.04045 else ((raw + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(foreground: str, background: str) -> float:
    """WCAG contrast ratio between two opaque colours."""
    light, dark = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def test_the_contrast_helper_matches_the_wcag_examples():
    assert _contrast("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)
    assert _contrast("#ffffff", "#ffffff") == pytest.approx(1.0, abs=0.01)
    assert _contrast("#767676", "#ffffff") == pytest.approx(4.54, abs=0.05)


def test_every_theme_keeps_its_text_legible():
    """Body text, hints and button labels clear 4.5:1 in all three themes.

    High contrast is held to 7:1, since being unreadable is the one thing it
    exists to fix.
    """
    failures = []
    for palette in (theme.DARK, theme.LIGHT, theme.CONTRAST):
        needed = 7.0 if palette.name == theme.THEME_CONTRAST else 4.5
        pairs = {
            "body text": (palette.text, palette.bg),
            "body text on a panel": (palette.text, palette.surface_solid),
            "dim text": (palette.text_dim, palette.bg),
            "faint text (hints, timestamps)": (palette.text_faint, palette.bg),
            "faint text on a panel": (palette.text_faint, palette.surface_solid),
            "button label": (palette.button_text, palette.button),
            "button label on hover": (palette.button_text, palette.button_hover),
            "button label when pressed": (palette.button_text, palette.button_pressed),
        }
        for status, (pill_text, pill_bg) in palette.pills.items():
            pairs[f"{status} pill"] = (pill_text, pill_bg)
        for what, (foreground, background) in pairs.items():
            ratio = _contrast(foreground, background)
            if ratio < needed:
                failures.append(f"{palette.name}: {what} is {ratio:.2f}:1")
    assert not failures, failures


def test_applying_a_theme_updates_the_shared_palette(qapp):  # noqa: ARG001 - Qt needs it

    theme.apply_theme(theme.THEME_LIGHT, 1.2)
    try:
        assert theme.current_palette().name == theme.THEME_LIGHT
        assert theme.current_text_scale() == pytest.approx(1.2)
        assert theme.is_dark() is False
        assert theme.status_colors()["done"] == theme.LIGHT.pills["done"][0]
        assert theme.px(10) == "12px"  # 10 * 1.2
        assert theme.faint_style() == f"color: {theme.LIGHT.text_faint};"
    finally:
        theme.apply_theme(theme.THEME_DARK, theme.DEFAULT_TEXT_SCALE)


#the window

@pytest.fixture()
def make_win(qapp, conn):
    created = []

    def factory(settings=None):
        from ui.dashboard import Dashboard

        win = Dashboard(conn=conn, persist_settings=False, settings=settings)
        win.timer.stop()
        created.append(win)
        return win

    yield factory
    for win in created:
        win.timer.stop()
        win.close()


def test_options_are_remembered_between_windows(make_win, settings):
    """Close the app, open it again: the choices are still there."""
    win = make_win(settings)
    win.model_combo.setCurrentText("small.en")
    win.language_combo.setCurrentIndex(win.language_combo.findData("hi"))
    win.translate_combo.setCurrentIndex(win.translate_combo.findData("bn"))
    win.align_check.setChecked(True)
    win._persist_queue_defaults()
    # Not closed here on purpose: the window owns the test's connection, and
    # closing it is what the fixture teardown does at the end.

    second = make_win(settings)
    assert second.model_combo.currentText() == "small.en"
    assert second.language_combo.currentData() == "hi"
    assert second.translate_combo.currentData() == "bn"
    assert second.align_check.isChecked() is True
    assert second._queue_options()["source_lang"] == "hi"


def test_saved_appearance_is_applied_at_startup(make_win, settings):
    prefs = Preferences.load(settings)
    prefs.theme = theme.THEME_CONTRAST
    prefs.text_scale = 1.5
    prefs.quiet = True
    prefs.save()

    win = make_win(settings)

    assert win.prefs.theme == theme.THEME_CONTRAST
    assert theme.current_text_scale() == pytest.approx(1.5)
    assert theme.CONTRAST.accent_bright in win.styleSheet()
    assert "font-size: 20px" in win.styleSheet()
    assert win.act_contrast.isChecked() is True


def test_text_size_shortcuts_scale_and_repaint(make_win):
    win = make_win()
    before = win.prefs.text_scale
    win.nudge_text_size(0.1)
    assert win.prefs.text_scale == pytest.approx(before + 0.1)
    assert f"font-size: {round(13 * win.prefs.text_scale)}px" in win.styleSheet()
    win.reset_text_size()
    assert win.prefs.text_scale == theme.DEFAULT_TEXT_SCALE


def test_quiet_mode_silences_the_beep_but_keeps_the_note(make_win, monkeypatch):
    win = make_win()
    beeps, alerts = [], []
    monkeypatch.setattr(QApplication, "beep", lambda *a: beeps.append(1))
    monkeypatch.setattr(QApplication, "alert", lambda *a, **k: alerts.append(1))

    win.notify_check.setChecked(True)
    win.prefs.quiet = True
    win._pending_before = 2
    win._maybe_notify([{"status": "done"}])
    assert beeps == [] and alerts == []
    assert win.last_notification is not None  # the status bar still says so

    win._pending_before = 1
    win.prefs.quiet = False
    win._maybe_notify([{"status": "done"}])
    assert len(beeps) == 1 and len(alerts) == 1


def test_the_checkbox_alone_decides_whether_anything_is_said(make_win, monkeypatch):
    win = make_win()
    beeps = []
    monkeypatch.setattr(QApplication, "beep", lambda *a: beeps.append(1))
    monkeypatch.setattr(QApplication, "alert", lambda *a, **k: None)

    win.notify_check.setChecked(False)
    win._pending_before = 1
    win._maybe_notify([{"status": "done"}])

    assert beeps == [] and win.last_notification is None


def test_main_controls_announce_themselves(make_win):
    win = make_win()
    for widget in (win.btn_add, win.btn_worker, win.btn_options, win.table,
                   win.preview, win.btn_translate, win.language_combo,
                   win.translate_combo, win.model_combo, win.diarize_check,
                   win.align_check, win.notify_check, win.progress):
        assert widget.accessibleName(), widget
    assert win.btn_add.accessibleDescription()
    assert win.table.accessibleDescription()


def test_the_shortcut_dialog_lists_the_new_keys(make_win):
    keys = dict(make_win().SHORTCUTS)
    assert keys["Ctrl+,"] == "Appearance and accessibility"
    assert keys["Ctrl+Shift+H"] == "High contrast theme"
    assert keys["Ctrl+= / Ctrl+-"] == "Larger / smaller text"
    assert keys["Ctrl+0"] == "Normal text size"
    assert keys["F1"] == "This list"


def test_translating_without_a_selection_says_so(make_win):
    win = make_win()
    win.table.clearSelection()
    win._preview_job_id = None
    assert win.translate_selected() is False
    assert "select a finished job" in win.statusBar().currentMessage()
    assert win.btn_translate.isEnabled() is False
