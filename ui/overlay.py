"""Live captions overlay (Phase 3): floating bar + background live worker.

The overlay is a frameless, always-on-top, translucent bar that shows the
stabilized live text: committed words dim, the changing tail bright — so the
eye can tell settled speech from in-flight guesses. Drag it anywhere; the
position is remembered.

`LiveWorker` runs the capture + sliding decode on a QThread so the UI never
blocks; both are small enough to test offscreen with fakes (no audio device
needed for the logic, one optional test exercises a real device).
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from captioner.live import (
    IDLE_STRIDE_SECONDS,
    STRIDE_SECONDS,
    LiveEngine,
    LiveState,
    MicCapture,
)

logger = logging.getLogger(__name__)

def _overlay_qss(font_px: int = 15) -> str:
    """Overlay stylesheet at a chosen caption size (settings-driven)."""
    return f"""
#OverlayRoot {{ background: rgba(13, 13, 13, 205); border-radius: 12px;
               border: 1px solid #1db954; }}
#OverlayCommitted {{ color: #7fbf95; font-size: {font_px}px; }}
#OverlayPartial  {{ color: #ffffff; font-size: {font_px}px; font-weight: 600; }}
#OverlayTranslation {{ color: #a5b4fc; font-size: {max(11, font_px - 1)}px; }}
#OverlayIdle     {{ color: #6f7378; font-size: {max(11, font_px - 1)}px; }}
#OverlayHeader   {{ color: #9aa0a6; font-size: {max(9, font_px - 4)}px; }}
"""


_OVERLAY_QSS = _overlay_qss()

# Appearance defaults; everything is user-tunable in the right-click menu.
DEFAULT_APPEARANCE = {"opacity": 80, "font_size": 15, "width": 760, "lock": False}
OPACITY_RANGE = (30, 100)
FONT_RANGE = (11, 24)
WIDTH_RANGE = (420, 1400)


class LiveWorker(QThread):
    """Capture + sliding-window decode off the UI thread.

    The engine and capture are injectable so tests can run the loop with
    fakes; by default a real `LiveEngine` + `MicCapture` are constructed.
    The loop honors ``engine.suggested_stride`` (short while speech plays,
    longer during silence), so a live session does not burn GPU on quiet.
    """

    state_ready = Signal(object)   # LiveState
    live_error = Signal(str)

    def __init__(self, device_id: str, engine: LiveEngine | None = None,
                 capture_factory: Callable[[str], MicCapture] | None = None,
                 model_size: str | None = None,
                 translate_to: str | None = None, parent=None):
        super().__init__(parent)
        self.device_id = device_id
        self._engine = engine
        self._model_size = model_size
        self._translate_to = translate_to
        self._capture_factory = capture_factory
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        stop = self._stop
        try:
            engine = self._engine or LiveEngine(
                model_size=self._model_size,
                translate_to=self._translate_to,
                # A dead translation backend must not take the captions down
                # with it: the overlay shows the note and keeps transcribing.
                on_translation_error=self.live_error.emit,
            )
            capture = (
                self._capture_factory(self.device_id)
                if self._capture_factory
                else MicCapture(self.device_id, on_error=self.live_error.emit)
            )
        except Exception as exc:
            self.live_error.emit(f"could not start capture: {exc}")
            return
        if hasattr(capture, "start"):
            capture.start()  # real capture threads itself; fakes may not
        try:
            # Calibrate the silence gate to this device's noise floor before
            # the first decode (2s of ambient). Test-injected fakes have no
            # calibrate method and skip this automatically.
            if hasattr(engine, "calibrate"):
                engine.calibrate(capture)
            while not stop.is_set():
                try:
                    state = engine.decode_once(capture)
                except Exception as exc:
                    logger.exception("live decode failed")
                    self.live_error.emit(f"decode failed: {exc}")
                    stop.wait(2.0)
                    continue
                self.state_ready.emit(state)
                stride = getattr(engine, "suggested_stride", STRIDE_SECONDS)
                stop.wait(min(max(stride, 0.5), IDLE_STRIDE_SECONDS))
        finally:
            if hasattr(capture, "stop"):
                capture.stop()
            if hasattr(engine, "close"):
                engine.close()  # stop the translation thread with the session


class TranscriptWindow(QWidget):
    """Read-back window: the whole live session so far, newest at the bottom.

    Opens alongside the overlay (which stays minimal); auto-scrolls unless
    the reader scrolled up. Updated from the same LiveState the overlay gets.
    """

    def __init__(self, persist_settings: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Captioner — live transcript")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.resize(520, 420)
        from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(10_000)  # bound memory on long sessions
        lay.addWidget(self.view)
        self._settings = QSettings("captioner", "transcript") if persist_settings else None
        if self._settings is not None and self._settings.contains("geometry"):
            self.restoreGeometry(self._settings.value("geometry"))
        self._rendered_text = ""

    def update_state(self, state: LiveState) -> None:
        """Mirror the session text; scroll only when already at the bottom.

        With live translation on, each finished sentence is shown as its
        source line followed by its translation, so the window reads as a
        bilingual transcript rather than two disconnected texts.
        """
        pairs = list(getattr(state, "translated_pairs", []) or [])
        if pairs:
            lines = [f"{source}\n    {translated}" for source, translated in pairs]
            text = "\n".join(lines)
            if state.partial:
                text = f"{text}\n{state.partial}"
        else:
            text = state.full_text
            if state.partial:
                text = f"{text} {state.partial}".strip()
        if text == self._rendered_text:
            return
        self._rendered_text = text
        bar = self.view.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        self.view.setPlainText(text)
        cursor = self.view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.view.setTextCursor(cursor)
        if at_bottom:
            bar.setValue(bar.maximum())

    def closeEvent(self, event) -> None:
        if self._settings is not None:
            self._settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


class LiveOverlay(QWidget):
    """Frameless always-on-top caption bar; drag to move, position remembered.

    Right-click offers copy/save of the live transcript and stopping the
    session — so the hotkey is not the only way to control it from afar.
    """

    stop_requested = Signal()

    def __init__(self, persist_settings: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("OverlayRoot")
        self.setStyleSheet(_OVERLAY_QSS)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._settings = QSettings("captioner", "overlay") if persist_settings else None
        self._drag_offset = None
        self._last_full_text = ""      # transcript so far, for copy
        self._last_full_words = []     # frozen (start, end, text) for SRT/VTT save
        self._appearance = dict(DEFAULT_APPEARANCE)
        self._load_appearance()
        self.apply_appearance()

        col = QVBoxLayout(self)
        col.setContentsMargins(14, 6, 14, 6)
        col.setSpacing(2)

        header = QHBoxLayout()
        header.setSpacing(8)
        self.lbl_device = QLabel("")
        self.lbl_device.setObjectName("OverlayHeader")
        header.addWidget(self.lbl_device)
        header.addStretch(1)
        self.lbl_level = QLabel("")
        self.lbl_level.setObjectName("OverlayHeader")
        self.lbl_level.setFixedWidth(80)
        self.lbl_level.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_level.setToolTip("Live input level — bars mean your mic/device hears sound")
        header.addWidget(self.lbl_level)
        col.addLayout(header)

        self.lbl_committed = QLabel("")
        self.lbl_committed.setObjectName("OverlayCommitted")
        self.lbl_committed.setWordWrap(True)
        self.lbl_committed.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.lbl_partial = QLabel("")
        self.lbl_partial.setObjectName("OverlayPartial")
        self.lbl_partial.setWordWrap(True)
        self.lbl_partial.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        # The translated line, shown only while live translation is on and a
        # sentence has come back from the translator.
        self.lbl_translation = QLabel("")
        self.lbl_translation.setObjectName("OverlayTranslation")
        self.lbl_translation.setWordWrap(True)
        self.lbl_translation.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.lbl_translation.setVisible(False)
        col.addWidget(self.lbl_committed)
        col.addWidget(self.lbl_partial)
        col.addWidget(self.lbl_translation)
        self.show_idle()

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self.resize(self._appearance["width"], 110)
        self._restore_position()
        self.transcript_window: TranscriptWindow | None = None

    # -- appearance (user customization)
    def _load_appearance(self) -> None:
        if self._settings is None:
            return
        s = self._settings
        try:
            self._appearance["opacity"] = int(s.value("opacity", DEFAULT_APPEARANCE["opacity"]))
            self._appearance["font_size"] = int(s.value("font_size", DEFAULT_APPEARANCE["font_size"]))
            self._appearance["width"] = int(s.value("width", DEFAULT_APPEARANCE["width"]))
            self._appearance["lock"] = s.value("lock", "false") in (True, "true", "1")
        except (TypeError, ValueError):
            pass  # corrupt entry: fall back to defaults

    def apply_appearance(self) -> None:
        """Push the current appearance settings onto the widget."""
        a = self._appearance
        op = max(OPACITY_RANGE[0], min(OPACITY_RANGE[1], a["opacity"])) / 100.0
        self.setWindowOpacity(op)
        self.setStyleSheet(_overlay_qss(
            max(FONT_RANGE[0], min(FONT_RANGE[1], a["font_size"]))
        ))
        w = max(WIDTH_RANGE[0], min(WIDTH_RANGE[1], a["width"]))
        self.resize(w, self.height())

    def save_appearance(self) -> None:
        if self._settings is None:
            return
        s = self._settings
        s.setValue("opacity", self._appearance["opacity"])
        s.setValue("font_size", self._appearance["font_size"])
        s.setValue("width", self._appearance["width"])
        s.setValue("lock", self._appearance["lock"])

    def open_settings_dialog(self) -> None:
        """Small non-modal-looking dialog: opacity, text size, width, lock."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Caption appearance")
        form = QFormLayout(dlg)

        op = QSlider(Qt.Orientation.Horizontal)
        op.setRange(*OPACITY_RANGE)
        op.setValue(self._appearance["opacity"])
        op_lbl = QLabel(f"{op.value()}%")
        op.valueChanged.connect(lambda v: op_lbl.setText(f"{v}%"))
        form.addRow("Overlay opacity", op)
        form.addRow("", op_lbl)

        fs = QSlider(Qt.Orientation.Horizontal)
        fs.setRange(*FONT_RANGE)
        fs.setValue(self._appearance["font_size"])
        fs_lbl = QLabel(f"{fs.value()} px")
        fs.valueChanged.connect(lambda v: fs_lbl.setText(f"{v} px"))
        form.addRow("Caption text size", fs)
        form.addRow("", fs_lbl)

        wd = QSpinBox()
        wd.setRange(*WIDTH_RANGE)
        wd.setValue(self._appearance["width"])
        wd.setSuffix(" px")
        form.addRow("Caption bar width", wd)

        lock = QCheckBox("Lock position (no dragging)")
        lock.setChecked(bool(self._appearance["lock"]))
        form.addRow(lock)

        def _live_preview() -> None:
            self._appearance.update(
                opacity=op.value(), font_size=fs.value(), width=wd.value(),
                lock=lock.isChecked(),
            )
            self.apply_appearance()

        for wdg in (op, fs, wd, lock):
            if isinstance(wdg, QCheckBox):
                wdg.toggled.connect(_live_preview)
            else:
                wdg.valueChanged.connect(_live_preview)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(
            lambda: (self._load_appearance(), self.apply_appearance(), dlg.reject())
        )
        form.addRow(buttons)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            _live_preview()
            self.save_appearance()

    # -- rendering
    def set_device_name(self, name: str) -> None:
        self.lbl_device.setText(name)

    def _render_level(self, level: float) -> str:
        """Eight-segment input meter: ▁▂▃ bars light up with the input level."""
        bars = 8
        lit = max(0, min(bars, int(round(level * bars * 2.5))))
        return "●" * lit + "○" * (bars - lit)

    def show_idle(self) -> None:
        self.lbl_committed.setText("")
        self.lbl_partial.setObjectName("OverlayIdle")
        self._repolish_partial()
        self.lbl_partial.setText("live captions — listening…")
        self.lbl_level.setText("")
        self.lbl_translation.clear()
        self.lbl_translation.setVisible(False)

    def update_state(self, state: LiveState) -> None:
        self.lbl_committed.setText(state.committed)
        self.lbl_partial.setObjectName("OverlayPartial")
        self._repolish_partial()
        self.lbl_partial.setText(state.partial or ("…" if state.committed else "listening…"))
        self.lbl_level.setText(self._render_level(state.level))
        translation = getattr(state, "translation", "")
        self.lbl_translation.setText(translation)
        self.lbl_translation.setVisible(bool(translation))
        if state.full_text:
            self._last_full_text = state.full_text
        if state.full_words:
            self._last_full_words = list(state.full_words)
        if self.transcript_window is not None:
            self.transcript_window.update_state(state)

    def show_error(self, message: str) -> None:
        self.lbl_partial.setObjectName("OverlayIdle")
        self._repolish_partial()
        self.lbl_partial.setText(f"⚠ {message}")

    # -- right-click menu
    def _show_menu(self, pos) -> None:
        menu = QMenu(self)
        transcript = (self._last_full_text or self.lbl_committed.text()).strip()
        act_copy = menu.addAction("⧉ Copy transcript")
        act_copy.setEnabled(bool(transcript))
        act_save = menu.addAction("💾 Save transcript…")
        act_save.setEnabled(bool(transcript))
        menu.addSeparator()
        act_settings = menu.addAction("⚙ Caption appearance…")
        act_transcript = menu.addAction(
            "⧉ Hide transcript window" if self.transcript_window is not None
            else "⧉ Show transcript window"
        )
        menu.addSeparator()
        act_stop = menu.addAction("⏹ Stop live captions")
        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen is act_copy:
            QApplication.clipboard().setText(transcript)
        elif chosen is act_save:
            words = list(self._last_full_words)
            path, _ = QFileDialog.getSaveFileName(
                self, "Save live transcript", "live_transcript.srt",
                "SubRip captions (*.srt);;WebVTT captions (*.vtt);;"
                "Plain text (*.txt);;All files (*)",
            )
            if path and words:
                from captioner.live import save_transcript

                try:
                    save_transcript(words, path)
                except OSError as exc:
                    self.show_error(f"could not save transcript: {exc}")
        elif chosen is act_settings:
            self.open_settings_dialog()
        elif chosen is act_transcript:
            self.toggle_transcript_window()
        elif chosen is act_stop:
            self.stop_requested.emit()

    def toggle_transcript_window(self) -> None:
        """Open/close the full-session read-back window."""
        if self.transcript_window is not None:
            self.transcript_window.close()
            self.transcript_window = None
            return
        self.transcript_window = TranscriptWindow()
        self.transcript_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.transcript_window.show()

    def close_transcript_window(self) -> None:
        if self.transcript_window is not None:
            self.transcript_window.close()
            self.transcript_window = None

    def _repolish_partial(self) -> None:
        # objectName changes only take effect on the styles after a polish cycle
        style = self.lbl_partial.style()
        style.unpolish(self.lbl_partial)
        style.polish(self.lbl_partial)

    # -- dragging
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._appearance["lock"]:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        self._save_position()

    # -- position persistence
    def _restore_position(self) -> None:
        if self._settings is None:
            return
        point = self._settings.value("position")
        if point is not None:
            self.move(point)
        else:
            screen = QGuiApplication.primaryScreen().availableGeometry()
            self.move(
                screen.center().x() - self.width() // 2,
                screen.bottom() - self.height() - 80,
            )

    def _save_position(self) -> None:
        if self._settings is not None:
            self._settings.setValue("position", self.pos())
