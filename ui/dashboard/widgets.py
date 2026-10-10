"""The playback control strip shown under the caption preview."""

from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import (
    QColor,
    QPalette,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from captioner.player import Player

from .formatting import _fmt_clock


class PlaybackControls(QWidget):
    """Play/pause/stop + seek + volume for the selected job's source audio.

    Audio only — no video surface is created, so an mp4 or mkv plays as sound.
    The ``Player`` is injected (defaulting to a real QtMultimedia one) so tests
    can drive the widget with a fake backend.

    ``load()`` points the player at a job's media, ``set_spans()`` hands over the
    caption timing so the matching line can be highlighted in the preview while
    the audio plays.
    """

    def __init__(self, player: Player | None = None, parent=None):
        super().__init__(parent)
        self.player = player if player is not None else Player(self)
        self._job_id: int | None = None
        self._path: str | None = None  # the file actually handed to the player
        self._requested: str | None = None  # last requested path (may be missing)
        self._spans: list = []
        self._preview: QPlainTextEdit | None = None
        self._highlight_block: int | None = None
        self._seeking = False
        self._duration_ms = 0
        self._pending_seek: int | None = None  # seek requested before load finished

        self._build()
        self._set_enabled(False)
        self.player.position_ms_changed.connect(self._on_position)
        self.player.duration_ms_changed.connect(self._on_duration)
        self.player.playback_state_changed.connect(self._on_state)
        self.player.player_error.connect(self._on_error)

    #construction

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(6)

        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setFixedWidth(86)
        self.btn_play.setToolTip("Play or pause the selected file's audio (Space)")
        self.btn_play.clicked.connect(self.toggle_play)
        row.addWidget(self.btn_play)

        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setFixedWidth(40)
        self.btn_stop.setProperty("secondary", True)
        self.btn_stop.setToolTip("Stop playback and rewind to the beginning (Ctrl+.)")
        self.btn_stop.clicked.connect(self.stop)
        row.addWidget(self.btn_stop)

        self.seek_bar = QSlider(Qt.Orientation.Horizontal)
        self.seek_bar.setRange(0, 0)
        self.seek_bar.setToolTip("Click or drag to choose which part to listen to")
        self.seek_bar.sliderPressed.connect(self._begin_seek)
        self.seek_bar.sliderMoved.connect(self._on_dragged)
        self.seek_bar.sliderReleased.connect(self._end_seek)
        self.seek_bar.installEventFilter(self)
        row.addWidget(self.seek_bar, stretch=1)

        self.lbl_time = QLabel("0:00 / --:--")
        self.lbl_time.setMinimumWidth(90)
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.lbl_time)

        self.btn_mute = QPushButton("Mute")
        self.btn_mute.setCheckable(True)
        self.btn_mute.setProperty("secondary", True)
        self.btn_mute.setToolTip("Mute or unmute the preview audio (Ctrl+M)")
        self.btn_mute.toggled.connect(self._on_mute_toggled)
        row.addWidget(self.btn_mute)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(self.player.volume)
        self.volume_slider.setFixedWidth(96)
        self.volume_slider.setToolTip(f"Volume: {self.player.volume}%")
        self.volume_slider.valueChanged.connect(self._on_volume)
        row.addWidget(self.volume_slider)
        outer.addLayout(row)

        info = QHBoxLayout()
        info.setSpacing(8)
        self.lbl_status = QLabel("No file loaded — select a job to listen to it.")
        self.lbl_status.setStyleSheet("color: palette(mid);")
        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: #e08a9e;")
        self.lbl_error.setVisible(False)
        self.lbl_error.setWordWrap(True)
        info.addWidget(self.lbl_status, stretch=1)
        info.addWidget(self.lbl_error, stretch=1)
        outer.addLayout(info)

    #public API

    @property
    def has_source(self) -> bool:
        """True once a playable media path has been loaded."""
        return self._path is not None

    @property
    def source_path(self) -> str | None:
        return self._path

    def attach_preview(self, preview: QPlainTextEdit) -> None:
        """Give the widget the caption pane it should highlight lines in."""
        self._preview = preview

    def load(self, job_id: int | None, path: str | None, spans=()) -> None:
        """Point the player at *path* (a job's own media file) and reset the UI.

        Re-selecting the same job is a no-op, so the one-second queue refresh
        never interrupts playback or restarts the audio.
        """
        path = str(path) if path else ""
        playable = bool(path) and Path(path).exists()
        if job_id == self._job_id and path == self._requested and playable:
            self.set_spans(spans)
            return
        self._job_id = job_id
        self._requested = path
        self._path = path if playable else None
        self.set_spans(spans)
        self._clear_error()
        self._duration_ms = 0
        self._pending_seek = None
        self.seek_bar.setRange(0, 0)
        self.seek_bar.setValue(0)
        self._update_time(0)
        self.btn_mute.setChecked(False)
        if not path:
            self._set_enabled(False)
            self.lbl_status.setText("No media file for this job.")
            return
        if not playable:
            self._set_enabled(False)
            self.lbl_status.setText(f"Source file is missing: {Path(path).name}")
            return
        self._set_enabled(True)
        self.lbl_status.setText(f"{Path(path).name} — loading…")
        self.player.open(path)

    def set_spans(self, spans) -> None:
        """Set (start_ms, end_ms, block_number) triples for line highlighting."""
        self._spans = sorted((int(s), int(e), int(b)) for s, e, b in spans)
        self._highlight_block = None
        self._apply_highlight(None)

    #transport

    def toggle_play(self) -> None:
        if self._path is None:
            return
        if self.player.is_playing:
            self.player.pause()
        else:
            self.player.play()

    def stop(self) -> None:
        self.player.stop()
        self._pending_seek = None
        self.seek_bar.setValue(0)
        self._update_time(0)
        self._highlight_block = None
        self._apply_highlight(None)

    def seek_to(self, position_ms: int) -> None:
        """Jump to an absolute position (clamped by the player)."""
        self._request_seek(position_ms)

    def _request_seek(self, position_ms: int) -> None:
        """Seek now, or hold the position until the source has finished loading.

        Seeking a source whose duration is not known yet silently snaps back
        to 0, so a seek issued while the file is still loading is remembered
        and applied the moment ``durationChanged`` arrives.
        """
        position_ms = max(0, int(position_ms))
        if self._duration_ms <= 0:
            self._pending_seek = position_ms
            self.seek_bar.setValue(position_ms)
            self._update_time(position_ms)
            self._highlight(position_ms)
            return
        self._pending_seek = None
        self.player.seek(position_ms)
        self._on_position(position_ms)

    def seek_relative(self, delta_ms: int) -> None:
        """Nudge the playhead by *delta_ms* (used by the menu shortcuts)."""
        if self._path is None:
            return
        target = self.player.position_ms + int(delta_ms)
        if self._duration_ms:
            target = min(target, self._duration_ms)
        self.seek_to(max(0, target))

    def toggle_mute(self) -> None:
        self.btn_mute.setChecked(not self.btn_mute.isChecked())

    #player signal handlers

    def _on_position(self, ms: int) -> None:
        if self._pending_seek is not None:
            return  # stale position from before the queued seek took effect
        if not self._seeking:
            if self._duration_ms and self.seek_bar.maximum() != self._duration_ms:
                self.seek_bar.setRange(0, self._duration_ms)
            self.seek_bar.setValue(int(ms))  # QSlider clamps out-of-range values
        self._update_time(ms)
        self._highlight(ms)

    def _on_duration(self, ms: int) -> None:
        self._duration_ms = max(0, int(ms))
        self.seek_bar.setRange(0, self._duration_ms)
        if self._pending_seek is not None:
            pending, self._pending_seek = self._pending_seek, None
            landing = min(pending, self._duration_ms)
            self.player.seek(landing)
            self._on_position(landing)
        else:
            self._update_time(self.player.position_ms)
        if self._path:
            self.lbl_status.setText(f"{Path(self._path).name} — ready")

    def _on_state(self, state: str) -> None:
        self.btn_play.setText("⏸  Pause" if state == "playing" else "▶  Play")
        if self._path:
            self.lbl_status.setText(f"{Path(self._path).name} — {state}")

    def _on_error(self, message: str) -> None:
        self.lbl_error.setText(message)
        self.lbl_error.setVisible(True)
        self.lbl_status.setText("Playback problem — see the message on the right.")

    def _clear_error(self) -> None:
        self.lbl_error.clear()
        self.lbl_error.setVisible(False)

    #volume / mute

    def _on_volume(self, value: int) -> None:
        self.player.volume = int(value)
        self.volume_slider.setToolTip(f"Volume: {int(value)}%")

    def _on_mute_toggled(self, checked: bool) -> None:
        self.player.muted = bool(checked)
        self.btn_mute.setText("Muted" if checked else "Mute")

    def eventFilter(self, obj, event):
        """Click anywhere on the seek bar jumps there (a plain QSlider pages)."""
        if (
            obj is self.seek_bar
            and event.type() == QEvent.Type.MouseButtonPress
            and self._duration_ms > 0
        ):
            click_x = event.position().x()
            handle_center = (
                self.seek_bar.width() * self.seek_bar.value() / max(1, self.seek_bar.maximum())
            )
            if abs(click_x - handle_center) > 12:  # on the track, not the handle
                self.seek_to(int(click_x / max(1, self.seek_bar.width()) * self._duration_ms))
                return True  # a jump, not the start of a drag
        return super().eventFilter(obj, event)

    #seeking

    def _begin_seek(self) -> None:
        self._seeking = True

    def _on_dragged(self, value: int) -> None:
        self._update_time(value)

    def _end_seek(self) -> None:
        self._seeking = False
        self._request_seek(self.seek_bar.value())

    def _update_time(self, position_ms: int) -> None:
        # An unknown duration reads as "--:--" rather than a misleading 0:00.
        total = _fmt_clock(self._duration_ms) if self._duration_ms > 0 else "--:--"
        self.lbl_time.setText(f"{_fmt_clock(position_ms)} / {total}")

    #caption highlighting

    def _highlight(self, position_ms: int) -> None:
        if self._preview is None or not self._spans:
            return
        block_no = None
        for start, end, block in self._spans:
            if start <= position_ms < end:
                block_no = block
                break
        if block_no == self._highlight_block:
            return
        self._highlight_block = block_no
        self._apply_highlight(block_no)

    def _apply_highlight(self, block_no: int | None) -> None:
        preview = self._preview
        if preview is None:
            return
        if block_no is None:
            preview.setExtraSelections([])
            return
        block = preview.document().findBlockByNumber(block_no)
        if not block.isValid():
            preview.setExtraSelections([])
            return
        dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(96, 165, 250, 90) if dark else QColor(253, 224, 71, 140))
        selection = QTextEdit.ExtraSelection()
        selection.cursor = QTextCursor(block)
        selection.format = fmt
        preview.setExtraSelections([selection])
        self._scroll_to(block)

    def _scroll_to(self, block) -> None:
        """Keep the highlighted caption visible without stealing focus."""
        preview = self._preview
        try:
            rect = preview.blockBoundingGeometry(block).translated(preview.contentOffset())
        except Exception:  # pragma: no cover - geometry is display-only
            return
        scroll = preview.verticalScrollBar()
        viewport = preview.viewport().height()
        if rect.top() < scroll.value() or rect.bottom() > scroll.value() + viewport:
            scroll.setValue(int(rect.top() - viewport / 3))

    def _set_enabled(self, on: bool) -> None:
        for widget in (self.btn_play, self.btn_stop, self.seek_bar, self.btn_mute):
            widget.setEnabled(on)
