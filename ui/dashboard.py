"""Dashboard window — Phase 1 batch UI.

Layout: menu bar, toolbar (model choice, add files, worker start/stop, retry,
remove, open output), a queue summary bar, a splitter with the job table above
and the caption preview below, and a status bar showing where the app keeps
its files.

The table re-reads the SQLite queue once a second (cheap, and WAL mode makes
concurrent read/write safe), so the worker can run in a background QThread or
as a separate console process — the UI does not care which.

Run standalone:
    python ui/dashboard.py                 # normal window
    python ui/dashboard.py --self-test     # opens and closes itself (sanity check)
"""
import html
import logging
import os
import sqlite3
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# Direct execution (`python ui/dashboard.py`) puts ui/ on sys.path, not the
# project root — add the root so `captioner` imports no matter how we start.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PySide6.QtCore import QEvent, QSettings, Qt, QThread, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QFontDatabase,
    QKeySequence,
    QPalette,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidgetAction,
    QWidget,
)

from captioner import config, db, editor, models, srt, worker
from captioner.player import Player
from captioner.translate import LANGUAGES

logger = logging.getLogger(__name__)

APP_NAME = "Captioner"
APP_VERSION = "0.6.0"
MEDIA_FILTER = (
    "Media files (*.mp4 *.mkv *.mov *.avi *.webm *.mp3 *.wav *.m4a *.flac *.aac *.ogg *.opus);;"
    "All files (*)"
)
# Extensions a dropped/picked FOLDER is scanned for (recursively).
MEDIA_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".m4v", ".mpg",
    ".mpeg", ".ts", ".3gp", ".mp3", ".wav", ".m4a", ".flac", ".aac",
    ".ogg", ".opus", ".wma", ".aiff", ".aif",
}
# Smallest → most accurate. distil-large-v3 = large accuracy at ~2.4× less
# waiting than medium (measured in scripts/model_sweep.py); large-v3 for when
# only the best will do. The batch default comes from config.
MODEL_CHOICES = ("tiny.en", "base.en", "small.en", "medium", "distil-large-v3", "large-v3")

COLUMNS = ("#", "File", "Status", "Model", "Progress", "Created", "Duration", "Details")

_STATUS_COLORS_LIGHT = {
    "queued": "#b9bdc7",
    "running": "#d6c9ff",
    "done": "#c9f0dd",
    "error": "#f2b8c6",
    "canceled": "#8b8b95",
}
_STATUS_COLORS_DARK = {
    "queued": "#fbbf24",
    "running": "#60a5fa",
    "done": "#34d399",
    "error": "#f87171",
    "canceled": "#9ca3af",
}

# A running job that has not checked in for this long is probably wedged.
STALL_WARNING_SECONDS = 30.0

# --- Visual direction: "violet glass" ---------------------------------------
# Deep near-black with a blue undertone (never flat #000), layered panel
# gradients that read as translucent sheets, one violet accent used sparingly
# (primary action, selection, progress, active states), soft greys and
# off-whites everywhere else. Destructive actions are muted rose that only
# surfaces on hover. Radii are generous (10–14px); worker/status badges are
# pills. Type: Segoe UI for everything, monospace reserved for timestamps.
_ACCENT = "#8b5cf6"          # violet 500 — the one accent
_ACCENT_BRIGHT = "#a78bfa"   # hover
_ACCENT_DIM = "#7c4fe0"      # pressed
_ACCENT_TEXT = "#0e0a18"     # text on accent
_ACCENT_SOFT = "#241d38"     # selection highlight (violet-tinted)
_BG = "#0b0b10"
_SURFACE = "rgba(255, 255, 255, 0.035)"   # glassy sheet over the dark bg
_SURFACE_SOLID = "#15151d"
_PANEL = "#111118"
_BORDER = "rgba(255, 255, 255, 0.08)"     # thin light border
_BORDER_HOVER = "rgba(167, 139, 250, 0.55)"
_TEXT = "#ececf1"
_TEXT_DIM = "#9b9ba6"
_TEXT_FAINT = "#6d6d78"
_DANGER = "#c96f85"          # muted rose — destructive, quiet until hover
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
    "queued": "#b9bdc7", "running": "#d6c9ff", "done": "#c9f0dd",
    "error": "#f2b8c6", "canceled": "#8b8b95",
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


def _elapsed(started: Optional[str], finished: Optional[str] = None) -> str:
    """Human-readable duration between two DB timestamps (to now if unfinished)."""
    if not started:
        return ""
    try:
        t0 = datetime.strptime(started, "%Y-%m-%d %H:%M:%S")
        t1 = (
            datetime.strptime(finished, "%Y-%m-%d %H:%M:%S")
            if finished
            else datetime.now()
        )
    except ValueError:
        return ""
    secs = max(0, int((t1 - t0).total_seconds()))
    if secs < 60:
        return f"{secs}s"
    mins, secs = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins:02d}m"


def _fmt_clock(ms: int) -> str:
    """Format milliseconds as m:ss, or h:mm:ss once past an hour."""
    total = max(0, int(ms)) // 1000
    mins, secs = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def _human_age(seconds: float) -> str:
    """Compact 'how long ago' text for the staleness hint."""
    secs = max(0, int(seconds))
    if secs < 60:
        return f"{secs}s"
    mins, secs = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins:02d}m"


def _stage_label(stage: Optional[str], progress: Optional[float]) -> str:
    """What a running job is doing right now, with a percentage when known."""
    percent = f"{int(round((progress or 0.0) * 100))}%" if progress is not None else ""
    if stage == "downloading":
        return (f"downloading model {percent}").strip()
    if stage == "loading":
        return "loading model"
    if stage == "transcribing":
        return (f"transcribing {percent}").strip()
    return f"processing {percent}".strip()


def _expand_media(paths) -> list:
    """Expand a mix of files and folders into media files (recursively).

    Folders are the natural unit for a lecture series or a pile of voice
    notes — dropping one should not mean hand-selecting every file inside.
    """
    found = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(
                sorted(
                    child
                    for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() in MEDIA_EXTS
                )
            )
    return found


class PlaybackControls(QWidget):
    """Play/pause/stop + seek + volume for the selected job's source audio.

    Audio only — no video surface is created, so an mp4 or mkv plays as sound.
    The ``Player`` is injected (defaulting to a real QtMultimedia one) so tests
    can drive the widget with a fake backend.

    ``load()`` points the player at a job's media, ``set_spans()`` hands over the
    caption timing so the matching line can be highlighted in the preview while
    the audio plays.
    """

    def __init__(self, player: Optional[Player] = None, parent=None):
        super().__init__(parent)
        self.player = player if player is not None else Player(self)
        self._job_id: Optional[int] = None
        self._path: Optional[str] = None  # the file actually handed to the player
        self._requested: Optional[str] = None  # last requested path (may be missing)
        self._spans: list = []
        self._preview: Optional[QPlainTextEdit] = None
        self._highlight_block: Optional[int] = None
        self._seeking = False
        self._duration_ms = 0
        self._pending_seek: Optional[int] = None  # seek requested before load finished

        self._build()
        self._set_enabled(False)
        self.player.position_ms_changed.connect(self._on_position)
        self.player.duration_ms_changed.connect(self._on_duration)
        self.player.playback_state_changed.connect(self._on_state)
        self.player.player_error.connect(self._on_error)

    # --- construction -------------------------------------------------------
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
        self.lbl_time.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
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

    # --- public API ---------------------------------------------------------
    @property
    def has_source(self) -> bool:
        """True once a playable media path has been loaded."""
        return self._path is not None

    @property
    def source_path(self) -> Optional[str]:
        return self._path

    def attach_preview(self, preview: QPlainTextEdit) -> None:
        """Give the widget the caption pane it should highlight lines in."""
        self._preview = preview

    def load(self, job_id: Optional[int], path: Optional[str], spans=()) -> None:
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

    # --- transport ----------------------------------------------------------
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

    # --- player signal handlers ---------------------------------------------
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

    # --- volume / mute ------------------------------------------------------
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
            handle_center = self.seek_bar.width() * self.seek_bar.value() / max(1, self.seek_bar.maximum())
            if abs(click_x - handle_center) > 12:  # on the track, not the handle
                self.seek_to(int(click_x / max(1, self.seek_bar.width()) * self._duration_ms))
                return True  # a jump, not the start of a drag
        return super().eventFilter(obj, event)

    # --- seeking ------------------------------------------------------------
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

    # --- caption highlighting -----------------------------------------------
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

    def _apply_highlight(self, block_no: Optional[int]) -> None:
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


class WorkerThread(QThread):
    """Runs the queue loop in the background; owns its own DB connection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = threading.Event()
        self.cancel_event = threading.Event()

    def stop(self) -> None:
        """Finish the current job, then stop polling."""
        self._stop.set()

    def cancel(self) -> None:
        """Abort the job being processed (checked between segments)."""
        self.cancel_event.set()

    def run(self) -> None:
        conn = db.connect()
        try:
            while not self._stop.is_set():
                self.cancel_event.clear()  # each new job starts uncanceled
                if not worker.run_once(conn, cancel=self.cancel_event):
                    self._stop.wait(config.WORKER_POLL_SECONDS)
        finally:
            conn.close()


class Dashboard(QMainWindow):

    def __init__(self, conn=None, persist_settings: bool = True):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — batch captions")
        self.resize(1040, 720)
        self.setAcceptDrops(True)
        self.setStyleSheet(_QSS)

        self.conn = conn if conn is not None else db.connect()
        self.settings = QSettings("captioner", "dashboard") if persist_settings else None
        self.worker_thread: Optional[WorkerThread] = None
        self._signature = None
        self._pending_before = 0
        self.last_notification: Optional[str] = None
        # Unsaved caption edits, keyed by job id: switching jobs keeps them
        # (a modal prompt on every selection change would be worse than a
        # visible reminder plus Ctrl+S).
        self._drafts: dict = {}
        self._preview_job_id: Optional[int] = None
        self._loading_preview = False
        self._cancel_pending = False
        self._live_worker = None   # LiveWorker when live captions run
        self._overlay = None       # LiveOverlay when live captions run
        self._live_device = None   # last device the user picked for live
        self._hotkey = None        # GlobalHotkey, created in _build_menus
        # A job left 'running' by a crashed worker has no owner and would stay
        # stuck forever; the heartbeat tells us which ones those are.
        self.recovered = worker.recover_stale(self.conn)

        config.ensure_dirs()

        # Central first: the menus bind shortcuts to the playback widget.
        self._build_central()
        self._build_menus()
        self._build_status_bar()
        self._restore_geometry()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self.refresh()
        if self.recovered:
            self.statusBar().showMessage(
                f"recovered {self.recovered} job(s) a crashed worker left running — "
                "they are queued again",
                10000,
            )

    # --- construction -------------------------------------------------------
    def _add_action(self, menu: QMenu, text: str, shortcut: Optional[str], slot, tip: str = "") -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.setStatusTip(tip or text)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._add_action(file_menu, "&Add files…", "Ctrl+O", self.add_files,
                         "Queue one or more media files for captioning")
        self._add_action(file_menu, "Add f&older…", "Ctrl+Shift+O", self.add_folder,
                         "Queue every media file inside a folder (recursive)")
        file_menu.addSeparator()
        self._add_action(file_menu, "E&xit", "Ctrl+Q", self.close)

        job_menu = self.menuBar().addMenu("&Job")
        self._add_action(job_menu, "&Retry selected", "Ctrl+R", self.retry_selected,
                         "Send the selected job back to the queue")
        self._add_action(job_menu, "&Remove selected", "Delete", self.remove_selected,
                         "Drop the selected job from the queue (output files stay on disk)")
        self._add_action(job_menu, "&Cancel running job", None, self.cancel_running_job,
                         "Stop the job being transcribed and leave it canceled")
        self._add_action(job_menu, "Recover &stuck jobs", None, self.recover_stuck,
                         "Requeue jobs a crashed worker left marked running")
        job_menu.addSeparator()
        self._add_action(job_menu, "&Save caption edits", "Ctrl+S", self.save_caption_edits,
                         "Rewrite captions.srt and captions.vtt with your edits")
        self._add_action(job_menu, "Dis&card caption edits", "Ctrl+Shift+Z",
                         self.revert_caption_edits,
                         "Throw away unsaved caption edits")
        job_menu.addSeparator()
        self._add_action(job_menu, "&Open output folder", "Ctrl+E", self.open_output)
        self._add_action(job_menu, "&Copy file path", "Ctrl+Shift+C", self.copy_path)
        self._add_action(job_menu, "Open &outputs directory", None, self.reveal_outputs_dir)

        play_menu = self.menuBar().addMenu("&Playback")
        self._add_action(play_menu, "&Play / Pause", "Space", self.toggle_play,
                         "Play the selected job's audio")
        self._add_action(play_menu, "&Stop", "Ctrl+.", self.stop_playback,
                         "Stop playback and rewind")
        play_menu.addSeparator()
        self._add_action(play_menu, "Back 5 seconds", "Ctrl+Left",
                         lambda: self.player_controls.seek_relative(-5000))
        self._add_action(play_menu, "Forward 5 seconds", "Ctrl+Right",
                         lambda: self.player_controls.seek_relative(5000))
        self._add_action(play_menu, "&Mute", "Ctrl+M", self.player_controls.toggle_mute,
                         "Mute or unmute the preview audio")

        live_menu = self.menuBar().addMenu("&Live")
        self._add_action(live_menu, "&Start live captions…", None, self.start_live,
                         "Caption microphone or system audio live in a floating bar")
        self.act_live_model = live_menu.addAction("Live &model: …")
        self.act_live_model.triggered.connect(self.choose_live_model)
        self._refresh_live_model_label()
        self.act_stop_live = self._add_action(live_menu, "Sto&p live captions", None,
                                              self.stop_live,
              "Stop the floating live caption bar")
        self.act_stop_live.setEnabled(False)
        self.act_toggle_live_menu = self._add_action(
            live_menu, "Toggle live captions any&where", "Ctrl+Alt+Y", self.toggle_live,
            "Same toggle as the system-wide hotkey (in-app shortcut)",
        )
        # The menu item above works inside the app (Ctrl+Alt+Y); the *global*
        # hotkey that works in other apps is registered with Windows below.
        if os.environ.get("CAPTIONER_NO_GLOBAL_HOTKEY") == "1":
            self._hotkey = None  # tests: never claim a system-wide combination
        else:
            try:
                from ui.hotkey import (
                    MOD_ALT,
                    MOD_CONTROL,
                    MOD_SHIFT,
                    VK_L,
                    GlobalHotkey,
                    describe,
                )

                candidates = [
                    (MOD_CONTROL | MOD_ALT, VK_L),              # Ctrl+Alt+L (preferred)
                    (MOD_CONTROL | MOD_ALT | MOD_SHIFT, VK_L),  # fallback
                ]
                for modifiers, vk in candidates:
                    hk = GlobalHotkey(modifiers, vk)
                    hk.activated.connect(self.toggle_live)
                    hk.registration_failed.connect(
                        lambda m: self.statusBar().showMessage(m, 8000)
                    )
                    hk.start()
                    if hk.error:
                        hk.stop()
                        continue  # try the next combination
                    self._hotkey = hk
                    # defer: the real status bar installs after the menus are built
                    QTimer.singleShot(
                        0,
                        lambda: self.statusBar().showMessage(
                            f"global hotkey ready: {describe(modifiers, vk)} "
                            "toggles live captions",
                            6000,
                        ),
                    )
                    break
                else:
                    logger.warning("no global hotkey combination was free")
            except Exception as exc:  # non-Windows or AV software blocking it
                logger.warning("global hotkey unavailable: %s", exc)
                self._hotkey = None

        help_menu = self.menuBar().addMenu("&Help")
        self._add_action(help_menu, "&About", None, self.show_about)

    def _build_central(self) -> None:
        central = QWidget()
        col = QVBoxLayout(central)
        col.setContentsMargins(20, 14, 20, 12)
        col.setSpacing(12)

        # --- calm header: primary actions + worker pill + settings ---------
        header = QHBoxLayout()
        header.setSpacing(10)

        self.btn_add = QPushButton("＋  Add files")
        self.btn_add.setToolTip("Queue media files for captioning (Ctrl+O) — or just drop them anywhere in this window")
        self.btn_add.clicked.connect(self.add_files)
        header.addWidget(self.btn_add)

        self.btn_worker = QPushButton("▶  Start worker")
        self.btn_worker.setToolTip(
            "Process queued jobs inside this window (Ctrl+R shows retry)"
        )
        self.btn_worker.clicked.connect(self.toggle_worker)
        header.addWidget(self.btn_worker)

        header.addSpacing(6)
        # worker status as a pill with a colored dot
        self.worker_state = QLabel("●  stopped")
        self.worker_state.setObjectName("WorkerPill")
        header.addWidget(self.worker_state)

        header.addStretch(1)

        # Secondary options tuck into a settings popover (a QMenu of widgets).
        self.btn_options = QPushButton("⚙  Options")
        self.btn_options.setProperty("secondary", True)
        self.btn_options.setToolTip("Model, speakers, watch folder, notifications")
        options_menu = QMenu(self.btn_options)
        self._options = QWidgetAction(options_menu)
        panel = QWidget()
        form = QVBoxLayout(panel)
        form.setContentsMargins(12, 8, 12, 10)
        form.setSpacing(8)

        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        model_label = QLabel("Model")
        model_label.setStyleSheet(f"color: {_TEXT_FAINT};")
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_CHOICES)
        preferred = config.BATCH_MODEL_SIZE if config.BATCH_MODEL_SIZE in MODEL_CHOICES else "medium"
        self.model_combo.setCurrentText(preferred)
        self.model_combo.setToolTip(
            "Model used for newly added files.\n"
            "tiny = fast draft (~8× realtime) · small/medium = accurate "
            "(~2.5–3× realtime) · distil = medium's accuracy at 2× medium's "
            "speed · large = slowest, most accurate.\n"
            "The first run with a model downloads its weights."
        )
        self.model_combo.setMinimumWidth(170)
        model_row.addWidget(model_label)
        model_row.addWidget(self.model_combo)
        form.addLayout(model_row)

        self.diarize_check = QCheckBox("Identify speakers")
        self.diarize_check.setToolTip(
            "Label caption lines with speakers ([SPEAKER_00]: …) using pyannote.\n"
            "Applies to newly added files; adds a diarization pass per job\n"
            "and needs the HF token in .env on first use."
        )
        form.addWidget(self.diarize_check)

        self.align_check = QCheckBox("Align word timings (wav2vec2)")
        self.align_check.setToolTip(
            "Re-time every word with a character alignment model before the\n"
            "captions are grouped — sharper subtitle in/out points than\n"
            "Whisper's own estimates. Adds a short pass per job; the first use\n"
            "downloads ~380 MB. Applies to newly added files."
        )
        form.addWidget(self.align_check)

        translate_row = QHBoxLayout()
        translate_row.setSpacing(8)
        translate_label = QLabel("Translate captions")
        translate_label.setStyleSheet(f"color: {_TEXT_FAINT};")
        self.translate_combo = QComboBox()
        self.translate_combo.addItem("Off", "")
        for code, name in sorted(LANGUAGES.items(), key=lambda kv: kv[1]):
            self.translate_combo.addItem(f"{name} ({code})", code)
        self.translate_combo.setToolTip(
            "Also write captions.<language>.srt/.vtt for newly added files.\n"
            "Each unique line is translated once and cached, so re-running a\n"
            "job costs nothing. First use downloads ~2.4 GB."
        )
        self.translate_combo.setMinimumWidth(170)
        translate_row.addWidget(translate_label)
        translate_row.addWidget(self.translate_combo)
        form.addLayout(translate_row)

        self.watch_check = QCheckBox("Watch uploads folder")
        self.watch_check.setToolTip(
            "Automatically queue media that appears in the uploads folder "
            f"({config.UPLOADS_DIR}).\n"
            "Removing a job while watching lets its file be queued again."
        )
        form.addWidget(self.watch_check)
        self.notify_check = QCheckBox("Notify when done")
        self.notify_check.setChecked(True)
        self.notify_check.setToolTip(
            "Beep and flash the taskbar entry when the queue drains"
        )
        form.addWidget(self.notify_check)
        self._options.setDefaultWidget(panel)
        options_menu.addAction(self._options)
        self._options_menu = options_menu
        self.btn_options.clicked.connect(
            lambda: options_menu.exec(
                self.btn_options.mapToGlobal(
                    self.btn_options.rect().bottomRight()
                )
            )
        )
        header.addWidget(self.btn_options)
        col.addLayout(header)

        # --- queue summary: slim progress, no duplicate counts row ----------
        summary = QHBoxLayout()
        summary.setSpacing(12)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("no jobs yet")
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setToolTip("Settled jobs (done + error) out of everything in the queue")
        self.counts_label = QLabel("")
        self.counts_label.setStyleSheet(
            f"color: {_TEXT_FAINT}; font-size: 12px; background: transparent;"
        )
        summary.addWidget(self.progress, stretch=1)
        summary.addWidget(self.counts_label)
        col.addLayout(summary)

        # --- table + preview splitter ---------------------------------------
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        header = self.table.horizontalHeader()
        for idx in (1, 7):
            header.setSectionResizeMode(idx, QHeaderView.ResizeMode.Stretch)
        for idx in (0, 2, 3, 4, 5, 6):
            header.setSectionResizeMode(idx, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.update_preview)
        self.table.cellDoubleClicked.connect(lambda *_: self.open_output())
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        splitter.addWidget(self.table)

        details = QWidget()
        dcol = QVBoxLayout(details)
        dcol.setContentsMargins(0, 2, 0, 0)
        dcol.setSpacing(8)
        # tidy summary card for the selected job (glassy, rounded)
        meta_card = QWidget()
        meta_card.setObjectName("SummaryCard")
        mcol = QVBoxLayout(meta_card)
        mcol.setContentsMargins(14, 10, 14, 10)
        self.meta = QLabel()
        self.meta.setWordWrap(True)
        self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mcol.addWidget(self.meta)
        dcol.addWidget(meta_card)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.preview.setPlaceholderText(
            "Drop media files or folders anywhere in this window, or press Ctrl+O.\n"
            "Select a finished job to read (and fix) its captions here."
        )
        self.preview.textChanged.connect(self._on_preview_edited)
        dcol.addWidget(self.preview, stretch=1)

        # Audio playback for the selected job — slim bar docked at the very
        # bottom of the window, below the splitter (spec: player docked low).
        self.player_controls = PlaybackControls(parent=self)
        self.player_controls.attach_preview(self.preview)
        self.player = self.player_controls.player

        btns = QHBoxLayout()
        btns.setSpacing(8)
        btn_open = QPushButton("Open output folder")
        btn_open.setProperty("secondary", True)
        btn_open.clicked.connect(self.open_output)
        btn_copy = QPushButton("Copy file path")
        btn_copy.setProperty("secondary", True)
        btn_copy.clicked.connect(self.copy_path)
        btns.addWidget(btn_open)
        btns.addWidget(btn_copy)
        btns.addStretch(1)
        self.edit_state = QLabel("")
        self.edit_state.setObjectName("EditStatePill")
        self.edit_state.hide()  # appears only when there are unsaved edits
        btns.addWidget(self.edit_state)
        self.btn_save_edits = QPushButton("Save edits")
        self.btn_save_edits.setToolTip(
            "Rewrite captions.srt and captions.vtt with your edited text (Ctrl+S)"
        )
        self.btn_save_edits.clicked.connect(self.save_caption_edits)
        self.btn_revert_edits = QPushButton("Revert")
        self.btn_revert_edits.setProperty("secondary", True)
        self.btn_revert_edits.setToolTip("Throw away unsaved caption edits (Ctrl+Shift+Z)")
        self.btn_revert_edits.clicked.connect(self.revert_caption_edits)
        btns.addWidget(self.btn_save_edits)
        btns.addWidget(self.btn_revert_edits)
        dcol.addLayout(btns)

        splitter.addWidget(details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.splitter = splitter
        col.addWidget(splitter, stretch=1)
        col.addWidget(self.player_controls)  # docked at the bottom

        self.setCentralWidget(central)

    def _set_worker_pill(self, running: bool) -> None:
        """Worker status as a pill: violet while running, neutral otherwise."""
        self.worker_state.setText("●  running" if running else "●  stopped")
        self.worker_state.setProperty("running", running)
        style = self.worker_state.style()
        style.unpolish(self.worker_state)
        style.polish(self.worker_state)

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.paths_label = QLabel(f"outputs: {config.OUTPUT_DIR}")
        self.paths_label.setToolTip(
            f"database: {config.DB_PATH}\noutputs: {config.OUTPUT_DIR}"
        )
        bar.addPermanentWidget(self.paths_label)
        bar.showMessage("Ready — add a file to caption.")

    def _restore_geometry(self) -> None:
        if self.settings is None:
            return
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self.settings.value("splitter")
        if state is not None:
            self.splitter.restoreState(state)

    # --- helpers ------------------------------------------------------------
    def selected_job_id(self) -> Optional[int]:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return int(item.text()) if item else None

    def _is_dark(self) -> bool:
        return self.palette().color(QPalette.ColorRole.Window).lightness() < 128

    def _status_color(self, status: str) -> QColor:
        palette = _STATUS_COLORS_DARK if self._is_dark() else _STATUS_COLORS_LIGHT
        return QColor(palette.get(status, "#888888"))

    # --- actions ------------------------------------------------------------
    def add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add media files", "", MEDIA_FILTER)
        if files:
            self.add_paths(files)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add every media file in a folder")
        if folder:
            self.add_paths([folder])

    def _queue_options(self) -> dict:
        """The per-job options the Options popover currently selects."""
        return {
            "diarize": self.diarize_check.isChecked(),
            "align": self.align_check.isChecked(),
            "translate_to": self.translate_combo.currentData() or None,
        }

    def add_paths(self, paths) -> int:
        """Queue media found in `paths` (files and folders); returns the count."""
        model = self.model_combo.currentText()
        options = self._queue_options()
        files = _expand_media(paths)
        for path in files:
            worker.enqueue(self.conn, str(path), model_size=model, **options)
        if files:
            self._signature = None
            self.refresh()
            extras = [name for name, on in (
                ("speakers", options["diarize"]),
                ("alignment", options["align"]),
                (f"translate:{options['translate_to']}", bool(options["translate_to"])),
            ) if on]
            suffix = f" [{', '.join(extras)}]" if extras else ""
            self.statusBar().showMessage(
                f"queued {len(files)} file(s) with {model}{suffix}", 4000
            )
        else:
            self.statusBar().showMessage("nothing added — no media files found", 4000)
        return len(files)

    def toggle_play(self) -> None:
        """Play/pause the selected job's audio (Space, or the Play button)."""
        if not self.player_controls.has_source:
            self.statusBar().showMessage("select a job with audio first", 3000)
            return
        self.player_controls.toggle_play()

    def stop_playback(self) -> None:
        self.player_controls.stop()

    # --- live captions (Phase 3) ---------------------------------------------
    @property
    def LIVE_MODEL_CHOICES(self):
        """Built-in sizes plus any locally fine-tuned (Phase 4/5) models."""
        return models.live_choices()

    def _refresh_live_model_label(self) -> None:
        model = self.settings.value("live_model", config.LIVE_MODEL_SIZE) if self.settings else config.LIVE_MODEL_SIZE
        self.act_live_model.setText(f"Live &model: {model}")

    def choose_live_model(self) -> None:
        """Pick the live caption model (remembered; takes effect next session).

        On clean audio tiny.en is already the most accurate live option
        (measured in scripts/bench_live_wer.py); base/small help when the
        source is noisy or far from the mic, at the cost of some latency.
        """
        current = self.settings.value("live_model", config.LIVE_MODEL_SIZE) if self.settings else config.LIVE_MODEL_SIZE
        names = [f"{m}{'  (current)' if m == current else ''}" for m in self.LIVE_MODEL_CHOICES]
        name, ok = QInputDialog.getItem(
            self, "Live captions model",
            "Model for live captions:\n"
            "tiny = fastest, best on clear audio · base/small = sturdier on "
            "noise, a little slower",
            names, max(0, list(self.LIVE_MODEL_CHOICES).index(current))
            if current in self.LIVE_MODEL_CHOICES else 0,
            False,
        )
        if not ok:
            return
        model = self.LIVE_MODEL_CHOICES[names.index(name)]
        if self.settings is not None:
            self.settings.setValue("live_model", model)
        self._refresh_live_model_label()
        self.statusBar().showMessage(
            f"live model: {model} (takes effect on the next live session)", 5000
        )

    def _launch_live(self, device: dict) -> None:
        """Start the overlay + worker for *device* (shared by menu and hotkey)."""
        from ui.overlay import LiveOverlay, LiveWorker

        self.stop_live()  # replace any running session
        self._overlay = LiveOverlay()
        self._overlay.set_device_name(f"● {device['name']}")
        self._overlay.stop_requested.connect(self.stop_live)
        self._overlay.show()
        model = self.settings.value("live_model", config.LIVE_MODEL_SIZE) if self.settings else config.LIVE_MODEL_SIZE
        try:
            model = models.resolve(model)
        except ValueError as exc:
            self.statusBar().showMessage(f"{exc} — using {config.LIVE_MODEL_SIZE}", 6000)
            model = config.LIVE_MODEL_SIZE
        self._live_worker = LiveWorker(device["id"], model_size=model)
        self._live_worker.state_ready.connect(self._overlay.update_state)
        self._live_worker.live_error.connect(self._overlay.show_error)
        self._live_worker.start()
        self.act_stop_live.setEnabled(True)
        self._live_device = device
        self.statusBar().showMessage(f"live captions on: {device['name']}", 5000)

    def _pick_live_device(self) -> Optional[dict]:
        """Ask which audio source to caption; None means the user canceled."""
        from captioner.live import list_input_devices

        try:
            devices = list_input_devices()
        except Exception as exc:
            QMessageBox.warning(self, "Live captions", f"Could not list audio devices:\n{exc}")
            return None
        if not devices:
            QMessageBox.warning(
                self, "Live captions",
                "No microphone or loopback device was found.\n"
                "Check Windows sound settings, then try again.",
            )
            return None
        names = [d["name"] for d in devices]
        name, ok = QInputDialog.getItem(
            self, "Live captions", "Caption audio from:", names, 0, False
        )
        if not ok:
            return None
        return devices[names.index(name)]

    def start_live(self) -> None:
        """Pick an audio source (remembered for next time), then go live."""
        device = self._pick_live_device()
        if device is not None:
            self._launch_live(device)

    def stop_live(self) -> None:
        if self._live_worker is not None and self._live_worker.isRunning():
            self._live_worker.stop()
            self._live_worker.wait(3000)
        self._live_worker = None
        if self._overlay is not None:
            self._overlay.close_transcript_window()
            self._overlay.close()
            self._overlay = None
        if getattr(self, "act_stop_live", None) is not None:
            self.act_stop_live.setEnabled(False)
            if self.isVisible():
                self.statusBar().showMessage("live captions stopped", 3000)

    def toggle_worker(self) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.stop_worker()
        else:
            self.start_worker()

    def start_worker(self) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            return
        self.worker_thread = WorkerThread(self)
        self.worker_thread.finished.connect(self._on_worker_finished)
        self.worker_thread.start()
        self.btn_worker.setText("⏹  Stop worker")
        self._set_worker_pill(True)
        self.statusBar().showMessage("worker started — queued jobs process in the background", 4000)

    def stop_worker(self) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.btn_worker.setEnabled(False)
            self.btn_worker.setText("⏹  Stopping…")
            self.statusBar().showMessage("stopping after the current job…", 4000)

    def _on_worker_finished(self) -> None:
        self.worker_thread = None
        self.btn_worker.setEnabled(True)
        self.btn_worker.setText("▶  Start worker")
        self._set_worker_pill(False)
        self.btn_cancel.setEnabled(False)
        self.statusBar().showMessage("worker stopped", 4000)

    def cancel_running_job(self) -> bool:
        """Abort the job this window's worker is running.

        A job owned by another process (run_worker.bat, the CLI) cannot be
        interrupted from here, so say so instead of pretending.
        """
        if self.worker_thread is None or not self.worker_thread.isRunning():
            running = self.conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'running'"
            ).fetchone()[0]
            if running:
                QMessageBox.information(
                    self, "Job owned by another process",
                    "A job is running outside this window (run_worker.bat or the CLI).\n\n"
                    "Stop that process, then use Job × Recover stuck jobs if the "
                    "row still says running.",
                )
            else:
                self.statusBar().showMessage("no job is running", 3000)
            return False
        self.worker_thread.cancel()
        self._cancel_pending = True
        self.btn_cancel.setEnabled(False)
        self.statusBar().showMessage(
            "canceling — the worker stops at the next segment boundary…", 6000
        )
        return True

    def recover_stuck(self) -> int:
        """Requeue jobs whose worker died mid-run."""
        count = worker.recover_stale(self.conn)
        self._signature = None
        self.refresh()
        self.statusBar().showMessage(
            f"requeued {count} stuck job(s)" if count else "no stuck jobs found", 5000
        )
        return count

    def retry_selected(self) -> None:
        job_id = self.selected_job_id()
        if job_id is None:
            self.statusBar().showMessage("select a job first", 3000)
            return
        row = self.conn.execute(
            "SELECT status FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return
        if row["status"] == "running":
            self.statusBar().showMessage(
                f"job #{job_id} is running — stop the worker to requeue it", 5000
            )
            return
        worker.retry(self.conn, job_id)
        self._signature = None
        self.refresh()
        self.statusBar().showMessage(f"job #{job_id} requeued", 3000)

    def remove_selected(self) -> None:
        job_id = self.selected_job_id()
        if job_id is None:
            self.statusBar().showMessage("select a job first", 3000)
            return
        row = self.conn.execute(
            "SELECT file_name, status FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return
        if row["status"] == "running":
            QMessageBox.information(
                self, "Job running",
                f"Job #{job_id} is being transcribed.\n\nStop the worker before removing it.",
            )
            return
        answer = QMessageBox.question(
            self, "Remove job",
            f"Remove job #{job_id} ({row['file_name']}) from the queue?\n\n"
            "Subtitle files already written to the outputs folder stay on disk.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._remove_job(job_id)

    def _remove_job(self, job_id: int) -> bool:
        removed = worker.delete_job(self.conn, job_id)
        self._signature = None
        self.refresh()
        self.statusBar().showMessage(
            f"job #{job_id} removed" if removed else f"job #{job_id} could not be removed",
            4000,
        )
        return removed

    def open_output(self) -> None:
        job_id = self.selected_job_id()
        if job_id is None:
            self.statusBar().showMessage("select a job first", 3000)
            return
        out = Path(config.OUTPUT_DIR) / f"job_{job_id}"
        if not out.exists():
            row = self.conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            status = row["status"] if row else "missing"
            QMessageBox.information(
                self, "No output yet",
                f"Job #{job_id} has no output folder yet (status: {status}).",
            )
            return
        if sys.platform == "win32":
            os.startfile(str(out))  # noqa: S606
        else:
            QMessageBox.information(self, "Output folder", str(out))

    def reveal_outputs_dir(self) -> None:
        config.ensure_dirs()
        out = Path(config.OUTPUT_DIR)
        if sys.platform == "win32":
            os.startfile(str(out))  # noqa: S606
        else:
            QMessageBox.information(self, "Outputs directory", str(out))

    def copy_path(self) -> None:
        job_id = self.selected_job_id()
        if job_id is None:
            self.statusBar().showMessage("select a job first", 3000)
            return
        row = self.conn.execute(
            "SELECT file_path FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row:
            QApplication.clipboard().setText(row["file_path"])
            self.statusBar().showMessage("file path copied", 2500)

    def show_about(self) -> None:
        try:
            import PySide6
            pyside = PySide6.__version__
        except Exception:  # pragma: no cover - version display only
            pyside = "?"
        try:
            import faster_whisper
            whisper = faster_whisper.__version__
        except Exception:  # pragma: no cover - version display only
            whisper = "?"
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> — local AI captions: batch + live + speakers<p>"
            f"version {APP_VERSION} · PySide6 {pyside} · faster-whisper {whisper}<br>"
            f"batch model default: {config.BATCH_MODEL_SIZE} · "
            f"live model default: {config.LIVE_MODEL_SIZE}"
            f"<p>database: {config.DB_PATH}<br>outputs: {config.OUTPUT_DIR}<br>"
            f"log: {config.DATA_DIR / 'logs' / 'captioner.log'}",
        )

    def _show_context_menu(self, point) -> None:
        """Row menu: all job actions live here instead of permanent buttons."""
        if self.selected_job_id() is None:
            return
        menu = QMenu(self)
        menu.addAction("Open output folder", self.open_output)
        menu.addAction("Copy file path", self.copy_path)
        menu.addSeparator()
        act_cancel = menu.addAction("Cancel running job", self.cancel_running_job)
        act_cancel.setEnabled(
            self.worker_thread is not None and self.worker_thread.isRunning()
        )
        menu.addAction("Retry job", self.retry_selected)
        act_remove = menu.addAction("Remove job", self.remove_selected)
        act_remove.setProperty("danger", True)  # muted rose in styled menus
        menu.exec(self.table.viewport().mapToGlobal(point))

    # --- refresh / preview ---------------------------------------------------
    def refresh(self) -> None:
        if self.watch_check.isChecked():
            self._scan_watch_folder()
        rows = self.conn.execute(
            "SELECT id, file_name, file_path, status, model_size, progress, stage, "
            "heartbeat_at, error, created_at, started_at, finished_at "
            "FROM jobs ORDER BY id"
        ).fetchall()
        # Rebuild only when something visible changed (avoids per-second flicker);
        # progress, stage, and a running job's elapsed/staleness are all
        # "visible", and staleness is bucketed to 15s so a quiet heartbeat
        # still refreshes the hint without redrawing every tick.
        signature = tuple(
            (
                r["id"], r["status"], r["error"], r["finished_at"], r["progress"],
                r["stage"],
                _elapsed(r["started_at"], r["finished_at"]) if r["status"] == "running" else "",
                int(worker.seconds_since(r["heartbeat_at"]) // 15)
                if r["status"] == "running"
                else "",
            )
            for r in rows
        )
        if signature != self._signature:
            self._signature = signature
            self._rebuild_table(rows)
        self._update_summary(rows)
        self._maybe_notify(rows)

    def _rebuild_table(self, rows) -> None:
        selected = self.selected_job_id()
        caption_counts = dict(
            self.conn.execute(
                "SELECT job_id, COUNT(*) FROM captions GROUP BY job_id"
            ).fetchall()
        )
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(rows))
            for i, r in enumerate(rows):
                duration = (
                    _elapsed(r["started_at"], r["finished_at"])
                    if r["status"] in ("running", "done", "error")
                    else ""
                )
                if r["error"]:
                    details = r["error"]
                elif r["status"] == "done":
                    n = caption_counts.get(r["id"], 0)
                    details = f"{n} caption{'s' if n != 1 else ''}"
                elif r["status"] == "running":
                    details = _stage_label(r["stage"], r["progress"])
                    age = worker.seconds_since(r["heartbeat_at"])
                    if age > STALL_WARNING_SECONDS:
                        details += f" · no update for {_human_age(age)}"
                elif r["status"] == "canceled":
                    details = "canceled — retry to transcribe again"
                else:
                    details = "waiting"
                if r["id"] in self._drafts:
                    details = f"{details} · unsaved edits"
                if r["status"] == "done":
                    percent = "100%"
                elif r["status"] == "running":
                    percent = (
                        f"{int(round((r['progress'] or 0.0) * 100))}%"
                        if r["progress"] is not None
                        else "starting…"
                    )
                else:
                    percent = ""
                values = (
                    str(r["id"]), r["file_name"], r["status"], r["model_size"],
                    percent, r["created_at"] or "", duration, details,
                )
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if col == 0:
                        item.setData(Qt.ItemDataRole.UserRole, r["id"])
                    if col == 1:
                        item.setToolTip(r["file_path"])
                    if col == 2:
                        item.setForeground(self._status_color(r["status"]))
                    if col == 7:
                        item.setToolTip(r["error"] or details)
                    self.table.setItem(i, col, item)
        finally:
            self.table.setUpdatesEnabled(True)

        restored = False
        if selected is not None:
            for i, r in enumerate(rows):
                if r["id"] == selected:
                    self.table.selectRow(i)
                    restored = True
                    break
        if not restored and rows:
            # keep the newest job visible so the preview is never blank by default
            self.table.selectRow(len(rows) - 1)
        # Re-select the same row does not re-emit itemSelectionChanged, so the
        # pane would otherwise keep a stale status (e.g. stay read-only after
        # the job it shows finished). update_preview is safe to call often: it
        # leaves an in-progress edit alone.
        self.update_preview()

    def _update_summary(self, rows) -> None:
        total = len(rows)
        keys = ("queued", "running", "done", "error", "canceled")
        counts = {key: 0 for key in keys}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        if total:
            settled = counts["done"] + counts["error"] + counts["canceled"]
            self.progress.setRange(0, total)
            self.progress.setValue(settled)
            self.progress.setFormat(f"{counts['done']}/{total} done")
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("no jobs yet")
        parts = [f"{key} {counts[key]}" for key in keys if counts[key]]
        self.counts_label.setText(" · ".join(parts))
        running = self.worker_thread is not None and self.worker_thread.isRunning()
        if not self.btn_worker.isEnabled() and not running:
            self.btn_worker.setEnabled(True)
        if self.btn_worker.isEnabled():
            self.btn_worker.setText("⏹  Stop worker" if running else "▶  Start worker")
        # worker status as a pill: violet while running, neutral otherwise
        self._set_worker_pill(running)
        if not counts["running"]:
            self._cancel_pending = False  # whatever we were canceling is done
        # Cancel lives in the row context menu now (no permanent button);
        # availability is still enforced when the menu opens.

    def _scan_watch_folder(self) -> int:
        """Queue media that appeared in the uploads folder since the last tick."""
        folder = Path(config.UPLOADS_DIR)
        if not folder.is_dir():
            return 0
        known = {
            row["file_path"]
            for row in self.conn.execute("SELECT file_path FROM jobs").fetchall()
        }
        new_files = [
            child
            for child in sorted(folder.rglob("*"))
            if child.is_file()
            and child.suffix.lower() in MEDIA_EXTS
            and str(child.resolve()) not in known
        ]
        if not new_files:
            return 0
        model = self.model_combo.currentText()
        options = self._queue_options()
        for path in new_files:
            worker.enqueue(self.conn, str(path), model_size=model, **options)
        self._signature = None
        self.statusBar().showMessage(
            f"watch folder: queued {len(new_files)} new file(s)", 4000
        )
        return len(new_files)

    def _maybe_notify(self, rows) -> None:
        """Beep/flash once when the queue drains (pending -> settled)."""
        pending = sum(1 for r in rows if r["status"] in ("queued", "running"))
        if (
            self.notify_check.isChecked()
            and rows
            and pending == 0
            and self._pending_before > 0
        ):
            self.last_notification = datetime.now().strftime("%H:%M:%S")
            QApplication.beep()
            QApplication.alert(self, 5000)
            self.statusBar().showMessage(
                f"batch finished at {self.last_notification} — nothing left to process",
                8000,
            )
        self._pending_before = pending

    def _build_preview(self, job_id: int):
        """Return (preview_text, spans) for a job's captions.

        ``spans`` maps each caption's time range to the document block holding
        its text, which is what the playback widget highlights in sync.
        """
        return editor.render(editor.load_drafts(self.conn, job_id))

    def _preview_text(self, job_id: int) -> str:
        return self._build_preview(job_id)[0]

    def _spans_from_text(self, text: str):
        """Highlight spans for whatever text is currently in the pane.

        Returns nothing while the buffer is mid-edit and unreadable — a stale
        highlight would be worse than none.
        """
        try:
            return editor.render(editor.parse_preview(text))[1]
        except editor.CaptionParseError:
            return []

    def _show_preview(self, text: str, *, editable: bool, job_id: Optional[int]) -> None:
        """Display *text* without disturbing an edit that is already in progress.

        The pane is only rewritten when its contents actually differ, so the
        once-a-second queue refresh cannot move the caret out from under you.
        """
        self._loading_preview = True
        try:
            self.preview.setReadOnly(not editable)
            self.preview.setToolTip(
                "Fix any misheard word here, then press Ctrl+S to rewrite "
                "captions.srt and captions.vtt"
                if editable
                else "Select a finished job to read (and fix) its captions."
            )
            if self.preview.toPlainText() != text:
                caret = self.preview.textCursor().position()
                self.preview.setPlainText(text)
                if caret:
                    cursor = self.preview.textCursor()
                    cursor.setPosition(min(caret, len(text)))
                    self.preview.setTextCursor(cursor)
        finally:
            self._loading_preview = False
        self._preview_job_id = job_id

    def _on_preview_edited(self) -> None:
        """Remember what the user typed; nothing is written until they save."""
        if self._loading_preview or self._preview_job_id is None:
            return
        if self.preview.isReadOnly():
            return
        job_id = self._preview_job_id
        text = self.preview.toPlainText()
        if self._drafts.get(job_id) == text:
            return
        was_clean = job_id not in self._drafts
        self._drafts[job_id] = text
        self._update_edit_buttons()
        if was_clean:
            self._signature = None  # let the Details column show the marker
            self.statusBar().showMessage(
                "editing captions — Ctrl+S writes captions.srt and captions.vtt", 5000
            )

    def _update_edit_buttons(self) -> None:
        job_id = self._preview_job_id
        dirty = job_id is not None and job_id in self._drafts
        self.btn_save_edits.setEnabled(dirty)
        self.btn_revert_edits.setEnabled(dirty)
        # "unsaved edits" shows once, as a violet pill; Save gets the accent
        # while dirty, Save/Revert sit quiet otherwise.
        self.edit_state.setText("\u25cf unsaved edits")
        self.edit_state.setVisible(dirty)
        self.btn_save_edits.setProperty("accentOutline" if dirty else "secondary", True)
        if dirty:
            self.btn_save_edits.setProperty("secondary", False)
        else:
            self.btn_save_edits.setProperty("accentOutline", False)
        for prop in ("secondary", "accentOutline"):
            style = self.btn_save_edits.style()
            style.unpolish(self.btn_save_edits)
            style.polish(self.btn_save_edits)

    def save_caption_edits(self) -> bool:
        """Write edited caption text to the database and re-export SRT + VTT."""
        job_id = self._preview_job_id
        if job_id is None or job_id not in self._drafts:
            self.statusBar().showMessage("no unsaved caption edits", 3000)
            return False
        try:
            drafts = editor.parse_preview(self.preview.toPlainText())
            result = editor.apply_edits(self.conn, job_id, drafts)
        except editor.CaptionParseError as exc:
            QMessageBox.warning(self, "Cannot save caption edits", str(exc))
            return False
        except (OSError, sqlite3.Error) as exc:
            QMessageBox.warning(
                self, "Cannot save caption edits", f"{type(exc).__name__}: {exc}"
            )
            return False
        self._drafts.pop(job_id, None)
        self._preview_job_id = None  # force a clean re-render from the database
        self._signature = None
        self.refresh()
        self.statusBar().showMessage(
            f"saved {result.changed} of {result.total} caption(s) → "
            f"{result.srt_path.name} + {result.vtt_path.name}",
            6000,
        )
        return True

    def revert_caption_edits(self) -> bool:
        job_id = self._preview_job_id
        if job_id is None or job_id not in self._drafts:
            self.statusBar().showMessage("no unsaved caption edits", 3000)
            return False
        self._drafts.pop(job_id, None)
        self._preview_job_id = None
        self._signature = None
        self.refresh()
        self.statusBar().showMessage("unsaved caption edits discarded", 4000)
        return True

    def update_preview(self) -> None:
        job_id = self.selected_job_id()
        if job_id is None:
            self.meta.setText(
                "No job selected — drop media files here, or press Ctrl+O to queue some."
            )
            self._show_preview("", editable=False, job_id=None)
            self.player_controls.load(None, "", ())
            self._update_edit_buttons()
            return
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            self.meta.setText("")
            self._show_preview("", editable=False, job_id=None)
            self.player_controls.load(None, "", ())
            self._update_edit_buttons()
            return
        out_dir = Path(config.OUTPUT_DIR) / f"job_{job_id}"
        summary = f"<b>#{row['id']} {html.escape(row['file_name'])}</b> — {row['status']}"
        if row["error"]:
            summary += f" · {html.escape(row['error'])}"
        lines = [
            summary,
            html.escape(row["file_path"]),
            html.escape(
                f"model {row['model_size']} · created {row['created_at']}"
                + (
                    f" · took {_elapsed(row['started_at'], row['finished_at'])}"
                    if row["started_at"]
                    else ""
                )
            ),
        ]
        if row["status"] == "done":
            lines.append(html.escape(str(out_dir)))
        text, spans = self._build_preview(job_id)
        editable = row["status"] == "done"
        draft = self._drafts.get(job_id)
        if draft is not None:
            # Mid-edit: show the user's buffer and map highlights from it, so
            # playback stays in sync with the text in front of them.
            text = draft
            spans = self._spans_from_text(draft)
            lines.append("<b>unsaved caption edits</b> — press Ctrl+S to save")
        self.meta.setText("<br>".join(lines))
        self._show_preview(text, editable=editable, job_id=job_id)
        # Play the job's own source file — it exists while a job is still
        # running, so you can listen along as the captions are produced.
        self.player_controls.load(job_id, row["file_path"], spans)
        self._update_edit_buttons()

    # --- drag & drop ---------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            files = _expand_media([u.toLocalFile() for u in event.mimeData().urls()])
            if files:
                event.acceptProposedAction()
                self.statusBar().showMessage(f"drop to queue {len(files)} file(s)")
                return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.statusBar().clearMessage()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        if self.add_paths(paths):
            event.acceptProposedAction()

    # --- lifecycle -----------------------------------------------------------
    def toggle_live(self) -> None:
        """Global hotkey target: start with the remembered device, or stop."""
        if self._live_worker is not None and self._live_worker.isRunning():
            self.stop_live()
            return
        device = self._live_device
        if device is None:
            device = self._pick_live_device()  # first press: ask (we have focus)
            if device is None:
                return
        self._launch_live(device)

    def closeEvent(self, event) -> None:
        if self._hotkey is not None:
            self._hotkey.stop()
            self._hotkey = None
        self.stop_live()
        if self._drafts:
            answer = QMessageBox.question(
                self, "Unsaved caption edits",
                f"{len(self._drafts)} job(s) have unsaved caption edits.\n\n"
                "Discard them and quit?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._drafts.clear()  # the user chose to discard them
        self.timer.stop()
        self.player_controls.stop()  # silence any preview audio before teardown
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait(5000)  # let the current job wind down
        if self.settings is not None:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("splitter", self.splitter.saveState())
        self.conn.close()
        super().closeEvent(event)


def _setup_logging() -> None:
    """Log to data/logs/captioner.log — pythonw has no console to print to."""
    try:
        log_dir = config.DATA_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "captioner.log", encoding="utf-8")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[handler],
        )
    except OSError:  # read-only install: still log somewhere
        logging.basicConfig(level=logging.INFO)


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    self_test = "--self-test" in argv
    argv = [a for a in argv if a != "--self-test"]
    if self_test:
        # A sanity run must never claim the system-wide Ctrl+Alt+L for 1.2s.
        os.environ.setdefault("CAPTIONER_NO_GLOBAL_HOTKEY", "1")

    _setup_logging()
    app = QApplication(argv)
    _apply_theme()  # violet-glass dark theme, regardless of the OS theme
    icon_path = _ROOT / "captioner.ico"
    if icon_path.exists():
        from PySide6.QtGui import QIcon

        app.setWindowIcon(QIcon(str(icon_path)))

    # Under pythonw an uncaught exception dies silently; log it and show a
    # dialog so a crash is never a mystery.
    def _excepthook(kind, value, tb) -> None:
        logger.critical("uncaught exception", exc_info=(kind, value, tb))
        traceback.print_exception(kind, value, tb, file=sys.stderr)
        try:
            QMessageBox.critical(
                None, "Captioner error",
                f"{kind.__name__}: {value}\n\n"
                f"Details were written to {config.DATA_DIR / 'logs' / 'captioner.log'}",
            )
        except Exception:
            pass

    sys.excepthook = _excepthook

    try:
        win = Dashboard()
        win.show()
    except Exception:
        _excepthook(*sys.exc_info())
        return 1
    if self_test:
        QTimer.singleShot(1200, app.quit)
    code = app.exec()
    if self_test:
        print("dashboard self-test OK: window built, rendered, and closed cleanly")
    return code


if __name__ == "__main__":
    sys.exit(main())
