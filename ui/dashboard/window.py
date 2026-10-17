"""The Dashboard window: menus, layout, status bar, close behaviour.

Behaviour is split across mixins by concern (jobs, preview, live, worker);
this module owns the frame they hang on and nothing else. The table re-reads
the SQLite queue once a second, so the worker can live in a background thread
or a separate console process - the UI does not care which.
"""

import logging
import os

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QFontDatabase,
    QKeySequence,
    QPalette,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from captioner import config, db, features, worker
from captioner.languages import SPOKEN_LANGUAGES, spoken_name
from captioner.translate import LANGUAGES

from .constants import APP_NAME, APP_VERSION, COLUMNS, MODEL_CHOICES
from .formatting import _expand_media
from .jobs import JobQueueMixin
from .live import LiveMixin
from .preferences import Preferences, PreferencesDialog
from .preview import PreviewMixin
from .services import ServiceMixin
from .theme import (
    apply_theme,
    faint_style,
    is_dark,
    px,
    status_colors,
)
from .widgets import PlaybackControls
from .worker import WorkerMixin, WorkerThread

logger = logging.getLogger(__name__)


class Dashboard(JobQueueMixin, PreviewMixin, LiveMixin, WorkerMixin, ServiceMixin, QMainWindow):
    """The batch captioning window."""

    def __init__(self, conn=None, persist_settings: bool = True, settings=None):
        """``settings`` overrides where preferences are stored (tests point it
        at a temporary file); ``persist_settings=False`` keeps everything in
        memory and never writes to the user's registry."""
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — batch captions")
        self.resize(1040, 720)
        self.setAcceptDrops(True)

        self.conn = conn if conn is not None else db.connect()
        if settings is not None:
            self.settings = settings
        else:
            self.settings = QSettings("captioner", "dashboard") if persist_settings else None
        # Appearance, accessibility and the queue defaults the user chose last
        # time; applied before the widgets are built so the first paint already
        # uses the right theme, text size and option values.
        self.prefs = Preferences.load(self.settings, model_default=config.BATCH_MODEL_SIZE)
        self._theme_labels: list = []  # labels whose style follows the theme
        self.apply_preferences()
        self.worker_thread: WorkerThread | None = None
        self._signature = None
        self._pending_before = 0
        self.last_notification: str | None = None
        # Unsaved caption edits, keyed by job id: switching jobs keeps them
        # (a modal prompt on every selection change would be worse than a
        # visible reminder plus Ctrl+S).
        self._drafts: dict = {}
        self._preview_job_id: int | None = None
        self._loading_preview = False
        self._cancel_pending = False
        self._live_worker = None  # LiveWorker when live captions run
        self._overlay = None  # LiveOverlay when live captions run
        self._live_device = None  # last device the user picked for live
        self._hotkey = None  # GlobalHotkey, created in _build_menus
        self._api_server = None  # captioner.api server while the web UI runs
        self._api_thread = None
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
        if self.prefs.autostart:
            self.start_worker()
        if self.recovered:
            self.statusBar().showMessage(
                f"recovered {self.recovered} job(s) a crashed worker left running — "
                "they are queued again",
                10000,
            )

    def _add_action(
        self, menu: QMenu, text: str, shortcut: str | None, slot, tip: str = ""
    ) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.setStatusTip(tip or text)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._add_action(
            file_menu,
            "&Add files…",
            "Ctrl+O",
            self.add_files,
            "Queue one or more media files for captioning",
        )
        self._add_action(
            file_menu,
            "Add f&older…",
            "Ctrl+Shift+O",
            self.add_folder,
            "Queue every media file inside a folder (recursive)",
        )
        file_menu.addSeparator()
        self._add_action(file_menu, "E&xit", "Ctrl+Q", self.close)

        job_menu = self.menuBar().addMenu("&Job")
        self._add_action(
            job_menu,
            "&Retry selected",
            "Ctrl+R",
            self.retry_selected,
            "Send the selected job back to the queue",
        )
        self._add_action(
            job_menu,
            "&Remove selected",
            "Delete",
            self.remove_selected,
            "Drop the selected job from the queue (output files stay on disk)",
        )
        self._add_action(
            job_menu,
            "&Cancel running job",
            None,
            self.cancel_running_job,
            "Stop the job being transcribed and leave it canceled",
        )
        self._add_action(
            job_menu,
            "Recover &stuck jobs",
            None,
            self.recover_stuck,
            "Requeue jobs a crashed worker left marked running",
        )
        job_menu.addSeparator()
        self._add_action(
            job_menu,
            "&Save caption edits",
            "Ctrl+S",
            self.save_caption_edits,
            "Rewrite captions.srt and captions.vtt with your edits",
        )
        self._add_action(
            job_menu,
            "Dis&card caption edits",
            "Ctrl+Shift+Z",
            self.revert_caption_edits,
            "Throw away unsaved caption edits",
        )
        job_menu.addSeparator()
        self._add_action(
            job_menu,
            "&Translate selection…",
            "Ctrl+T",
            self.translate_selected,
            "Write subtitles in another language, without transcribing again",
        )
        job_menu.addSeparator()
        self._add_action(job_menu, "&Open output folder", "Ctrl+E", self.open_output)
        self._add_action(job_menu, "&Copy file path", "Ctrl+Shift+C", self.copy_path)
        self._add_action(job_menu, "Open &outputs directory", None, self.reveal_outputs_dir)

        play_menu = self.menuBar().addMenu("&Playback")
        self._add_action(
            play_menu, "&Play / Pause", "Space", self.toggle_play, "Play the selected job's audio"
        )
        self._add_action(
            play_menu, "&Stop", "Ctrl+.", self.stop_playback, "Stop playback and rewind"
        )
        play_menu.addSeparator()
        self._add_action(
            play_menu,
            "Back 5 seconds",
            "Ctrl+Left",
            lambda: self.player_controls.seek_relative(-5000),
        )
        self._add_action(
            play_menu,
            "Forward 5 seconds",
            "Ctrl+Right",
            lambda: self.player_controls.seek_relative(5000),
        )
        self._add_action(
            play_menu,
            "&Mute",
            "Ctrl+M",
            self.player_controls.toggle_mute,
            "Mute or unmute the preview audio",
        )

        live_menu = self.menuBar().addMenu("&Live")
        self._add_action(
            live_menu,
            "&Start live captions…",
            None,
            self.start_live,
            "Caption microphone or system audio live in a floating bar",
        )
        self.act_live_model = live_menu.addAction("Live &model: …")
        self.act_live_model.triggered.connect(self.choose_live_model)
        self._refresh_live_model_label()
        self.act_live_translate = live_menu.addAction("Live translation: off")
        self.act_live_translate.triggered.connect(self.choose_live_language)
        self._refresh_live_language_label()
        self.act_stop_live = self._add_action(
            live_menu,
            "Sto&p live captions",
            None,
            self.stop_live,
            "Stop the floating live caption bar",
        )
        self.act_stop_live.setEnabled(False)
        self.act_toggle_live_menu = self._add_action(
            live_menu,
            "Toggle live captions any&where",
            "Ctrl+Alt+Y",
            self.toggle_live,
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
                    (MOD_CONTROL | MOD_ALT, VK_L),  # Ctrl+Alt+L (preferred)
                    (MOD_CONTROL | MOD_ALT | MOD_SHIFT, VK_L),  # fallback
                ]
                for modifiers, vk in candidates:
                    hk = GlobalHotkey(modifiers, vk)
                    hk.activated.connect(self.toggle_live)
                    hk.registration_failed.connect(lambda m: self.statusBar().showMessage(m, 8000))
                    hk.start()
                    if hk.error:
                        hk.stop()
                        continue  # try the next combination
                    self._hotkey = hk
                    # defer: the real status bar installs after the menus are built
                    QTimer.singleShot(
                        0,
                        lambda mods=modifiers, key=vk: self.statusBar().showMessage(
                            f"global hotkey ready: {describe(mods, key)} toggles live captions",
                            6000,
                        ),
                    )
                    break
                else:
                    logger.warning("no global hotkey combination was free")
            except Exception as exc:  # non-Windows or AV software blocking it
                logger.warning("global hotkey unavailable: %s", exc)
                self._hotkey = None

        view_menu = self.menuBar().addMenu("&View")
        self._add_action(
            view_menu,
            "&Appearance and accessibility…",
            "Ctrl+,",
            self.open_preferences,
            "Theme, text size and quiet mode",
        )
        view_menu.addSeparator()
        self._add_action(
            view_menu,
            "&Larger text",
            "Ctrl+=",
            lambda: self.nudge_text_size(0.1),
            "Scale the whole interface up by 10%",
        )
        self._add_action(
            view_menu,
            "&Smaller text",
            "Ctrl+-",
            lambda: self.nudge_text_size(-0.1),
            "Scale the whole interface down by 10%",
        )
        self._add_action(
            view_menu,
            "&Normal text size",
            "Ctrl+0",
            self.reset_text_size,
            "Back to 100%",
        )
        view_menu.addSeparator()
        self.act_contrast = self._add_action(
            view_menu,
            "&High contrast theme",
            "Ctrl+Shift+H",
            self.toggle_contrast,
            "Opaque surfaces, brighter text, stronger focus rings",
        )
        self.act_contrast.setCheckable(True)
        self.act_contrast.setChecked(self.prefs.theme == "contrast")

        tools_menu = self.menuBar().addMenu("&Tools")
        self.act_api = self._add_action(
            tools_menu,
            "Toggl&e web API + UI",
            None,
            self.toggle_api,
            "Serve the queue over HTTP on this machine, with a browser UI",
        )

        help_menu = self.menuBar().addMenu("&Help")
        self._add_action(
            help_menu,
            "&Keyboard shortcuts",
            "F1",
            self.show_shortcuts,
            "Every shortcut, and what it does",
        )
        self._add_action(help_menu, "&About", None, self.show_about)

    def _build_central(self) -> None:
        central = QWidget()
        col = QVBoxLayout(central)
        col.setContentsMargins(20, 14, 20, 12)
        col.setSpacing(12)

        #calm header: primary actions + worker pill + settings
        header = QHBoxLayout()
        header.setSpacing(10)

        self.btn_add = QPushButton("＋  Add files")
        self.btn_add.setToolTip(
            "Queue media files for captioning (Ctrl+O) — or just drop them anywhere in this window"
        )
        self.btn_add.setAccessibleName("Add files")
        self.btn_add.setAccessibleDescription(
            "Queue media files for captioning. Shortcut Control O."
        )
        self.btn_add.clicked.connect(self.add_files)
        header.addWidget(self.btn_add)

        self.btn_worker = QPushButton("▶  Start worker")
        self.btn_worker.setToolTip("Process queued jobs inside this window")
        self.btn_worker.setAccessibleName("Start or stop the worker")
        self.btn_worker.setAccessibleDescription(
            "Processes queued jobs inside this window. Turns into Stop worker while running."
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
        self.btn_options.setAccessibleName("Options for newly added files")
        self.btn_options.setToolTip(
            "Model, spoken language, translation, speakers, watch folder, "
            "notifications (Alt+O)"
        )
        options_menu = QMenu(self.btn_options)
        self._options = QWidgetAction(options_menu)
        panel = QWidget()
        form = QVBoxLayout(panel)
        form.setContentsMargins(12, 8, 12, 10)
        form.setSpacing(8)

        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        model_label = QLabel("Model")
        self._style_faint(model_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_CHOICES)
        self.model_combo.setCurrentText(self.prefs.model)
        self.model_combo.setAccessibleName("Model for newly added files")
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

        language_row = QHBoxLayout()
        language_row.setSpacing(8)
        language_label = QLabel("Spoken language")
        self._style_faint(language_label)
        self.language_combo = QComboBox()
        self.language_combo.addItem("Detect automatically", "")
        for code, name in sorted(SPOKEN_LANGUAGES.items(), key=lambda kv: (kv[1], kv[0])):
            self.language_combo.addItem(f"{name} ({code})", code)
        index = self.language_combo.findData(self.prefs.language)
        if index > 0:
            self.language_combo.setCurrentIndex(index)
        self.language_combo.setAccessibleName("Spoken language of new files")
        self.language_combo.setToolTip(
            f"What language is spoken in the recording, for newly added files "
            f"({len(SPOKEN_LANGUAGES)} are recognised).\n"
            "Detecting is the default and is right almost always; naming it\n"
            "skips the detection step and is more accurate for a language\n"
            "Whisper might confuse with another - and it is what tells the\n"
            "translator what it is reading."
        )
        self.language_combo.setMinimumWidth(190)
        language_row.addWidget(language_label)
        language_row.addWidget(self.language_combo)
        self.language_hint = QLabel("")
        self.language_hint.setWordWrap(True)
        self._style_faint(self.language_hint)
        self.language_hint.hide()
        form.addLayout(language_row)
        form.addWidget(self.language_hint)

        self.diarize_check = QCheckBox("Identify speakers")
        self.diarize_check.setChecked(self.prefs.diarize)
        self.diarize_check.setAccessibleName("Identify speakers")
        self.diarize_check.setToolTip(
            "Label caption lines with speakers ([SPEAKER_00]: …) using pyannote.\n"
            "Applies to newly added files; adds a diarization pass per job\n"
            "and needs the HF token in .env on first use."
        )
        form.addWidget(self.diarize_check)

        self.align_check = QCheckBox("Align word timings (wav2vec2)")
        self.align_check.setChecked(self.prefs.align)
        self.align_check.setAccessibleName("Align word timings with wav2vec2")
        self.align_check.setToolTip(
            "Re-time every word with a character alignment model before the\n"
            "captions are grouped — sharper subtitle in/out points than\n"
            "Whisper's own estimates. Adds a short pass per job; the first use\n"
            "downloads ~380 MB. Applies to newly added files."
        )
        form.addWidget(self.align_check)

        translate_row = QHBoxLayout()
        translate_row.setSpacing(8)
        self.translate_label = self._style_faint(QLabel("Translate captions"))
        self.translate_combo = QComboBox()
        self.translate_combo.addItem("Off", "")
        for code, name in sorted(LANGUAGES.items(), key=lambda kv: (kv[1], kv[0])):
            self.translate_combo.addItem(f"{name} ({code})", code)
        index = self.translate_combo.findData(self.prefs.translate_to)
        if index > 0:
            self.translate_combo.setCurrentIndex(index)
        self.translate_combo.setAccessibleName("Translate new captions into")
        self.translate_combo.setToolTip(
            f"Also write captions.<language>.srt/.vtt for newly added files "
            f"({len(LANGUAGES)} languages).\n"
            "Each unique line is translated once and cached, so re-running a\n"
            "job costs nothing. First use downloads ~2.4 GB.\n"
            "The transcript is read as the spoken language above, so a Hindi\n"
            "recording can go straight to French."
        )
        self.translate_combo.setMinimumWidth(190)
        translate_row.addWidget(self.translate_label)
        translate_row.addWidget(self.translate_combo)
        form.addLayout(translate_row)

        # A build can be missing what an option needs (the slim installer leaves
        # PyTorch out on purpose). Say so on the control instead of letting a job
        # discover it: a disabled switch with the reason in its label and tooltip
        # is a feature list, a job that fails in a log is a support ticket.
        self._mark_unavailable(self.diarize_check, "speakers")
        self._mark_unavailable(self.align_check, "align")
        self._mark_unavailable(self.translate_combo, "translate_local",
                               label_widget=self.translate_label)

        self.watch_check = QCheckBox("Watch uploads folder")
        self.watch_check.setChecked(self.prefs.watch)
        self.watch_check.setAccessibleName("Watch the uploads folder")
        self.watch_check.setToolTip(
            "Automatically queue media that appears in the uploads folder "
            f"({config.UPLOADS_DIR}).\n"
            "Removing a job while watching lets its file be queued again."
        )
        form.addWidget(self.watch_check)
        self.notify_check = QCheckBox("Notify when done")
        self.notify_check.setChecked(self.prefs.notify)
        self.notify_check.setAccessibleName("Notify when the queue finishes")
        self.notify_check.setToolTip(
            "Beep and flash the taskbar entry when the queue drains.\n"
            "Quiet mode in Appearance and accessibility… turns this off \n"
            "for good."
        )
        form.addWidget(self.notify_check)
        self._wire_option_persistence()
        if features.unavailable():
            # A remembered option from a different install (or a different
            # machine) must not stay in the preferences pointing at a feature
            # that is not here.
            self._persist_queue_defaults()
        self._options.setDefaultWidget(panel)
        options_menu.addAction(self._options)
        self._options_menu = options_menu
        self.btn_options.clicked.connect(
            lambda: options_menu.exec(
                self.btn_options.mapToGlobal(self.btn_options.rect().bottomRight())
            )
        )
        header.addWidget(self.btn_options)
        col.addLayout(header)

        #queue summary: slim progress, no duplicate counts row
        summary = QHBoxLayout()
        summary.setSpacing(12)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("no jobs yet")
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setToolTip("Settled jobs (done + error) out of everything in the queue")
        self.progress.setAccessibleName("Queue progress")
        self.counts_label = QLabel("")
        self._style_faint(self.counts_label, extra=f"font-size: {px(12)}; ")
        self.counts_label.setAccessibleName("Queue counts")
        summary.addWidget(self.progress, stretch=1)
        summary.addWidget(self.counts_label)
        col.addLayout(summary)

        #table + preview splitter
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
        self.table.setAccessibleName("Job queue")
        self.table.setAccessibleDescription(
            "One row per job. Use the arrow keys to move, Retry and Delete from the Job menu."
        )
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
        self.preview.setAccessibleName("Captions of the selected job")
        self.preview.setAccessibleDescription(
            "Editable captions. Control S saves your edits, Control Shift Z discards them."
        )
        self.preview.textChanged.connect(self._on_preview_edited)
        dcol.addWidget(self.preview, stretch=1)

        # Audio playback for the selected job - slim bar docked at the very
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
        self.btn_save_edits.setAccessibleName("Save caption edits")
        self.btn_save_edits.clicked.connect(self.save_caption_edits)
        self.btn_revert_edits = QPushButton("Revert")
        self.btn_revert_edits.setProperty("secondary", True)
        self.btn_revert_edits.setToolTip("Throw away unsaved caption edits (Ctrl+Shift+Z)")
        self.btn_revert_edits.setAccessibleName("Discard caption edits")
        self.btn_revert_edits.clicked.connect(self.revert_caption_edits)
        btns.addWidget(self.btn_save_edits)
        btns.addWidget(self.btn_revert_edits)
        self.btn_translate = QPushButton("Translate…")
        self.btn_translate.setProperty("secondary", True)
        self.btn_translate.setToolTip(
            "Write captions.<language>.srt/.vtt for the selected job without\n"
            "transcribing it again (also in the Job menu)"
        )
        self.btn_translate.setAccessibleName("Translate the selected job")
        self.btn_translate.clicked.connect(self.translate_selected)
        btns.addWidget(self.btn_translate)
        dcol.addLayout(btns)

        splitter.addWidget(details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.splitter = splitter
        col.addWidget(splitter, stretch=1)
        col.addWidget(self.player_controls)  # docked at the bottom

        self.setCentralWidget(central)
        # A predictable keyboard path: header actions, then the queue, then the
        # captions, then the buttons under them, then the player.
        for first, second in (
            (self.btn_add, self.btn_worker),
            (self.btn_worker, self.btn_options),
            (self.btn_options, self.table),
            (self.table, self.preview),
            (self.preview, self.btn_save_edits),
            (self.btn_save_edits, self.btn_revert_edits),
            (self.btn_revert_edits, self.btn_translate),
            (self.btn_translate, self.player_controls),
        ):
            self.setTabOrder(first, second)

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.paths_label = QLabel(f"outputs: {config.OUTPUT_DIR}")
        self.paths_label.setToolTip(f"database: {config.DB_PATH}\noutputs: {config.OUTPUT_DIR}")
        self.paths_label.setAccessibleName(f"Output folder: {config.OUTPUT_DIR}")
        bar.addPermanentWidget(self.paths_label)
        bar.showMessage("Ready — add a file to caption.")
        # The status bar is the live region of this window: it is where every
        # long-running action reports itself, so a screen reader follows it.
        bar.setAccessibleName("Status")

    #preferences, appearance and accessibility

    def _style_faint(self, label: QLabel, extra: str = "") -> QLabel:
        """Give a label the theme's muted colour and remember it for re-theming."""
        label.setStyleSheet(faint_style(extra))
        self._theme_labels.append((label, extra))
        return label

    def apply_preferences(self) -> None:
        """Apply the theme, the text scale and the queue defaults.

        Called at startup, whenever the shortcuts change something, and live
        from the preferences dialog as the user drags the text-size slider.
        """
        self.setStyleSheet(apply_theme(self.prefs.theme, self.prefs.text_scale))
        for label, extra in getattr(self, "_theme_labels", []):
            label.setStyleSheet(faint_style(extra))
        action = getattr(self, "act_contrast", None)
        if action is not None:
            action.setChecked(self.prefs.theme == "contrast")
        self._update_language_hint()
        # Deliberately no save here: this runs on every repaint, including the
        # live preview inside the preferences dialog, where Cancel has to be
        # able to put everything back. The option widgets save when they change.

    def nudge_text_size(self, delta: float) -> None:
        """Scale the interface up or down (Ctrl+= / Ctrl+-)."""
        self.apply_preferences_with_scale(self.prefs.scale_text(delta))

    def reset_text_size(self) -> None:
        self.prefs.reset_text_scale()
        self.apply_preferences_with_scale("100%")

    def apply_preferences_with_scale(self, label: str) -> None:
        """Repaint after a change made outside the dialog, then say so."""
        self.apply_preferences()
        self.statusBar().showMessage(f"text size {label}", 3000)

    def toggle_contrast(self) -> None:
        """Ctrl+Shift+H: high contrast on or off."""
        name = self.prefs.toggle_contrast()
        self.apply_preferences()
        self.statusBar().showMessage(f"{name} theme", 3000)

    def open_preferences(self) -> None:
        """Ctrl+,: appearance and accessibility, previewed live."""
        dialog = PreferencesDialog(self.prefs, self)
        dialog.exec()
        self.apply_preferences()

    SHORTCUTS = (
        ("Ctrl+O", "Add files"),
        ("Ctrl+Shift+O", "Add a folder"),
        ("Ctrl+R", "Retry the selected job"),
        ("Delete", "Remove the selected job from the queue"),
        ("Ctrl+S", "Save caption edits"),
        ("Ctrl+Shift+Z", "Discard caption edits"),
        ("Ctrl+E", "Open the output folder"),
        ("Ctrl+T", "Translate the selected job"),
        ("Space", "Play or pause the preview"),
        ("Ctrl+.", "Stop playback"),
        ("Ctrl+Left / Ctrl+Right", "Seek 5 seconds"),
        ("Ctrl+M", "Mute the preview"),
        ("Ctrl+Alt+Y", "Toggle live captions"),
        ("Ctrl+,", "Appearance and accessibility"),
        ("Ctrl+= / Ctrl+-", "Larger / smaller text"),
        ("Ctrl+0", "Normal text size"),
        ("Ctrl+Shift+H", "High contrast theme"),
        ("Alt+O", "Open the Options popover"),
        ("Tab / Shift+Tab", "Move between controls"),
        ("F1", "This list"),
    )

    def show_shortcuts(self) -> None:
        """F1: the keyboard map, in a dialog small enough to read."""
        rows = "".join(
            f"<tr><td style='padding:2px 14px 2px 0'><b>{key}</b></td>"
            f"<td style='padding:2px 0'>{text}</td></tr>"
            for key, text in self.SHORTCUTS
        )
        box = QMessageBox(self)
        box.setWindowTitle("Keyboard shortcuts")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(f"<b>Keyboard shortcuts</b><table>{rows}</table>")
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _mark_unavailable(self, control, extra_key: str, label_widget=None) -> None:
        """Grey out an option this build cannot run, and explain it in place.

        The feature keeps its own name (so it is still searchable, and a screen
        reader still reads the feature rather than a footnote), with the reason
        after it: *not in this build*. The tooltip then says which packages are
        missing, what happens instead, and how to install them.
        """
        extra = features.find(extra_key)
        if features.available(extra):
            return
        text = features.label(extra)
        if isinstance(control, QCheckBox):
            control.setChecked(False)
            control.setText(text)
        elif label_widget is not None:
            label_widget.setText(text)
        if isinstance(control, QComboBox):
            control.setCurrentIndex(0)   # "Off": a job must not ask for it
        control.setEnabled(False)
        control.setToolTip(features.tooltip(extra, control.toolTip()))
        control.setAccessibleName(text)
        control.setAccessibleDescription(features.tooltip(extra, ""))

    def _wire_option_persistence(self) -> None:
        """Every option control saves its value as soon as it changes.

        The options double as defaults for the next file, so remembering them
        is the whole point: the popover is a preference editor that happens to
        also apply to the current drop.
        """
        self.model_combo.currentTextChanged.connect(self._persist_queue_defaults)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        self.translate_combo.currentIndexChanged.connect(self._persist_queue_defaults)
        self.diarize_check.toggled.connect(self._persist_queue_defaults)
        self.align_check.toggled.connect(self._persist_queue_defaults)
        self.watch_check.toggled.connect(self._persist_queue_defaults)
        self.notify_check.toggled.connect(self._persist_queue_defaults)

    def _on_language_changed(self) -> None:
        self._update_language_hint()
        self._persist_queue_defaults()

    def _update_language_hint(self) -> None:
        """Say so when the spoken language cannot be translated.

        The recogniser knows 100 languages and the translator another 100; the
        overlap is smaller than either. Telling the user before they queue a
        file beats a job that finishes with a note instead of a translation.
        """
        hint = getattr(self, "language_hint", None)
        if hint is None:
            return
        code = self.language_combo.currentData() or ""
        target = self.translate_combo.currentData() or ""
        if code and target and code not in LANGUAGES:
            hint.setText(
                f"{spoken_name(code)} can be transcribed, but the translator has no "
                f"{spoken_name(code)} model yet - set Spoken language to a language it "
                "knows, or leave the translation off."
            )
            hint.show()
        else:
            hint.hide()

    def _persist_queue_defaults(self) -> None:
        """Copy the Options popover into the preferences and save them."""
        if not hasattr(self, "model_combo"):
            return
        self.prefs.model = self.model_combo.currentText()
        self.prefs.language = self.language_combo.currentData() or ""
        self.prefs.translate_to = self.translate_combo.currentData() or ""
        self.prefs.diarize = self.diarize_check.isChecked()
        self.prefs.align = self.align_check.isChecked()
        self.prefs.watch = self.watch_check.isChecked()
        self.prefs.notify = self.notify_check.isChecked()
        self.prefs.save()

    def _restore_geometry(self) -> None:
        if self.settings is None:
            return
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self.settings.value("splitter")
        if state is not None:
            self.splitter.restoreState(state)

    def _is_dark(self) -> bool:
        """Today's answer comes from the theme, not from the palette guess."""
        if self.prefs is not None:
            return is_dark(self.prefs.theme)
        return self.palette().color(QPalette.ColorRole.Window).lightness() < 128

    def _status_color(self, status: str) -> QColor:
        return QColor(status_colors().get(status, "#888888"))

    def about_text(self) -> str:
        """The body of Help → About: identity, versions, this build, paths.

        Kept as a function rather than built inside the dialog so a test can
        read it without opening a modal, and so the title card and the dialog
        credit the same project.
        """
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
        return (
            f"<b>{APP_NAME}</b> — local AI captions: batch + live + speakers<br>"
            f"<span style='{faint_style()}'>project by sharat</span><p>"
            f"version {APP_VERSION} · PySide6 {pyside} · faster-whisper {whisper}<br>"
            f"batch model default: {config.BATCH_MODEL_SIZE} · "
            f"live model default: {config.LIVE_MODEL_SIZE}"
            # What this build is, not just what the project can do: the two
            # differ on purpose (see captioner/features.py).
            f"<p>{features.summary()}"
            f"<p>database: {config.DB_PATH}<br>outputs: {config.OUTPUT_DIR}<br>"
            f"log: {config.DATA_DIR / 'logs' / 'captioner.log'}"
        )

    def show_about(self) -> None:
        QMessageBox.about(self, f"About {APP_NAME}", self.about_text())

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

    def closeEvent(self, event) -> None:
        self.close_services()  # stop the API listener with the window
        if self._hotkey is not None:
            self._hotkey.stop()
            self._hotkey = None
        self.stop_live()
        if self._drafts:
            answer = QMessageBox.question(
                self,
                "Unsaved caption edits",
                f"{len(self._drafts)} job(s) have unsaved caption edits.\n\nDiscard them and quit?",
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
        self._persist_queue_defaults()  # last chance to remember the options
        self.conn.close()
        super().closeEvent(event)
