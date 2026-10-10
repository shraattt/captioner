"""The dashboard package: one window, split by concern.

    constants    identity, column layout, media filters
    theme        palette, status colors and the stylesheet
    formatting   duration/age/stage helpers
    widgets      the playback control strip
    jobs         queue actions and the table (JobQueueMixin)
    preview      caption preview, playback and editing (PreviewMixin)
    live         live caption sessions (LiveMixin)
    worker       the in-app worker thread and its controls (WorkerMixin)
    window       the Dashboard class that assembles the mixins

Public names are re-exported here, so `from ui.dashboard import Dashboard`
keeps working whichever module a caller wants. Run it with:

    python -m ui.dashboard                # normal window
    python -m ui.dashboard --self-test    # opens and closes itself
"""

import logging
import os
import sys
import traceback
from pathlib import Path

# A bare `python ui/dashboard/__init__.py` run: make the project importable.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from captioner import config

from .constants import APP_NAME, APP_VERSION
from .theme import _apply_theme
from .widgets import PlaybackControls
from .window import Dashboard

__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "Dashboard",
    "PlaybackControls",
    "main",
]

logger = logging.getLogger(__name__)


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
                None,
                "Captioner error",
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
