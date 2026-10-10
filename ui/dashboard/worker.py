"""The in-app worker thread, and the controls that drive it.

Two ways to run the queue exist by design: this thread (so a single-window
user never opens a console) and the separate `python cli.py work` process.
Both go through captioner.worker, so the queue semantics are identical.
"""

import threading

from PySide6.QtCore import QThread

from captioner import config, db, worker


class WorkerThread(QThread):
    """Runs the queue loop in the background; owns its own DB connection."""


"""Runs the queue loop in the background; owns its own DB connection."""


class WorkerMixin:
    """In-app worker control (start, stop, cancel)."""

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

    def _set_worker_pill(self, running: bool) -> None:
        """Worker status as a pill: violet while running, neutral otherwise."""
        self.worker_state.setText("●  running" if running else "●  stopped")
        self.worker_state.setProperty("running", running)
        style = self.worker_state.style()
        style.unpolish(self.worker_state)
        style.polish(self.worker_state)

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
