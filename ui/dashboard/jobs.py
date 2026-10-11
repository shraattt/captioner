"""Queue actions: adding files, retrying, removing, refreshing the table."""

import os
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMenu,
    QMessageBox,
    QTableWidgetItem,
)

from captioner import automation, config, worker

from .constants import MEDIA_FILTER, STALL_WARNING_SECONDS
from .formatting import _elapsed, _expand_media, _human_age, _stage_label


class JobQueueMixin:
    """Queue actions, the table and its refresh loop."""

    def selected_job_id(self) -> int | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return int(item.text()) if item else None

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
        """Queue media found in `paths` (files and folders); returns the count.

        A file whose content is already queued, running or finished is skipped:
        re-adding the same recording under a second name should not buy it a
        second transcription. The status bar names what was skipped, so it is
        never a silent no-op.
        """
        model = self.model_combo.currentText()
        options = self._queue_options()
        skipped = []
        files = []
        for path in _expand_media(paths):
            existing = worker.find_duplicate(self.conn, path)
            if existing is None:
                files.append(path)
            else:
                skipped.append((path.name, existing["id"]))
        for path in files:
            worker.enqueue(self.conn, str(path), model_size=model, **options)
        if files:
            self._signature = None
            self.refresh()
            extras = [
                name
                for name, on in (
                    ("speakers", options["diarize"]),
                    ("alignment", options["align"]),
                    (f"translate:{options['translate_to']}", bool(options["translate_to"])),
                )
                if on
            ]
            suffix = f" [{', '.join(extras)}]" if extras else ""
            note = f", skipped {len(skipped)} already queued" if skipped else ""
            self.statusBar().showMessage(
                f"queued {len(files)} file(s) with {model}{suffix}{note}", 4000
            )
        elif skipped:
            name, job_id = skipped[0]
            self.statusBar().showMessage(
                f"already queued: {name} (job #{job_id}) — nothing added", 6000
            )
        else:
            self.statusBar().showMessage("nothing added — no media files found", 4000)
        return len(files)

    def cancel_running_job(self) -> bool:
        """Abort the running job, whoever owns it.

        This window's own worker is stopped through its event. A job owned by
        another process (run_worker.bat, the CLI, the API) is asked to stop
        through the cancel_requested column, which that worker polls between
        segments — so the button finally works for a worker started elsewhere.
        """
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self._cancel_pending = True
            self.btn_cancel.setEnabled(False)
            self.statusBar().showMessage(
                "canceling — the worker stops at the next segment boundary…", 6000
            )
            return True

        row = self.conn.execute(
            "SELECT id FROM jobs WHERE status = 'running' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            self.statusBar().showMessage("no job is running", 3000)
            return False
        outcome = worker.request_cancel(self.conn, row["id"])
        if outcome == "requested":
            self._signature = None
            self.refresh()
            self.statusBar().showMessage(
                f"asked job #{row['id']} (running in another process) to stop — "
                "it finishes the segment it is on",
                6000,
            )
            return True
        self.statusBar().showMessage(
            f"job #{row['id']} already finished; nothing to stop", 4000
        )
        self.refresh()
        return False

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
        row = self.conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
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
                self,
                "Job running",
                f"Job #{job_id} is being transcribed.\n\nStop the worker before removing it.",
            )
            return
        answer = QMessageBox.question(
            self,
            "Remove job",
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
            row = self.conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            status = row["status"] if row else "missing"
            QMessageBox.information(
                self,
                "No output yet",
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
        row = self.conn.execute("SELECT file_path FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row:
            QApplication.clipboard().setText(row["file_path"])
            self.statusBar().showMessage("file path copied", 2500)

    def _show_context_menu(self, point) -> None:
        """Row menu: all job actions live here instead of permanent buttons."""
        if self.selected_job_id() is None:
            return
        menu = QMenu(self)
        menu.addAction("Open output folder", self.open_output)
        menu.addAction("Copy file path", self.copy_path)
        menu.addSeparator()
        act_cancel = menu.addAction("Cancel running job", self.cancel_running_job)
        act_cancel.setEnabled(self.worker_thread is not None and self.worker_thread.isRunning())
        menu.addAction("Retry job", self.retry_selected)
        act_remove = menu.addAction("Remove job", self.remove_selected)
        act_remove.setProperty("danger", True)  # muted rose in styled menus
        menu.exec(self.table.viewport().mapToGlobal(point))

    #refresh / preview

    def refresh(self) -> None:
        if self.watch_check.isChecked():
            self._scan_watch_folder()
        rows = self.conn.execute(
            "SELECT id, file_name, file_path, status, model_size, progress, stage, "
            "heartbeat_at, error, created_at, started_at, finished_at, "
            "cancel_requested FROM jobs ORDER BY id"
        ).fetchall()
        # Rebuild only when something visible changed (avoids per-second flicker);
        # progress, stage, and a running job's elapsed/staleness are all
        # "visible", and staleness is bucketed to 15s so a quiet heartbeat
        # still refreshes the hint without redrawing every tick.
        signature = tuple(
            (
                r["id"],
                r["status"],
                r["error"],
                r["finished_at"],
                r["progress"],
                r["stage"],
                r["cancel_requested"],
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
            self.conn.execute("SELECT job_id, COUNT(*) FROM captions GROUP BY job_id").fetchall()
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
                    if r["cancel_requested"]:
                        details += " · stopping at the next boundary"
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
                    str(r["id"]),
                    r["file_name"],
                    r["status"],
                    r["model_size"],
                    percent,
                    r["created_at"] or "",
                    duration,
                    details,
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

    def _scan_watch_folder(self) -> int:
        """Queue media that appeared in a watched folder since the last tick.

        The scan lives in captioner.automation, the same code the headless
        `python cli.py watch` uses: data/uploads always counts, and
        CAPTIONER_WATCH_DIRS adds folders. A file that was written to a moment
        ago is left for the next tick, so a half-copied download is never
        queued.
        """
        new_files = automation.find_new_media(self.conn)
        if not new_files:
            return 0
        model = self.model_combo.currentText()
        options = self._queue_options()
        for path in new_files:
            worker.enqueue(self.conn, str(path), model_size=model, **options)
        self._signature = None
        self.statusBar().showMessage(f"watch folder: queued {len(new_files)} new file(s)", 4000)
        return len(new_files)

    def _maybe_notify(self, rows) -> None:
        """Beep/flash once when the queue drains (pending -> settled)."""
        pending = sum(1 for r in rows if r["status"] in ("queued", "running"))
        if self.notify_check.isChecked() and rows and pending == 0 and self._pending_before > 0:
            self.last_notification = datetime.now().strftime("%H:%M:%S")
            QApplication.beep()
            QApplication.alert(self, 5000)
            self.statusBar().showMessage(
                f"batch finished at {self.last_notification} — nothing left to process",
                8000,
            )
        self._pending_before = pending
