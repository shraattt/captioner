"""The caption preview pane, its playback controls and caption edits."""

import html
import sqlite3
from pathlib import Path

from PySide6.QtWidgets import (
    QMessageBox,
)

from captioner import config, editor

from .formatting import _elapsed


class PreviewMixin:
    """The preview pane, playback and caption edits."""

    def toggle_play(self) -> None:
        """Play/pause the selected job's audio (Space, or the Play button)."""
        if not self.player_controls.has_source:
            self.statusBar().showMessage("select a job with audio first", 3000)
            return
        self.player_controls.toggle_play()

    def stop_playback(self) -> None:
        self.player_controls.stop()

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

    def _show_preview(self, text: str, *, editable: bool, job_id: int | None) -> None:
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
        style = self.btn_save_edits.style()  # re-read the new properties
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
            QMessageBox.warning(self, "Cannot save caption edits", f"{type(exc).__name__}: {exc}")
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
        # Play the job's own source file - it exists while a job is still
        # running, so you can listen along as the captions are produced.
        self.player_controls.load(job_id, row["file_path"], spans)
        self._update_edit_buttons()
