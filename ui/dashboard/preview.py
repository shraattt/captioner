"""The caption preview pane, its playback controls and caption edits."""

import html
import logging
import sqlite3
from pathlib import Path

from PySide6.QtWidgets import (
    QInputDialog,
    QMessageBox,
)

from captioner import config, editor
from captioner.languages import spoken_name, translation_name
from captioner.translate import LANGUAGES, translate_subtitles

from .formatting import _elapsed

logger = logging.getLogger(__name__)


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

    def _language_line(self, row) -> str:
        """One line describing what was heard and what it was turned into.

        The *detected* language is shown even when the user named one, and it is
        what the translator is handed when the user did not: a detected language
        that disagrees with the audio is something you can see and fix.
        """
        keys = set(row.keys())
        spoken = row["source_lang"] if "source_lang" in keys else ""
        heard = row["detected_lang"] if "detected_lang" in keys else ""
        target = row["translate_to"] or ""
        parts = []
        if spoken:
            parts.append(f"spoken {spoken_name(spoken)}")
        if heard and heard != spoken:
            parts.append(f"heard {spoken_name(heard)}")
        if not spoken and not heard:
            parts.append("language not determined yet" if row["status"] != "done"
                         else "language not recorded (transcribed before 0.8)")
        if target:
            parts.append(f"translated to {translation_name(target)}")
        if not parts:
            return ""
        return "language: " + " · ".join(parts)

    def _update_edit_buttons(self) -> None:
        job_id = self._preview_job_id
        dirty = job_id is not None and job_id in self._drafts
        self.btn_save_edits.setEnabled(dirty)
        self.btn_revert_edits.setEnabled(dirty)
        # Translating reads captions.srt, so it needs a finished job - and the
        # button says which job it would act on.
        done = False
        if job_id is not None:
            row = self.conn.execute(
                "SELECT status, file_name FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            done = row is not None and row["status"] == "done"
            self.btn_translate.setText("Translate…")
            self.btn_translate.setToolTip(
                f"Write a translated copy of job #{job_id}'s captions without "
                "transcribing it again"
                if done
                else "Available once a job has finished (translation reads its captions.srt)"
            )
        self.btn_translate.setEnabled(done)
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

    def translate_selected(self) -> bool:
        """Translate a finished job into another language, no re-transcribe.

        The transcript is already on disk, so this is seconds of work for a
        language the model has cached and a few minutes the first time. Any
        unsaved caption edits are saved first: translating the file on disk
        while the pane shows something else would translate the wrong text.
        """
        job_id = self.selected_job_id()
        if job_id is None:
            self.statusBar().showMessage("select a finished job first", 3000)
            return False
        row = self.conn.execute(
            "SELECT id, file_name, status, diarize, source_lang, detected_lang, "
            "translate_to FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return False
        if row["status"] != "done":
            self.statusBar().showMessage(
                f"job #{job_id} is {row['status']} - translation needs a finished "
                "transcript",
                4000,
            )
            return False
        if job_id in self._drafts and not self.save_caption_edits():
            return False

        # Sorted by display name, like every other language picker here.
        pairs = sorted(LANGUAGES.items(), key=lambda kv: (kv[1], kv[0]))
        labels = [f"{name} ({code})" for code, name in pairs]
        code_of = {label: code for (code, _), label in zip(pairs, labels, strict=True)}
        choice, accepted = QInputDialog.getItem(
            self,
            f"Translate job #{job_id}",
            f"Language for {row['file_name']}",
            labels,
            labels.index(f"{translation_name(row['translate_to'])} ({row['translate_to']})")
            if row["translate_to"] in LANGUAGES else 0,
            False,
        )
        if not accepted:
            return False
        target = code_of[choice]

        out_dir = Path(config.OUTPUT_DIR) / f"job_{job_id}"
        # Read it as the language it was spoken in; English only as a fallback.
        source = row["source_lang"] or row["detected_lang"] or None
        self.statusBar().showMessage(
            f"translating job #{job_id} to {translation_name(target)}…"
        )
        try:
            paths = translate_subtitles(
                out_dir, target, speaker_names=bool(row["diarize"]), source=source
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fatal to the app
            logger.exception("translation of job %s failed", job_id)
            QMessageBox.warning(
                self,
                "Translation failed",
                f"{type(exc).__name__}: {exc}\n\nThe captions themselves are untouched.",
            )
            self.statusBar().showMessage("translation failed", 5000)
            return False
        self.conn.execute(
            "UPDATE jobs SET translate_to = ? WHERE id = ?", (target, job_id)
        )
        self.conn.commit()
        self._signature = None
        self.refresh()
        self.statusBar().showMessage(
            f"wrote {', '.join(path.name for path in paths)} for job #{job_id}", 6000
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
        language = self._language_line(row)
        if language:
            lines.append(html.escape(language))
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
