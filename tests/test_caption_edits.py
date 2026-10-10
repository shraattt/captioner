"""Tests for caption editing and re-export.

Two layers:

* Pure logic — render/parse round-tripping, and the validation that stops an
  edit from corrupting the subtitle files.
* The dashboard flow — the pane is editable only for finished jobs, typing
  marks a draft, Ctrl+S writes the database *and* both subtitle files, and the
  one-second queue refresh never yanks the text out from under the user.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from captioner import config, db, editor, srt, worker
from captioner.segmenter import Block, Word
from ui.dashboard import Dashboard
from ui.dashboard import widgets as dashboard_widgets

CAPTIONS = [
    (0, 1000, "Hello and welcome."),
    (1500, 2500, "This is a test of the captioning pipeline."),
    (3000, 4000, "Thank you for listening."),
]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class FakePlayer(QObject):
    """Minimal stand-in so no real QtMultimedia backend is constructed."""

    position_ms_changed = Signal(int)
    duration_ms_changed = Signal(int)
    player_error = Signal(str)
    playback_state_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.opened = []
        self.volume = 100
        self.muted = False
        self.position_ms = 0
        self.is_playing = False

    def open(self, path):
        self.opened.append(str(path))

    def play(self):
        self.is_playing = True

    def pause(self):
        self.is_playing = False

    def stop(self):
        self.is_playing = False

    def seek(self, ms):
        self.position_ms = int(ms)


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ui.db")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    connection = db.connect()
    yield connection
    connection.close()


@pytest.fixture()
def make_win(qapp, conn, monkeypatch):
    created = []

    def factory():
        fake = FakePlayer()
        monkeypatch.setattr(dashboard_widgets, "Player", lambda parent=None: fake)
        win = Dashboard(conn=conn, persist_settings=False)
        win.timer.stop()
        win.fake_player = fake
        created.append(win)
        return win

    yield factory
    for win in created:
        win.timer.stop()
        win._drafts.clear()  # teardown must never hit the "unsaved edits" prompt
        win.close()


def seed_job(conn, tmp_path, status="done", captions=CAPTIONS):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"not real media")
    job_id = worker.enqueue(conn, str(src), model_size="tiny.en")
    conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
    conn.commit()
    if captions:
        worker._store_captions(
            conn,
            job_id,
            [Block([Word(text, start / 1000, end / 1000)]) for start, end, text in captions],
        )
    return job_id, src


def drafts(**overrides):
    result = [
        editor.CaptionDraft(seq=i, start_ms=s, end_ms=e, text=t)
        for i, (s, e, t) in enumerate(CAPTIONS, start=1)
    ]
    for seq, text in overrides.items():
        result[int(seq.lstrip("s")) - 1].text = text
    return result


#render / parse round-trip


def test_render_shape_matches_the_old_preview_format():
    text, spans = editor.render(drafts())
    assert "1\n00:00:00,000 --> 00:00:01,000\nHello and welcome.\n" in text
    assert "2\n00:00:01,500 --> 00:00:02,500\n" in text
    assert len(spans) == 3


def test_render_and_parse_are_inverse():
    original = drafts()
    text, _ = editor.render(original)
    parsed = editor.parse_preview(text)
    assert parsed == original
    assert editor.render(parsed)[0] == text


def test_spans_point_at_the_caption_bodies():
    text, spans = editor.render(drafts())
    blocks = text.split("\n")
    for (start, end, block_no), (exp_start, exp_end, expected) in zip(
        spans, CAPTIONS, strict=True
    ):
        assert blocks[block_no - 1].startswith("00:")  # the line above is the timing
        assert blocks[block_no] == expected
        assert (start, end) == (exp_start, exp_end)
    assert [s[0] for s in spans] == [c[0] for c in CAPTIONS]


def test_multi_line_caption_bodies_survive_a_round_trip():
    multi = drafts()
    multi[0].text = "Hello and welcome.\nSecond display line."
    text, spans = editor.render(multi)
    blocks = text.split("\n")
    assert blocks[spans[0][2] + 1] == "Second display line."
    # the extra body line shifts every later caption down by one block
    assert blocks[spans[1][2]] == "This is a test of the captioning pipeline."
    assert editor.parse_preview(text) == multi


def test_render_of_nothing_is_empty():
    assert editor.render([]) == ("", [])


#parse validation


def test_parse_rejects_a_caption_with_no_number():
    text, _ = editor.render(drafts())
    broken = text.replace("1\n00:00:00,000", "one\n00:00:00,000")
    with pytest.raises(editor.CaptionParseError, match="caption number"):
        editor.parse_preview(broken)


def test_parse_rejects_a_smashed_timestamp_line():
    text, _ = editor.render(drafts())
    broken = text.replace("-->", "->", 1)
    with pytest.raises(editor.CaptionParseError, match="caption 1"):
        editor.parse_preview(broken)


def test_parse_rejects_an_emptied_caption():
    text = (
        "1\n00:00:00,000 --> 00:00:01,000\n\n"
        "2\n00:00:01,500 --> 00:00:02,500\nSecond.\n"
    )
    with pytest.raises(editor.CaptionParseError, match="caption 1 has no text"):
        editor.parse_preview(text)


def test_parse_rejects_an_empty_editor():
    with pytest.raises(editor.CaptionParseError, match="no captions"):
        editor.parse_preview("   \n\n  \n")


#apply_edits


def test_apply_edits_updates_the_database(conn, tmp_path):
    job_id, _ = seed_job(conn, tmp_path)
    result = editor.apply_edits(conn, job_id, drafts(s2="This is a test of captioning."))

    rows = conn.execute(
        "SELECT seq, text FROM captions WHERE job_id = ? ORDER BY seq", (job_id,)
    ).fetchall()
    assert [r["text"] for r in rows] == [
        "Hello and welcome.",
        "This is a test of captioning.",
        "Thank you for listening.",
    ]
    assert (result.changed, result.total) == (1, 3)


def test_apply_edits_rewrites_both_subtitle_files(conn, tmp_path):
    job_id, _ = seed_job(conn, tmp_path)
    result = editor.apply_edits(conn, job_id, drafts(s1="Hi there."))

    srt_text = result.srt_path.read_text(encoding="utf-8")
    vtt_text = result.vtt_path.read_text(encoding="utf-8")
    assert "Hi there." in srt_text and "Hi there." in vtt_text
    assert "Hello and welcome." not in srt_text
    assert srt_text.startswith("1\n00:00:00,000 --> 00:00:01,000\n")
    assert vtt_text.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n")


def test_exported_files_match_a_fresh_render_of_the_database(conn, tmp_path):
    """The files on disk and the captions table must not drift apart."""
    job_id, _ = seed_job(conn, tmp_path)
    result = editor.apply_edits(conn, job_id, drafts(s3="Thanks for listening."))

    stored = editor.load_drafts(conn, job_id)
    blocks = [Block.from_text(d.text, d.start_ms / 1000, d.end_ms / 1000) for d in stored]
    assert result.srt_path.read_text(encoding="utf-8") == srt.to_srt(blocks)
    assert result.vtt_path.read_text(encoding="utf-8") == srt.to_vtt(blocks)


def test_apply_edits_recreates_deleted_output_files(conn, tmp_path):
    job_id, _ = seed_job(conn, tmp_path)
    out_dir = tmp_path / "outputs" / f"job_{job_id}"
    if out_dir.exists():
        for child in out_dir.iterdir():
            child.unlink()
        out_dir.rmdir()

    result = editor.apply_edits(conn, job_id, drafts(s1="Re-exported."))

    assert result.srt_path.exists() and result.vtt_path.exists()


def test_apply_edits_leaves_no_temp_files_behind(conn, tmp_path):
    job_id, _ = seed_job(conn, tmp_path)
    result = editor.apply_edits(conn, job_id, drafts(s1="Clean write."))
    leftovers = list(result.srt_path.parent.glob("*.tmp"))
    assert leftovers == []


def test_apply_edits_rejects_a_changed_timestamp(conn, tmp_path):
    job_id, _ = seed_job(conn, tmp_path)
    bad = drafts()
    bad[1].end_ms = 9_999

    with pytest.raises(editor.CaptionParseError, match="timestamps must stay"):
        editor.apply_edits(conn, job_id, bad)
    assert conn.execute(
        "SELECT text FROM captions WHERE job_id = ? AND seq = 2", (job_id,)
    ).fetchone()["text"] == "This is a test of the captioning pipeline."


def test_apply_edits_rejects_renumbering(conn, tmp_path):
    job_id, _ = seed_job(conn, tmp_path)
    with pytest.raises(editor.CaptionParseError, match="numbers were changed"):
        editor.apply_edits(conn, job_id, list(reversed(drafts())))


def test_apply_edits_rejects_a_blank_caption(conn, tmp_path):
    job_id, _ = seed_job(conn, tmp_path)
    with pytest.raises(editor.CaptionParseError, match="caption 2 is empty"):
        editor.apply_edits(conn, job_id, drafts(s2="   "))


def test_apply_edits_refuses_a_job_without_captions(conn, tmp_path):
    job_id, _ = seed_job(conn, tmp_path, captions=None)
    with pytest.raises(editor.CaptionParseError, match="no captions to edit"):
        editor.apply_edits(conn, job_id, drafts())


#dashboard flow


def test_preview_is_editable_only_for_finished_jobs(make_win, conn, tmp_path):
    done_id, _ = seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()
    assert win.preview.isReadOnly() is False

    conn.execute("UPDATE jobs SET status = 'queued' WHERE id = ?", (done_id,))
    conn.commit()
    win.refresh()
    assert win.preview.isReadOnly() is True


def test_typing_marks_a_draft_and_enables_save(make_win, conn, tmp_path):
    seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()
    assert win.btn_save_edits.isEnabled() is False

    win.preview.setPlainText(win.preview.toPlainText().replace("Hello", "Hullo"))

    assert win.btn_save_edits.isEnabled() is True
    assert win.btn_revert_edits.isEnabled() is True
    assert "unsaved" in win.edit_state.text()


def test_save_writes_the_database_and_the_files(make_win, conn, tmp_path):
    job_id, _ = seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()

    win.preview.setPlainText(
        win.preview.toPlainText().replace("Thank you for listening.", "Thanks for watching.")
    )
    assert win.save_caption_edits() is True

    text = conn.execute(
        "SELECT text FROM captions WHERE job_id = ? AND seq = 3", (job_id,)
    ).fetchone()["text"]
    assert text == "Thanks for watching."
    assert "Thanks for watching." in (
        tmp_path / "outputs" / f"job_{job_id}" / "captions.srt"
    ).read_text(encoding="utf-8")
    # the pane re-renders from the database and is clean again
    assert "Thanks for watching." in win.preview.toPlainText()
    assert win.btn_save_edits.isEnabled() is False
    assert win._drafts == {}


def test_save_reports_a_changed_timestamp_instead_of_writing(make_win, conn, tmp_path, monkeypatch):
    job_id, _ = seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()

    shown = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: shown.append(args[-1])
    )
    win.preview.setPlainText(
        win.preview.toPlainText().replace("00:00:00,000 --> 00:00:01,000",
                                          "00:00:00,000 --> 00:00:09,000")
    )
    assert win.save_caption_edits() is False
    assert shown and "timestamps" in shown[0]
    # nothing was written
    assert conn.execute(
        "SELECT text FROM captions WHERE job_id = ? AND seq = 1", (job_id,)
    ).fetchone()["text"] == "Hello and welcome."
    assert win._drafts  # the user's buffer is preserved so they can fix it


def test_revert_restores_the_generated_text(make_win, conn, tmp_path):
    seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()
    original = win.preview.toPlainText()

    win.preview.setPlainText("mistake")
    assert win.preview.toPlainText() == "mistake"

    assert win.revert_caption_edits() is True
    assert win.preview.toPlainText() == original
    assert win._drafts == {}


def test_unsaved_edits_survive_switching_jobs(make_win, conn, tmp_path):
    first, _ = seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()
    win.preview.setPlainText(win.preview.toPlainText().replace("Hello", "Hullo"))
    edited = win.preview.toPlainText()

    # a second job arrives and gets selected instead
    second = worker.enqueue(conn, str(tmp_path / "other.mp4"), model_size="tiny.en")
    conn.commit()
    win.refresh()
    win.table.selectRow(
        [i for i in range(win.table.rowCount()) if win.table.item(i, 0).text() == str(second)][0]
    )
    win.update_preview()
    assert win._drafts[first] == edited

    # back to the first job: the edit is still there and still saveable
    win.table.selectRow(
        [i for i in range(win.table.rowCount()) if win.table.item(i, 0).text() == str(first)][0]
    )
    win.update_preview()
    assert win.preview.toPlainText() == edited
    assert win.btn_save_edits.isEnabled() is True


def test_details_column_flags_unsaved_edits(make_win, conn, tmp_path):
    seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()

    win.preview.setPlainText(win.preview.toPlainText() + "\nExtra line.")
    win.refresh()

    assert "unsaved edits" in win.table.item(0, 7).text()


def test_queue_refresh_does_not_clobber_an_in_progress_edit(make_win, conn, tmp_path):
    """The 1/second refresh must never rewrite the buffer under the caret."""
    job_id, _ = seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()

    win.preview.setPlainText(win.preview.toPlainText().replace("Hello", "Hullo"))
    typed = win.preview.toPlainText()
    caret = win.preview.textCursor().position()

    conn.execute("UPDATE jobs SET progress = 0.5 WHERE id = ?", (job_id,))
    conn.commit()
    win.refresh()  # signature changed -> table rebuild -> update_preview

    assert win.preview.toPlainText() == typed
    assert win.preview.textCursor().position() == caret


def test_editing_does_not_restart_playback(make_win, conn, tmp_path):
    job_id, src = seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()
    assert win.fake_player.opened == [str(src)]

    win.preview.setPlainText(win.preview.toPlainText().replace("Hello", "Hullo"))
    win.refresh()
    win.save_caption_edits()

    assert win.fake_player.opened == [str(src)]  # opened exactly once


def test_highlight_still_lines_up_after_an_edit(make_win, conn, tmp_path):
    seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()

    win.preview.setPlainText(win.preview.toPlainText().replace("Hello and welcome.", "Hi!"))
    win.save_caption_edits()

    win.fake_player.position_ms_changed.emit(500)  # inside caption 1
    selections = win.preview.extraSelections()
    assert len(selections) == 1
    assert selections[0].cursor.block().text() == "Hi!"


def test_save_without_edits_is_a_no_op(make_win, conn, tmp_path):
    seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()
    assert win.save_caption_edits() is False
    assert win.revert_caption_edits() is False


def test_closing_with_unsaved_edits_asks_first(make_win, conn, tmp_path, monkeypatch):
    from PySide6.QtGui import QCloseEvent

    seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()
    win.preview.setPlainText(win.preview.toPlainText() + "\nExtra line.")

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted() is False  # the window stayed open

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted() is True
    # a confirmed close drops the drafts, so teardown cannot prompt again
    assert win._drafts == {}


def test_writing_a_caption_edit_and_reopening_the_db_keeps_it(make_win, conn, tmp_path):
    """A saved edit is durable: reopening the connection shows the new text."""
    job_id, _ = seed_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()
    win.preview.setPlainText(win.preview.toPlainText().replace("Hello", "Hullo"))
    win.save_caption_edits()

    fresh = db.connect()
    try:
        assert fresh.execute(
            "SELECT text FROM captions WHERE job_id = ? AND seq = 1", (job_id,)
        ).fetchone()["text"] == "Hullo and welcome."
    finally:
        fresh.close()
