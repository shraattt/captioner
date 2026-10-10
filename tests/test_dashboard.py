"""Offscreen UI tests for the dashboard window.

Qt's 'offscreen' platform plugin lets us build the real window and drive it
without a display, so these cover what a user actually clicks through: rows
render, the preview loads captions, adding files queues jobs with the chosen
model, retry and remove go through the worker layer, and running jobs are
protected from accidental deletion.
"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from captioner import config, db, worker
from captioner.segmenter import Block, Word
from ui.dashboard import Dashboard


def _settle(path, seconds=5):
    """Backdate a file: the watch scanner skips files still being written."""
    import os
    import time

    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ui.db")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    connection = db.connect()
    yield connection
    connection.close()


@pytest.fixture()
def make_win(qapp, conn):
    """Factory for dashboard windows; closes everything afterwards."""
    created = []

    def factory():
        win = Dashboard(conn=conn, persist_settings=False)
        win.timer.stop()  # tests refresh explicitly
        created.append(win)
        return win

    yield factory
    for win in created:
        win.timer.stop()
        win.close()


def seed_job(conn, tmp_path, name="clip.mp4", status="done", error=None, captions=True):
    src = tmp_path / name
    src.write_bytes(b"not real media")
    job_id = worker.enqueue(conn, str(src), model_size="tiny.en")
    # A running job carries a fresh heartbeat: that is what marks it as owned
    # by a live worker, and what stops startup recovery from requeueing it.
    heartbeat = (
        time.strftime("%Y-%m-%d %H:%M:%S") if status == "running" else None
    )
    conn.execute(
        "UPDATE jobs SET status = ?, started_at = ?, finished_at = ?, error = ?, "
        "heartbeat_at = ? WHERE id = ?",
        (
            status,
            "2026-01-01 10:00:00",
            "2026-01-01 10:00:05" if status in ("done", "error") else None,
            error,
            heartbeat,
            job_id,
        ),
    )
    conn.commit()
    if captions and status == "done":
        worker._store_captions(
            conn, job_id, [Block([Word("Hello", 0.0, 0.4), Word("world.", 0.5, 1.0)])]
        )
    return job_id


def test_rows_and_preview_render(make_win, conn, tmp_path):
    job_id = seed_job(conn, tmp_path)
    win = make_win()

    assert win.table.rowCount() == 1
    assert win.table.item(0, 0).text() == str(job_id)
    assert win.table.item(0, 2).text() == "done"
    assert win.table.item(0, 3).text() == "tiny.en"
    assert win.table.item(0, 7).text() == "1 caption"
    assert win.table.currentRow() == 0  # newest job is auto-selected

    preview = win.preview.toPlainText()
    assert "00:00:00,000 --> 00:00:01,000" in preview
    assert "Hello world." in preview
    assert "clip.mp4" in win.meta.text()
    assert win.progress.format() == "1/1 done"


def test_add_paths_queues_with_selected_model(make_win, conn, tmp_path):
    win = make_win()
    win.model_combo.setCurrentText("small.en")
    first = tmp_path / "one.mp4"
    second = tmp_path / "two.wav"
    first.write_bytes(b"x")
    second.write_bytes(b"x")

    queued = win.add_paths([str(first), str(second), str(tmp_path / "missing.mp4")])

    assert queued == 2  # the missing path is skipped
    rows = conn.execute(
        "SELECT file_name, model_size, status FROM jobs ORDER BY id"
    ).fetchall()
    assert [r["file_name"] for r in rows] == ["one.mp4", "two.wav"]
    assert all(r["model_size"] == "small.en" for r in rows)
    assert all(r["status"] == "queued" for r in rows)
    assert win.table.rowCount() == 2


def test_retry_then_remove(make_win, conn, tmp_path):
    job_id = seed_job(
        conn, tmp_path, status="error", error="RuntimeError: boom", captions=False
    )
    win = make_win()
    assert win.table.item(0, 7).text() == "RuntimeError: boom"

    win.table.selectRow(0)
    win.retry_selected()
    row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "queued"

    assert win._remove_job(job_id) is True
    assert conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone() is None
    assert win.table.rowCount() == 0


def test_remove_refuses_running_job(make_win, conn, tmp_path):
    job_id = seed_job(conn, tmp_path, status="running", captions=False)
    win = make_win()
    win.table.selectRow(0)

    assert win._remove_job(job_id) is False
    assert conn.execute(
        "SELECT * FROM jobs WHERE id = ?", (job_id,)
    ).fetchone() is not None
    assert win.table.item(0, 2).text() == "running"


def test_folders_are_expanded_recursively(make_win, conn, tmp_path):
    folder = tmp_path / "course"
    (folder / "week1").mkdir(parents=True)
    (folder / "intro.mp4").write_bytes(b"x")
    (folder / "week1" / "part1.wav").write_bytes(b"x")
    (folder / "week1" / "slides.pdf").write_bytes(b"x")  # not media
    win = make_win()

    assert win.add_paths([str(folder)]) == 2
    names = [
        r["file_name"]
        for r in conn.execute("SELECT file_name FROM jobs ORDER BY id").fetchall()
    ]
    assert names == ["intro.mp4", "part1.wav"]


def test_progress_column_tracks_running_jobs(make_win, conn, tmp_path):
    job_id = seed_job(conn, tmp_path, status="running", captions=False)
    conn.execute("UPDATE jobs SET progress = 0.42 WHERE id = ?", (job_id,))
    conn.commit()
    win = make_win()

    # column order: # file status model progress created duration details
    assert win.table.item(0, 4).text() == "42%"

    conn.execute("UPDATE jobs SET progress = NULL WHERE id = ?", (job_id,))
    conn.commit()
    win.refresh()
    assert win.table.item(0, 4).text() == "starting…"


def test_watch_folder_queues_new_files_once(make_win, conn, tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(config, "UPLOADS_DIR", uploads)
    win = make_win()
    win.watch_check.setChecked(True)

    lecture = uploads / "lecture.mp4"
    lecture.write_bytes(b"x")
    (uploads / "notes.txt").write_bytes(b"x")  # ignored
    _settle(lecture)  # the scanner leaves a file that was just written alone
    win.refresh()

    names = [
        r["file_name"]
        for r in conn.execute("SELECT file_name FROM jobs ORDER BY id").fetchall()
    ]
    assert names == ["lecture.mp4"]

    win.refresh()  # a second tick must not queue the same file again
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_batch_finished_notification_fires_once(make_win, conn, tmp_path):
    job_id = seed_job(conn, tmp_path, status="queued", captions=False)
    win = make_win()
    assert win.last_notification is None  # a job is still pending

    conn.execute(
        "UPDATE jobs SET status = 'done', finished_at = ? WHERE id = ?",
        ("2026-01-01 10:00:05", job_id),
    )
    conn.commit()
    win.refresh()

    assert win.last_notification is not None
    first = win.last_notification
    win.refresh()  # no pending->settled transition left: no second alert
    assert win.last_notification == first


def test_notification_respects_checkbox(make_win, conn, tmp_path):
    job_id = seed_job(conn, tmp_path, status="queued", captions=False)
    win = make_win()
    win.notify_check.setChecked(False)

    conn.execute("UPDATE jobs SET status = 'done' WHERE id = ?", (job_id,))
    conn.commit()
    win.refresh()

    assert win.last_notification is None


def test_queue_options_come_from_the_options_popover(make_win, conn, tmp_path):
    """Align / translate / speakers selected in Options land on the job rows."""
    win = make_win()
    win.diarize_check.setChecked(True)
    win.align_check.setChecked(True)
    win.translate_combo.setCurrentIndex(
        win.translate_combo.findData("hi")
    )
    media = tmp_path / "talk.wav"
    media.write_bytes(b"x")

    win.add_paths([str(media)])

    row = conn.execute(
        "SELECT diarize, align, translate_to FROM jobs"
    ).fetchone()
    assert (row["diarize"], row["align"], row["translate_to"]) == (1, 1, "hi")
    assert "translate:hi" in win.statusBar().currentMessage()


def test_options_default_to_off(make_win):
    win = make_win()
    assert win._queue_options() == {"diarize": False, "align": False, "translate_to": None}
    assert win.translate_combo.itemData(0) == ""  # "Off" is first
    assert win.translate_combo.count() > 1
