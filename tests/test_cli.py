"""Tests for the command-line interface.

The CLI is the headless way to drive the queue — a script or a scheduled task
uses it instead of the window — so its exit codes and its cancel path are worth
pinning down. Every test goes through ``cli.main``, which opens the database the
same way a real run does.
"""

import pytest

import cli
from captioner import config, db, worker


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "cli.db")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    return tmp_path


def seed(tmp_path, status="queued"):
    """Put one job in the queue, optionally already advanced to ``status``."""
    conn = db.connect()
    try:
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"not real media")
        job_id = worker.enqueue(conn, str(path))
        if status != "queued":
            conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
            conn.commit()
        return job_id
    finally:
        conn.close()


def status_of(job_id):
    conn = db.connect()
    try:
        return conn.execute(
            "SELECT status FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()["status"]
    finally:
        conn.close()


def test_cancel_calls_off_a_queued_job(cli_env, capsys):
    job_id = seed(cli_env)
    assert cli.main(["cancel", str(job_id)]) == 0
    assert "canceled before it started" in capsys.readouterr().out
    assert status_of(job_id) == "canceled"


def test_cancel_asks_a_running_job_to_stop(cli_env, capsys):
    job_id = seed(cli_env, status="running")
    assert cli.main(["cancel", str(job_id)]) == 0
    assert "next segment boundary" in capsys.readouterr().out
    assert status_of(job_id) == "running"  # the worker decides when it stops


def test_cancel_reports_an_unknown_job(cli_env, capsys):
    assert cli.main(["cancel", "999"]) == 1
    assert "no job #999" in capsys.readouterr().out


def test_cancel_of_a_finished_job_is_not_an_error(cli_env, capsys):
    """Nothing to stop is worth a message, but it is not a failure."""
    job_id = seed(cli_env, status="done")
    assert cli.main(["cancel", str(job_id)]) == 0
    assert "already done" in capsys.readouterr().out


def test_add_reports_the_job_that_already_covers_the_file(cli_env, capsys):
    """The same audio under a second name is not queued twice."""
    original = cli_env / "lecture.mp4"
    original.write_bytes(b"one lecture")
    copy = cli_env / "lecture-copy.mp4"
    copy.write_bytes(b"one lecture")

    assert cli.main(["add", str(original)]) == 0
    assert "queued job #" in capsys.readouterr().out

    assert cli.main(["add", str(copy)]) == 0
    out = capsys.readouterr().out
    assert "already has a job: #1 (queued)" in out

    conn = db.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    finally:
        conn.close()


def test_status_marks_a_job_that_is_stopping(cli_env, capsys):
    job_id = seed(cli_env, status="running")
    conn = db.connect()
    try:
        worker.request_cancel(conn, job_id)
    finally:
        conn.close()

    assert cli.main(["status"]) == 0
    assert "(stopping)" in capsys.readouterr().out
