"""Tests for watch folders and webhooks.

The webhook is exercised against a real HTTP server on loopback, so the
request the receiver actually sees (method, headers, JSON body) is asserted
rather than a stubbed call, and the worker test proves a settled job fires it.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from captioner import automation, config, db, worker


@pytest.fixture
def conn(tmp_path, monkeypatch):
    """A real database in a temp directory, with the data paths redirected."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "a.db")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.delenv(automation.WATCH_ENV, raising=False)
    monkeypatch.delenv(automation.WEBHOOK_ENV, raising=False)
    monkeypatch.delenv(automation.WEBHOOK_SECRET_ENV, raising=False)
    config.ensure_dirs()  # the app creates its data tree; the scanner only reads it
    connection = db.connect()
    yield connection
    connection.close()


def media_file(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    old = time.time() - 60
    import os

    os.utime(path, (old, old))  # old enough that the "still copying" guard passes
    return path


class Receiver:
    """A one-endpoint HTTP server that records what it was sent."""

    def __init__(self, status=200):
        self.status = status
        self.bodies = []
        self.headers = []
        self._server = None
        self.url = ""

    def __enter__(self):
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - stdlib naming
                length = int(self.headers.get("Content-Length", 0))
                receiver.bodies.append(json.loads(self.rfile.read(length).decode("utf-8")))
                receiver.headers.append(dict(self.headers))
                self.send_response(receiver.status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):  # keep the test output clean
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/hook"
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


#watch folders

def test_watch_dirs_includes_uploads_and_env(tmp_path, monkeypatch, conn):
    extra = tmp_path / "shows"
    extra.mkdir()
    monkeypatch.setenv(automation.WATCH_ENV, f"{extra}{__import__('os').pathsep}{tmp_path / 'missing'}")
    dirs = automation.watch_dirs()
    assert config.UPLOADS_DIR in dirs
    assert extra in dirs
    assert tmp_path / "missing" not in dirs  # non-existent folders are dropped


def test_watch_dirs_deduplicates(tmp_path, monkeypatch, conn):
    monkeypatch.setenv(automation.WATCH_ENV, f"{config.UPLOADS_DIR};{config.UPLOADS_DIR}")
    dirs = automation.watch_dirs()
    assert dirs.count(config.UPLOADS_DIR) == 1


def test_find_new_media_skips_what_is_not_ready(conn, tmp_path):
    uploads = config.UPLOADS_DIR
    uploads.mkdir(parents=True, exist_ok=True)
    good = media_file(uploads / "clip.mp4")
    media_file(uploads / "notes.txt")
    media_file(uploads / "video.mp4.part")
    media_file(uploads / ".hidden.mp4")
    fresh = uploads / "still-copying.mp4"
    fresh.write_bytes(b"x")  # mtime is now: the scanner should wait
    nested = media_file(uploads / "season 1" / "episode.mp3")

    found = automation.find_new_media(conn)
    assert found == [good, nested] or found == [nested, good]
    assert set(found) == {good, nested}


def test_find_new_media_is_idempotent(conn):
    uploads = config.UPLOADS_DIR
    uploads.mkdir(parents=True, exist_ok=True)
    clip = media_file(uploads / "clip.mp4")
    assert automation.find_new_media(conn) == [clip]
    worker.enqueue(conn, str(clip))
    assert automation.find_new_media(conn) == []


def test_find_new_media_returns_oldest_first(conn):
    uploads = config.UPLOADS_DIR
    uploads.mkdir(parents=True, exist_ok=True)
    older = media_file(uploads / "b-first.mp4")
    newer = media_file(uploads / "a-second.mp4")
    import os

    now = time.time()
    os.utime(older, (now - 500, now - 500))
    os.utime(newer, (now - 100, now - 100))
    assert automation.find_new_media(conn) == [older, newer]


def test_find_new_media_survives_a_missing_folder(conn):
    assert automation.find_new_media(conn, [config.UPLOADS_DIR / "nope"]) == []


#webhooks

def test_notify_is_off_until_configured(conn):
    conn.execute("INSERT INTO jobs (file_path, file_name, model_size, created_at, status) "
                 "VALUES ('/x/a.mp4', 'a.mp4', 'small.en', '2026-01-01 00:00:00', 'done')")
    conn.commit()
    assert automation.webhook_url() == ""
    assert automation.notify_job(conn, 1) is False


def test_job_event_describes_the_job_and_its_outputs(conn, tmp_path):
    job_id = worker.enqueue(conn, str(media_file(tmp_path / "clip.mp4")))
    out_dir = config.OUTPUT_DIR / f"job_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "captions.srt").write_text("1\n", encoding="utf-8")
    (out_dir / "captions.hi.vtt").write_text("WEBVTT\n", encoding="utf-8")
    conn.execute(
        "INSERT INTO captions (job_id, seq, start_ms, end_ms, text) VALUES (?, 1, 0, 1500, 'hi')",
        (job_id,),
    )
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = ?", (job_id,))
    conn.commit()

    event = automation.job_event(conn, job_id)
    assert event["event"] == "job.done"
    job = event["job"]
    assert job["file_name"] == "clip.mp4"
    assert job["captions"] == 1
    assert job["duration_ms"] == 1500
    assert job["outputs"] == ["captions.hi.vtt", "captions.srt"]
    assert job["options"] == {"diarize": False, "align": False, "translate_to": None}


def test_job_event_is_none_for_a_missing_row(conn):
    assert automation.job_event(conn, 999) is None


def test_notify_delivers_json_with_the_secret_header(conn, monkeypatch, tmp_path):
    job_id = worker.enqueue(conn, str(media_file(tmp_path / "clip.mp4")))
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = ?", (job_id,))
    conn.commit()
    monkeypatch.setenv(automation.WEBHOOK_SECRET_ENV, "s3cret")

    with Receiver() as receiver:
        assert automation.notify_job(conn, job_id, url=receiver.url) is True

    assert receiver.bodies[0]["job"]["id"] == job_id
    assert receiver.headers[0]["X-Captioner-Secret"] == "s3cret"
    assert receiver.headers[0]["Content-Type"] == "application/json"


def test_notify_reports_a_refused_webhook(conn, tmp_path):
    job_id = worker.enqueue(conn, str(media_file(tmp_path / "clip.mp4")))
    with Receiver(status=500) as receiver:
        assert automation.notify_job(conn, job_id, url=receiver.url) is False


def test_notify_survives_a_dead_endpoint(conn, tmp_path):
    job_id = worker.enqueue(conn, str(media_file(tmp_path / "clip.mp4")))
    # Nothing listens on this port: a failed webhook must return False, not raise.
    assert automation.notify_job(conn, job_id, url="http://127.0.0.1:9/hook") is False


def test_worker_fires_the_webhook_when_a_job_settles(conn, monkeypatch, tmp_path):
    from captioner.segmenter import Word

    monkeypatch.setattr(
        worker, "transcribe_words",
        lambda path, size=None, on_progress=None, on_stage=None, should_cancel=None: [
            Word(text="hello", start=0.0, end=1.0)
        ],
    )
    sent = []
    monkeypatch.setattr(
        automation, "notify_job",
        lambda connection, job_id, url=None, transport=None: sent.append(job_id) or True,
    )
    clip = media_file(tmp_path / "clip.mp4")
    worker.enqueue(conn, str(clip))

    assert worker.run_once(conn) is True
    row = conn.execute("SELECT status FROM jobs WHERE id = 1").fetchone()
    assert row["status"] == "done"
    assert sent == [1]
    assert (config.OUTPUT_DIR / "job_1" / "captions.srt").exists()


def test_worker_still_settles_when_the_webhook_explodes(conn, monkeypatch, tmp_path):
    from captioner.segmenter import Word

    monkeypatch.setattr(
        worker, "transcribe_words",
        lambda path, size=None, on_progress=None, on_stage=None, should_cancel=None: [
            Word(text="hello", start=0.0, end=1.0)
        ],
    )

    def boom(*args, **kwargs):
        raise RuntimeError("webhook backend is broken")

    monkeypatch.setattr(automation, "notify_job", boom)
    worker.enqueue(conn, str(media_file(tmp_path / "clip.mp4")))

    assert worker.run_once(conn) is True
    assert conn.execute("SELECT status FROM jobs WHERE id = 1").fetchone()["status"] == "done"
