"""Worker pipeline integration test — no models needed (transcriber mocked).

Exercises the real queue semantics (claim atomicity, status flips, error
recording, retry) and verifies captions land in the DB and subtitle files
land on disk. Also covers the long-run machinery: cancellation, the heartbeat
that distinguishes a slow job from a dead one, and recovery of jobs orphaned
by a crash.
"""
import threading
import time
from pathlib import Path

import pytest

from captioner import config, db, worker
from captioner.segmenter import Word
from captioner.transcribe import TranscriptionCancelled


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    conn = db.connect()
    yield conn
    conn.close()


@pytest.fixture()
def fake_words(monkeypatch):
    words = [
        Word("Hello", 0.0, 0.4),
        Word("beautiful", 0.5, 0.8),
        Word("world.", 0.9, 1.2),
        Word("Second", 3.5, 3.9),
        Word("sentence.", 4.0, 4.5),
    ]
    monkeypatch.setattr(
        worker, "transcribe_words", lambda path, size, on_progress=None, **_: words
    )
    return words


def test_enqueue_creates_queued_job(tmp_db):
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "queued"
    assert row["file_name"] == "sample.mp4"


def test_diarized_job_writes_speaker_labels(tmp_db, fake_words, monkeypatch):
    """A diarize=1 job tags captions with [SPEAKER_XX]: prefixes."""
    from captioner.diarize import SpeakerTurn

    turns = [
        SpeakerTurn(0.0, 1.3, "SPEAKER_00"),
        SpeakerTurn(3.0, 5.0, "SPEAKER_01"),
    ]
    monkeypatch.setattr(worker, "_diarize_turns", lambda path: turns)
    worker.enqueue(tmp_db, "sample.mp4")
    tmp_db.execute("UPDATE jobs SET diarize = 1")
    tmp_db.commit()
    job = worker.claim_next(tmp_db)
    worker.run_pipeline(tmp_db, job)

    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job["id"],)).fetchone()
    assert row["status"] == "done"
    caps = tmp_db.execute(
        "SELECT * FROM captions WHERE job_id = ? ORDER BY seq", (job["id"],)
    ).fetchall()
    assert [c["text"] for c in caps] == [
        "[SPEAKER_00]: Hello beautiful world.",
        "[SPEAKER_01]: Second sentence.",
    ]
    # the exported SRT carries the same labels
    srt_text = (
        Path(config.OUTPUT_DIR) / f"job_{job['id']}" / "captions.srt"
    ).read_text(encoding="utf-8")
    assert "[SPEAKER_00]: Hello" in srt_text


def test_full_pipeline_produces_captions_and_files(tmp_db, fake_words):
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    job = worker.claim_next(tmp_db)
    assert job["id"] == job_id
    worker.run_pipeline(tmp_db, job)

    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "done"
    assert row["error"] is None

    caps = tmp_db.execute(
        "SELECT * FROM captions WHERE job_id = ? ORDER BY seq", (job_id,)
    ).fetchall()
    assert [c["text"] for c in caps] == ["Hello beautiful world.", "Second sentence."]
    assert caps[0]["start_ms"] == 0

    out = Path(config.OUTPUT_DIR) / f"job_{job_id}"
    assert (out / "captions.srt").exists()
    assert (out / "captions.vtt").exists()
    srt_text = (out / "captions.srt").read_text(encoding="utf-8")
    assert "Hello beautiful world." in srt_text


def test_aligned_job_realigns_words_before_segmenting(tmp_db, fake_words, monkeypatch):
    """align=1 runs the alignment pass and the tightened times reach the DB."""
    seen = {}

    def fake_realign(path, words, **kwargs):
        seen["path"] = path
        return [Word(w.text, w.start + 0.1, w.end + 0.1) for w in words]

    monkeypatch.setattr("captioner.align.realign_file", fake_realign)
    job_id = worker.enqueue(tmp_db, "sample.mp4", align=True)
    job = worker.claim_next(tmp_db)
    worker.run_pipeline(tmp_db, job)

    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "done"
    assert seen["path"].endswith("sample.mp4")
    first = tmp_db.execute(
        "SELECT start_ms FROM captions WHERE job_id = ? ORDER BY seq LIMIT 1", (job_id,)
    ).fetchone()
    assert first["start_ms"] == 100  # 0.0 + 0.1, proving the aligned words were used


def test_unflagged_job_skips_alignment(tmp_db, fake_words, monkeypatch):
    called = []
    monkeypatch.setattr(
        "captioner.align.realign_file",
        lambda *a, **k: called.append(True) or [],
    )
    worker.enqueue(tmp_db, "sample.mp4")
    worker.run_pipeline(tmp_db, worker.claim_next(tmp_db))
    assert called == []


def test_translate_option_writes_extra_subtitles(tmp_db, fake_words, monkeypatch):
    # Never load the real model here: the pass-through below stands in for it.
    from captioner import translate as translate_mod

    monkeypatch.setattr(
        translate_mod.Translator,
        "translate_many",
        lambda self, texts, target: [f"[{target}]{t}" for t in texts],
    )
    job_id = worker.enqueue(tmp_db, "sample.mp4", translate_to="hi")
    job = worker.claim_next(tmp_db)
    worker.run_pipeline(tmp_db, job)

    out = Path(config.OUTPUT_DIR) / f"job_{job_id}"
    assert (out / "captions.srt").exists()
    assert (out / "captions.hi.srt").exists()
    assert (out / "captions.hi.vtt").exists()
    row = tmp_db.execute("SELECT status, error FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "done"
    assert row["error"] is None


def test_translation_failure_still_finishes_the_transcript(tmp_db, fake_words, monkeypatch):
    """A missing NLLB download must not flip a good transcript to error."""
    def boom(*args, **kwargs):
        raise RuntimeError("model not available")

    monkeypatch.setattr("captioner.translate.translate_subtitles", boom)
    job_id = worker.enqueue(tmp_db, "sample.mp4", translate_to="hi")
    worker.run_pipeline(tmp_db, worker.claim_next(tmp_db))

    row = tmp_db.execute("SELECT status, error FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "done"
    assert "translation to hi failed" in row["error"]
    out = Path(config.OUTPUT_DIR) / f"job_{job_id}"
    assert (out / "captions.srt").exists()


def test_claim_is_atomic(tmp_db, fake_words):
    worker.enqueue(tmp_db, "a.mp4")
    first = worker.claim_next(tmp_db)
    second = worker.claim_next(tmp_db)
    assert first and not second  # a job can only be claimed once


def test_error_is_caught_and_recorded(tmp_db, monkeypatch):
    worker.enqueue(tmp_db, "broken.mp4")
    job = worker.claim_next(tmp_db)

    def boom(path, size, on_progress=None, **kwargs):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(worker, "transcribe_words", boom)
    worker.run_pipeline(tmp_db, job)

    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job["id"],)).fetchone()
    assert row["status"] == "error"
    assert "ffmpeg exploded" in row["error"]


def test_retry_requeues_error_job(tmp_db, fake_words):
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    job = worker.claim_next(tmp_db)
    worker.run_pipeline(tmp_db, job)
    worker.retry(tmp_db, job_id)
    row = tmp_db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "queued"


def test_delete_job_removes_row_and_captions(tmp_db, fake_words):
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    worker.run_pipeline(tmp_db, worker.claim_next(tmp_db))

    assert worker.delete_job(tmp_db, job_id) is True
    assert tmp_db.execute(
        "SELECT * FROM jobs WHERE id = ?", (job_id,)
    ).fetchone() is None
    assert tmp_db.execute(
        "SELECT * FROM captions WHERE job_id = ?", (job_id,)
    ).fetchall() == []


def test_delete_refuses_running_and_missing_jobs(tmp_db, fake_words):
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    worker.claim_next(tmp_db)  # status is now 'running'

    assert worker.delete_job(tmp_db, job_id) is False
    still = tmp_db.execute(
        "SELECT status FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    assert still["status"] == "running"
    assert worker.delete_job(tmp_db, 9999) is False


def test_progress_is_persisted_mid_flight(tmp_db, monkeypatch):
    """A running job reports progress while its segments decode."""
    job_id = worker.enqueue(tmp_db, "sample.mp4")

    def read_progress():
        return tmp_db.execute(
            "SELECT progress FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()["progress"]

    observed = []

    def fake(path, size, on_progress=None, **kwargs):
        on_progress(0.25)
        observed.append(read_progress())
        on_progress(0.75)
        observed.append(read_progress())
        return [Word("Hello", 0.0, 1.0)]

    monkeypatch.setattr(worker, "transcribe_words", fake)
    worker.run_pipeline(tmp_db, worker.claim_next(tmp_db))

    assert observed == [0.25, 0.75]
    settled = tmp_db.execute(
        "SELECT status, progress FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    assert settled["status"] == "done"
    assert settled["progress"] is None  # cleared once the job settles


def test_set_progress_throttles_and_retry_resets(tmp_db):
    job_id = worker.enqueue(tmp_db, "sample.mp4")

    def read_progress():
        return tmp_db.execute(
            "SELECT progress FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()["progress"]

    worker._set_progress(tmp_db, job_id, 0.20)
    assert read_progress() == 0.20
    worker._set_progress(tmp_db, job_id, 0.204)  # still 20%: not rewritten
    assert read_progress() == 0.20
    worker._set_progress(tmp_db, job_id, 0.21)  # 21%: written
    assert read_progress() == 0.21
    worker._set_progress(tmp_db, job_id, -0.5)  # clamped to 0.0
    assert read_progress() == 0.0
    worker._set_progress(tmp_db, job_id, 1.5)  # clamped to 1.0
    assert read_progress() == 1.0

    worker.retry(tmp_db, job_id)
    assert read_progress() is None


def test_retry_clears_live_run_bookkeeping(tmp_db):
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    worker.claim_next(tmp_db)
    tmp_db.execute(
        "UPDATE jobs SET stage = 'downloading', heartbeat_at = ? WHERE id = ?",
        (time.strftime("%Y-%m-%d %H:%M:%S"), job_id),
    )
    tmp_db.commit()

    worker.retry(tmp_db, job_id)

    row = tmp_db.execute(
        "SELECT status, stage, heartbeat_at, started_at FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    assert row["status"] == "queued"
    assert row["stage"] is None
    assert row["heartbeat_at"] is None
    assert row["started_at"] is None


#cancellation


def test_cancel_marks_the_job_canceled_and_writes_nothing(tmp_db, monkeypatch):
    """A canceled job is neither done nor an error, and produces no output."""
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    job = worker.claim_next(tmp_db)
    cancel = threading.Event()

    def fake(path, size, on_progress=None, on_stage=None, should_cancel=None):
        cancel.set()
        assert should_cancel is not None and should_cancel() is True
        raise TranscriptionCancelled("canceled by the user")

    monkeypatch.setattr(worker, "transcribe_words", fake)
    worker.run_pipeline(tmp_db, job, cancel=cancel)

    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "canceled"
    assert row["progress"] is None and row["stage"] is None
    assert row["finished_at"] is not None
    assert tmp_db.execute(
        "SELECT COUNT(*) FROM captions WHERE job_id = ?", (job_id,)
    ).fetchone()[0] == 0
    assert not (Path(config.OUTPUT_DIR) / f"job_{job_id}").exists()


def test_canceled_job_can_be_retried(tmp_db, monkeypatch):
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    job = worker.claim_next(tmp_db)

    def fake(path, size, on_progress=None, on_stage=None, should_cancel=None):
        raise TranscriptionCancelled("canceled")

    monkeypatch.setattr(worker, "transcribe_words", fake)
    worker.run_pipeline(tmp_db, job, cancel=threading.Event())
    assert tmp_db.execute(
        "SELECT status FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()["status"] == "canceled"

    worker.retry(tmp_db, job_id)
    assert tmp_db.execute(
        "SELECT status FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()["status"] == "queued"


def test_canceled_job_is_not_claimed_again(tmp_db, monkeypatch):
    """Canceling must not turn into a hot loop that re-runs the same job."""
    job_id = worker.enqueue(tmp_db, "sample.mp4")

    def fake(path, size, on_progress=None, on_stage=None, should_cancel=None):
        raise TranscriptionCancelled("canceled")

    monkeypatch.setattr(worker, "transcribe_words", fake)
    conn = tmp_db
    assert worker.run_once(conn, cancel=threading.Event()) is True
    assert worker.run_once(conn, cancel=threading.Event()) is False  # nothing left
    assert conn.execute(
        "SELECT status FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()["status"] == "canceled"


def test_another_process_can_cancel_a_running_job(tmp_db, monkeypatch):
    """The API or CLI can stop a worker that lives in a different process.

    The flag is written through a *second* connection, which is exactly what a
    separate process looks like from here: nothing is shared but the database.
    No in-process event is passed, so only the DB path can be doing the work.
    """
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    job = worker.claim_next(tmp_db)
    elsewhere = db.connect()

    def fake(path, size, on_progress=None, on_stage=None, should_cancel=None):
        assert worker.request_cancel(elsewhere, job_id) == "requested"
        assert should_cancel is not None and should_cancel() is True
        raise TranscriptionCancelled("asked to stop")

    monkeypatch.setattr(worker, "transcribe_words", fake)
    try:
        worker.run_pipeline(tmp_db, job)
    finally:
        elsewhere.close()

    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "canceled"
    assert row["cancel_requested"] == 0  # honored, so it is cleared


def test_request_cancel_calls_off_a_queued_job(tmp_db):
    """A job that has not started needs nobody's cooperation to stop."""
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    assert worker.request_cancel(tmp_db, job_id) == "canceled"

    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "canceled" and row["finished_at"] is not None
    assert worker.claim_next(tmp_db) == {}  # and it is never picked up


def test_request_cancel_on_a_running_job_only_asks(tmp_db):
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    worker.claim_next(tmp_db)
    assert worker.request_cancel(tmp_db, job_id) == "requested"

    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "running"  # the worker decides when it stops
    assert row["cancel_requested"] == 1


def test_request_cancel_reports_a_settled_job_as_nothing_to_stop(tmp_db):
    for status in ("done", "error", "canceled"):
        job_id = worker.enqueue(tmp_db, "sample.mp4")
        worker.claim_next(tmp_db)
        tmp_db.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
        tmp_db.commit()
        assert worker.request_cancel(tmp_db, job_id) is None, status
        assert tmp_db.execute(
            "SELECT status FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()["status"] == status


def test_retry_clears_a_stale_cancel_request(tmp_db):
    """A cancel that lands just as a job dies must not kill the retry."""
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    worker.claim_next(tmp_db)
    worker.request_cancel(tmp_db, job_id)

    worker.retry(tmp_db, job_id)
    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "queued"
    assert row["cancel_requested"] == 0


def test_recover_stale_honours_a_cancel_request(tmp_db):
    """A job canceled while its worker was down is canceled, not run again."""
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    worker.claim_next(tmp_db)
    worker.request_cancel(tmp_db, job_id)
    tmp_db.execute(
        "UPDATE jobs SET heartbeat_at = '2000-01-01 00:00:00' WHERE id = ?", (job_id,)
    )
    tmp_db.commit()

    assert worker.recover_stale(tmp_db) == 0  # nothing was requeued
    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "canceled"
    assert row["cancel_requested"] == 0


#heartbeat


def test_heartbeat_keeps_ticking_while_a_job_is_blocked(tmp_db, monkeypatch):
    """A job stuck in model loading still checks in.

    This is the whole point of the heartbeat: without it, "working on a 1.5 GB
    download" and "the process died" look identical in the database.
    """
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    job = worker.claim_next(tmp_db)
    beats, stages = [], []

    def fake(path, size, on_progress=None, on_stage=None, should_cancel=None):
        if on_stage is not None:
            on_stage("loading", None)
        deadline = time.time() + 1.6
        while time.time() < deadline:
            row = tmp_db.execute(
                "SELECT heartbeat_at, stage FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            beats.append(row["heartbeat_at"])
            stages.append(row["stage"])
            time.sleep(0.2)
        return [Word("Hello", 0.0, 1.0)]

    monkeypatch.setattr(worker, "transcribe_words", fake)
    worker.run_pipeline(tmp_db, job)

    assert len(set(beats)) > 1, f"heartbeat never moved: {beats}"
    assert "loading" in stages, f"stage was never reported: {stages}"
    settled = tmp_db.execute(
        "SELECT status, stage FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    assert settled["status"] == "done" and settled["stage"] is None


def test_heartbeat_never_touches_a_settled_job(tmp_db):
    """A late flush must not resurrect progress on a finished job."""
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    worker.claim_next(tmp_db)
    tmp_db.execute(
        "UPDATE jobs SET status = 'done', progress = NULL, stage = NULL WHERE id = ?",
        (job_id,),
    )
    tmp_db.commit()

    writer = worker.JobStatusWriter(tmp_db, job_id, interval=0.05)
    writer.note(stage="transcribing", fraction=0.5)
    writer.start()
    time.sleep(0.3)
    writer.stop()

    row = tmp_db.execute(
        "SELECT status, stage, progress FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    assert (row["status"], row["stage"], row["progress"]) == ("done", None, None)


def test_seconds_since_reports_age(tmp_db):
    from captioner.worker import seconds_since

    assert seconds_since(None) > 1_000_000  # unknown counts as very old
    assert seconds_since("not a timestamp") > 1_000_000
    assert seconds_since(time.strftime("%Y-%m-%d %H:%M:%S")) < 5


#crash recovery


def test_recover_stale_leaves_a_live_job_alone(tmp_db):
    worker.enqueue(tmp_db, "sample.mp4")
    worker.claim_next(tmp_db)  # heartbeat_at = now
    assert worker.recover_stale(tmp_db) == 0


def test_recover_stale_requeues_an_orphaned_job(tmp_db):
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    worker.claim_next(tmp_db)
    tmp_db.execute(
        "UPDATE jobs SET heartbeat_at = '2020-01-01 00:00:00' WHERE id = ?", (job_id,)
    )
    tmp_db.commit()

    assert worker.recover_stale(tmp_db) == 1

    row = tmp_db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "queued"
    assert row["heartbeat_at"] is None and row["started_at"] is None
    assert row["error"] is None and row["progress"] is None


def test_recover_stale_handles_a_missing_heartbeat(tmp_db):
    """Rows written before heartbeats existed (or hard-killed) have no stamp."""
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    worker.claim_next(tmp_db)
    tmp_db.execute("UPDATE jobs SET heartbeat_at = NULL WHERE id = ?", (job_id,))
    tmp_db.commit()

    assert worker.recover_stale(tmp_db) == 1


def test_recover_stale_only_touches_running_jobs(tmp_db):
    done_id = worker.enqueue(tmp_db, "done.mp4")
    error_id = worker.enqueue(tmp_db, "error.mp4")
    tmp_db.execute("UPDATE jobs SET status = 'done' WHERE id = ?", (done_id,))
    tmp_db.execute("UPDATE jobs SET status = 'error' WHERE id = ?", (error_id,))
    tmp_db.commit()

    assert worker.recover_stale(tmp_db) == 0
    assert tmp_db.execute(
        "SELECT status FROM jobs WHERE id = ?", (done_id,)
    ).fetchone()["status"] == "done"
    assert tmp_db.execute(
        "SELECT status FROM jobs WHERE id = ?", (error_id,)
    ).fetchone()["status"] == "error"


def test_recovered_job_can_be_processed_again(tmp_db, fake_words):
    """The end-to-end story: crash, recover, transcribe successfully."""
    job_id = worker.enqueue(tmp_db, "sample.mp4")
    worker.claim_next(tmp_db)
    tmp_db.execute(
        "UPDATE jobs SET heartbeat_at = '2020-01-01 00:00:00' WHERE id = ?", (job_id,)
    )
    tmp_db.commit()

    assert worker.recover_stale(tmp_db) == 1
    assert worker.run_once(tmp_db) is True
    assert tmp_db.execute(
        "SELECT status FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()["status"] == "done"
