"""Batch worker: claims queued jobs and runs the transcription pipeline.

Runs as a separate PROCESS from the UI (deliberately): a crash in ffmpeg or
model inference must never take the dashboard down. This is the Celery-worker
pattern in miniature — the SQLite jobs table stands in for Redis, and the
claim step is a guarded transaction so N workers can run concurrently
without ever taking the same job.

Pipeline per job:
  mark running -> transcribe to words -> segment into caption blocks
  -> store captions -> write SRT/VTT -> mark done (or error + message).
"""
import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path

from . import automation, config, srt
from .engines import asr as asr_engine
from .segmenter import Block, segment_words
from .transcribe import TranscriptionCancelled

logger = logging.getLogger(__name__)

POLL_SECONDS = config.WORKER_POLL_SECONDS


def transcribe_words(
    audio_path: str,
    model_size: str = None,
    on_progress=None,
    on_stage=None,
    should_cancel=None,
):
    """Transcribe through the configured ASR backend.

    The backend is chosen by ``captioner.engines`` (``CAPTIONER_ASR_ENGINE``,
    default local faster-whisper). This stays a module-level function so the
    pipeline and the tests both keep calling one name.
    """
    return asr_engine().transcribe_words(
        audio_path,
        model_size,
        on_progress=on_progress,
        on_stage=on_stage,
        should_cancel=should_cancel,
    )

# A running job whose heartbeat is older than this has no live owner: the
# process that claimed it is gone (crash, closed window, killed terminal).
STALE_AFTER_SECONDS = 120.0


class JobStatusWriter:
    """Keeps a running job's heartbeat and stage columns fresh.

    Downloading model weights and loading a model can block for minutes with
    no progress callback at all, so "is this job alive?" cannot be read off
    the progress column. This writer owns its own SQLite connection for two
    reasons: sqlite3 objects belong to the thread that made them, and the
    reports arrive from whichever thread has news (the weight downloader
    counts bytes from its own).

    Writes are guarded with ``status='running'`` so a late flush can never
    resurrect a job that has already settled.
    """

    INTERVAL = 1.0

    def __init__(self, conn, job_id: int, interval: float = INTERVAL):
        self.job_id = job_id
        self.interval = interval
        self._path = conn.execute("PRAGMA database_list").fetchone()[2]
        self._lock = threading.Lock()
        self._stage: str | None = None
        self._fraction: float | None = None
        self._dirty = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    #called from any thread
    def note(self, stage: str | None = None, fraction: float | None = None) -> None:
        """Record the job's latest stage (and fraction, if one is known)."""
        with self._lock:
            if stage is not None:
                self._stage = stage
            if fraction is not None:
                self._fraction = max(0.0, min(1.0, float(fraction)))
            self._dirty = True

    #lifecycle
    def start(self) -> None:
        if not self._path:
            return  # in-memory database (tests): nothing to heartbeat
        self._flush()  # do not wait a whole interval for the first beat
        self._thread = threading.Thread(
            target=self._run, name=f"job-{self.job_id}-heartbeat", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        try:
            conn = sqlite3.connect(self._path, timeout=10.0)
        except sqlite3.Error:  # pragma: no cover - defensive
            return
        try:
            while not self._stop.wait(self.interval):
                self._flush(conn)
        finally:
            conn.close()

    def _flush(self, conn: sqlite3.Connection | None = None) -> None:
        own = conn is None
        if own:
            try:
                conn = sqlite3.connect(self._path, timeout=10.0)
            except sqlite3.Error:  # pragma: no cover - defensive
                return
        with self._lock:
            stage, fraction, dirty = self._stage, self._fraction, self._dirty
            self._dirty = False
        try:
            if dirty:
                conn.execute(
                    "UPDATE jobs SET heartbeat_at = ?, stage = ?, progress = ? "
                    "WHERE id = ? AND status = 'running'",
                    (_now(), stage, fraction, self.job_id),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND status = 'running'",
                    (_now(), self.job_id),
                )
            conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - bookkeeping only
            logger.debug("heartbeat write failed for job %s: %s", self.job_id, exc)
        finally:
            if own:
                conn.close()


DIGEST_CHUNK = 1024 * 1024
# Statuses that mean "this content is already handled". Error and canceled are
# deliberately absent: those rows exist to be retried, not skipped.
HANDLED_STATUSES = ("running", "queued", "done")


def file_digest(path: str | Path) -> str | None:
    """A fast fingerprint of a file's content, or None if it cannot be read.

    Size plus the first and last megabyte, not the whole file: a two-hour
    lecture would otherwise cost a full disk read before it could even be
    queued. That is enough to tell one recording from another, and it is only
    ever compared against other rows - delete a job to transcribe its audio
    again.
    """
    target = Path(path)
    try:
        size = target.stat().st_size
        digest = hashlib.sha256(str(size).encode("utf-8"))
        with target.open("rb") as handle:
            digest.update(handle.read(DIGEST_CHUNK))
            if size > DIGEST_CHUNK:
                handle.seek(max(DIGEST_CHUNK, size - DIGEST_CHUNK))
                digest.update(handle.read(DIGEST_CHUNK))
    except OSError:  # missing, unreadable, or a directory
        return None
    return digest.hexdigest()


def enqueue(
    conn,
    file_path: str,
    model_size: str = None,
    diarize: bool = False,
    align: bool = False,
    translate_to: str | None = None,
) -> int:
    """Add a file to the queue; returns the job id.

    ``align`` / ``translate_to`` / ``diarize`` are stored on the row rather
    than passed to the worker, so a job keeps the options it was queued with
    even if the UI checkboxes change (or another worker picks it up). The row
    also carries a content digest, which is what ``find_duplicate`` matches on.
    """
    p = Path(file_path)
    cur = conn.execute(
        "INSERT INTO jobs (file_path, file_name, file_hash, model_size, "
        "created_at, diarize, align, translate_to) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(p.resolve()),
            p.name,
            file_digest(p),
            model_size or config.BATCH_MODEL_SIZE,
            _now(),
            1 if diarize else 0,
            1 if align else 0,
            translate_to or None,
        ),
    )
    conn.commit()
    return cur.lastrowid


def find_duplicate(conn, file_path: str | Path) -> dict | None:
    """An existing job for the same content, if the queue has already seen it.

    The same lecture copied under a second name, or downloaded twice into the
    watch folder, should not cost another full transcription. Callers decide
    what to do about it: the API answers with the job it already has, the CLI
    and the dashboard say so and skip the file. A path that cannot be read has
    no digest and therefore never matches.

    Finished jobs count as duplicates. Re-transcribing the same audio is
    "delete the job, then queue it again", which keeps the rule simple.
    """
    digest = file_digest(file_path)
    if digest is None:
        return None
    row = conn.execute(
        "SELECT * FROM jobs WHERE file_hash = ? AND status IN (?, ?, ?) "
        "ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END, "
        "id DESC LIMIT 1",
        (digest, *HANDLED_STATUSES),
    ).fetchone()
    return dict(row) if row is not None else None


def claim_next(conn) -> dict:
    """Atomically claim the oldest queued job, or return {}."""
    with conn:  # implicit transaction; rolls back on error
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            return {}
        now = _now()
        conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, heartbeat_at = ?, "
            "error = NULL, progress = NULL, stage = NULL "
            "WHERE id = ? AND status = 'queued'",
            (now, now, row["id"]),
        )
    return dict(row)


def run_pipeline(conn, job: dict, cancel: threading.Event | None = None) -> None:
    """Execute one claimed job; flips it to done, error, or canceled.

    Two ways to stop a job, checked between segments so a long file stops
    within a few seconds instead of finishing first:

    * ``cancel`` — an in-process event the dashboard sets. Only useful when
      this process owns the job.
    * the ``cancel_requested`` column — set by any client (the API, the CLI,
      another dashboard) and polled here, so a worker started in its own
      console can be stopped by a client that never imported it.
    """
    job_id = job["id"]
    status = JobStatusWriter(conn, job_id)
    status.start()

    def on_progress(fraction: float) -> None:
        _set_progress(conn, job_id, fraction)  # caller's thread: visible at once
        status.note(stage="transcribing", fraction=fraction)

    def on_stage(stage: str, fraction: float | None) -> None:
        status.note(stage=stage, fraction=fraction)

    def should_cancel() -> bool:
        if cancel is not None and cancel.is_set():
            return True
        return cancel_requested(conn, job_id)

    try:
        words = transcribe_words(
            job["file_path"],
            job["model_size"],
            on_progress=on_progress,
            on_stage=on_stage,
            should_cancel=should_cancel,
        )

        # Phase 2: optional wav2vec2 alignment. Runs on Whisper's words before
        # segmentation so the blocks are built from the tighter timings.
        if bool(job.get("align")):
            from .align import realign_file

            on_stage("aligning", None)
            words = realign_file(job["file_path"], words, on_stage=on_stage)

        # Phase 2: optional speaker labels ("[SPEAKER_00]: text" lines).
        diarize_flag = bool(job.get("diarize"))
        if diarize_flag:
            from .diarize import speaker_blocks

            on_stage("diarizing", None)
            turns = _diarize_turns(job["file_path"])
            blocks = speaker_blocks(words, turns)
        else:
            blocks = segment_words(words)

        _store_captions(conn, job_id, blocks, diarize=diarize_flag)
        out_dir = Path(config.OUTPUT_DIR) / f"job_{job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "captions.srt").write_text(
            srt.to_srt(blocks, speaker_names=diarize_flag), encoding="utf-8"
        )
        (out_dir / "captions.vtt").write_text(
            srt.to_vtt(blocks, speaker_names=diarize_flag), encoding="utf-8"
        )

        # Phase 2: optional translation, written as extra caption files. A
        # failed translation must not throw away a finished transcript, so it
        # is reported in the log and on the job row rather than failing it.
        translate_to = job.get("translate_to")
        warning = None
        if translate_to:
            warning = _translate_output(out_dir, translate_to, diarize_flag, on_stage)

        _settle(conn, job_id, "done", warning)
        logger.info("Job %d done: %d caption blocks%s", job_id, len(blocks),
                    f" ({warning})" if warning else "")
    except TranscriptionCancelled:
        _settle(conn, job_id, "canceled")
        logger.info("Job %d canceled by the user", job_id)
    except Exception as exc:  # noqa: BLE001 - worker must survive anything
        logger.exception("Job %d failed", job_id)
        _settle(conn, job_id, "error", f"{type(exc).__name__}: {exc}")
    finally:
        status.stop()
        # A settled job is the automation trigger. The webhook is optional and
        # best-effort, so it runs last and cannot change the job's outcome.
        notify_webhook(conn, job_id)


def notify_webhook(conn, job_id: int) -> bool:
    """Fire the configured webhook for a settled job (never raises)."""
    try:
        return automation.notify_job(conn, job_id)
    except Exception as exc:  # noqa: BLE001 - notification only
        logger.warning("webhook notification failed for job %s: %s", job_id, exc)
        return False


def _translate_output(out_dir: Path, target: str, diarized: bool,
                      on_stage) -> str | None:
    """Write ``captions.<lang>.srt/.vtt`` for a finished job (best effort).

    Returns a warning string when translation failed: the transcript itself is
    already finished and valid at this point, so a missing translation model
    must never flip the job to error.
    """
    from .translate import translate_subtitles

    try:
        on_stage("translating", None)
        translate_subtitles(out_dir, target, speaker_names=diarized)
        return None
    except Exception as exc:  # noqa: BLE001 - the transcript is already valid
        logger.warning("translation to %s failed: %s: %s", target, type(exc).__name__, exc)
        return f"translation to {target} failed: {type(exc).__name__}: {exc}"


def _diarize_turns(path: str):
    """Speaker turns for *path*, isolated so pyannote stays optional."""
    from .diarize import diarize

    return diarize(path)


def _settle(conn, job_id: int, status: str, error: str | None = None) -> None:
    """Mark a job finished, clearing the live-run bookkeeping columns.

    ``cancel_requested`` is cleared too: it has been honored by now, and a
    leftover flag would stop the job the moment it was retried.
    """
    conn.execute(
        "UPDATE jobs SET status = ?, progress = NULL, stage = NULL, "
        "error = ?, finished_at = ?, cancel_requested = 0 WHERE id = ?",
        (status, error, _now(), job_id),
    )
    conn.commit()


def cancel_requested(conn, job_id: int) -> bool:
    """Has any client asked for this job to stop?"""
    try:
        row = conn.execute(
            "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    except sqlite3.Error as exc:  # pragma: no cover - bookkeeping only
        logger.debug("cancel check failed for job %s: %s", job_id, exc)
        return False
    return bool(row is not None and row["cancel_requested"])


def request_cancel(conn, job_id: int) -> str | None:
    """Ask a job to stop, from any process.

    Returns ``"canceled"`` when a queued job was called off before it started
    (nothing has to cooperate), ``"requested"`` when a running job will stop at
    the next segment boundary, or ``None`` when the job has already settled and
    there is nothing left to stop.
    """
    with conn:  # both updates are single statements, so the race is benign
        changed = conn.execute(
            "UPDATE jobs SET status = 'canceled', finished_at = ?, "
            "progress = NULL, stage = NULL, heartbeat_at = NULL "
            "WHERE id = ? AND status = 'queued'",
            (_now(), job_id),
        ).rowcount
        if changed:
            return "canceled"
        changed = conn.execute(
            "UPDATE jobs SET cancel_requested = 1 WHERE id = ? AND status = 'running'",
            (job_id,),
        ).rowcount
        if changed:
            return "requested"
    return None


def recover_stale(conn, max_age_seconds: float = STALE_AFTER_SECONDS) -> int:
    """Requeue jobs left 'running' by a worker that is no longer alive.

    A crash cannot flip its own job back to queued, and the UI deliberately
    refuses to retry or delete a running job (a live worker owns it) — which
    used to leave such a row stuck forever with no way out but hand-written
    SQL. The heartbeat is what separates the two cases: a job whose heartbeat
    stopped moving has no owner.
    """
    now = time.time()
    stale = [
        row["id"]
        for row in conn.execute("SELECT id, heartbeat_at FROM jobs WHERE status = 'running'")
        if _age_seconds(row["heartbeat_at"], now) > max_age_seconds
    ]
    requeued = 0
    for job_id in stale:
        if cancel_requested(conn, job_id):
            # The user asked for a stop and the worker died before honoring it;
            # honoring it now is what they wanted, not another attempt.
            conn.execute(
                "UPDATE jobs SET status = 'canceled', progress = NULL, "
                "stage = NULL, heartbeat_at = NULL, finished_at = ?, "
                "cancel_requested = 0 WHERE id = ?",
                (_now(), job_id),
            )
            logger.info("Job %d was canceled while its worker was down", job_id)
            continue
        conn.execute(
            "UPDATE jobs SET status = 'queued', error = NULL, progress = NULL, "
            "stage = NULL, heartbeat_at = NULL, started_at = NULL, "
            "finished_at = NULL WHERE id = ?",
            (job_id,),
        )
        requeued += 1
    conn.commit()
    return requeued


def _age_seconds(stamp: str | None, now: float) -> float:
    """Seconds since a DB timestamp; unreadable or missing counts as forever."""
    if not stamp:
        return _FOREVER
    try:
        return max(0.0, now - time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S")))
    except ValueError:
        return _FOREVER


# Large but finite: "very old" must stay comparable and printable (infinite
# floats blow up int() when the UI buckets them).
_FOREVER = 1e9


def seconds_since(stamp: str | None) -> float:
    """How long ago a DB timestamp was written (for staleness display)."""
    return _age_seconds(stamp, time.time())


def _store_captions(conn, job_id: int, blocks: list[Block], diarize: bool = False) -> None:
    conn.execute("DELETE FROM captions WHERE job_id = ?", (job_id,))
    conn.executemany(
        "INSERT INTO captions (job_id, seq, start_ms, end_ms, text) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (job_id, i, int(b.start * 1000), int(b.end * 1000),
             (f"[{b.words[0].speaker}]: {b.text}"
              if diarize and b.words and getattr(b.words[0], "speaker", None)
              else b.text))
            for i, b in enumerate(blocks, start=1)
        ],
    )
    conn.commit()


def _set_progress(conn, job_id: int, fraction: float) -> None:
    """Persist transcription progress, stored and compared as whole percents.

    The UI polls this table once a second, so writing only when the percent
    actually changes keeps the write rate trivial even for hour-long files
    (WAL mode means those small writes never block the readers). Comparing in
    integer percent space avoids floating-point noise deciding whether a
    real update gets skipped.
    """
    percent = int(round(max(0.0, min(1.0, float(fraction))) * 100))
    row = conn.execute(
        "SELECT progress FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    if row is not None and row["progress"] is not None:
        if int(round(row["progress"] * 100)) == percent:
            return
    conn.execute(
        "UPDATE jobs SET progress = ? WHERE id = ?", (percent / 100.0, job_id)
    )
    conn.commit()


def run_once(conn, cancel: threading.Event | None = None) -> bool:
    """Claim and process a single job. Returns True if work was done."""
    job = claim_next(conn)
    if not job:
        return False
    run_pipeline(conn, job, cancel=cancel)
    return True


def run_forever(conn, cancel: threading.Event | None = None) -> None:
    """Poll the queue until killed. Multiple instances may run safely."""
    logger.info("Worker polling every %.1fs...", POLL_SECONDS)
    while True:
        if not run_once(conn, cancel=cancel):
            time.sleep(POLL_SECONDS)


def retry(conn, job_id: int) -> None:
    """Queue a settled job again, clearing any stale cancel request."""
    conn.execute(
        "UPDATE jobs SET status = 'queued', error = NULL, progress = NULL, "
        "stage = NULL, heartbeat_at = NULL, started_at = NULL, "
        "finished_at = NULL, cancel_requested = 0 WHERE id = ?",
        (job_id,),
    )
    conn.commit()


def delete_job(conn, job_id: int) -> bool:
    """Remove a job and its captions. Returns False if missing or running.

    Refusing 'running' jobs keeps the DB consistent with a worker that is
    mid-pipeline (its final UPDATE would target a row that no longer exists).
    Subtitle files already written to data/outputs are left on disk —
    removing a queue entry should never destroy something the user has
    already shared or copied.
    """
    row = conn.execute(
        "SELECT status FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    if row is None or row["status"] == "running":
        return False
    with conn:
        conn.execute("DELETE FROM captions WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return True


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
