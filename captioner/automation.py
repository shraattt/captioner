"""Watch folders and outbound webhooks.

Two pieces of glue that let the queue run unattended:

* ``watch_dirs()`` / ``find_new_media()`` - folders scanned for new media.
  ``data/uploads`` is always watched (that is what the dashboard's watch
  toggle scans); ``CAPTIONER_WATCH_DIRS`` adds more, separated by the
  platform's path separator or a semicolon.
* ``notify_job()`` - one JSON POST when a job settles, so a script, a Slack
  relay or an automation tool can react. Off unless ``CAPTIONER_WEBHOOK_URL``
  is set.

Both are stdlib-only and best-effort on purpose: a webhook that times out must
never change a job's outcome, and an unreadable folder must never stop the
queue.
"""
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

WATCH_ENV = "CAPTIONER_WATCH_DIRS"
WEBHOOK_ENV = "CAPTIONER_WEBHOOK_URL"
WEBHOOK_SECRET_ENV = "CAPTIONER_WEBHOOK_SECRET"
WEBHOOK_TIMEOUT = 5.0

# Media extensions the scanner accepts (same set the dashboard drag-drop uses).
MEDIA_EXTS = frozenset({
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".m4v", ".mpg", ".mpeg",
    ".ts", ".3gp", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus",
    ".wma", ".aiff", ".aif",
})

# Files still being written by a browser download or a copy are not ready.
SKIP_SUFFIXES = (".part", ".tmp", ".crdownload", ".filepart", ".download", ".partial")
SKIP_PREFIXES = (".", "~$")

# A file younger than this may still be growing; waiting avoids queueing half
# a download and getting a decode error for no reason.
MIN_AGE_SECONDS = 2.0

# A transport takes a prepared request and a timeout, returning (status, body).
Transport = Callable[[urllib.request.Request, float], tuple[int, bytes]]


def _split_env_list(raw: str) -> list[str]:
    """Split a path list on the platform separator or a semicolon."""
    prepared = raw.replace(";", os.pathsep)
    return [chunk.strip().strip('"') for chunk in prepared.split(os.pathsep) if chunk.strip()]


def watch_dirs(extra: Sequence[str] = None) -> list[Path]:
    """Folders to scan for new media, existing ones only, duplicates removed.

    ``config.UPLOADS_DIR`` is read at call time rather than captured, so a
    caller (or a test) that points it somewhere else is respected.
    """
    candidates: list[str] = [str(config.UPLOADS_DIR)]
    candidates += _split_env_list(os.environ.get(WATCH_ENV, ""))
    candidates += list(extra or [])
    seen, out = set(), []
    for item in candidates:
        path = Path(item).expanduser()
        try:
            key = path.resolve()
        except OSError:  # pragma: no cover - pathological path
            continue
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir():
            out.append(path)
        else:
            logger.debug("watch folder does not exist, skipping: %s", path)
    return out


def _is_ready_media(path: Path, min_age_seconds: float) -> bool:
    name = path.name
    if path.suffix.lower() not in MEDIA_EXTS:
        return False
    if name.startswith(SKIP_PREFIXES) or name.lower().endswith(SKIP_SUFFIXES):
        return False
    try:
        if min_age_seconds and time.time() - path.stat().st_mtime < min_age_seconds:
            return False
    except OSError:
        return False
    return True


def find_new_media(
    conn,
    dirs: Sequence[Path] = None,
    min_age_seconds: float = MIN_AGE_SECONDS,
) -> list[Path]:
    """Media in the watched folders that the queue has not seen yet.

    Two things are skipped, so re-scanning is idempotent: paths already in the
    jobs table, and files whose *content* is already queued or done - a lecture
    copied in under a second name is the same lecture. Oldest first, which for
    a drop-and-walk-away folder is the order the files finished arriving.
    """
    from .worker import file_digest  # local: worker imports this module

    known = {
        str(Path(row[0]).resolve()) if row[0] else ""
        for row in conn.execute("SELECT file_path FROM jobs")
    }
    # Only content the queue has actually handled counts, matching
    # worker.find_duplicate: a file whose earlier attempt errored is worth
    # trying again rather than silently skipping forever.
    seen_content = {
        row[0] for row in conn.execute(
            "SELECT file_hash FROM jobs "
            "WHERE file_hash IS NOT NULL AND status IN ('running', 'queued', 'done')"
        )
    }
    found: list[Path] = []
    for folder in (dirs if dirs is not None else watch_dirs()):
        try:
            entries = list(folder.rglob("*"))
        except OSError as exc:  # pragma: no cover - permissions/locked folder
            logger.warning("could not scan %s: %s", folder, exc)
            continue
        for path in entries:
            if not path.is_file() or not _is_ready_media(path, min_age_seconds):
                continue
            try:
                key = str(path.resolve())
            except OSError:  # pragma: no cover
                continue
            if key in known:
                continue
            digest = file_digest(path)
            if digest is not None and digest in seen_content:
                logger.info("skipping %s: the same content is already queued", path.name)
                known.add(key)
                continue
            known.add(key)
            if digest is not None:
                seen_content.add(digest)
            found.append(path)
    found.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return found


def webhook_url() -> str:
    """Configured webhook target, or an empty string when disabled."""
    return os.environ.get(WEBHOOK_ENV, "").strip()


def post_json(
    url: str,
    payload: dict,
    timeout: float = WEBHOOK_TIMEOUT,
    transport: Transport = None,
    headers: dict[str, str] = None,
) -> bool:
    """POST ``payload`` as JSON. Returns whether the receiver said OK.

    Never raises: automation is a notification, not a step in the pipeline.
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    all_headers = {
        "Content-Type": "application/json",
        "User-Agent": "captioner-webhook/1.0",
    }
    all_headers.update(headers or {})
    request = urllib.request.Request(url, data=body, headers=all_headers, method="POST")
    try:
        if transport is not None:
            status, raw = transport(request, timeout)
        else:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status, raw = response.status, response.read()
            except urllib.error.HTTPError as exc:
                status, raw = exc.code, exc.read()
    except Exception as exc:  # noqa: BLE001 - a webhook failure is not job failure
        logger.warning("webhook to %s failed: %s", url, exc)
        return False
    if not 200 <= status < 300:
        logger.warning("webhook to %s returned HTTP %s: %s",
                       url, status, raw.decode("utf-8", "replace")[:200])
        return False
    return True


def job_event(conn, job_id: int) -> dict | None:
    """The JSON body describing one job, or None when the row is gone."""
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    keys = set(row.keys())
    out_dir = Path(config.OUTPUT_DIR) / f"job_{job_id}"
    outputs = sorted(path.name for path in out_dir.glob("captions*")) if out_dir.is_dir() else []
    caption_count = conn.execute(
        "SELECT COUNT(*) FROM captions WHERE job_id = ?", (job_id,)
    ).fetchone()[0]
    last_end_ms = conn.execute(
        "SELECT MAX(end_ms) FROM captions WHERE job_id = ?", (job_id,)
    ).fetchone()[0]
    job = {
        "id": job_id,
        "file_name": row["file_name"],
        "file_path": row["file_path"],
        "status": row["status"],
        "model": row["model_size"],
        "error": row["error"],
        "created_at": row["created_at"],
        "started_at": row["started_at"] if "started_at" in keys else None,
        "finished_at": row["finished_at"],
        "captions": int(caption_count or 0),
        "duration_ms": int(last_end_ms or 0),
        "outputs": outputs,
        "output_dir": str(out_dir) if out_dir.is_dir() else None,
        "options": {
            "diarize": bool(row["diarize"]),
            "align": bool(row["align"]),
            "translate_to": row["translate_to"],
        },
    }
    return {
        "event": f"job.{row['status']}",
        "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "job": job,
    }


def notify_job(
    conn,
    job_id: int,
    url: str = None,
    transport: Transport = None,
) -> bool:
    """POST one settled job's event. False when disabled or undeliverable."""
    target = (url or webhook_url()).strip()
    if not target:
        return False
    payload = job_event(conn, job_id)
    if payload is None:
        return False
    headers = {}
    secret = os.environ.get(WEBHOOK_SECRET_ENV, "").strip()
    if secret:
        headers["X-Captioner-Secret"] = secret
    sent = post_json(target, payload, transport=transport, headers=headers)
    if sent:
        logger.info("webhook delivered for job %s (%s)", job_id, payload["event"])
    return sent
