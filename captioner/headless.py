"""One caption job with no window: the packaged app's ``--job`` mode.

A packaged build is a double-clicked application, but the thing it is for -
a recording turned into subtitles - is also what a script, a scheduled task,
or somebody already at a command prompt wants without a window opening. This
module is that path:

    Captioner.exe --job lecture.mp3 --align --translate-to fr

It queues the file with the options given, claims *that* job (not the oldest
one in the queue, which ``worker.run_once`` would take) and runs exactly the
pipeline the dashboard's worker runs. Nothing here is a second implementation:
the queue, the segmenter, the SRT/VTT writer, alignment and translation are the
same functions.

It doubles as the honest test of a build. A slim build has no torch at all, so
running a job with ``--align`` fails in the same place a user's job would,
which is worth more than reporting that a module could be imported.

Exit codes: 0 the job finished, 1 it failed or could not start, 2 the arguments
were wrong. A finished job with a translation warning is still 0 - the
transcript is what was asked for.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import config, db, worker
from .languages import spoken_language

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Captioner --job",
        description="Caption one file without opening the window.",
    )
    parser.add_argument("file", help="the media file to caption")
    parser.add_argument("--align", action="store_true",
                        help="re-time words with wav2vec2 forced alignment "
                             "(needs the full build)")
    parser.add_argument("--diarize", action="store_true",
                        help="label speakers (needs the full build and a token)")
    parser.add_argument("--translate-to", metavar="LANG", default=None,
                        help="also write captions.<lang>.srt/.vtt "
                             "(needs the full build)")
    parser.add_argument("--language", "-l", metavar="CODE", default=None,
                        help="spoken language, e.g. hi or de (default: detect it)")
    parser.add_argument("--model", default=None,
                        help="override the batch model, e.g. small.en for a quick "
                             "pass (default: the app's distil-large-v3)")
    return parser


def _problem(args) -> str | None:
    """Everything that can be wrong before a job is queued, in one place."""
    path = Path(args.file)
    if not path.exists():
        return f"file not found: {path}"
    if not path.is_file():
        return f"not a file: {path}"
    try:
        spoken_language(args.language)
        if args.translate_to:
            from .translate import language_code

            language_code(args.translate_to)
    except ValueError as exc:
        return str(exc)
    return None


def _setup_logging() -> None:
    """Log to the app's log file, and to the console when there is one.

    A windowed executable started by double-click would otherwise swallow every
    message, including the model download that explains a long first run.
    """
    handlers: list[logging.Handler] = []
    try:
        log_dir = config.DATA_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "captioner.log", encoding="utf-8"))
    except OSError:
        pass
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    if handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=handlers,
        )
        # The hub's HTTP client logs every HEAD request at INFO - 17 KB of them
        # before the first word of a cold run. The progress bars and the
        # warnings still come through; the request log belongs in a debug log.
        for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _out_dir(job_id: int) -> Path:
    return Path(config.OUTPUT_DIR) / f"job_{job_id}"


def _report(job_id: int, row, seconds: float) -> int:
    """Say what happened, where the files are, and whether that is a success."""
    name = row["file_name"]
    if row["status"] != "done":
        reason = row["error"] or f"status {row['status']}"
        print(f"job #{job_id} ({name}) failed: {reason}", file=sys.stderr,
              flush=True)
        return 1

    heard = row["detected_lang"] or row["source_lang"]
    facts = [f"{seconds:.0f}s" if seconds >= 1 else "<1s"]
    if heard:
        facts.append(f"heard {heard}")
    if row["align"]:
        facts.append("aligned")
    if row["diarize"]:
        facts.append("speakers labelled")
    if row["translate_to"]:
        facts.append(f"translated to {row['translate_to']}")
    # Flushed as it goes: the packaged entry may leave the process right after
    # this, and output still sitting in a block buffer would be lost.
    print(f"job #{job_id} ({name}) done in {', '.join(facts)}", flush=True)

    files = sorted(_out_dir(job_id).glob("captions*"))
    for path in files:
        print(f"  {path}", flush=True)
    if row["error"]:  # a warning: the transcript was written, the extra step failed
        print(f"  warning: {row['error']}", file=sys.stderr, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    problem = _problem(args)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2

    _setup_logging()
    config.ensure_dirs()
    conn = db.connect()
    try:
        job_id = worker.enqueue(
            conn,
            str(Path(args.file).resolve()),
            model_size=args.model,
            diarize=args.diarize,
            align=args.align,
            translate_to=args.translate_to,
            source_lang=spoken_language(args.language),
        )
        job = worker.claim_job(conn, job_id)
        if not job:
            print(f"error: job #{job_id} could not be started", file=sys.stderr)
            return 1
        extras = [name for name, on in (
            ("align", args.align), ("speakers", args.diarize),
            (f"translate:{args.translate_to}", bool(args.translate_to)),
        ) if on]
        print(f"job #{job_id}: {Path(args.file).name} "
              f"({job['model_size']}{', ' + ', '.join(extras) if extras else ''})",
              flush=True)
        started = time.monotonic()
        worker.run_pipeline(conn, job)
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _report(job_id, row, time.monotonic() - started)
    finally:
        conn.close()
