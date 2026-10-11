"""Command-line interface for the captioner.

Lets you drive the whole pipeline without a GUI:
  python cli.py add <file>        queue a file
  python cli.py status            show jobs
  python cli.py run-once          process one queued job
  python cli.py work              poll the queue forever (the worker process)
  python cli.py cancel <job_id>   stop a queued or running job
  python cli.py retry <job_id>    requeue a failed job
  python cli.py translate <job>   add subtitles in another language
  python cli.py languages         list translation targets
  python cli.py watch             queue new files from the watched folders
  python cli.py api               serve the HTTP API and browser UI
"""
import argparse
import sys
import time
from pathlib import Path

from captioner import config, db, worker


def cmd_add(args, conn) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"error: file not found: {path}")
        return 1
    if args.translate_to:
        from captioner.translate import language_code

        try:
            language_code(args.translate_to)
        except ValueError as exc:
            print(f"error: {exc}")
            return 2
    existing = worker.find_duplicate(conn, path)
    if existing is not None:
        print(f"already has a job: #{existing['id']} ({existing['status']}) - "
              "delete it first to transcribe this audio again")
        return 0
    job_id = worker.enqueue(
        conn, str(path),
        diarize=args.diarize, align=args.align, translate_to=args.translate_to,
    )
    extras = [name for name, on in
              (("speakers", args.diarize), ("alignment", args.align),
               (f"translate:{args.translate_to}", bool(args.translate_to))) if on]
    suffix = f" [{', '.join(extras)}]" if extras else ""
    print(f"queued job #{job_id}: {path.name}{suffix}")
    return 0


def cmd_status(args, conn) -> int:
    rows = conn.execute(
        "SELECT id, file_name, status, error, created_at, finished_at, "
        "cancel_requested FROM jobs ORDER BY id"
    ).fetchall()
    if not rows:
        print("no jobs yet - add one with: python cli.py add <file>")
        return 0
    width = max(len(r["file_name"]) for r in rows)
    for r in rows:
        line = f"#{r['id']:<4} {r['status']:<8} {r['file_name']:<{width}}"
        if r["error"]:
            line += f"  ({r['error']})"
        if r["cancel_requested"]:
            # A running job only notices at the next segment boundary.
            line += "  (stopping)"
        print(line)
    return 0


def cmd_run_once(args, conn) -> int:
    did_work = worker.run_once(conn)
    print("processed one job" if did_work else "queue is empty")
    return 0


def cmd_work(args, conn) -> int:
    print("worker running — Ctrl+C to stop")
    try:
        worker.run_forever(conn)
    except KeyboardInterrupt:
        print("\nworker stopped")
    return 0


def cmd_cancel(args, conn) -> int:
    """Call off a queued job, or ask a running one to stop at the next boundary."""
    row = conn.execute(
        "SELECT id, status FROM jobs WHERE id = ?", (args.job_id,)
    ).fetchone()
    if row is None:
        print(f"error: no job #{args.job_id}")
        return 1
    outcome = worker.request_cancel(conn, args.job_id)
    if outcome is None:
        print(f"job #{row['id']} is already {row['status']} - nothing to stop")
        return 0
    if outcome == "canceled":
        print(f"job #{row['id']} canceled before it started")
    else:
        print(f"job #{row['id']} will stop at the next segment boundary")
    return 0


def cmd_retry(args, conn) -> int:
    worker.retry(conn, args.job_id)
    print(f"job #{args.job_id} requeued")
    return 0


def cmd_translate(args, conn) -> int:
    """Translate a finished job's captions without re-transcribing."""
    from captioner.translate import Translator, translate_subtitles

    row = conn.execute(
        "SELECT id, file_name, status, diarize FROM jobs WHERE id = ?", (args.job_id,)
    ).fetchone()
    if row is None:
        print(f"error: no job #{args.job_id}")
        return 1
    out_dir = Path(config.OUTPUT_DIR) / f"job_{row['id']}"
    translator = Translator(model_name=args.model) if args.model else None
    try:
        paths = translate_subtitles(
            out_dir, args.to, translator=translator,
            speaker_names=bool(row["diarize"]),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"translated job #{row['id']} ({row['file_name']}) -> "
          + ", ".join(p.name for p in paths))

    # Remember the language on the row so the dashboard can show it too.
    conn.execute("UPDATE jobs SET translate_to = ? WHERE id = ?", (args.to, row["id"]))
    conn.commit()
    return 0


def cmd_languages(args, conn) -> int:
    from captioner.translate import LANGUAGES

    print("translation targets:")
    for code, name in sorted(LANGUAGES.items(), key=lambda kv: kv[1]):
        print(f"  {code:<4} {name}")
    return 0


def cmd_watch(args, conn) -> int:
    """Queue new media from the watched folders, forever.

    The scan is idempotent: a path already in the jobs table is never queued
    twice, so the loop can run for days and survive a restart.
    """
    from captioner import automation

    folders = automation.watch_dirs(args.dir)
    if not folders:
        print("no watched folders exist; create one or pass --dir")
        return 1
    print("watching:")
    for folder in folders:
        print(f"  {folder}")
    print(f"polling every {args.interval:g}s - Ctrl+C to stop")
    try:
        while True:
            for path in automation.find_new_media(conn, folders):
                job_id = worker.enqueue(
                    conn, str(path), diarize=args.diarize, align=args.align,
                    translate_to=args.translate_to,
                )
                print(f"queued job #{job_id}: {path.name}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped watching")
    return 0


def cmd_api(args, conn) -> int:
    """Serve the HTTP API and its browser UI until interrupted."""
    from captioner import api

    return api.serve(
        host=args.host, port=args.port, token=args.token, open_browser=args.open
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="captioner")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add", help="queue a media file for captioning")
    p.add_argument("file")
    p.add_argument("--diarize", action="store_true", help="label speakers (pyannote)")
    p.add_argument("--align", action="store_true",
                   help="re-time words with wav2vec2 forced alignment")
    p.add_argument("--translate-to", metavar="LANG", default=None,
                   help="also write captions.<lang>.srt/.vtt (see: languages)")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("status", help="list all jobs")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("run-once", help="process a single queued job")
    p.set_defaults(func=cmd_run_once)

    p = sub.add_parser("work", help="run the worker loop (polls forever)")
    p.set_defaults(func=cmd_work)

    p = sub.add_parser("cancel", help="stop a queued job, or a running one")
    p.add_argument("job_id", type=int)
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("retry", help="requeue a failed/finished job")
    p.add_argument("job_id", type=int)
    p.set_defaults(func=cmd_retry)

    p = sub.add_parser("translate", help="translate a finished job's captions")
    p.add_argument("job_id", type=int)
    p.add_argument("--to", required=True, help="target language code, e.g. hi")
    p.add_argument("--model", default=None,
                   help="override the MT model (default facebook/m2m100_418M; "
                        "try facebook/m2m100_1.2B for better quality, ~4.8 GB)")
    p.set_defaults(func=cmd_translate)

    p = sub.add_parser("languages", help="list translation targets")
    p.set_defaults(func=cmd_languages)

    p = sub.add_parser("watch", help="queue new files from the watched folders")
    p.add_argument("--dir", action="append", default=None, metavar="PATH",
                   help="extra folder to watch (repeatable; CAPTIONER_WATCH_DIRS "
                        "adds more)")
    p.add_argument("--interval", type=float, default=5.0, metavar="SECONDS",
                   help="how often to rescan (default: 5)")
    p.add_argument("--diarize", action="store_true", help="label speakers")
    p.add_argument("--align", action="store_true", help="re-time words with forced alignment")
    p.add_argument("--translate-to", metavar="LANG", default=None,
                   help="also write captions.<lang>.srt/.vtt")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("api", help="serve the HTTP API + browser UI")
    p.add_argument("--host", default=None, help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="port (default: 8765)")
    p.add_argument("--token", default=None,
                   help="require this bearer token (default: CAPTIONER_API_TOKEN)")
    p.add_argument("--open", action="store_true", help="open the browser UI")
    p.set_defaults(func=cmd_api)

    args = parser.parse_args(argv)
    config.ensure_dirs()
    conn = db.connect()
    try:
        return args.func(args, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
