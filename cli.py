"""Command-line interface for the captioner.

Lets you drive the whole pipeline without a GUI:
  python cli.py add <file>        queue a file
  python cli.py status            show jobs
  python cli.py run-once          process one queued job
  python cli.py work              poll the queue forever (the worker process)
  python cli.py retry <job_id>    requeue a failed job
  python cli.py translate <job>   add subtitles in another language
  python cli.py languages         list translation targets
"""
import argparse
import sys
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
        "SELECT id, file_name, status, error, created_at, finished_at "
        "FROM jobs ORDER BY id"
    ).fetchall()
    if not rows:
        print("no jobs yet - add one with: python cli.py add <file>")
        return 0
    width = max(len(r["file_name"]) for r in rows)
    for r in rows:
        line = f"#{r['id']:<4} {r['status']:<8} {r['file_name']:<{width}}"
        if r["error"]:
            line += f"  ({r['error']})"
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
    for code, (_nllb, name) in sorted(LANGUAGES.items(), key=lambda kv: kv[1][1]):
        print(f"  {code:<4} {name}")
    return 0


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

    args = parser.parse_args(argv)
    config.ensure_dirs()
    conn = db.connect()
    try:
        return args.func(args, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
