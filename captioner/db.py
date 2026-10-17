"""SQLite data layer.

The jobs table doubles as the queue (the local stand-in for Redis+Celery):
workers claim jobs by flipping status 'queued' -> 'running', and the claim is
guarded by an immediate DB transaction so two workers can never take the same
job. Everything is plain SQL — no ORM — so the schema stays inspectable and
ports 1:1 to Postgres if this ever becomes a service.
"""
import sqlite3
from pathlib import Path

from . import config

# A job is settled once it can no longer become something else by itself.
SETTLED_STATUSES = ("done", "error", "canceled")

# Column order for the jobs table. Kept in one place so the migration that
# recreates the table copies whatever the old shape had.
JOB_COLUMNS = (
    "id", "file_path", "file_name", "file_hash", "status", "error",
    "model_size", "progress", "stage", "heartbeat_at",
    "created_at", "started_at", "finished_at",
)
# Per-job processing options added over time (Phase 2 alignment/translation,
# speaker labels). Existing databases gain them through the ALTER TABLE loop in
# _migrate, so each tuple is (column name, DDL) and must stay additive.
OPTIONAL_JOB_COLUMNS = (
    ("diarize", "INTEGER NOT NULL DEFAULT 0"),
    ("align", "INTEGER NOT NULL DEFAULT 0"),
    ("translate_to", "TEXT"),
    # Spoken language: NULL/empty means detect it. detected_lang is what the
    # recogniser actually heard, which is worth showing even when the answer
    # was "you told me" - it is how a wrong guess becomes visible.
    ("source_lang", "TEXT"),
    ("detected_lang", "TEXT"),
    # Set by any client (API, CLI, dashboard) to ask for a stop; the worker
    # running the job polls it, so the request crosses process boundaries.
    ("cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path     TEXT NOT NULL,
    file_name     TEXT NOT NULL,
    file_hash     TEXT,
    status        TEXT NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued','running','done','error','canceled')),
    error         TEXT,
    model_size    TEXT NOT NULL DEFAULT 'medium',
    progress      REAL,
    stage         TEXT,
    heartbeat_at  TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    diarize       INTEGER NOT NULL DEFAULT 0,
    align         INTEGER NOT NULL DEFAULT 0,
    translate_to  TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    source_lang   TEXT,
    detected_lang TEXT
);

CREATE TABLE IF NOT EXISTS captions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER NOT NULL REFERENCES jobs(id),
    seq         INTEGER NOT NULL,
    start_ms    INTEGER NOT NULL,
    end_ms      INTEGER NOT NULL,
    text        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_captions_job ON captions(job_id, seq);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection with sane pragmas and the schema applied.

    Row values come back as sqlite3.Row (indexable by column name). WAL mode
    lets the UI read the database while the worker writes without blocking
    either side.
    """
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent upgrades for databases made by older versions.

    CREATE TABLE IF NOT EXISTS only helps brand-new files; an app update that
    changes the schema has to patch existing databases in place. Adding a
    column is a metadata-only ALTER TABLE; widening the status CHECK constraint
    cannot be done that way and needs the table recreated.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
    ).fetchone()
    if row is not None and "'canceled'" not in (row["sql"] or ""):
        _rebuild_jobs_table(conn)

    columns = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    for name, ddl in (("progress", "REAL"), ("stage", "TEXT"), ("heartbeat_at", "TEXT")) \
            + OPTIONAL_JOB_COLUMNS:
        if name not in columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {ddl}")
    conn.commit()


def _rebuild_jobs_table(conn: sqlite3.Connection) -> None:
    """Recreate ``jobs`` to widen its status CHECK constraint.

    SQLite cannot alter a CHECK, so the documented recreate-and-copy dance is
    the only option. The existing columns are copied in place, ids are
    preserved (the captions table references them), and foreign key
    enforcement is switched off for the swap so dropping the old table cannot
    cascade into the captions.
    """
    old_columns = [r["name"] for r in conn.execute("PRAGMA table_info(jobs)")]
    keep = [c for c in JOB_COLUMNS if c in old_columns]
    count_before = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    new_table = """
    CREATE TABLE jobs_migrated (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path     TEXT NOT NULL,
        file_name     TEXT NOT NULL,
        file_hash     TEXT,
        status        TEXT NOT NULL DEFAULT 'queued'
                      CHECK (status IN ('queued','running','done','error','canceled')),
        error         TEXT,
        model_size    TEXT NOT NULL DEFAULT 'medium',
        progress      REAL,
        stage         TEXT,
        heartbeat_at  TEXT,
        created_at    TEXT NOT NULL,
        started_at    TEXT,
        finished_at   TEXT
    )
    """
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        with conn:
            conn.execute("DROP TABLE IF EXISTS jobs_migrated")
            conn.execute(new_table)
            conn.execute(
                f"INSERT INTO jobs_migrated ({', '.join(keep)}) "
                f"SELECT {', '.join(keep)} FROM jobs"
            )
            conn.execute("DROP TABLE jobs")
            conn.execute("ALTER TABLE jobs_migrated RENAME TO jobs")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            _reset_autoincrement(conn)
    finally:
        conn.execute("PRAGMA foreign_keys=ON")

    count_after = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    if count_after != count_before:  # pragma: no cover - defensive
        raise RuntimeError(
            f"jobs migration lost rows: {count_before} -> {count_after}"
        )


def _reset_autoincrement(conn: sqlite3.Connection) -> None:
    """Keep autoincrement ids above the highest existing job id."""
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'sqlite_sequence'"
    ).fetchone():
        return
    highest = conn.execute("SELECT COALESCE(MAX(id), 0) FROM jobs").fetchone()[0]
    updated = conn.execute(
        "UPDATE sqlite_sequence SET seq = ? WHERE name = 'jobs'", (highest,)
    ).rowcount
    if not updated:
        conn.execute(
            "INSERT INTO sqlite_sequence (name, seq) VALUES ('jobs', ?)", (highest,)
        )
