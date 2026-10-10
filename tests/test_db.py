"""Database layer tests, focused on upgrading databases made by older versions.

An app update that changes the schema must patch existing databases in place —
CREATE TABLE IF NOT EXISTS only helps brand-new files. Adding a column is a
metadata-only ALTER TABLE; widening the status CHECK constraint is not, and
needs the table recreated. These tests pin both, so a schema change can never
silently break an existing queue.
"""
import sqlite3

import pytest

from captioner import config, db

# The CHECK constraint comes last: this machine's SQLite (3.50.4) rejects a
# table-level constraint followed by another column definition.
LEGACY_JOBS_TABLE = (
    "CREATE TABLE jobs ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " file_path TEXT NOT NULL,"
    " file_name TEXT NOT NULL,"
    " status TEXT NOT NULL DEFAULT 'queued',"
    " error TEXT,"
    " model_size TEXT NOT NULL DEFAULT 'medium',"
    " created_at TEXT NOT NULL,"
    " started_at TEXT,"
    " finished_at TEXT,"
    " CHECK (status IN ('queued','running','done','error')))"
)

LEGACY_CAPTIONS_TABLE = (
    "CREATE TABLE captions ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " job_id INTEGER NOT NULL REFERENCES jobs(id),"
    " seq INTEGER NOT NULL,"
    " start_ms INTEGER NOT NULL,"
    " end_ms INTEGER NOT NULL,"
    " text TEXT NOT NULL)"
)


def _write_legacy_db(path, jobs=2, captions_per_job=3):
    """A database as an older version of this app would have left it."""
    raw = sqlite3.connect(path)
    raw.execute(LEGACY_JOBS_TABLE)
    raw.execute(LEGACY_CAPTIONS_TABLE)
    for i in range(1, jobs + 1):
        raw.execute(
            "INSERT INTO jobs (file_path, file_name, status, created_at) "
            "VALUES (?, ?, ?, '2026-01-01 00:00:00')",
            (f"file{i}.mp4", f"file{i}.mp4", "done" if i == 1 else "queued"),
        )
        for seq in range(1, captions_per_job + 1):
            raw.execute(
                "INSERT INTO captions (job_id, seq, start_ms, end_ms, text) "
                "VALUES (?, ?, ?, ?, ?)",
                (i, seq, seq * 1000, seq * 1000 + 500, f"line {seq}"),
            )
    raw.commit()
    raw.close()
    return path


def test_fresh_database_has_progress_column(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "fresh.db")
    conn = db.connect()
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        assert "progress" in columns
    finally:
        conn.close()


def test_legacy_database_is_upgraded_in_place(tmp_path):
    path = tmp_path / "legacy.db"
    raw = sqlite3.connect(path)
    raw.execute(
        "CREATE TABLE jobs ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " file_path TEXT NOT NULL,"
        " file_name TEXT NOT NULL,"
        " status TEXT NOT NULL DEFAULT 'queued',"
        " error TEXT,"
        " model_size TEXT NOT NULL DEFAULT 'medium',"
        " created_at TEXT NOT NULL,"
        " started_at TEXT,"
        " finished_at TEXT)"
    )
    raw.execute(
        "INSERT INTO jobs (file_path, file_name, created_at) "
        "VALUES ('a.mp4', 'a.mp4', '2026-01-01 00:00:00')"
    )
    raw.commit()
    raw.close()

    conn = db.connect(path)
    try:
        row = conn.execute("SELECT progress FROM jobs WHERE id = 1").fetchone()
        assert row["progress"] is None  # existing rows survive with NULL progress
    finally:
        conn.close()


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "twice.db"
    for _ in range(3):  # connecting repeatedly must not raise
        conn = db.connect(path)
        conn.close()


def test_canceled_status_needs_the_table_rebuilt(tmp_path):
    """A pre-`canceled` database rejects the status until it is migrated."""
    path = _write_legacy_db(tmp_path / "legacy.db")
    raw = sqlite3.connect(path)
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute(
            "INSERT INTO jobs (file_path, file_name, status, created_at) "
            "VALUES ('x.mp4', 'x.mp4', 'canceled', '2026-01-01 00:00:00')"
        )
    raw.close()

    conn = db.connect(path)  # runs the migration
    try:
        conn.execute(
            "INSERT INTO jobs (file_path, file_name, status, created_at) "
            "VALUES ('x.mp4', 'x.mp4', 'canceled', '2026-01-01 00:00:00')"
        )
        conn.commit()
    finally:
        conn.close()


def test_rebuilding_jobs_preserves_rows_ids_and_captions(tmp_path):
    path = _write_legacy_db(tmp_path / "legacy.db")
    before = sqlite3.connect(path)
    before.row_factory = sqlite3.Row
    jobs_before = [dict(r) for r in before.execute("SELECT * FROM jobs ORDER BY id")]
    captions_before = before.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
    before.close()

    conn = db.connect(path)
    try:
        jobs_after = [dict(r) for r in conn.execute("SELECT * FROM jobs ORDER BY id")]
        assert [j["id"] for j in jobs_after] == [j["id"] for j in jobs_before]
        assert [j["file_path"] for j in jobs_after] == [j["file_path"] for j in jobs_before]
        assert [j["status"] for j in jobs_after] == [j["status"] for j in jobs_before]
        # captions and the foreign key to jobs must survive the rebuild
        assert conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0] == captions_before
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        # a rebuilt jobs table still hands out fresh ids
        conn.execute(
            "UPDATE jobs SET status = 'queued', heartbeat_at = ?, stage = NULL WHERE id = ?",
            ("2026-01-01 00:00:00", 1),
        )
        conn.commit()
        highest = conn.execute("SELECT MAX(id) FROM jobs").fetchone()[0]
        conn.execute(
            "INSERT INTO jobs (file_path, file_name, model_size, created_at) "
            "VALUES ('new.mp4', 'new.mp4', 'tiny.en', '2026-01-01 00:00:00')"
        )
        conn.commit()
        assert conn.execute("SELECT MAX(id) FROM jobs").fetchone()[0] > highest
    finally:
        conn.close()


def test_rebuild_adds_the_new_columns(tmp_path):
    path = _write_legacy_db(tmp_path / "legacy.db")
    conn = db.connect(path)
    try:
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
        assert {"progress", "stage", "heartbeat_at"} <= columns
    finally:
        conn.close()


def test_rebuild_runs_only_once(tmp_path):
    path = _write_legacy_db(tmp_path / "legacy.db")
    for _ in range(3):
        conn = db.connect(path)
        try:
            assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0] == 6
        finally:
            conn.close()


def test_settled_statuses_cover_every_terminal_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "fresh.db")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    conn = db.connect()
    try:
        for status in db.SETTLED_STATUSES:
            conn.execute(
                "INSERT INTO jobs (file_path, file_name, status, created_at) "
                "VALUES ('x', 'x', ?, '2026-01-01 00:00:00')",
                (status,),
            )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 3
    finally:
        conn.close()
