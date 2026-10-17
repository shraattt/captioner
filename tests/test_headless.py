"""Tests for captioner/headless.py: the packaged app's `--job` mode.

The pipeline itself is covered by test_worker.py; what matters here is the
seam: a file named on the command line is queued *with its options*, that job
and not an older one is run, and the console is told where the subtitles went -
or why there are none.
"""
from pathlib import Path

import pytest

from captioner import config, db, headless, worker
from captioner.segmenter import Word


@pytest.fixture()
def media(tmp_path):
    """A stand-in media file: the transcriber is mocked, only the path matters."""
    path = tmp_path / "talk.wav"
    path.write_bytes(b"not really audio")
    return path


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """Point the app's data folder at tmp_path, and keep logging out of it."""
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "DB_PATH", data / "captioner.db")
    monkeypatch.setattr(config, "UPLOADS_DIR", data / "uploads")
    monkeypatch.setattr(config, "OUTPUT_DIR", data / "outputs")
    monkeypatch.setattr(headless, "_setup_logging", lambda: None)
    return data


@pytest.fixture()
def words(monkeypatch):
    spoken = [Word("Hello", 0.0, 0.4), Word("world.", 0.5, 0.9),
              Word("Second", 3.5, 3.9), Word("sentence.", 4.0, 4.5)]
    monkeypatch.setattr(
        worker, "transcribe_words", lambda path, size, on_progress=None, **_: spoken
    )
    return spoken


def _job(data: Path, job_id: int = 1):
    conn = db.connect(data / "captioner.db")
    try:
        return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    finally:
        conn.close()


def test_a_job_writes_captions_and_says_where(project, media, words, capsys):
    assert headless.main([str(media), "--model", "tiny.en"]) == 0

    out = capsys.readouterr().out
    assert "job #1: talk.wav" in out
    assert "done" in out
    srt = project / "outputs" / "job_1" / "captions.srt"
    assert srt.exists()
    assert "Hello world." in srt.read_text(encoding="utf-8")
    assert str(srt) in out

    row = _job(project)
    assert row["status"] == "done"
    assert row["model_size"] == "tiny.en"
    # The file is stored resolved, so a job survives the working directory moving.
    assert Path(row["file_path"]).is_absolute()


def test_the_options_reach_the_job_the_pipeline_runs(project, media, words,
                                                     monkeypatch):
    """--align and --translate-to are stored on the row, not applied later."""
    seen = {}
    monkeypatch.setattr(
        "captioner.align.realign_file",
        lambda path, spoken, **_: (seen.setdefault("aligned", path), spoken)[1],
    )

    def fake_translate(out_dir, target, diarized, on_stage, source=None):
        assert target == "fr"
        seen["translated"] = target
        (Path(out_dir) / f"captions.{target}.srt").write_text("bonjour\n",
                                                              encoding="utf-8")
        return None

    monkeypatch.setattr(worker, "_translate_output", fake_translate)
    assert headless.main([str(media), "--align", "--translate-to", "fr"]) == 0

    row = _job(project)
    assert row["align"] == 1 and row["translate_to"] == "fr"
    assert seen["aligned"] == row["file_path"]
    assert seen["translated"] == "fr"


def test_a_failing_job_exits_nonzero_and_says_why(project, media, monkeypatch,
                                                  capsys):
    def refuse(path, size, on_progress=None, **_):
        raise RuntimeError("Aligning word timings is not available in this "
                           "installation: torch is not installed.")

    monkeypatch.setattr(worker, "transcribe_words", refuse)
    assert headless.main([str(media)]) == 1

    err = capsys.readouterr().err
    assert "failed" in err and "torch is not installed" in err
    assert _job(project)["status"] == "error"


def test_a_missing_file_is_refused_before_anything_is_queued(project, capsys):
    assert headless.main([str(Path(project) / "absent.mp3")]) == 2
    assert "file not found" in capsys.readouterr().err
    assert not (project / "captioner.db").exists()


def test_an_unknown_language_is_refused_at_the_command_line(project, media,
                                                            capsys):
    assert headless.main([str(media), "--translate-to", "zz"]) == 2
    assert "unknown target language" in capsys.readouterr().err
    assert not (project / "captioner.db").exists()


def test_the_named_job_runs_without_taking_an_older_one(project, media, words,
                                                        tmp_path, capsys):
    """--job means *this* file: another queued job must be left alone."""
    other = tmp_path / "earlier.mp3"
    other.write_bytes(b"different audio")
    conn = db.connect(project / "captioner.db")
    try:
        earlier = worker.enqueue(conn, str(other))
    finally:
        conn.close()

    assert headless.main([str(media)]) == 0
    assert f"job #{earlier + 1}: talk.wav" in capsys.readouterr().out

    conn = db.connect(project / "captioner.db")
    try:
        rows = {r["id"]: r["status"] for r in
                conn.execute("SELECT id, status FROM jobs").fetchall()}
    finally:
        conn.close()
    assert rows[earlier] == "queued"
    assert rows[earlier + 1] == "done"
