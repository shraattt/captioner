"""Tests for the .env loader and HF token resolution."""
import os

from captioner import secrets


def test_load_env_file_sets_missing_vars(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        'HF_TOKEN=hf_test_123\n# a comment\nQUOTED="abc"\n', encoding="utf-8"
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    loaded = secrets.load_env_file(env)
    assert loaded == {"HF_TOKEN": "hf_test_123", "QUOTED": "abc"}
    assert os.environ["HF_TOKEN"] == "hf_test_123"
    assert os.environ["QUOTED"] == "abc"


def test_real_env_wins_over_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=from_file\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "from_env")

    secrets.load_env_file(env)
    assert os.environ["HF_TOKEN"] == "from_env"


def test_malformed_lines_are_ignored(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("no equals sign here\n\n  \n=emptykey\nOK=fine\n", encoding="utf-8")
    monkeypatch.delenv("OK", raising=False)

    loaded = secrets.load_env_file(env)
    assert loaded == {"OK": "fine"}


def test_missing_file_is_noop(tmp_path):
    assert secrets.load_env_file(tmp_path / "nope.env") == {}


def test_hf_token_fallback_chain(tmp_path, monkeypatch):
    for var in secrets._TOKEN_VARS:
        monkeypatch.delenv(var, raising=False)

    env = tmp_path / ".env"
    env.write_text("HUGGINGFACE_TOKEN=hf_fallback\n", encoding="utf-8")
    assert secrets.hf_token(env) == "hf_fallback"


def test_hf_token_none_when_absent(tmp_path, monkeypatch):
    for var in secrets._TOKEN_VARS:
        monkeypatch.delenv(var, raising=False)
    assert secrets.hf_token(tmp_path / "nope.env") is None
