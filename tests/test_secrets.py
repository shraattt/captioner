"""Tests for the .env loader and HF token resolution."""
import os
import sys

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


def test_an_empty_env_var_does_not_shadow_the_file(tmp_path, monkeypatch):
    """`set CAPTIONER_DATA_DIR=` (empty) is how a launcher clears a variable, so
    it must not stop the file's value from being used."""
    env = tmp_path / ".env"
    env.write_text("CAPTIONER_DATA_DIR=D:/stick/data\n", encoding="utf-8")
    monkeypatch.setenv("CAPTIONER_DATA_DIR", "")

    assert secrets.load_env_file(env) == {"CAPTIONER_DATA_DIR": "D:/stick/data"}
    assert os.environ["CAPTIONER_DATA_DIR"] == "D:/stick/data"


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


def test_a_packaged_build_reads_the_env_beside_its_own_exe(tmp_path, monkeypatch):
    """What makes a portable install portable: the .env travels with the app.

    The installer writes that file for `--portable`, and it is also the file a
    user can see and edit, so it has to win over the profile's copy.
    """
    folder = tmp_path / "Captioner"
    folder.mkdir()
    (folder / ".env").write_text("HF_TOKEN=from_the_folder\n", encoding="utf-8")
    (tmp_path / "Local" / "Captioner").mkdir(parents=True)
    (tmp_path / "Local" / "Captioner" / ".env").write_text(
        "HF_TOKEN=from_the_profile\n", encoding="utf-8")
    for var in secrets._TOKEN_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(folder / "Captioner.exe"))

    paths = secrets.env_paths()
    assert paths[0] == folder / ".env"
    assert secrets.hf_token() == "from_the_folder"


def test_a_checkout_does_not_look_beside_its_own_interpreter(tmp_path, monkeypatch):
    """Only a packaged build reads "next to the executable": from a checkout
    that path is .venv/Scripts, where nobody keeps a token."""
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert all("Scripts" not in str(path) for path in secrets.env_paths())
