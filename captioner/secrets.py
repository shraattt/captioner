"""Secrets handling: load a project-local .env file and expose the HF token.

The pyannote diarization models (Phase 2) are gated on Hugging Face: they
require a token from an account that accepted the model conditions. We keep
that token in captioner/.env (gitignored) or the ambient environment — it
must never be committed or hardcoded.

Deliberately dependency-free: parses the tiny subset of .env syntax we need
(KEY=value, optional quotes, # comments). Real environment variables always
win over the file.
"""
import os
import sys
from pathlib import Path

# Project root = the folder the code ships in: normally the parent of the
# captioner/ package directory, and under PyInstaller the folder the build was
# unpacked into (`sys._MEIPASS`), which is where a packaged build keeps the
# `.env` it was built with.
PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))

_TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def env_paths() -> list[Path]:
    """Where a `.env` may live, most specific first.

    From a checkout that is the repository root, which is what every line of the
    documentation means by "the .env file". A packaged build has no checkout to
    write into, so it also looks in the folder the app keeps its data in
    (`%LOCALAPPDATA%/Captioner/.env`, created as a commented template by the
    installer) - without that, a user who installed the app could not give it a
    Hugging Face token at all, and every gated model would stay unreachable.
    `CAPTIONER_ENV_FILE` names one file explicitly and wins over both.
    """
    candidates: list[Path] = []
    override = os.environ.get("CAPTIONER_ENV_FILE", "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    if getattr(sys, "frozen", False):
        # A packaged build is told the Hugging Face token by a `.env` sitting
        # next to its own executable: that is the file an installer writes for a
        # portable copy, and the one place a user can see and edit it.
        candidates.append(Path(sys.executable).resolve().parent / ".env")
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        candidates.append(Path(base) / "Captioner" / ".env")
    candidates.append(PROJECT_ROOT / ".env")
    return candidates


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Load KEY=value pairs from .env into os.environ (existing env wins).

    Returns the dict of variables actually set by the file(s). With no argument
    every candidate from `env_paths()` is read, in order, so a packaged install
    finds the user's file and a checkout finds its own.
    """
    paths = [Path(path)] if path is not None else env_paths()
    loaded: dict[str, str] = {}
    for env_path in paths:
        loaded.update(_load_one(env_path))
    return loaded


def _load_one(env_path: Path) -> dict[str, str]:
    """KEY=value pairs from one file, skipping anything already in the env."""
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # An empty value in the environment is not an override: a launcher that
        # exported CAPTIONER_DATA_DIR="" must not shadow the file's real value.
        if key and not os.environ.get(key, "").strip():
            os.environ[key] = value
            loaded[key] = value
    return loaded


def hf_token(env_path: str | Path | None = None) -> str | None:
    """Return the Hugging Face token from env or .env, loading the file once."""
    load_env_file(env_path)
    for var in _TOKEN_VARS:
        token = os.environ.get(var)
        if token:
            return token
    return None
