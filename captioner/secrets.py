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
from pathlib import Path

# Project root = parent of the captioner/ package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Load KEY=value pairs from .env into os.environ (existing env wins).

    Returns the dict of variables actually set by the file.
    """
    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
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
        if key and key not in os.environ:
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
