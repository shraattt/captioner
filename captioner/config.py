"""Central configuration for the captioner app.

All paths are relative to the project root (the folder containing this
package) and are created on demand. Keeping every knob in one module means
the UI, worker, and tests all agree on where things live.
"""
import os
from pathlib import Path

#Layout
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "captioner.db"

#Models
# Batch uses the accurate-but-slow model; live (later phase) uses tiny.
# distil-large-v3: medium's accuracy measured at ~2.4x its speed on the
# RTX 4050, weights ~1.5GB (see scripts/model_sweep.py for the comparison).
BATCH_MODEL_SIZE = "distil-large-v3"
LIVE_MODEL_SIZE = "tiny.en"  # live captions need latency, not accuracy

# Phase 4/5: fine-tuned checkpoints converted to CTranslate2 land in
# data/ft/<name>_ct2 and are offered by captioner/models.py automatically.
FT_DIR = DATA_DIR / "ft"

# "auto" picks cuda if available, else cpu. Override with "cuda"/"cpu".
DEVICE = "auto"
COMPUTE_TYPE = "auto"  # auto -> float16 on GPU, int8 on CPU

#Caption segmentation tunables
SEGMENT_MAX_CHARS = 84        # two comfortable subtitle lines
SEGMENT_MAX_SECONDS = 5.0     # readable dwell time
SEGMENT_MAX_GAP = 0.8         # a silence this long always ends a block
SEGMENT_MIN_CHARS = 12        # don't emit one-word blocks unless forced
SEGMENT_MIN_DURATION = 1.2    # minimum on-screen time for a block

#Worker
WORKER_POLL_SECONDS = 1.0

#Live translation
# Target language for live captions (two letters, see translate.LANGUAGES), or
# empty for off. The dashboard overrides this per session; the environment
# variable is what makes a headless `python -m captioner.live`-style session
# translatable without a UI.
LIVE_TRANSLATE_TO = os.environ.get("CAPTIONER_LIVE_TRANSLATE_TO", "").strip()


def ensure_dirs() -> None:
    """Create the data directory tree if missing."""
    for d in (DATA_DIR, UPLOADS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


#Secrets
# Hugging Face token for gated model downloads (Phase 2 diarization). Loaded
# from the project-local .env (gitignored) so the token never has to be
# committed or hardcoded. The file is parsed by captioner.secrets and is
# only read once per variable.
try:
    from . import secrets as _secrets
except ImportError:  # pragma: no cover - top-level run path
    import sys
    from pathlib import Path

    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from secrets import secrets as _secrets  # type: ignore


def hf_token() -> str:
    """Return the Hugging Face token (loaded from .env on first call)."""
    load_env_file()
    for var in _secrets._TOKEN_VARS:
        token = _secrets.os.environ.get(var)
        if token:
            return token
    return ""


def load_env_file(path=None):
    """Load KEY=value pairs from .env into os.environ (env wins). Idempotent.

    Returns the dict of variables actually set by the file.
    """
    return _secrets.load_env_file(path)
