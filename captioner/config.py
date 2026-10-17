"""Central configuration for the captioner app.

Every path the app writes is decided here and created on demand, so the UI, the
worker, the CLI and the tests all agree on where things live. Run from a
checkout, they live in its `data/` folder; run from a packaged build, they live
in the user's profile - see `data_home` for why that has to be different.
"""
import os
import sys
from pathlib import Path

#Secrets, read before anything else
# The `.env` file is loaded *here*, first, because it can decide where the app
# keeps its data (`CAPTIONER_DATA_DIR`) - and DATA_DIR below is fixed at import,
# so reading the file any later would be too late. A portable install is exactly
# that: a `.env` beside the executable, pointing the data folder at the stick it
# travels on. Real environment variables always win over the file.
try:
    from . import secrets as _secrets
except ImportError:  # pragma: no cover - top-level run path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from secrets import secrets as _secrets  # type: ignore

_secrets.load_env_file()

#Layout
# Where the code lives, and with it the files that ship beside it (the icon,
# `ui/web`). Under PyInstaller that is the folder the build was unpacked into,
# which is what `sys._MEIPASS` names; from a checkout it is the parent of this
# package. Nothing here is writable in a packaged install, which is why no path
# below this line is derived from it except PROJECT_ROOT itself.
PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def frozen() -> bool:
    """Is this a packaged build, with no Python and no checkout around it?"""
    return bool(getattr(sys, "frozen", False))


def data_home() -> Path:
    """The folder the app writes to: database, outputs, logs, fine-tuned models.

    Development keeps the checkout's `data/`, so every script and every line of
    the documentation still points where it always did. A packaged build must
    *not* write beside its own executable: that is Program Files (needs
    elevation) and, worse, a folder an uninstall or a repair may delete -
    the user's jobs and transcripts would go with it. So a packaged build keeps
    its data in the per-user profile, and `CAPTIONER_DATA_DIR` moves it anywhere
    (a portable stick, a second profile, a network share).
    """
    override = os.environ.get("CAPTIONER_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if frozen():
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        return (Path(base) if base else Path.home()) / "Captioner" / "data"
    return PROJECT_ROOT / "data"


DATA_DIR = data_home()
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "captioner.db"

#Models
# Batch uses the accurate-but-slow model; live uses the smallest one that
# still hears the words (see the LIVE_MODEL_SIZE note below).
# distil-large-v3: medium's accuracy measured at ~2.4x its speed on the
# RTX 4050, weights ~1.5GB (see scripts/model_sweep.py for the comparison).
BATCH_MODEL_SIZE = "distil-large-v3"
# Live needs latency, not accuracy - but tiny.en was measurably *too* small.
# On two real lecture recordings (README, "On real audio") it disagreed with
# the batch transcript on 28% / 23% of words and small.en on 15% / 14%, for
# twice the decode cost that still fits the 1.5s stride (p95 1.05-1.32s).
# Live -> Live model still offers tiny.en for a machine that cannot keep up.
LIVE_MODEL_SIZE = "small.en"

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
# Hugging Face token for gated model downloads (Phase 2 diarization). The file
# itself was read at the top of this module (it can move the data folder); these
# are the accessors the rest of the app uses.
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
