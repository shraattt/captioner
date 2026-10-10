"""Which models exist: shipped faster-whisper sizes, plus fine-tuned ones.

Phase 4/5 produce a fine-tuned Whisper checkpoint and convert it to
CTranslate2; this module is the single place that turns "a model the user can
pick" into "something faster-whisper can load". ``transcribe`` already accepts
a local directory wherever it accepts a size name, so resolution is just
label -> path, and everything downstream (batch, live, WER checks) keeps
working unchanged.

A fine-tuned model is discovered as ``data/ft/<name>_ct2``: the converter's
output directory, carrying ``model.bin`` and the tokenizer files. Nothing is
registered by hand, so dropping in a converted model is enough to make it
appear in the UI.
"""
import json
import logging
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

# Sizes that come from the Hub (faster-whisper repos).
BUILTIN_SIZES = ("tiny.en", "base.en", "small.en", "medium", "distil-large-v3")
LIVE_BUILTIN = ("tiny.en", "base.en", "small.en")
BATCH_BUILTIN = ("tiny.en", "base.en", "small.en", "medium", "distil-large-v3")

FINETUNED_SUFFIX = "_ct2"


def finetuned_root() -> Path:
    return Path(config.DATA_DIR) / "ft"


def _meta_for(directory: Path) -> dict:
    """Training metadata for a converted checkpoint, if any.

    The converter copies ``ft_meta.json`` into the CTranslate2 directory; an
    older conversion (or one made with --keep) may only have it next to the
    HF checkpoint the converter read, so both places are checked.
    """
    label = directory.name[: -len(FINETUNED_SUFFIX)] or directory.name
    candidates = [directory / "ft_meta.json", directory.parent / label / "ft_meta.json"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.debug("unreadable metadata at %s", path)
    return {}


def find_finetuned(root: Path | None = None) -> list[dict]:
    """Locally fine-tuned CTranslate2 models, newest name order.

    Each entry: ``{"label", "path", "meta"}``. A directory without
    ``model.bin`` is incomplete (conversion died halfway) and is skipped
    rather than offered as a model that would fail to load.
    """
    root = Path(root) if root is not None else finetuned_root()
    found: list[dict] = []
    if not root.is_dir():
        return found
    for child in sorted(root.glob(f"*{FINETUNED_SUFFIX}")):
        if not child.is_dir() or not (child / "model.bin").exists():
            continue
        found.append({
            "label": child.name[: -len(FINETUNED_SUFFIX)],
            "path": str(child),
            "meta": _meta_for(child),
        })
    return found


def is_builtin(label: str) -> bool:
    return label in BUILTIN_SIZES


def resolve(label: str) -> str:
    """Turn a picker label into something faster-whisper can load.

    Built-in sizes pass through unchanged; a fine-tuned label becomes the
    absolute path of its converted directory. Unknown labels raise, because
    silently loading a *different* model than the user chose is worse than
    saying so.
    """
    if not label:
        return config.LIVE_MODEL_SIZE
    if is_builtin(label):
        return label
    if Path(label).exists():
        return label  # an explicit path always wins over the registry
    for entry in find_finetuned():
        if entry["label"] == label:
            return entry["path"]
    raise ValueError(f"unknown model {label!r}")


def live_choices(root: Path | None = None) -> list[str]:
    """Labels for the live-caption picker: built-ins first, then fine-tuned."""
    return list(LIVE_BUILTIN) + [e["label"] for e in find_finetuned(root)]


def batch_choices(root: Path | None = None) -> list[str]:
    """Labels for the batch model picker (same idea, batch defaults)."""
    return list(BATCH_BUILTIN) + [e["label"] for e in find_finetuned(root)]


def describe(label: str, root: Path | None = None) -> str:
    """One-line human description for a picker entry or status line."""
    if is_builtin(label):
        return label
    for entry in find_finetuned(root):
        if entry["label"] == label:
            meta = entry["meta"]
            base = meta.get("base_model", "fine-tuned")
            wer = meta.get("eval_wer")
            bits = [f"fine-tuned from {base}"]
            if meta.get("train_hours"):
                bits.append(f"{meta['train_hours']} h noisy speech")
            if isinstance(wer, (int, float)):
                bits.append(f"synthetic WER {wer:.1f}%")
            return f"{label} ({', '.join(bits)})"
    return label
