"""Whisper transcription wrapper (faster-whisper / CTranslate2).

One place knows about models so the rest of the app never imports
faster_whisper directly. Word timestamps are required — the segmenter
groups words, not Whisper's native sentence segments.

Device/compute policy:
  - cuda available  -> float16 (best speed/accuracy on the 4050)
  - otherwise       -> int8 on CPU (slow but functional fallback)
"""
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional

from faster_whisper import WhisperModel

from . import config
from .segmenter import Word

logger = logging.getLogger(__name__)

# Cache model instances so repeated jobs don't reload weights.
_model_cache = {}

# Keep os.add_dll_directory handles alive for the process lifetime.
_dll_dirs = []

MODEL_REPO = "Systran/faster-whisper-{size}"

# Models that live outside the Systran/faster-whisper-* naming scheme.
REPO_OVERRIDES = {
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}


def repo_id(model_size: str) -> str:
    """HF repo for a size name (distil models have their own repos)."""
    return REPO_OVERRIDES.get(model_size, MODEL_REPO.format(size=model_size))


def model_ref(model_size: str) -> str:
    """What faster-whisper should be handed for this "model size".

    Three shapes are accepted everywhere a size name is:
      * a **local directory** — a fine-tuned CTranslate2 model from Phase 5
        (``data/ft/<name>_ct2``), used as-is;
      * an **explicit repo id** (``namespace/name``) — someone else's
        converted model on the Hub;
      * a plain **size name** (``tiny.en``) — mapped to its Systran repo.

    Without this, a local path was turned into
    ``Systran/faster-whisper-D:/path/to/model`` and every load failed.
    """
    if Path(model_size).exists():
        return str(Path(model_size))
    if "/" in model_size:
        return model_size
    return repo_id(model_size)

# ``on_stage(stage, fraction)`` — fraction is None when it is not knowable.
StageCallback = Callable[[str, Optional[float]], None]


class TranscriptionCancelled(RuntimeError):
    """Raised when the caller asked to abort a running transcription."""


def preload_cuda_dlls() -> None:
    """Make the CUDA runtime DLLs loadable for ctranslate2 on Windows.

    The ctranslate2 wheel bundles cuDNN but not cuBLAS, and Windows' DLL
    search never looks inside site-packages — so on a machine with a driver
    but no CUDA Toolkit, encoding dies with "cublas64_12.dll is not found".
    Preloading the libs by absolute path registers them under their base
    names, which is what ctranslate2's later LoadLibrary call resolves
    against. Sources, in order: CAPTIONER_CUDA_BIN, the NVIDIA pip wheels
    (nvidia-cublas-cu12), a torch install, then the CUDA Toolkit.
    """
    if sys.platform != "win32":
        return
    import ctypes
    import site

    dirs = []
    override = os.environ.get("CAPTIONER_CUDA_BIN")
    if override:
        dirs.append(Path(override))
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        dirs += [Path(sp) / "nvidia" / "cublas" / "bin", Path(sp) / "torch" / "lib"]
    if os.environ.get("CUDA_PATH"):
        dirs.append(Path(os.environ["CUDA_PATH"]) / "bin")

    for name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"):
        for d in dirs:
            dll = d / name
            if not dll.exists():
                continue
            try:
                ctypes.WinDLL(str(dll))
                _dll_dirs.append(os.add_dll_directory(str(d)))
                logger.debug("Preloaded %s from %s", name, d)
            except OSError as exc:  # pragma: no cover - machine dependent
                logger.warning("Could not preload %s: %s", dll, exc)
            break


def resolve_device() -> str:
    if config.DEVICE != "auto":
        return config.DEVICE
    try:
        import ctranslate2  # lightweight import, no model load
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def resolve_compute_type(device: str) -> str:
    if config.COMPUTE_TYPE != "auto":
        return config.COMPUTE_TYPE
    return "float16" if device == "cuda" else "int8"


def _model_cache_dir(model_size: str) -> Optional[Path]:
    """Where huggingface_hub would keep this model, or None if unknown."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:  # pragma: no cover - hub always ships with faster-whisper
        return None
    repo = model_ref(model_size)
    if Path(repo).exists():
        return None  # local directory: "cached" is not a question
    return Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"


def model_is_cached(model_size: str) -> bool:
    """True when the weights are already on disk (no download needed)."""
    local = Path(model_size)
    if local.exists():
        # A local directory is only usable if it actually holds CT2 weights;
        # a half-finished conversion would otherwise fail inside WhisperModel.
        if local.is_dir():
            return (local / "model.bin").exists()
        return True
    cache = _model_cache_dir(model_size)
    if cache is None or not cache.is_dir():
        return False
    return any((snap / "model.bin").exists() for snap in (cache / "snapshots").glob("*"))


def _cached_bytes(model_size: str) -> int:
    cache = _model_cache_dir(model_size)
    if cache is None or not cache.is_dir():
        return 0
    total = 0
    for path in cache.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:  # pragma: no cover - file vanished mid-walk
            continue
    return total


def _repo_bytes(model_size: str) -> Optional[int]:
    """Total size of the model repo, for a percentage; None if unknown."""
    try:
        from huggingface_hub import HfApi

        ref = model_ref(model_size)
        if Path(ref).exists():
            return None
        info = HfApi().model_info(ref, files_metadata=True)
    except Exception as exc:
        logger.debug("Could not size %s: %s", model_size, exc)
        return None
    total = sum(getattr(sib, "size", 0) or 0 for sib in (info.siblings or []))
    return total or None


def ensure_model_downloaded(model_size: str, on_stage: Optional[StageCallback] = None) -> None:
    """Fetch weights ahead of loading them, reporting how much has arrived.

    faster-whisper downloads lazily *inside* WhisperModel(), which leaves the
    UI with nothing to show for however long the transfer takes — and a first
    `medium` run is ~1.5 GB, so "looks hung for ten minutes" is the normal
    experience without this. Downloading here lets the job report
    ``downloading`` with real byte progress; the constructor then finds
    everything cached and only has to load it.

    Failure is not fatal: if this cannot run, WhisperModel() downloads the
    model itself and reports the real error, we just lose the progress bar.
    """
    if model_is_cached(model_size):
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception:  # pragma: no cover - hub always ships with faster-whisper
        return

    total = _repo_bytes(model_size)
    if on_stage is not None:
        on_stage("downloading", 0.0 if total else None)

    stop = threading.Event()

    def report() -> None:
        while not stop.wait(0.5):
            if on_stage is not None and total:
                on_stage("downloading", min(0.99, _cached_bytes(model_size) / total))

    reporter = threading.Thread(target=report, name="model-download-progress", daemon=True)
    reporter.start()
    try:
        snapshot_download(
            repo_id=model_ref(model_size),
            token=config.hf_token() or None,
        )
    except Exception as exc:
        logger.warning(
            "Pre-download of %s failed (%s); letting faster-whisper fetch it",
            model_size, exc,
        )
    finally:
        stop.set()
        reporter.join(timeout=1.0)


def get_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    key = (model_size, device, compute_type)
    if key not in _model_cache:
        if device == "cuda":
            preload_cuda_dlls()
        token = config.hf_token()
        if token:
            # huggingface_hub reads HF_TOKEN for downloads: higher rate limits,
            # and required for the gated pyannote models coming in Phase 2.
            os.environ.setdefault("HF_TOKEN", token)
        logger.info("Loading Whisper %s on %s (%s)...", model_size, device, compute_type)
        _model_cache[key] = WhisperModel(
            model_ref(model_size), device=device, compute_type=compute_type
        )
    return _model_cache[key]


def transcribe_words(
    audio_path: str,
    model_size: str = None,
    on_progress: Optional[Callable[[float], None]] = None,
    on_stage: Optional[StageCallback] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> List[Word]:
    """Transcribe an audio/video file into timed words.

    Accepts anything ffmpeg can open (mp4, mkv, mov, mp3, wav, m4a...);
    faster-whisper demuxes and resamples internally.

    ``on_progress`` (if given) is called with a 0.0-1.0 fraction as segments
    finish decoding, so a long file can report where it is while it runs.
    ``on_stage`` reports which phase the work is in (``downloading``,
    ``loading``, ``transcribing``) for the UI.
    ``should_cancel`` is polled between segments: raising
    :class:`TranscriptionCancelled` within a few seconds of a long file beats
    making the user wait for the whole thing.
    """
    model_size = model_size or config.BATCH_MODEL_SIZE
    device = resolve_device()
    compute_type = resolve_compute_type(device)

    _check_cancel(should_cancel)
    ensure_model_downloaded(model_size, on_stage=on_stage)
    _check_cancel(should_cancel)
    if on_stage is not None:
        on_stage("loading", None)
    model = get_model(model_size, device, compute_type)
    _check_cancel(should_cancel)
    if on_stage is not None:
        on_stage("transcribing", 0.0)

    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,          # trims silence, fewer wasted windows
        vad_parameters={"min_silence_duration_ms": 500},
        language="en",
        # Carrying context between segments makes Whisper repeat itself on
        # repetitive audio (tiny's main failure mode); dropping it measured
        # equal accuracy here and slightly faster, and is the standard
        # anti-hallucination remedy.
        condition_on_previous_text=False,
    )

    words: List[Word] = []
    for seg in segments:  # generator — transcription runs lazily here
        _check_cancel(should_cancel)  # checked per segment: prompt on long files
        for w in seg.words:
            text = w.word.strip()
            if text:
                words.append(Word(text=text, start=w.start, end=w.end))
        if on_progress is not None and info.duration:
            on_progress(min(1.0, seg.end / info.duration))
    logger.info("Transcribed %s (%.1fs audio, %d words)", audio_path, info.duration, len(words))
    return words


def _check_cancel(should_cancel: Optional[Callable[[], bool]]) -> None:
    if should_cancel is not None and should_cancel():
        raise TranscriptionCancelled("canceled by the user")
