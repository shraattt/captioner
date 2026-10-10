"""Compare Whisper model sizes on the same audio: speed, VRAM, agreement.

Answers two user questions with measurements instead of guesses:
  1. can `tiny.en` be made more viable? (variant flags below)
  2. why is `medium` slow, and is there a faster near-medium option?
     (`distil-large-v3`: fewer decoder layers, near-large accuracy)

Reference honesty: the audio is synthetic; nobody wrote down ground truth.
Each model is scored against the *medium* model's output (the most accurate
model in the set), so "WER 0" means "agrees with medium", not "perfect".
The full text of every run is printed so a human can judge the rest.

Usage:
    .venv/Scripts/python scripts/model_sweep.py [--sizes tiny.en,base.en,...]
"""
import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

from captioner import transcribe  # noqa: E402
from captioner.segmenter import Word  # noqa: E402

DEFAULT_SIZES = "tiny.en,base.en,small.en,distil-large-v3,medium"
AUDIO = ROOT / "data" / "uploads" / "long_speech.wav"

# distil models live outside the Systran/faster-whisper-* naming scheme.
REPO_OVERRIDES = {
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}


def repo_id(size: str) -> str:
    return REPO_OVERRIDES.get(size, transcribe.MODEL_REPO.format(size=size))


def nvidia_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def words_text(words) -> str:
    return " ".join(w.text for w in words)


def normalize(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def wer(ref_words, hyp_words) -> float:
    """Word error rate by Levenshtein distance over word lists."""
    r, h = ref_words, hyp_words
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / max(1, len(r))


def run_once(model, audio: str, **kwargs):
    """One timed transcription; returns (words, seconds, vram_delta, audio_dur)."""
    proc = psutil.Process()
    peak_rss = [0]
    stop = threading.Event()
    vram = {"base": nvidia_used_mib(), "peak": 0}

    def monitor():
        last = 0.0
        while not stop.is_set():
            try:
                peak_rss[0] = max(peak_rss[0], proc.memory_info().rss)
            except Exception:
                pass
            if time.monotonic() - last >= 0.5:
                last = time.monotonic()
                v = nvidia_used_mib()
                if v is not None:
                    vram["peak"] = max(vram["peak"], v)
            stop.wait(0.2)

    t0 = time.perf_counter()
    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()
    try:
        segments, info = model.transcribe(audio, language="en", **kwargs)
        words = []
        for seg in segments:
            for w in seg.words or []:
                if w.word.strip():
                    words.append(Word(text=w.word.strip(), start=w.start, end=w.end))
    finally:
        elapsed = time.perf_counter() - t0
        stop.set()
        mon.join()
    return words, elapsed, (vram["peak"] - vram["base"]) if vram["base"] is not None else None, info.duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=DEFAULT_SIZES)
    ap.add_argument("--audio", default=str(AUDIO))
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    audio = args.audio

    results = []  # (label, words, seconds, vram_delta, audio_dur, note)
    reference_words = None

    for size in sizes:
        rid = repo_id(size)
        if not transcribe.model_is_cached(size) or size in REPO_OVERRIDES:
            from huggingface_hub import snapshot_download
            print(f"[setup] ensuring {rid} is downloaded ...", flush=True)
            snapshot_download(rid)
        device = transcribe.resolve_device()
        compute = transcribe.resolve_compute_type(device)
        t0 = time.perf_counter()
        model = transcribe.get_model(rid, device, compute)
        print(f"[setup] loaded {rid} on {device}/{compute} in {time.perf_counter() - t0:.1f}s", flush=True)

        variants = [("default", {})]
        if size == "tiny.en":
            variants = [
                ("default", {}),
                ("cond_off", {"condition_on_previous_text": False}),
                ("cond_off_novad", {"condition_on_previous_text": False, "vad_filter": False}),
            ]
        for name, extra in variants:
            kwargs = dict(
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                beam_size=5,
            )
            kwargs.update(extra)
            words, secs, vram, adur = run_once(model, audio, **kwargs)
            note = " | ".join(f"{k}={v}" for k, v in extra.items()) or "app defaults"
            results.append((f"{size}:{name}", words, secs, vram, adur, note))
            print(f"[run] {size}:{name:<14} {secs:6.1f}s  {adur / secs:5.1f}x  vram {vram} MiB  {len(words)} words", flush=True)
            if size == "medium" and name == "default":
                reference_words = normalize(words_text(words))

        # Free the weights before the next model loads: keeping every model
        # resident would overflow the 6GB card by the time distil loads.
        transcribe._model_cache.clear()
        del model
        import gc
        gc.collect()
        time.sleep(1.0)

    if reference_words is None:
        # fall back: use the most accurate present model's own text
        reference_words = normalize(words_text(results[-1][1]))

    print("=" * 78)
    print(f"{'run':<28}{'wall':>8}{'realtime':>10}{'VRAM':>10}{'WER*':>8}  note")
    for label, words, secs, vram, adur, note in results:
        hyp = normalize(words_text(words))
        err = wer(reference_words, hyp)
        vram_s = f"{vram:+d} MiB" if vram is not None else "n/a"
        print(f"{label:<28}{secs:>7.1f}s{adur / secs:>9.1f}x{vram_s:>12}{err:>8.1%}  {note}")
    print("* WER vs medium's own output on this audio (no human ground truth).")
    print("=" * 78)
    for label, words, *_ in results:
        print(f"\n--- {label} ---")
        print(words_text(words)[:600])


if __name__ == "__main__":
    main()
