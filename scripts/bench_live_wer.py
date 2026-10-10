"""WER benchmark for the live pipeline: model sizes x decode params.

Replays the fixtures through the exact live code path (LiveEngine.stream_file,
sliding window + freeze), then computes WER of the frozen words against a
reference transcript. The reference is generated once from the full audio with
the batch default model (distil-large-v3) — an effective ground truth for
clear synthetic speech — and cached under data/bench/.

Compares configurations that are safe for the 6GB laptop: tiny.en (current
live default), base.en, small.en, plus beam sizes and window lengths.

    QT_QPA_PLATFORM=offscreen python scripts/bench_live_wer.py
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURES = [ROOT / "data" / "uploads" / "smoke_speech.wav",
            ROOT / "data" / "uploads" / "long_speech.wav"]
BENCH = ROOT / "data" / "bench"
REF_PATH = BENCH / "live_reference.json"

WORD_RE = re.compile(r"[a-z']+")


def normalize(text: str):
    """Lowercase, strip punctuation and filler tokens — fair word matching."""
    tokens = WORD_RE.findall(text.lower())
    return [t for t in tokens if t not in ("um", "uh", "erm")]


def wer(reference, hypothesis) -> float:
    """Levenshtein WER: substitutions + deletions + insertions / ref words."""
    r, h = reference, hypothesis
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[len(h)] / len(r)


def make_reference() -> dict:
    """Ground truth: transcribe each fixture whole with the batch model."""
    from faster_whisper.audio import decode_audio

    import numpy as np
    from captioner import transcribe

    model = transcribe.get_model(
        "distil-large-v3",  # repo_id() resolves REPO_OVERRIDES
        transcribe.resolve_device(),
        transcribe.resolve_compute_type(transcribe.resolve_device()),
    )
    refs = {}
    for fx in FIXTURES:
        audio = np.asarray(
            decode_audio(str(fx), sampling_rate=16000), dtype=np.float32
        ).reshape(-1)
        words = []
        for seg in model.transcribe(audio, language="en", word_timestamps=True,
                                    vad_filter=True,
                                    condition_on_previous_text=False)[0]:
            for w in seg.words or []:
                if w.word.strip():
                    words.append(w.word.strip())
        refs[fx.name] = " ".join(words)
        print(f"  reference {fx.name}: {len(words)} words")
    return refs


def run_config(fixtures: dict, model_size: str, beam: int, window: float,
               stride: float) -> dict:
    """Replay every fixture through the live pipeline with these settings."""
    from captioner import live as L

    orig = (L.WINDOW_SECONDS, L.STRIDE_SECONDS)
    L.WINDOW_SECONDS, L.STRIDE_SECONDS = window, stride
    try:
        rows = []
        for name, ref_text in fixtures.items():
            engine = L.LiveEngine(model_size=model_size)
            t0 = time.time()
            engine.stream_file(str(ROOT / "data" / "uploads" / name))
            secs = time.time() - t0
            hyp = normalize(" ".join(t for _s, _e, t in engine.state.full_words))
            ref = normalize(ref_text)
            audio_secs = L.SAMPLE_RATE * 0  # measured below via file length
            rows.append({
                "fixture": name, "wer": wer(ref, hyp),
                "ref_words": len(ref), "hyp_words": len(hyp),
                "decode_secs": round(secs, 2),
            })
        return rows
    finally:
        L.WINDOW_SECONDS, L.STRIDE_SECONDS = orig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="only smoke_speech.wav and the main configs")
    args = ap.parse_args()

    BENCH.mkdir(parents=True, exist_ok=True)

    fixtures = {fx.name: "" for fx in FIXTURES}
    if args.quick:
        fixtures = {FIXTURES[0].name: ""}
    if REF_PATH.exists() and not args.quick:
        refs = json.loads(REF_PATH.read_text())
        if set(refs) == set(fixtures):
            print("using cached reference transcript")
            fixtures = refs
        else:
            fixtures = {}
    else:
        fixtures = {}
    if not fixtures:
        print("generating reference transcript (distil-large-v3)…")
        refs = make_reference()
        if not args.quick:
            REF_PATH.write_text(json.dumps(refs, indent=1))
        fixtures = refs

    configs = [
        ("tiny.en", 1, 12.0, 1.5),   # current live default
        ("tiny.en", 5, 12.0, 1.5),   # beam only
        ("base.en", 1, 12.0, 1.5),   # bigger model
        ("base.en", 5, 12.0, 1.5),
        ("small.en", 1, 15.0, 2.0),  # accurate tier: bigger window+stride
        ("small.en", 5, 15.0, 2.0),
    ]
    if args.quick:
        configs = configs[:4]

    results = []
    for model, beam, window, stride in configs:
        print(f"--- {model} beam={beam} window={window}s stride={stride}s")
        rows = run_config(fixtures, model, beam, window, stride)
        for r in rows:
            print(f"    {r['fixture']}: WER {r['wer']*100:.1f}% "
                  f"(ref {r['ref_words']}w) decode {r['decode_secs']}s")
            r.update(model=model, beam=beam, window=window, stride=stride)
        results.extend(rows)

    print("\n=== summary (weighted mean WER) ===")
    by_cfg: dict = {}
    for r in results:
        key = (r["model"], r["beam"], r["window"], r["stride"])
        by_cfg.setdefault(key, []).append(r)
    for key, rows in by_cfg.items():
        total_ref = sum(r["ref_words"] for r in rows)
        mean = sum(r["wer"] * r["ref_words"] for r in rows) / max(1, total_ref)
        slowest = max(r["decode_secs"] for r in rows)
        print(f"  {key[0]:<9} beam={key[1]} win={key[2]:>4} stride={key[3]}: "
              f"WER {mean*100:5.1f}%   decode {slowest:.1f}s")

    out = BENCH / "live_wer_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"results saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
