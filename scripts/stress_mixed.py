"""Contention stress: batch transcribe (distil) + live stream (tiny) at once.

A user can start live captions while a batch job runs; both share the 6GB
GPU. This script runs one of each concurrently and reports per-lane timing
against the solo baselines from data/bench, so slowdown or OOM shows up as
numbers, not a support ticket.

    QT_QPA_PLATFORM=offscreen python scripts/stress_mixed.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

results = {"batch": {}, "live": {}}


def batch_lane() -> None:
    from captioner import transcribe

    t0 = time.time()
    try:
        words = transcribe.transcribe_words(str(ROOT / "data/uploads/long_speech.wav"), "distil-large-v3")
        results["batch"] = {"ok": True, "words": len(words), "secs": round(time.time() - t0, 1)}
    except Exception as exc:
        results["batch"] = {"ok": False, "error": str(exc), "secs": round(time.time() - t0, 1)}


def live_lane() -> None:
    from captioner import live as L

    t0 = time.time()
    sessions = 0
    try:
        while time.time() - t0 < 75:  # keep captioning while batch works
            engine = L.LiveEngine(model_size="tiny.en")
            engine.stream_file(str(ROOT / "data/uploads/smoke_speech.wav"))
            sessions += 1
        results["live"] = {"ok": True, "sessions": sessions,
                           "secs_per_session": round((time.time() - t0) / max(1, sessions), 2)}
    except Exception as exc:
        results["live"] = {"ok": False, "error": str(exc), "sessions": sessions}


def main() -> int:
    import threading

    tb = threading.Thread(target=batch_lane)
    tl = threading.Thread(target=live_lane)
    t0 = time.time()
    tb.start(); tl.start()
    tb.join(); tl.join()
    results["total_secs"] = round(time.time() - t0, 1)

    print(json.dumps(results, indent=1))
    out = ROOT / "data" / "bench" / "stress_mixed_results.json"
    out.write_text(json.dumps(results, indent=1))
    ok = results["batch"].get("ok") and results["live"].get("ok")
    print("MIXED STRESS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
