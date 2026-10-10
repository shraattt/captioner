"""Stress the live pipeline: repeated sessions, memory + VRAM monitoring.

Runs many consecutive stream_file sessions (tiny.en, GPU) and reports:
  - per-session decode time (slowdown = leak/fragmentation symptom)
  - process RSS after each session (monotonic growth = leak)
  - GPU utilization/memory sampled from nvidia-smi during decodes
  - transcript sanity per session (word count in a plausible range)

    QT_QPA_PLATFORM=offscreen python scripts/stress_live.py [--sessions 30]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURES = ["data/uploads/smoke_speech.wav", "data/uploads/long_speech.wav"]
# plausible frozen-word range per fixture (smoke ~31 incl. tail flush; long ~189)
EXPECTED_WORDS = {"smoke_speech.wav": (20, 45), "long_speech.wav": (150, 220)}

try:
    import psutil
except ImportError:
    psutil = None


def gpu_sample() -> dict:
    """One nvidia-smi reading; {} when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        util, mem = out.splitlines()[0].split(", ")
        return {"util": int(util), "mem_mib": int(mem)}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=30)
    args = ap.parse_args()

    from captioner import live as L

    proc = psutil.Process() if psutil else None
    rows = []
    gpu_peak = {"util": 0, "mem_mib": 0}

    for i in range(args.sessions):
        fixture = FIXTURES[i % len(FIXTURES)]
        engine = L.LiveEngine(model_size="tiny.en")
        t0 = time.time()
        engine.stream_file(str(ROOT / fixture))
        secs = time.time() - t0
        n_words = len(engine.state.full_words)

        # sample GPU right after the session's final decode
        g = gpu_sample()
        if g:
            gpu_peak["util"] = max(gpu_peak["util"], g["util"])
            gpu_peak["mem_mib"] = max(gpu_peak["mem_mib"], g["mem_mib"])

        rss = proc.memory_info().rss // (1024 * 1024) if proc else 0
        ok_range = EXPECTED_WORDS[Path(fixture).name]
        ok = ok_range[0] <= n_words <= ok_range[1]
        rows.append({
            "session": i + 1, "fixture": Path(fixture).name,
            "words": n_words, "words_ok": ok,
            "decode_secs": round(secs, 2), "rss_mib": rss,
        })
        flag = "" if ok else "  <-- WORD COUNT OUT OF RANGE"
        print(f"session {i+1:>3}: {Path(fixture).name:<18} words={n_words:>3} "
              f"decode={secs:5.2f}s rss={rss}MiB gpu={g.get('util','-')}%/{g.get('mem_mib','-')}MiB{flag}")

    #analysis
    bad = [r for r in rows if not r["words_ok"]]
    first, last = rows[0], rows[-1]
    rss_growth = (last["rss_mib"] - first["rss_mib"]) if rows and first["rss_mib"] else 0
    decodes = [r["decode_secs"] for r in rows]

    print("\n=== stress summary ===")
    print(f"sessions: {len(rows)}   word-count failures: {len(bad)}")
    print(f"decode time: first {decodes[0]:.2f}s, last {decodes[-1]:.2f}s, "
          f"max {max(decodes):.2f}s (growth would indicate a leak)")
    print(f"RSS: {first['rss_mib']} -> {last['rss_mib']} MiB "
          f"(growth {rss_growth} MiB across the run)")
    print(f"GPU peak during sessions: {gpu_peak['util']}% util, "
          f"{gpu_peak['mem_mib']} MiB used")

    out = ROOT / "data" / "bench" / "stress_live_results.json"
    out.write_text(json.dumps({
        "rows": rows, "gpu_peak": gpu_peak,
        "rss_growth_mib": rss_growth,
        "word_failures": len(bad),
    }, indent=1))
    print(f"saved: {out}")
    # leak heuristic per fixture: a session taking 3x the fixture's own
    # median (plus slack) means slowdown, not just a longer file
    slow = any(
        x > 3 * _median(vals) + 2.0
        for name in EXPECTED_WORDS
        for vals in [[y["decode_secs"] for y in rows if y["fixture"] == name]]
        for x in vals
    )
    return 1 if bad or slow else 0


def _median(values) -> float:
    v = sorted(values)
    return v[len(v) // 2] if v else 0.0


if __name__ == "__main__":
    sys.exit(main())
