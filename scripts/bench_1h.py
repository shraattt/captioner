"""One-hour proof benchmark for the Captioner batch pipeline.

Synthesizes ~1 hour of speech by looping the 13.9 s smoke fixture (same
format throughout), then runs the real worker pipeline on it with monitors
sampling wall-clock, process RAM, and GPU VRAM. Prints an honest report —
no stage is faked; if transcription fails, the report says so.

The bench uses its own database and output directory under data/bench/ so
the user's real queue is untouched.

Usage:
    .venv/Scripts/python scripts/bench_1h.py [--size tiny.en] [--hours 1.0]
"""
import argparse
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402  (bench-only dependency)

from captioner import config, db, worker  # noqa: E402

SMOKE = ROOT / "data" / "uploads" / "smoke_speech.wav"
BENCH_DIR = ROOT / "data" / "bench"
BENCH_WAV = BENCH_DIR / "bench_1h.wav"


def build_bench_audio(hours: float):
    """Loop the smoke fixture until we have ~hours of speech. Returns (path, seconds)."""
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    with wave.open(str(SMOKE), "rb") as src:
        params = src.getparams()
        frames = src.readframes(src.getnframes())
        rate = src.getframerate()
    src_seconds = params.nframes / rate
    loops = max(1, round(hours * 3600 / src_seconds))
    want = loops * src_seconds

    if BENCH_WAV.exists():
        try:
            with wave.open(str(BENCH_WAV), "rb") as w:
                if abs(w.getnframes() / w.getframerate() - want) < 1.0 and w.getframerate() == rate:
                    return BENCH_WAV, want
        except Exception:
            pass  # unreadable leftover — rebuild it

    tmp = BENCH_WAV.with_suffix(".tmp.wav")
    with wave.open(str(tmp), "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        for _ in range(loops):
            out.writeframes(frames)
    os.replace(tmp, BENCH_WAV)
    return BENCH_WAV, want


def nvidia_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny.en")
    ap.add_argument("--hours", type=float, default=1.0)
    args = ap.parse_args()

    audio_path, audio_seconds = build_bench_audio(args.hours)
    print(f"audio: {audio_path.name}  {audio_seconds:.0f}s ({audio_seconds / 3600:.2f} h)")

    # Isolated queue + outputs: the user's real database stays untouched.
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = BENCH_DIR / "bench.db"
    config.OUTPUT_DIR = BENCH_DIR / "outputs"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    conn = db.connect()

    baseline_vram = nvidia_used_mib()
    proc = psutil.Process()
    peak_rss = 0
    peak_vram = 0
    stop = threading.Event()

    def monitor():
        nonlocal peak_rss, peak_vram
        last_v = 0.0
        while not stop.is_set():
            try:
                peak_rss = max(peak_rss, proc.memory_info().rss)
            except Exception:
                pass
            if time.monotonic() - last_v >= 1.0:
                last_v = time.monotonic()
                v = nvidia_used_mib()
                if v is not None:
                    peak_vram = max(peak_vram, v)
            stop.wait(0.5)

    def progress_watcher(job_id):
        """Follow the DB row so the log shows the real stage transitions."""
        # Own connection: sqlite connections cannot cross threads.
        watch = db.connect()
        last = None
        try:
            while not stop.is_set():
                try:
                    row = watch.execute(
                        "SELECT status, stage, progress FROM jobs WHERE id = ?", (job_id,)
                    ).fetchone()
                    if row is not None:
                        mark = (row["status"], row["stage"], round(row["progress"] or 0, 2))
                        if mark != last:
                            last = mark
                            print(f"[{time.strftime('%H:%M:%S')}] {mark[0]} stage={mark[1]} progress={mark[2]}", flush=True)
                except Exception:
                    pass
                stop.wait(3.0)
        finally:
            watch.close()

    job_id = worker.enqueue(conn, str(audio_path), model_size=args.size)
    job = worker.claim_next(conn)
    assert job is not None and job["id"] == job_id, "bench job was not claimed"

    watcher = threading.Thread(target=progress_watcher, args=(job_id,), daemon=True)
    watcher.start()
    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()

    print(f"running pipeline: {args.size} on {audio_path.name}", flush=True)
    t0 = time.perf_counter()
    failure = None
    try:
        worker.run_pipeline(conn, job)
    except Exception as exc:  # report honestly, then re-raise for a nonzero exit
        failure = exc
        print(f"PIPELINE FAILED: {exc!r}", flush=True)
    finally:
        elapsed = time.perf_counter() - t0
        stop.set()
        mon.join()
        watcher.join()

    row = conn.execute(
        "SELECT status, error FROM jobs WHERE id = ?", (job["id"],)
    ).fetchone()
    n_seg = conn.execute(
        "SELECT COUNT(*) FROM captions WHERE job_id = ?", (job["id"],)
    ).fetchone()[0]

    vram_note = "n/a" if baseline_vram is None or peak_vram == 0 else (
        f"baseline {baseline_vram} MiB -> peak {peak_vram} MiB "
        f"(delta {peak_vram - baseline_vram:+d} MiB)"
    )
    print("=" * 64)
    print(f"model:            {args.size}")
    print(f"pipeline status:  {row['status']}" + (f"  error: {row['error']}" if row["error"] else ""))
    print(f"audio duration:   {audio_seconds:.0f}s ({audio_seconds / 3600:.2f} h)")
    print(f"wall clock:       {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    if not failure:
        print(f"realtime factor:  {audio_seconds / elapsed:.2f}x audio-speed")
    print(f"peak process RAM: {peak_rss / 2 ** 20:.0f} MiB")
    print(f"GPU VRAM:         {vram_note}")
    print(f"segments stored:  {n_seg}")
    print(f"outputs:          {config.OUTPUT_DIR}")
    print("=" * 64)
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
