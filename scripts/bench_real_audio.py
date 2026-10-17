"""Real-audio benchmark: what the app actually does on a recording you name.

Every headline number in the README used to come from synthesized speech or a
13.9 s fixture. This measures the same things on a real file:

  batch   wall clock, real-time factor, word count (the default batch model)
  live    decode cost per window, caption age in audio time, and the
          agreement between the live and batch transcripts
  quality repetition loops, cue overlaps, coverage

    python scripts/bench_real_audio.py "D:\\lectures\\merge-sort.mp4"
    python scripts/bench_real_audio.py lecture.wav --reference lecture.txt
    python scripts/bench_real_audio.py lecture.mp4 --live-models tiny.en,small.en

Reading the numbers
-------------------
The live path is replayed by `live.LiveEngine.stream_file`, which decodes the
sliding window without pacing by default. That is deliberate: the pacing of a
file replay (sleep, then decode) is not how the app runs, where a capture
thread keeps the audio current and a timer schedules decodes. Two numbers
survive that difference, and they are the ones worth quoting:

  decode cost   seconds of GPU per 12 s window. The app keeps up while this
                stays under the stride (1.5 s with speech, 3 s when quiet).
  word lag      how long after a word is spoken the model first reports it,
                in audio time. This is the *display* latency: the overlay
                shows the unstable tail immediately, so a word reaches the
                screen about this long after it is spoken, and only settles
                into the transcript COMMIT_MARGIN later (caption age).
  caption age   how stale a word is when it is *frozen into the transcript*
                (the dim, stable line and the saved captions). Words settle
                ~COMMIT_MARGIN later than they appear, because the sliding
                window gets that long to revise them first.

`--speed 1.0` replays in real time instead; its wall-clock caption age is
meaningless for the app (it accumulates the decode time the app hides in the
background), so the script labels it a replay backlog.

The reference is optional, and what it changes matters:
  without it  the live transcript is scored against the batch model's own
              transcript of the same audio, the method
              scripts/bench_live_wer.py already uses. That is comparable to
              the existing synthetic table, but it measures *agreement
              between two models*, not accuracy.
  with it     both paths are scored against ground truth you typed, and the
              numbers become real WER.
A character-level similarity is reported too: on technical speech the small
live models split and re-join compounds ("subarrays" -> "sub arrays"), so
word-level agreement reads worse than the transcription actually sounds.
"""
import argparse
import difflib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_wer_check():
    """Reuse the WER gate's normalisation so both scripts score identically."""
    path = ROOT / "scripts" / "wer_check.py"
    spec = importlib.util.spec_from_file_location("_captioner_wer_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_WER = _load_wer_check()


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
        "max": ordered[-1],
    }


def _fmt(stat: dict, unit: str = "s") -> str:
    if not stat.get("n"):
        return "n/a"
    return (f"{stat['mean']:.2f}{unit} mean / {stat['median']:.2f}{unit} median / "
            f"{stat['p95']:.2f}{unit} p95 / {stat['max']:.2f}{unit} max")


def _word_key(text: str) -> str:
    """Case/punctuation-insensitive word, for matching a word across decodes."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _char_similarity(a: str, b: str) -> float:
    """Character agreement with spaces removed: insensitive to how a model
    splits compound words, which word-level WER punishes twice."""
    left = "".join(a.lower().split())
    right = "".join(b.lower().split())
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()


def _repetition(tokens: list[str], size: int = 4) -> dict:
    """Repeated-phrase loops: the hallucination shape Whisper is prone to."""
    if len(tokens) < size * 2:
        return {"worst_ngram": "", "worst_repeats": 0, "max_same_word_run": 0}
    counts: dict[tuple, int] = {}
    for i in range(len(tokens) - size + 1):
        key = tuple(tokens[i:i + size])
        counts[key] = counts.get(key, 0) + 1
    worst = max(counts.items(), key=lambda kv: kv[1])
    run = best = 1
    for prev, cur in zip(tokens, tokens[1:], strict=False):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    return {
        "worst_ngram": " ".join(worst[0]),
        "worst_repeats": worst[1],
        "max_same_word_run": best,
    }


def _sanity(words: list[tuple]) -> dict:
    """Cue health, in both orders that matter.

    `as_written` is the order the words would be written to a saved transcript,
    which is what a player walks. `sorted` is the timeline with the jitter
    removed, where overlaps mean two cues genuinely claim the same instant.
    """
    usable = [(float(s), float(e)) for s, e, _t in words if s is not None and e is not None]
    by_time = sorted(usable)

    def measure(seq: list[tuple]) -> tuple[int, int, int]:
        backwards = sum(1 for a, b in zip(seq, seq[1:], strict=False) if b[0] < a[0])
        overlaps = sum(1 for a, b in zip(seq, seq[1:], strict=False) if b[0] < a[1])
        zero = sum(1 for a, b in seq if b <= a)
        return backwards, overlaps, zero

    backwards, overlaps, zero = measure(usable)
    sorted_overlaps = measure(by_time)[1]
    return {
        "words": len(usable),
        "as_written_backwards": backwards,
        "as_written_overlaps": overlaps,
        "sorted_overlaps": sorted_overlaps,
        "zero_length": zero,
        "span": (by_time[0][0], by_time[-1][1]) if by_time else None,
    }


def _coverage(words: list[tuple], duration: float) -> float:
    """Fraction of the file's timeline that carries at least one caption word."""
    spans = sorted((float(s), float(e)) for s, e, _t in words
                   if s is not None and e is not None)
    if not spans or duration <= 0:
        return 0.0
    merged = 0.0
    start, end = spans[0]
    for next_start, next_end in spans[1:]:
        if next_start > end:
            merged += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    merged += end - start
    return merged / duration


def _cue_health(full_words: list[tuple]) -> dict:
    """Score the .srt the app would actually save from this session.

    The live UI's "Save live transcript" writes grouped, timed captions
    (`live.transcript_to_srt`), so the numbers that matter are in the written
    file: how many cues overlap or run backwards, which is what a player
    shows as text jumping around.
    """
    from captioner import live as L
    from captioner import srt as srt_mod

    text = L.transcript_to_srt(full_words)
    blocks = srt_mod.parse_srt(text)
    overlaps = backwards = zero = 0
    previous = None
    for block in blocks:
        if block.end <= block.start:
            zero += 1
        if previous is not None:
            if block.start < previous.end:
                overlaps += 1
            if block.start < previous.start:
                backwards += 1
        previous = block
    return {"cues": len(blocks), "overlaps": overlaps,
            "out_of_order": backwards, "zero_length": zero}


def decode_audio_mono(path: str):
    """Decode once so the duration and every timeline share an audio clock."""
    import numpy as np
    from faster_whisper.audio import decode_audio

    audio = np.asarray(decode_audio(path, sampling_rate=16_000), dtype=np.float32)
    return audio.reshape(-1)


def run_batch(audio_path: str, model_size: str | None) -> dict:
    from captioner import config, transcribe

    started = time.perf_counter()
    words = transcribe.transcribe_words(audio_path, model_size)
    wall = time.perf_counter() - started
    return {
        "model": model_size or config.BATCH_MODEL_SIZE,
        "wall_seconds": wall,
        "words": len(words),
        "text": " ".join(w.text for w in words),
    }


def run_live(audio_path: str, model_size: str, speed: float) -> dict:
    from captioner import live as L

    engine = L.LiveEngine(model_size=model_size)
    seen = 0
    events: list[tuple[float, list[tuple], float, float | None]] = []
    # When was each word first reported, and by which window?
    first_seen: dict[tuple[str, int], float] = {}

    def on_update(state) -> None:
        nonlocal seen
        now = time.perf_counter()
        # The newest audio this decode heard: the tail of the window it just
        # transcribed. Comparing a word against it gives its age in audio
        # time, with no dependence on how the replay is paced.
        newest = max((seg["end"] for seg in state.segments), default=None)
        for seg in state.segments:
            key = (_word_key(seg["text"]), int(round(float(seg["start"]) * 10)))
            first_seen.setdefault(key, float(state.window_end))
        fresh = list(state.full_words[seen:])
        seen = len(state.full_words)
        if not fresh:
            return
        events.append((now, fresh, float(engine.suggested_stride), newest))

    started = time.perf_counter()
    state = engine.stream_file(str(audio_path), on_update=on_update, speed=speed)
    wall = time.perf_counter() - started

    # Callbacks fire right after each decode_once. Unpaced, the gap between two
    # of them *is* the decode; paced, the loop slept for one stride in between.
    intervals, decode_costs, ages = [], [], []
    for (prev_t, _p, _s, _n_prev), (this_t, _t, stride, _n) in zip(
        events, events[1:], strict=False
    ):
        elapsed = this_t - prev_t
        intervals.append(elapsed)
        decode_costs.append(max(0.0, elapsed - stride) if speed > 0 else elapsed)
    for _t, fresh, _stride, newest in events:
        if newest is None:
            continue
        for _start, end, _text in fresh:
            if end is not None:
                ages.append((float(end), max(0.0, newest - end)))

    # A word cannot be frozen before COMMIT_MARGIN has passed, so words in the
    # first moments carry an age that says nothing about steady state. Quote
    # steady state; the opening is a fixed cost of the design.
    steady = [age for end, age in ages if end >= L.WINDOW_SECONDS]
    opening = [age for end, age in ages if end < L.WINDOW_SECONDS]
    full = [tuple(w) for w in state.full_words]
    # A word's display lag: the window that first reported it, minus its end.
    # Words the session never reported at all have no lag to report.
    lags = []
    for start, end, text in full:
        if end is None or float(end) < L.WINDOW_SECONDS:
            continue
        bucket = int(round(float(start) * 10))
        for near in range(bucket - 10, bucket + 11):
            reported = first_seen.get((_word_key(text), near))
            if reported is not None:
                lags.append(max(0.0, reported - float(end)))
                break
    return {
        "model": model_size,
        "wall_seconds": wall,
        "words": len(full),
        "text": " ".join(t for _s, _e, t in full),
        "words_raw": full,
        "decode_interval": _stats(intervals),
        "decode_cost": _stats(decode_costs),
        "word_lag": _stats(lags),
        "caption_age": _stats(steady),
        "caption_age_opening": _stats(opening),
        "late_drops": int(getattr(engine, "late_drops", 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--reference", default=None,
                    help="ground-truth transcript (.txt) or subtitles (.srt/.vtt)")
    ap.add_argument("--model", default=None,
                    help="batch model (default: config.BATCH_MODEL_SIZE)")
    ap.add_argument("--live-models", default=None,
                    help="comma-separated live models to compare "
                         "(default: config.LIVE_MODEL_SIZE)")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="replay pacing: 0 = decode as fast as the GPU allows "
                         "(default), 1.0 = real time")
    ap.add_argument("--json", default=str(ROOT / "data" / "bench" / "real_audio_results.json"))
    ap.add_argument("--skip-batch", action="store_true",
                    help="live paths only (no agreement baseline)")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"audio file missing: {audio_path}")
        return 1

    from captioner import config, models
    from captioner import live as L

    audio = decode_audio_mono(str(audio_path))
    duration = len(audio) / 16_000
    print(f"file      : {audio_path.name}")
    print(f"duration  : {duration:.1f}s  ({duration/60:.2f} min)")
    print(f"replay    : {'as fast as the GPU allows' if not args.speed else f'{args.speed}x real time'}\n")

    report: dict = {
        "file": str(audio_path),
        "duration_seconds": duration,
        "speed": args.speed,
    }

    # ---- batch -----------------------------------------------------------
    batch = None
    batch_tokens: list[str] = []
    if not args.skip_batch:
        print("running the batch pipeline...")
        batch = run_batch(str(audio_path), args.model)
        batch["realtime_factor"] = duration / batch["wall_seconds"] if batch["wall_seconds"] else 0.0
        batch_tokens = _WER.normalize(batch["text"])
        print(f"batch ({batch['model']}): {batch['wall_seconds']:.1f}s wall, "
              f"{batch['realtime_factor']:.1f}x real time, {batch['words']} words\n")
        report["batch"] = {k: v for k, v in batch.items() if k != "text"}

    # ---- live, one pass per model ---------------------------------------
    wanted = (args.live_models or config.LIVE_MODEL_SIZE).split(",")
    resolved = []
    for label in wanted:
        try:
            resolved.append((label.strip(), models.resolve(label.strip())))
        except ValueError as exc:
            print(f"live model: {exc}")
            return 2

    truth = None
    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"reference file missing: {ref_path}")
            return 1
        truth = _WER.normalize(_WER.load_reference(ref_path))
        print(f"reference : {len(truth)} words from {ref_path.name}\n")

    report["live"] = {}
    summary = []
    for label, live_model in resolved:
        print(f"replaying through the live pipeline ({label})...")
        live = run_live(str(audio_path), live_model, args.speed)
        live_tokens = _WER.normalize(live["text"])
        live_words = [(float(s), float(e), t) for s, e, t in live["words_raw"]]
        entry = {
            "model": live_model,
            "wall_seconds": live["wall_seconds"],
            "words": live["words"],
            "decode_cost": live["decode_cost"],
            "word_lag": live["word_lag"],
            "caption_age": live["caption_age"],
            "realtime_factor": duration / live["wall_seconds"] if live["wall_seconds"] else 0.0,
            "caption_age_opening": live["caption_age_opening"],
            "late_drops": live["late_drops"],
            "words_raw": live["words_raw"],
            "sanity": _sanity(live_words),
            "saved_srt": _cue_health(live_words),
            "coverage": _coverage(live_words, duration),
            "words_per_min": len(live_tokens) / (duration / 60) if duration else 0.0,
            "repetition": _repetition(live_tokens),
        }
        print(f"\n  {label}: {live['wall_seconds']:.1f}s wall, {live['words']} words")
        print(f"    decode cost  : {_fmt(live['decode_cost'])}")
        print(f"    word lag     : {_fmt(live['word_lag'])} behind the window edge "
              f"- how long a word takes to reach the screen")
        print(f"    caption age  : {_fmt(live['caption_age'])} - how long until a "
              f"word settles into the transcript and stops being revised")
        print(f"    opening window: caption age "
              f"{live['caption_age_opening'].get('max', 0):.1f}s max "
              f"(the first strides cannot reach the commit margin)")
        print(f"    late re-writes refused: {live['late_drops']} "
              f"(a second, later estimate of audio already captioned)")
        if batch is not None:
            wer_score, sub, dele, ins = _WER.wer(batch_tokens, live_tokens)
            char_sim = _char_similarity(batch["text"], live["text"])
            entry["agreement"] = {"wer": wer_score, "sub": sub, "del": dele, "ins": ins,
                                  "char_similarity": char_sim}
            print(f"    vs batch     : {wer_score*100:.1f}% word disagreement "
                  f"({sub} sub, {dele} del, {ins} ins), "
                  f"{char_sim*100:.1f}% character agreement")
        if truth is not None:
            wer_score, sub, dele, ins = _WER.wer(truth, live_tokens)
            entry["wer_vs_reference"] = wer_score
            print(f"    vs reference : WER {wer_score*100:.1f}% "
                  f"({sub} sub, {dele} del, {ins} ins)")
            if len(truth) < 0.5 * len(live_tokens):
                print(f"    [!] that reference covers only {len(truth)} words against "
                      f"{len(live_tokens)} transcribed: it scores a preamble, and "
                      f"every word past it counts as an insertion. Reference the "
                      f"whole file (or trim the audio) for a number worth quoting.")
        flag = "OK" if live["decode_cost"].get("p95", 9.9) < L.STRIDE_SECONDS else "TOO SLOW"
        print(f"    keeps up     : decode p95 {live['decode_cost'].get('p95', 0):.2f}s "
              f"vs {L.STRIDE_SECONDS}s stride -> {flag}")
        srt_health = entry["saved_srt"]
        print(f"    saved .srt   : {srt_health['cues']} cues, "
              f"{srt_health['overlaps']} overlapping, "
              f"{srt_health['out_of_order']} out of order, "
              f"{srt_health['zero_length']} zero-length")
        report["live"][label] = entry
        summary.append((label, entry))

    if len(summary) > 1:
        ranked = sorted(summary, key=lambda kv: kv[1].get("agreement", {}).get("wer", 1.0))
        print("\nlive models by word disagreement vs batch "
              "(lower is better, quote decode cost alongside):")
        for label, entry in ranked:
            agreement = entry.get("agreement", {}).get("wer")
            shown = f"{agreement*100:.1f}%" if agreement is not None else "n/a"
            print(f"  {label:<10} {shown:>6}  decode p95 "
                  f"{entry['decode_cost'].get('p95', 0):.2f}s")

    # ---- batch quality, for the same reference-free signals -------------
    rep_batch = _repetition(batch_tokens) if batch_tokens else {}
    if rep_batch:
        print(f"\nbatch repetition: worst 4-gram x{rep_batch['worst_repeats']} "
              f"({rep_batch['worst_ngram']!r}), longest same-word run "
              f"{rep_batch['max_same_word_run']}")
        report["repetition_batch"] = rep_batch
    for label, entry in summary:
        rep = entry["repetition"]
        sanity = entry["sanity"]
        print(f"live {label}: repetition x{rep['worst_repeats']} "
              f"({rep['worst_ngram']!r}), longest same-word run {rep['max_same_word_run']}")
        print(f"  raw frozen list (before the export sorts it): "
              f"{sanity['as_written_overlaps']} overlaps in decode order, "
              f"{sanity['sorted_overlaps']} on the sorted timeline, "
              f"{entry['coverage']*100:.1f}% covered, {entry['words_per_min']:.0f} words/min")

    for label, entry in summary:
        health = entry["saved_srt"]
        broken = health["overlaps"] + health["out_of_order"] + health["zero_length"]
        if health["cues"] and broken:
            print(f"\n[!] {label}: the transcript this session would save has "
                  f"{health['zero_length']} cues whose end precedes their start, "
                  f"{health['overlaps']} overlapping and {health['out_of_order']} "
                  f"out-of-order cues, out of {health['cues']}.")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nresults saved: {out}")
    if truth is None and batch is not None:
        print("no --reference given: every live number above is model agreement, "
              "not accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
