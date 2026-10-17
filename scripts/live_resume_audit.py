"""How late are the first captions after a pause, on a recording you name?

The wait between live decodes is a *period* (`ui.overlay.stride_wait`): every
STRIDE_SECONDS while somebody is talking, IDLE_STRIDE_SECONDS once they stop, so
the GPU rests through a silence. That back-off is a good guess about the audio
and a bad one about the speaker - the words they say on returning are already in
the ring, and the loop sleeps over them until the nap it chose ends. The loop now
listens instead (`ui.overlay.wait_for_sound`), waking the moment the capture's
50 ms meter hears a block at the engine's silence gate.

The same recording is replayed through the shipping loop
(`ui.overlay.run_live_session`) in real time, once per *pass*, all on the same
audio and machine:

    before   the loop as it was: full 12 s windows, no wake (a capture whose
             meter never crosses the gate)
    wake     the idle stride ends on sound, still decoding full windows
    short    the wake plus a 4 s window for the first decode after a pause.
             Measured: it does not make the first caption after a pause arrive
             earlier, so it is not what ships (README, "On real audio")
    burst    the wake, plus the short cadence afterwards: for the first
             RESUME_BURST_SECONDS of resumed audio the loop decodes every
             RESUME_STRIDE_SECONDS on that 4 s window, because the words the
             model will not yet report are caught by whatever decode comes next
             and one stride is too long to wait for it

Each configuration is replayed *several* times (`--repeats`, default 3) with the
pass order rotating, because one pass per configuration cannot answer what the
comparison is asked: two repeats of the *same* configuration differ by as much
as some of the differences being compared. Rotation gives every configuration
every slot in turn, so whatever a pass inherits from the pass before it (a warm
GPU, a fan that has spun up) is not charged to one of them.

    python scripts/live_resume_audit.py "D:\\lectures\\merge-sort.mp4"
    python scripts/live_resume_audit.py lecture.mp4 --max-seconds 60   # smoke test
    python scripts/live_resume_audit.py lecture.mp4 --repeats 1        # one look, no error bar
    python scripts/live_resume_audit.py lecture.mp4 --passes before,wake,burst --baseline wake
    python scripts/live_resume_audit.py lecture.mp4 --passes before,wake
    python scripts/live_resume_audit.py lecture.mp4 --window-cost   # just the cost

Reading the numbers
-------------------
A *resume* is a pause of at least PAUSE_SECONDS - the length that makes the loop
back off - followed by speech, counted only once the window has filled. For each
one, and for each pass:

  to caption     wall seconds from the resumed audio reaching the ring to the
                 decode that first transcribes it. The overlay prints a decode's
                 text as soon as it returns, so this is screen latency.
  loop waits     the same, minus the decode itself: when the loop *starts* the
                 decode that catches the words. This is the part the wake can
                 change; the decode's cost varies with the machine, and the
                 model's own lag before it will report a word is neither.
  to transcript  wall seconds to the decode that freezes it into the saved text.

The control pass is the same measurement without the wake. The difference is what
the wake buys. Cost is reported as decode *count* rather than GPU seconds: the
per-decode cost between two passes of the same 4-minute recording differed by
more than the extra decodes did (0.68 s vs 0.54 s median on one recording, 0.49
vs 0.51 on the other, following the order the passes ran rather than the wake),
so a GPU-second total would say more about the machine's mood than about this.

Across repeats a configuration's numbers are *pooled*, but per pause they are
averaged first: repeats disagree about how long a given pause took, and pooling
them would also change which pauses are being compared every time a repeat missed
a caption. The difference is then reported with its spread, a 95% interval over
pauses and an exact sign test - the p-value says whether the *direction* is
consistent, the spread says whether the size is worth having, and the
repeat-to-repeat spread of a configuration's own mean is the floor both are read
against. A single run cannot measure that floor, which is the whole reason for
the repeats.

Results are appended to `data/bench/live_resume_results.json` under the file's
name: pooled numbers in `passes`, one entry per repeat in `per_repeat`, and the
per-pause rows for each repeat in `rows`.
"""
import argparse
import importlib.util
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_bench():
    """Reuse the real-audio benchmark's stats, formatting and audio decode, so
    this script's numbers are measured and printed exactly like the README's."""
    path = ROOT / "scripts" / "bench_real_audio.py"
    spec = importlib.util.spec_from_file_location("_captioner_bench_real_audio", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BENCH = _load_bench()


class FileReplay(threading.Thread):
    """Feed a recording into a capture ring in real time, like a device would.

    Pacing is the point: the wait between decodes is wall-clock, so the audio has
    to arrive at the rate it was recorded. Every write is stamped, which is how a
    moment in the file maps to the moment the loop could first have heard it -
    measured rather than assumed, because a sleep is never exactly its argument.
    """

    def __init__(self, capture, audio, blocksize_ms: float = 100.0):
        from captioner.live import SAMPLE_RATE

        super().__init__(daemon=True, name="resume-replay")
        self.capture = capture
        self.audio = audio
        self.block = max(1, int(blocksize_ms * SAMPLE_RATE / 1000))
        self.marks: list[tuple[float, float]] = []  # (wall clock, seconds into the file)
        self._stop = threading.Event()

    def heard_at(self, content_seconds: float) -> float:
        """Wall clock at which *content_seconds* of the file reached the ring."""
        for wall, content in self.marks:
            if content >= content_seconds:
                return wall
        return float("inf")

    def run(self) -> None:
        from captioner.live import SAMPLE_RATE

        started = time.perf_counter()
        pos = 0
        while pos < len(self.audio) and not self._stop.is_set():
            chunk = self.audio[pos:pos + self.block]
            self.capture.write_frames(chunk)
            pos += len(chunk)
            content = pos / SAMPLE_RATE
            self.marks.append((time.perf_counter(), content))
            slack = (started + content) - time.perf_counter()  # hold the ring to real time
            if slack > 0:
                self._stop.wait(slack)


def _capture_for(wake: bool):
    """The real capture, or the pre-wake one whose meter never crosses the gate."""
    from captioner.live import MicCapture

    if wake:
        return MicCapture(device_id="unused")

    class DeafCapture(MicCapture):
        """The live loop before the wake: a meter that never hears anything."""

        @property
        def sound_level(self) -> float:
            return 0.0

    return DeafCapture(device_id="unused")


def _decode_seconds(model, audio, options: dict) -> tuple:
    """Run one transcription and return ``(seconds, segment count)``.

    `WhisperModel.transcribe` hands back a *generator*: calling it only builds
    a segment iterator, and the GPU work happens as it is consumed. Timing the
    call alone measures nothing - which is how a first version of this script
    reported 0.06 s for a 12 s window the live loop pays 0.5 s for.
    """
    began = time.perf_counter()
    segments = list(model.transcribe(audio, **options)[0])
    return time.perf_counter() - began, len(segments)


def warm_model(model_label: str, audio) -> float:
    """Load the model and decode once before anything is timed.

    The first decode of a session pays the weight load and the CUDA allocator.
    In the smoke run that was a 5.5 s outlier against a 0.7 s median, and it
    lands in whichever pass goes first - which would put the comparison's error
    on one side of it. Returns what the warm-up cost, for the record.
    """
    from captioner import transcribe
    from captioner.live import SAMPLE_RATE

    device = transcribe.resolve_device()
    model = transcribe.get_model(model_label, device, transcribe.resolve_compute_type(device))
    seconds, _segments = _decode_seconds(model, audio[: int(3.0 * SAMPLE_RATE)],
                                        {"language": "en", "word_timestamps": True})
    return seconds


def run_pass(audio, model_label: str, wake: bool, resume_window: float | None,
             resume_stride: float | None = None) -> dict:
    """One real-time pass over the recording: what every decode ran and read."""
    from captioner.live import LiveEngine
    from ui.overlay import run_live_session

    engine = LiveEngine(model_size=model_label, resume_window_seconds=resume_window,
                        resume_stride_seconds=resume_stride)
    capture = _capture_for(wake)
    decodes: list[dict] = []
    errors: list[str] = []
    stop = threading.Event()
    seen_frozen = 0
    real_decode = engine.decode_once

    def timed_decode(source):
        """Wrap the engine's decode so the audit knows when it ran and what it
        read - all of it from the same public state the overlay renders."""
        nonlocal seen_frozen
        started = time.perf_counter()
        state = real_decode(source)
        ended = time.perf_counter()
        decodes.append({
            "started": started,
            "ended": ended,
            # words the model reported, with absolute times (what reaches the screen)
            "seen": [(float(seg["start"]), float(seg["end"]), str(seg["text"]))
                     for seg in state.segments],
            # words this decode froze into the transcript
            "frozen": [(float(w[0]), str(w[2])) for w in state.full_words[seen_frozen:]],
        })
        seen_frozen = len(state.full_words)
        return state

    engine.decode_once = timed_decode
    replay = FileReplay(capture, audio)
    loop = threading.Thread(
        target=run_live_session,
        args=(engine, capture, stop, lambda _state: None, errors.append),
        daemon=True,
    )
    loop.start()
    started = time.perf_counter()
    replay.start()
    replay.join()
    time.sleep(1.5)  # the decode in flight still has to land
    stop.set()
    loop.join(timeout=30)
    engine.close()
    return {
        "decodes": decodes,
        "errors": errors,
        "replay": replay,
        "wall_seconds": time.perf_counter() - started,
        "gpu_seconds": sum(d["ended"] - d["started"] for d in decodes),
        "late_drops": int(getattr(engine, "late_drops", 0)),
    }


def window_cost(audio, model_label: str, resume_window: float, full_window: float,
                reps: int = 6) -> dict:
    """What a short resume window saves per decode, measured in isolation.

    A session's GPU totals cannot answer this: two passes of the *same*
    recording and configuration differed by 0.15 s of mean decode cost (0.44
    against 0.59) purely from when they ran, which is bigger than the window.
    So this decodes the same slices at both lengths, alternately, the short one
    being the tail of the long one (which is what a resume decode reads):
    alternating cancels a drift, and nothing else differs between the two sides.
    """
    from captioner import transcribe
    from captioner.live import SAMPLE_RATE

    device = transcribe.resolve_device()
    model = transcribe.get_model(model_label, device,
                                 transcribe.resolve_compute_type(device))
    options = {"language": "en", "word_timestamps": True, "vad_filter": True,
               "vad_parameters": {"min_silence_duration_ms": 350},
               "condition_on_previous_text": False, "beam_size": 1}
    full_frames = int(full_window * SAMPLE_RATE)
    short_frames = int(resume_window * SAMPLE_RATE)
    span = max(1, len(audio) - full_frames)
    times: dict[str, list[float]] = {"full": [], "short": []}
    for step in range(1, reps + 1):
        start = int(span * step / (reps + 1))
        for name, frames in (("full", full_frames), ("short", short_frames)):
            bout = audio[start:start + frames]
            if len(bout) < frames:
                continue
            seconds, _segments = _decode_seconds(model, bout, options)
            times[name].append(seconds)
    return {name: _BENCH._stats(values) for name, values in times.items()}


def speech_runs(levels, gate: float, min_quiet: float, after: float) -> list[dict]:
    """Quiet stretches long enough to stop the loop, and when speech returned."""
    from captioner.live import SPEECH_BLOCK_SECONDS

    above = levels >= gate
    events: list[dict] = []
    index = 0
    while index < len(above):
        if above[index]:
            index += 1
            continue
        start = index
        while index < len(above) and not above[index]:
            index += 1
        quiet = (index - start) * SPEECH_BLOCK_SECONDS
        resume = index * SPEECH_BLOCK_SECONDS
        # `index` is past the end on a trailing silence: nobody came back, so
        # there is no resume to time. `after` skips the opening, where the window
        # is still filling and the loop has not settled into a stride.
        if index < len(above) and quiet >= min_quiet and resume >= after:
            events.append({"quiet": quiet, "resume": resume})
    return events


def latencies(result: dict, event: dict, tolerance: float) -> dict:
    """How long one resume took to reach the screen and the transcript.

    The caption is the first word the model reports from the resumed speech;
    the word itself is kept in the result, because a suspiciously fast number
    is usually a timestamp that drifted over the boundary and reading the word
    tells those apart from a genuine first word.
    """
    resumed_at = event["resume"]
    heard = result["replay"].heard_at(resumed_at)
    found: dict = {"to_caption": None, "to_caption_start": None, "to_transcript": None,
                   "word": "", "word_start": None}
    for decode in result["decodes"]:
        if decode["started"] < heard:
            continue  # this decode read the ring before the speaker was back
        if found["to_caption"] is None:
            for start, _end, text in decode["seen"]:
                if start >= resumed_at - tolerance:
                    found["to_caption"] = decode["ended"] - heard
                    found["to_caption_start"] = decode["started"] - heard
                    found["word"], found["word_start"] = text, start
                    break
        if found["to_transcript"] is None and any(
                s >= resumed_at - tolerance for s, _t in decode["frozen"]):
            found["to_transcript"] = decode["ended"] - heard
        if found["to_caption"] is not None and found["to_transcript"] is not None:
            break
    return found


def collect(result: dict, events: list[dict], tolerance: float) -> dict:
    """Per-resume latencies and the pass's own cost."""
    rows, captions, starts, transcripts = [], [], [], []
    for event in events:
        found = latencies(result, event, tolerance)
        rows.append({"resume": event["resume"], "quiet": event["quiet"], **found})
        if found["to_caption"] is not None:
            captions.append(found["to_caption"])
            starts.append(found["to_caption_start"])
        if found["to_transcript"] is not None:
            transcripts.append(found["to_transcript"])
    decode_cost = [d["ended"] - d["started"] for d in result["decodes"]]
    # The whole transcript this pass wrote, so a change to *what* is decoded
    # (not just when) cannot hide behind the latency numbers.
    written = [w for d in result["decodes"] for w in d["frozen"]]
    return {
        "rows": rows,
        "values": {"to_caption": captions, "to_caption_start": starts,
                   "to_transcript": transcripts, "decode_cost": decode_cost},
        "to_caption": _BENCH._stats(captions),
        "to_caption_start": _BENCH._stats(starts),
        "to_transcript": _BENCH._stats(transcripts),
        "missed": len(events) - len(captions),
        "decodes": len(result["decodes"]),
        "decode_cost": _BENCH._stats(decode_cost),
        "gpu_seconds": result["gpu_seconds"],
        "wall_seconds": result["wall_seconds"],
        "errors": result["errors"],
        "words": len(written),
        "late_drops": result["late_drops"],
        "text": " ".join(str(t) for _s, t in written),
        "timeline": [(round(d["started"], 3), round(d["ended"] - d["started"], 3))
                     for d in result["decodes"]],
    }


def rotation(names: list[str], repeat: int) -> list[str]:
    """This repeat's pass order: every configuration takes every slot in turn.

    Rotating cannot remove drift *within* one repeat, but it spreads it evenly
    over the configurations across repeats, which is what makes the repeats'
    differences comparable with each other. Without it, the configuration that
    always runs second wears whatever the pass before it left behind, and the
    comparison carries that as if it were the change being measured.
    """
    shift = repeat % len(names) if names else 0
    return names[shift:] + names[:shift]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _sd(values: list[float]) -> float:
    """Sample standard deviation; 0.0 for a lone value - nothing to vary over."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def sign_test(faster: int, slower: int) -> float:
    """Two-sided exact binomial p-value for a sign test, ties dropped.

    Pauses come in tens, not thousands, so the exact test beats a normal
    approximation: it is simply the chance of this lopsided a split if a pause
    were as likely to get slower as faster. It speaks to the *direction* being
    consistent, never to the size - read it beside the mean and the spread.
    """
    n = faster + slower
    if not n:
        return 1.0
    tail = min(faster, slower)
    p = 2 * sum(math.comb(n, i) for i in range(tail + 1)) / 2 ** n
    return min(1.0, p)


def per_pause(runs: list[dict], key: str) -> list[float | None]:
    """One value per pause: the mean of the repeats that managed to time it.

    Repeats of the same pause differ by the machine's mood - a decode landing
    either side of the loop's stride - rather than by the pause, so averaging
    them is what turns several runs into one number per pause. A pause no
    repeat timed comes back as None and drops out of the comparisons.
    """
    values: list[float | None] = []
    for index in range(len(runs[0]["rows"])):
        seen = [run["rows"][index][key] for run in runs
                if run["rows"][index][key] is not None]
        values.append(_mean(seen) if seen else None)
    return values


def pause_words(runs: list[dict], index: int) -> str:
    """The first word any repeat printed for this pause.

    Kept beside the timings because a suspiciously fast caption is nearly
    always a drifted timestamp from before the pause, and reading the word is
    how those are told from a genuine first word.
    """
    for run in runs:
        word = run["rows"][index]["word"]
        if word:
            return word
    return ""


def paired(before: list[float | None], after: list[float | None]) -> dict:
    """The per-pause differences between two configurations' pauses.

    Positive means the second configuration got the caption out first. A pause
    either side missed drops out of the pair, so the same pauses are compared
    throughout - which is the point of pairing them at all.
    """
    removed = [left - right for left, right in zip(before, after, strict=False)
               if left is not None and right is not None]
    faster = sum(1 for value in removed if value > 0)
    slower = sum(1 for value in removed if value < 0)
    spread = _sd(removed)
    return {
        "removed": removed,
        "mean": _mean(removed),
        "median": statistics.median(removed) if removed else 0.0,
        "sd": spread,
        "interval": 1.96 * spread / len(removed) ** 0.5 if removed else 0.0,
        "faster": faster,
        "slower": slower,
        "same": len(removed) - faster - slower,
        "p": sign_test(faster, slower),
    }


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def repeat_means(runs: list[dict], key: str) -> list[float]:
    """Each repeat's own mean for one measurement - the drift, made visible."""
    return [run[key]["mean"] for run in runs if run[key].get("n")]


def repeat_spread(runs: list[dict], key: str) -> float:
    means = repeat_means(runs, key)
    return max(means) - min(means) if len(means) > 1 else 0.0


def repeat_range(runs: list[dict], key: str) -> str:
    means = repeat_means(runs, key)
    if not means:
        return "repeats: n/a"
    if len(means) == 1:
        return f"repeats: {means[0]:.2f}s (one run - no spread to measure)"
    return ("repeats: " + " / ".join(f"{value:.2f}s" for value in means)
            + f"  (spread {max(means) - min(means):.2f}s)")


def pool(runs: list[dict]) -> dict:
    """Several repeats of one configuration, reported as one.

    The repeats are as interchangeable as this audit can make them, so the
    safest summary of a configuration is all of its measurements at once. The
    per-repeat means stay available beside it: the distance between them is the
    drift a single run cannot see past, and the reason this script runs more
    than one pass.
    """
    def flat(key: str) -> list[float]:
        return [value for run in runs for value in run["values"][key]]

    pooled = {key: _BENCH._stats(flat(key))
              for key in ("to_caption", "to_caption_start", "to_transcript",
                          "decode_cost")}
    pooled.update(
        missed=sum(run["missed"] for run in runs),
        decodes=sum(run["decodes"] for run in runs),
        gpu_seconds=sum(run["gpu_seconds"] for run in runs),
        wall_seconds=sum(run["wall_seconds"] for run in runs),
        words=sum(run["words"] for run in runs),
        late_drops=sum(run["late_drops"] for run in runs),
        errors=sorted({error for run in runs for error in run["errors"]}),
    )
    return pooled


def main() -> int:
    from captioner import config, models
    from captioner.live import (
        PAUSE_SECONDS,
        RESUME_STRIDE_SECONDS,
        RESUME_WINDOW_SECONDS,
        SAMPLE_RATE,
        WINDOW_SECONDS,
        LiveEngine,
        speech_levels,
    )

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--model", default=None,
                    help="live model (default: config.LIVE_MODEL_SIZE)")
    ap.add_argument("--passes", default="before,wake,short,burst",
                    help="comma-separated passes: before (no wake, full window), "
                         "wake (wake, full window), short (wake and the resume "
                         "window), burst (wake and a short cadence for the first "
                         "moments after the pause)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="how many times to replay each configuration, with the "
                         "pass order rotating (default 3: one run per "
                         "configuration cannot measure the drift it is trying "
                         "to see past, so it cannot settle a small effect)")
    ap.add_argument("--baseline", default="before",
                    help="pass the others are compared against (default: before)")
    ap.add_argument("--resume-window", type=float, default=None,
                    help="the short pass's window in seconds "
                         "(default: live.RESUME_WINDOW_SECONDS)")
    ap.add_argument("--window-cost", action="store_true",
                    help="only measure what the short window saves per decode, "
                         "interleaved against the full one, and exit")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="replay only the first N seconds (smoke test)")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="a word this far before the resume still counts as the "
                         "resumed speech (kept tight: a loose one lets a drifted "
                         "timestamp from before the pause count as a caption)")
    ap.add_argument("--json", default=str(ROOT / "data" / "bench" / "live_resume_results.json"))
    args = ap.parse_args()

    path = Path(args.audio)
    if not path.exists():
        print(f"audio file missing: {path}")
        return 1
    if args.repeats < 1:
        print(f"--repeats must be at least 1, not {args.repeats}")
        return 2

    try:
        model = models.resolve(args.model) if args.model else config.LIVE_MODEL_SIZE
    except ValueError as exc:
        print(f"live model: {exc}")
        return 2

    audio = _BENCH.decode_audio_mono(str(path))
    if args.max_seconds > 0:
        audio = audio[: int(args.max_seconds * SAMPLE_RATE)]
    duration = len(audio) / SAMPLE_RATE
    resume_window = (args.resume_window if args.resume_window is not None
                     else RESUME_WINDOW_SECONDS)

    if args.window_cost:
        print(f"decoding {WINDOW_SECONDS:.0f}s and {resume_window:.1f}s windows of the "
              f"same audio, alternately ({model}, {duration:.0f}s of it):")
        costs = window_cost(audio, model, resume_window, WINDOW_SECONDS)
        for name, stat in costs.items():
            print(f"  {name:<6} {_BENCH._fmt(stat)}")
        if costs["full"].get("mean") and costs["short"].get("mean"):
            print(f"  a resume decode costs {costs['short']['mean'] / costs['full']['mean']:.2f}x "
                  f"a full one ({costs['full']['mean'] - costs['short']['mean']:.2f}s less "
                  f"per pause, for the one decode the wake adds)")
        return 0

    # The gate has to be the engine's: it is both the level a window is decoded
    # at and the level the wake listens for. A file has no room tone to calibrate
    # against, so this is the engine's default gate.
    gate = LiveEngine(model_size=model).silence_rms
    levels = speech_levels(audio)
    events = speech_runs(levels, gate, PAUSE_SECONDS, WINDOW_SECONDS)
    speech_share = float((levels >= gate).mean()) if len(levels) else 0.0

    print(f"file        : {path.name}")
    print(f"duration    : {duration:.1f}s")
    print(f"live model  : {model}")
    print(f"silence gate: {gate:.4f} (engine default) - {speech_share * 100:.1f}% of the "
          f"recording is above it")
    if not events:
        print("resumes     : none found - no pause reached "
              f"{PAUSE_SECONDS:.1f}s after the window filled")
    else:
        quiet = _BENCH._stats([e["quiet"] for e in events])
        print(f"resumes     : {len(events)} pauses of >= {PAUSE_SECONDS:.1f}s after the "
              f"window filled ({_BENCH._fmt(quiet)} of quiet)")
    print("note        : 'to caption' includes the decode itself, which is the "
          "machine's speed; 'loop waits' is the part this measures")

    # wake = the loop stops napping over a returning speaker; resume_window = the
    # first decode after a pause reads only a few seconds; resume_stride = the
    # loop keeps coming back that often for the first moments of the resumed
    # utterance. `before` is the loop every measurement in the README's earlier
    # sections ran on.
    configurations = {
        "before": (False, None, None),
        "wake": (True, None, None),
        "short": (True, resume_window, None),
        "burst": (True, None, RESUME_STRIDE_SECONDS),
    }
    wanted = [name.strip() for name in args.passes.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in configurations]
    if unknown:
        print(f"unknown pass: {', '.join(unknown)} (known: {', '.join(configurations)})")
        return 2
    print(f"passes      : baseline {args.baseline}, resume window "
          f"{resume_window:.1f}s for 'short'")
    print(f"repeats     : {args.repeats} per configuration, pass order rotating - "
          f"{args.repeats * len(wanted)} real-time passes, about "
          f"{args.repeats * len(wanted) * duration / 60:.0f} min")

    runs: dict[str, list[dict]] = {name: [] for name in wanted}
    for repeat in range(args.repeats):
        order = rotation(wanted, repeat)
        print(f"\nrepeat {repeat + 1}/{args.repeats}: " + " -> ".join(order))
        for name in order:
            wake, window, stride = configurations[name]
            print(f"replaying in real time ({name}: "
                  f"{'wake' if wake else 'no wake'}, "
                  f"{'full' if window is None else f'{window:.1f}s'} resume window"
                  f"{'' if stride is None else f', {stride:.1f}s cadence after it'})...")
            warm = warm_model(model, audio)
            stats = collect(
                run_pass(audio, model, wake=wake, resume_window=window,
                         resume_stride=stride),
                events, args.tolerance)
            runs[name].append(stats)
            if stats["errors"]:
                print(f"  [!] decode errors: {stats['errors'][:3]}")
            print(f"  {stats['decodes']} decodes, {stats['gpu_seconds']:.1f}s of GPU, "
                  f"decode {_BENCH._fmt(stats['decode_cost'])} (warm-up {warm:.1f}s)")
            print(f"  to caption    : {_BENCH._fmt(stats['to_caption'])}")
            print(f"  loop waits    : {_BENCH._fmt(stats['to_caption_start'])}")
            print(f"  to transcript : {_BENCH._fmt(stats['to_transcript'])}")
            print(f"  text          : {stats['words']} words written, "
                  f"{stats['late_drops']} refused as too late to be speech")
            if stats["missed"]:
                print(f"  [!] {stats['missed']} resumes were never captioned in this pass")

    summary = report(runs, events, args.baseline, args.repeats)

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing[path.name] = {
        "file": str(path),
        "duration_seconds": duration,
        "model": model,
        "silence_gate": gate,
        "speech_share": speech_share,
        "resumes": events,
        "repeats": args.repeats,
        # the order each repeat ran in, so a result can be read against what
        # the passes inherited from each other
        "order": [rotation(wanted, repeat) for repeat in range(args.repeats)],
        "passes": {name: {k: v for k, v in stats.items() if k not in ("rows", "values")}
                   for name, stats in summary.items()},
        # `timeline` is kept per repeat: when each decode ran and what it cost is
        # how a change to the *cadence* is told apart from a change to the wait.
        "per_repeat": {name: [{k: v for k, v in run.items()
                               if k not in ("values", "rows")}
                              for run in runs[name]]
                       for name in summary},
        "texts": {name: [run["text"] for run in runs[name]] for name in summary},
        "rows": {name: [run["rows"] for run in runs[name]] for name in summary},
    }
    out.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    print(f"\nresults saved: {out}")
    return 0


def report(runs: dict[str, list[dict]], events: list[dict], base: str,
           repeats: int) -> dict:
    """Print the comparison, and return the pooled numbers per configuration.

    Split out of `main` so the arithmetic can be run against recorded passes
    without replaying a recording: this is where the audit's own numbers are
    made, and a bug in it would read as a result.
    """
    summary = {name: pool(config) for name, config in runs.items()}
    if len(summary) > 1 and base in summary:
        perkeys = {name: {key: per_pause(runs[name], key)
                          for key in ("to_caption", "to_caption_start", "to_transcript")}
                   for name in summary}

        print(f"\nper configuration (pooled over {plural(repeats, 'repeat')}):")
        for key, label in (("to_caption", "to the screen"),
                           ("to_caption_start", "loop's own wait"),
                           ("to_transcript", "to the transcript")):
            print(f"  {label}:")
            for name in summary:
                stat = summary[name][key]
                shown = ("n/a" if not stat.get("n") else
                         f"n={stat['n']} {stat['mean']:.2f}s mean / "
                         f"{stat['median']:.2f}s median / {stat['p95']:.2f}s p95")
                print(f"    {name:<7} {shown}")
                print(f"      {repeat_range(runs[name], key)}")

        print("\nper pause (repeats averaged into one number per pause: caption, "
              "the loop's own wait, and the word that printed it):")
        for index in range(len(events)):
            cells = []
            for name in summary:
                caption = perkeys[name]["to_caption"][index]
                wait = perkeys[name]["to_caption_start"][index]
                shown_caption = " n/a " if caption is None else f"{caption:5.2f}"
                shown_wait = " n/a " if wait is None else f"{wait:4.2f}"
                word = pause_words(runs[name], index)[:11]
                cells.append(f"{name} {shown_caption} ({shown_wait}) {word!r}")
            quiet = events[index]["quiet"]
            print(f"  at {events[index]['resume']:7.2f}s quiet {quiet:4.1f}s  "
                  + "  ".join(cells))

        print("\npaired per pause (each pause's mean over the repeats, then the "
              "difference; positive means that pause got faster):")
        for key, label in (("to_caption", "to the screen"),
                           ("to_caption_start", "loop's own wait"),
                           ("to_transcript", "to the transcript")):
            print(f"  {label}:")
            for name in summary:
                if name == base:
                    continue
                stats = paired(perkeys[base][key], perkeys[name][key])
                if not stats["removed"]:
                    continue
                print(f"    {name} vs {base}: removed {stats['mean']:+.2f}s mean / "
                      f"{stats['median']:+.2f}s median / sd {stats['sd']:.2f}s / "
                      f"+/-{stats['interval']:.2f}s 95% CI  over "
                      f"{plural(len(stats['removed']), 'pause')}")
                moved = len(stats["removed"])
                # A recording has tens of pauses, not thousands, so the test's
                # own floor matters: with five pauses even unanimity cannot
                # reach p<0.05, and reading that as "no effect" would be wrong.
                note = (f" (with {plural(moved, 'pause')} the test's floor is "
                        f"p={2 / 2 ** moved:.3f})" if 2 / 2 ** moved > 0.05 else "")
                print(f"      {stats['faster']} faster / {stats['slower']} slower / "
                      f"{stats['same']} unchanged - sign test p={stats['p']:.3f} "
                      f"(direction, not size){note}")

        # The reason for the repeats: an effect has to clear the drift of the
        # thing it is measured against, and only repeats can measure that.
        floor = max(repeat_spread(runs[name], "to_caption_start") for name in summary)
        print(f"\nnoise floor : {floor:.2f}s - the widest repeat-to-repeat spread of "
              f"one configuration's own mean loop wait")
        for name in summary:
            if name == base:
                continue
            stats = paired(perkeys[base]["to_caption_start"],
                           perkeys[name]["to_caption_start"])
            if not stats["removed"]:
                continue
            moved = len(stats["removed"])
            agreeing = max(stats["faster"], stats["slower"])
            if abs(stats["mean"]) <= floor:
                verdict = ("smaller than the drift between runs, so not settled by these "
                           "repeats")
            elif agreeing == moved:
                verdict = "larger than it, and every pause moved the same way"
            elif agreeing * 4 >= moved * 3:  # three of four, four of five, ...
                verdict = (f"larger than it, and {agreeing} of {moved} pauses moved the "
                           f"same way")
            else:
                verdict = (f"larger than it, but only {agreeing} of {moved} pauses moved "
                           f"the same way - the mean is carried by the rest")
            print(f"  {name:<7} {abs(stats['mean']):.2f}s of loop wait moved against that "
                  f"floor - {verdict}")

        print("\ncost and text:")
        for name in summary:
            stats = summary[name]
            print(f"  {name:<7} {stats['decodes']} decodes / {stats['gpu_seconds']:.1f}s GPU "
                  f"over {plural(repeats, 'repeat')} / decode "
                  f"{stats['decode_cost'].get('mean', 0):.2f}s mean / "
                  f"{stats['words']} words / {stats['late_drops']} refused as late")
        for name in summary:
            if name == base:
                continue
            scores = []
            for index in range(repeats):
                reference = _BENCH._WER.normalize(runs[base][index]["text"])
                tokens = _BENCH._WER.normalize(runs[name][index]["text"])
                score, _sub, _dele, _ins = _BENCH._WER.wer(reference, tokens)
                scores.append(score * 100)
            print(f"  {name} vs {base}: " + " / ".join(f"{score:.1f}%" for score in scores)
                  + " of words differ, one number per repeat - what it said, not just when")
    return summary


if __name__ == "__main__":
    sys.exit(main())
