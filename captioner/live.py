"""Live captioning core (Phase 3): capture, ring buffer, sliding decode.

Design for the 6GB laptop:
  - capture runs on its own thread via SoundCard (WASAPI): microphones and
    output-device loopback (what the system is playing) both feed the same
    16 kHz mono ring buffer;
  - a decode timer re-transcribes the last WINDOW_SECONDS every STRIDE_SECONDS
    with the small live model, so spoken words appear within ~1-2 s;
  - text stabilizes as the window slides past it: words whose start time falls
    before the "committed" frontier are frozen (no more flicker), and only the
    tail of the window is re-written every stride.

Everything hardware-touching is isolated here; the overlay UI only sees
strings. `LiveEngine.stream_file` replays a file through the exact same
pipeline, so the logic is testable without an audio device.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, srt, transcribe
from .segmenter import Block, Word, segment_words

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 12.0   # audio handed to Whisper each decode
STRIDE_SECONDS = 1.5    # how often a decode runs while speech is active
IDLE_STRIDE_SECONDS = 3.0  # silence: decode less often, the GPU gets a rest
SILENCE_RMS = 0.004     # below this RMS the stream counts as quiet
COMMIT_MARGIN = 3.0     # words starting this close to the window edge freeze
DUPLICATE_TOLERANCE = 0.7  # same word re-seen within this window = duplicate

AudioCallback = Callable[[np.ndarray], None]  # float32 mono frames


# devices

def list_input_devices() -> list[dict]:
    """Microphones plus every output device's loopback (system audio)."""
    import soundcard as sc

    devices: list[dict] = []
    for mic in sc.all_microphones(include_loopback=False):
        devices.append({"id": mic.id, "name": mic.name, "kind": "mic", "is_loopback": False})
    for spk in sc.all_speakers():
        try:
            loop = sc.get_microphone(id=str(spk.name), include_loopback=True)
        except Exception:
            loop = None
        if loop is not None:
            devices.append(
                {"id": loop.id, "name": f"{spk.name} (system audio)", "kind": "loopback", "is_loopback": True}
            )
    return devices


# capture

class MicCapture(threading.Thread):
    """Record a device into a fixed ring buffer on a background thread.

    SoundCard's recorder blocks per chunk, so the thread only ever copies
    frames into the ring — no allocation in the audio path, no GIL pressure.
    """

    def __init__(self, device_id: str, ring_seconds: float = 60.0,
                 blocksize_ms: float = 100.0, on_error: Callable[[str], None] | None = None):
        super().__init__(daemon=True, name="live-capture")
        self.device_id = device_id
        self.on_error = on_error
        self.ring = np.zeros(int(ring_seconds * SAMPLE_RATE), dtype=np.float32)
        self._write_pos = 0
        self._total_frames = 0
        self._level = 0.0  # RMS of the newest block; feeds the overlay meter
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._blocksize = max(1, int(blocksize_ms * SAMPLE_RATE / 1000))

    # ring accounting
    @property
    def total_frames(self) -> int:
        with self._lock:
            return self._total_frames

    @property
    def level(self) -> float:
        """RMS (0..~1) of the newest audio block — a live input meter."""
        with self._lock:
            return self._level

    def read_tail(self, seconds: float) -> np.ndarray:
        """The newest *seconds* of audio, chronological."""
        want = min(int(seconds * SAMPLE_RATE), len(self.ring))
        with self._lock:
            total = self._total_frames
            take = min(want, total)
            start = (self._write_pos - take) % len(self.ring)
            if start + take <= len(self.ring):
                out = self.ring[start:start + take].copy()
            else:
                out = np.concatenate([self.ring[start:], self.ring[:take - (len(self.ring) - start)]])
        return out

    # thread
    def run(self) -> None:
        import soundcard as sc

        try:
            mic = sc.get_microphone(id=self.device_id, include_loopback=True)
        except Exception as exc:
            logger.error("capture device %r not found: %s", self.device_id, exc)
            if self.on_error:
                self.on_error(f"device not found: {exc}")
            return
        try:
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=self._blocksize) as rec:
                while not self._stop.is_set():
                    frames = rec.record(numframes=self._blocksize)  # (n, 1) float32
                    mono = frames.reshape(-1)[: len(self.ring)]
                    with self._lock:
                        end = self._write_pos + len(mono)
                        if end <= len(self.ring):
                            self.ring[self._write_pos:end] = mono
                        else:
                            cut = len(self.ring) - self._write_pos
                            self.ring[self._write_pos:] = mono[:cut]
                            self.ring[:end - len(self.ring)] = mono[cut:]
                        self._write_pos = end % len(self.ring)
                        self._total_frames += len(mono)
                        if len(mono):
                            self._level = float(np.sqrt(np.mean(np.square(mono))))
        except Exception as exc:
            logger.error("capture failed: %s", exc)
            if self.on_error:
                self.on_error(f"capture failed: {exc}")

    def stop(self) -> None:
        self._stop.set()

    def write_frames(self, mono: np.ndarray) -> None:
        """Push pre-recorded frames into the ring (used by the file stream)."""
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)
        with self._lock:
            step = len(mono)
            end = self._write_pos + step
            if end <= len(self.ring):
                self.ring[self._write_pos:end] = mono
            else:
                cut = len(self.ring) - self._write_pos
                self.ring[self._write_pos:] = mono[:cut]
                self.ring[:end - len(self.ring)] = mono[cut:]
            self._write_pos = end % len(self.ring)
            self._total_frames += step
            if step:
                self._level = float(np.sqrt(np.mean(np.square(mono))))


# stabilized live text

@dataclass
class LiveState:
    """What the overlay renders: frozen text plus the changing tail."""
    committed: str = ""
    partial: str = ""
    full_text: str = ""  # every frozen word since the session started (uncapped)
    full_words: list[tuple] = field(default_factory=list)  # (start, end, text) seconds
    level: float = 0.0   # input RMS 0..~1, for the overlay's level meter
    segments: list[dict] = field(default_factory=list)
    # Translation of the last committed sentence, plus the session's
    # (source, translation) pairs in order. Empty when translation is off.
    translation: str = ""
    translated_pairs: list[tuple] = field(default_factory=list)

    @property
    def text(self) -> str:
        if self.committed and self.partial:
            return f"{self.committed} {self.partial}".strip()
        return (self.committed or self.partial).strip()


class TranslationRunner:
    """Translates finished sentences on a background thread.

    Live captions cannot wait for a translation model: a 418M-parameter batch
    takes longer than the next decode window. So finished sentences are handed
    over through a queue and the audio loop never blocks on them, and only
    *sentences* are translated - translating the unstable tail would both waste
    work and churn the display.

    A backend that cannot be built (weights missing, API key absent) is
    reported once through ``on_error`` and then disabled, so a session keeps
    captioning in the source language instead of failing.
    """

    def __init__(self, target: str, translator=None, on_error=None,
                 poll_seconds: float = 0.2):
        self.target = target
        self._translator = translator
        self._on_error = on_error
        self._poll = float(poll_seconds)
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pairs: list[tuple] = []
        self._latest = ""
        self._disabled = False
        self._thread = threading.Thread(
            target=self._run, name="live-translation", daemon=True
        )
        self._thread.start()

    def submit(self, sentence: str) -> None:
        """Queue one finished sentence (no-op once disabled)."""
        text = sentence.strip()
        if not text or self._disabled:
            return
        self._queue.put(text)

    def snapshot(self) -> tuple:
        """``(latest translation, [(source, translation), ...])``."""
        with self._lock:
            return self._latest, list(self._pairs)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _backend(self):
        if self._translator is None:
            from .engines import mt

            self._translator = mt()
        return self._translator

    def _disable(self, message: str) -> None:
        self._disabled = True
        logger.warning("live translation disabled: %s", message)
        if self._on_error is not None:
            self._on_error(message)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sentence = self._queue.get(timeout=self._poll)
            except queue.Empty:
                continue
            if sentence is None:
                break
            try:
                translator = self._backend()
                available, why = getattr(translator, "available", lambda: (True, ""))()
                if not available:
                    self._disable(f"translation unavailable: {why}")
                    continue
                translated = translator.translate_many([sentence], self.target)[0]
            except Exception as exc:  # noqa: BLE001 - a session must survive this
                self._disable(f"translation failed: {type(exc).__name__}: {exc}")
                continue
            with self._lock:
                self._latest = str(translated).strip()
                self._pairs.append((sentence, self._latest))


class LiveEngine:
    """Sliding-window decode over a ring buffer.

    Each `decode_once` re-transcribes the last WINDOW_SECONDS. Words whose
    start time has slid COMMIT_MARGIN seconds behind the window edge are
    frozen into `committed` (exactly once, deduplicated across windows);
    everything newer is the `partial` tail that may still be rewritten.
    """

    COMMITTED_KEEP_CHARS = 400  # cap the frozen string; live captions are ephemeral
    DUPLICATE_TOLERANCE = 0.7   # same word re-seen within this window = duplicate
    SILENCE_HOLD_SECONDS = 2.0  # keep decoding this long after speech stops
    # Whisper's confidence floor: segments below this avg logprob are the
    # classic silence/noise hallucination, not speech the user said.
    MIN_SEGMENT_LOGPROB = -1.0
    # Whisper's own "this isn't speech" score; combined with a mediocre
    # logprob it reliably marks music/background-noise hallucinations.
    MAX_NO_SPEECH_PROB = 0.7
    # A run of at least this many words exactly repeating the frozen tail is
    # a re-proposal (timestamps drifted past the per-word tolerance), not
    # someone genuinely saying the same 3+ words again.
    REPEAT_RUN_WORDS = 3
    #per-session calibration (calibrate)
    GATE_FLOOR_MULTIPLIER = 2.5  # gate sits this far above the noise floor
    GATE_MAX_RMS = 0.03          # never gate above this: real speech must pass
    NOISY_FLOOR_RMS = 0.01       # above this, tighten the confidence filter too

    def __init__(self, model_size: str | None = None, beam_size: int = 1,
                 window_seconds: float = WINDOW_SECONDS,
                 translate_to: str | None = None, translator=None,
                 on_translation_error: Callable[[str], None] | None = None):
        self.model_size = model_size or config.LIVE_MODEL_SIZE
        self.beam_size = int(beam_size)
        self.window_seconds = float(window_seconds)
        self.translate_to = (translate_to or config.LIVE_TRANSLATE_TO or "").strip()
        self._translation: TranslationRunner | None = None
        if self.translate_to:
            self._translation = TranslationRunner(
                self.translate_to, translator=translator,
                on_error=on_translation_error,
            )
        self.state = LiveState()
        self.last_window_rms: float = 1.0   # decays only via measurements
        self.suggested_stride: float = STRIDE_SECONDS
        self._full: list[str] = []         # every frozen word, uncapped (text only)
        self._full_words: list[tuple] = []  # frozen (start, end, text), seconds
        self._frontier: float = 0.0        # absolute ring-seconds: words starting before this freeze
        self._frozen_recent: list[tuple] = []  # (abs_start, norm_text) for dup matching
        self._speech_recently = True   # decode until proven silent
        self._silence_since: float | None = None
        self._sentence_buffer: list[tuple] = []  # frozen words awaiting a sentence end
        # Per-session calibration (see calibrate); start from safe defaults.
        self.silence_rms: float = SILENCE_RMS
        self.min_logprob: float = self.MIN_SEGMENT_LOGPROB
        self.calibrated = False
        self._device = transcribe.resolve_device()
        self._compute = transcribe.resolve_compute_type(self._device)

    def _append_committed(self, texts: list[str]) -> None:
        """Append words to the frozen text, keeping it bounded."""
        committed = (self.state.committed + " " + " ".join(texts)).strip()
        self.state.committed = committed[-self.COMMITTED_KEEP_CHARS:]

    def _sync_translation(self) -> None:
        """Copy the translation worker's progress into the state the UI reads."""
        if self._translation is None:
            return
        latest, pairs = self._translation.snapshot()
        self.state.translation = latest
        self.state.translated_pairs = pairs

    def close(self) -> None:
        """Stop background work (the translation thread) at session end."""
        if self._translation is not None:
            self._translation.stop()

    @staticmethod
    def _ends_sentence(text: str) -> bool:
        return bool(text) and text[-1] in ".!?"

    def _commit_sentence(self) -> None:
        """Promote buffered words into the frozen (green) text as one unit.

        The sentence buffer is the only path into `committed`, so text moves
        to the green line sentence-by-sentence and never duplicates. A final
        guard skips a sentence identical to the one just committed — the
        last resort against a hallucination loop that survived the filters.
        """
        if not self._sentence_buffer:
            return
        texts = [t for _s, _e, t in self._sentence_buffer]
        if len(texts) >= self.REPEAT_RUN_WORDS:
            last = self.state.committed.split()[-len(texts):]
            if last == texts:
                self._sentence_buffer.clear()
                return
        self._append_committed(texts)
        self._sentence_buffer.clear()
        # A finished sentence is the unit live translation works on.
        if self._translation is not None:
            self._translation.submit(" ".join(texts))

    @staticmethod
    def _norm(text: str) -> str:
        """Case/punctuation-insensitive form for duplicate matching."""
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _is_dup(self, word: tuple) -> bool:
        """True if *word* matches something already frozen.

        Word timestamps jitter between overlapping decode windows (VAD
        boundaries move as the window slides), so exact (start, end, text)
        keys re-freeze the same spoken word every stride. A word is a
        duplicate when its normalized text matches a frozen word whose
        absolute start is within DUPLICATE_TOLERANCE seconds.
        """
        start, _end, text = word
        key = self._norm(text)
        if not key:
            return True
        for fstart, ftext in self._frozen_recent:
            if abs(fstart - start) <= self.DUPLICATE_TOLERANCE and ftext == key:
                return True
        return False

    def _strip_repeat_run(self, fresh: list[tuple]) -> list[tuple]:
        """Drop runs that exactly repeat the frozen tail, at any position.

        On real TV/mic audio, word timestamps drift well past the per-word
        tolerance, and whole phrases come back as "new": the transcript
        fills with "We'll dive in. We'll dive in." Any contiguous run of
        >= REPEAT_RUN_WORDS fresh words that matches the most-recent frozen
        words verbatim is a re-proposal, not someone genuinely saying the
        same phrase again — it can sit anywhere in *fresh* (a repeat is
        often followed by genuinely new words).
        """
        if not fresh or len(self._frozen_recent) < self.REPEAT_RUN_WORDS:
            return fresh
        frozen_texts = [t for _s, t in self._frozen_recent]
        fresh_texts = [self._norm(t) for _s, _e, t in fresh]
        keep: list[tuple] = []
        i = 0
        while i < len(fresh):
            matched = False
            max_run = min(len(frozen_texts), len(fresh) - i, 12)
            for n in range(max_run, self.REPEAT_RUN_WORDS - 1, -1):
                if frozen_texts[-n:] == fresh_texts[i:i + n]:
                    i += n  # skip the whole re-proposed run
                    matched = True
                    break
            if not matched:
                keep.append(fresh[i])
                i += 1
        return keep

    def _freeze(self, words: list[tuple], frontier: float) -> list[tuple]:
        """Return the not-yet-frozen words starting before *frontier*, and mark them frozen."""
        fresh = [w for w in words if w[0] < frontier and not self._is_dup(w)]
        fresh = self._strip_repeat_run(fresh)
        for w in fresh:
            self._frozen_recent.append((w[0], self._norm(w[2])))
        # keep the matching list bounded: the window only ever re-proposes
        # words from the last WINDOW_SECONDS, so a slice of that is plenty
        if len(self._frozen_recent) > 400:
            del self._frozen_recent[:200]
        return fresh

    def decode_once(self, capture: MicCapture) -> LiveState:
        """Transcribe the current window and update the stabilized state."""
        audio = capture.read_tail(self.window_seconds)
        if len(audio) < SAMPLE_RATE * 0.6:  # nothing meaningful yet
            return self.state
        window_start_t = (capture.total_frames - len(audio)) / SAMPLE_RATE

        # Quiet stream? Decode less often: live captions are about *speech*,
        # and burning GPU on silence shortens laptop battery for nothing.
        # The gate itself is per-session calibrated (see calibrate).
        rms = float(np.sqrt(np.mean(np.square(audio))))
        self.last_window_rms = rms
        self.suggested_stride = (
            STRIDE_SECONDS if rms >= self.silence_rms else IDLE_STRIDE_SECONDS
        )

        # Silence gate: decode only when speech energy is present anywhere
        # in the window. Whisper hallucinates fluent filler on near-silence
        # ("Thank you for watching."), and the commit step would freeze it
        # - never decoding a silent window removes the whole failure class.
        if rms < self.silence_rms and not self._speech_recently:
            return self.state
        if rms >= self.silence_rms:
            self._speech_recently = True
            self._silence_since = None
        else:
            if self._silence_since is None:
                self._silence_since = time.time()
            elif time.time() - self._silence_since >= self.SILENCE_HOLD_SECONDS:
                self._speech_recently = False  # long pause: gate shut again

        model = transcribe.get_model(self.model_size, self._device, self._compute)
        segments, _info = model.transcribe(
            audio,
            language="en",
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
            condition_on_previous_text=False,  # each window stands alone
            beam_size=self.beam_size,  # beam 1 measured as accurate as beam 5
        )
        words: list[tuple] = []
        for seg in segments:
            # Whisper hallucination filters: on silence/music/noise the model
            # emits fluent filler. A segment this unsure - or one Whisper
            # itself flags as likely-not-speech (music on TV, room noise on
            # a mic) - is almost never speech the user actually said.
            if seg.avg_logprob is not None and seg.avg_logprob < self.min_logprob:
                logger.debug("dropped low-confidence segment (%.2f): %r",
                             seg.avg_logprob, seg.text)
                continue
            if (getattr(seg, "no_speech_prob", None) is not None
                    and seg.no_speech_prob > self.MAX_NO_SPEECH_PROB
                    and (seg.avg_logprob or 0.0) < -0.5):
                logger.debug("dropped likely-not-speech segment (%.2f): %r",
                             seg.no_speech_prob, seg.text)
                continue
            for w in seg.words or []:
                if w.word.strip():
                    words.append(
                        (window_start_t + w.start, window_start_t + w.end, w.word.strip())
                    )

        frontier = window_start_t + len(audio) / SAMPLE_RATE - COMMIT_MARGIN
        if frontier > self._frontier:
            self._frontier = frontier

        frozen = self._freeze(words, self._frontier)
        if frozen:
            self._full.extend(t for _s, _e, t in frozen)
            self._full_words.extend(frozen)
        self.state.full_text = " ".join(self._full)
        self.state.full_words = list(self._full_words)
        # Display: whole sentences at once. Frozen-but-unfinished words sit
        # in _sentence_buffer and render as part of the partial tail; only
        # a completed sentence (or a long pause) commits to the green text.
        self._sentence_buffer.extend(frozen)
        last_text = frozen[-1][2] if frozen else ""
        if (self._ends_sentence(last_text)
                or (self.state.level < self.silence_rms and self._speech_recently is False)):
            self._commit_sentence()
        self.state.partial = " ".join(
            t for s, _e, t in words if s >= self._frontier
        )
        if self._sentence_buffer:
            buffered = " ".join(t for _s, _e, t in self._sentence_buffer)
            self.state.partial = f"{buffered} {self.state.partial}".strip()
        # last-block RMS (not the 12 s window): the meter should move instantly
        self.state.level = float(getattr(capture, "level", 0.0) or 0.0)
        self.state.segments = [
            {"start": s, "end": e, "text": t} for s, e, t in words[-60:]
        ]
        self._sync_translation()
        return self.state

    def calibrate(self, capture: MicCapture, seconds: float = 2.0) -> dict:
        """Tune the silence gate to this session's actual noise floor.

        Mics differ hugely: a hot mic's room tone sits above the generic
        SILENCE_RMS, which defeats the anti-hallucination gate (decode never
        stops), while a quiet mic makes it far too sensitive a threshold.

        Samples *seconds* of whatever the device hears right now and takes
        the 25th percentile of block levels as the noise floor — speech
        spikes land above the 25th percentile, so they don't poison it. The
        gate becomes floor × GATE_FLOOR_MULTIPLIER, clamped so it never
        exceeds GATE_MAX_RMS (a shouted room could otherwise swallow real
        speech). On a noisy mic, the confidence filter also tightens: a
        hissy signal makes hallucinations more likely at any given energy.
        """
        levels: list[float] = []
        block = max(1, int(0.05 * SAMPLE_RATE))
        n_blocks = max(1, int(seconds * SAMPLE_RATE / block))
        for _ in range(n_blocks):
            frames = capture.read_tail(0.05)
            if len(frames):
                levels.append(float(np.sqrt(np.mean(np.square(frames)))))
        if not levels:
            return {"floor": None, "silence_rms": self.silence_rms,
                    "min_logprob": self.min_logprob, "calibrated": False}
        floor = float(np.percentile(sorted(levels), 25))
        self.silence_rms = min(
            max(floor * self.GATE_FLOOR_MULTIPLIER, SILENCE_RMS),
            self.GATE_MAX_RMS,
        )
        if floor > self.NOISY_FLOOR_RMS:
            self.min_logprob = self.MIN_SEGMENT_LOGPROB - 0.2  # stricter
        self.calibrated = True
        logger.info("calibrated: noise floor %.4f -> silence gate %.4f, "
                    "min_logprob %.2f", floor, self.silence_rms, self.min_logprob)
        return {"floor": floor, "silence_rms": self.silence_rms,
                "min_logprob": self.min_logprob, "calibrated": True}

    def flush_once(self, capture: MicCapture) -> LiveState:
        """Final decode that freezes the tail words still marked partial."""
        audio = capture.read_tail(self.window_seconds)
        if len(audio) < SAMPLE_RATE * 0.6:
            return self.state
        window_start_t = (capture.total_frames - len(audio)) / SAMPLE_RATE
        model = transcribe.get_model(self.model_size, self._device, self._compute)
        segments, _info = model.transcribe(
            audio, language="en", word_timestamps=True,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 350},
            condition_on_previous_text=False,
        )
        words = []
        for seg in segments:
            if seg.avg_logprob is not None and seg.avg_logprob < self.min_logprob:
                continue
            if (getattr(seg, "no_speech_prob", None) is not None
                    and seg.no_speech_prob > self.MAX_NO_SPEECH_PROB
                    and (seg.avg_logprob or 0.0) < -0.5):
                continue
            for w in seg.words or []:
                if w.word.strip():
                    words.append((window_start_t + w.start, window_start_t + w.end, w.word.strip()))
        # freeze everything seen in this window (dedup keeps repeats out);
        # end of session - commit the sentence buffer too, nothing stays pending
        fresh = [w for w in words if not self._is_dup(w)]
        for w in fresh:
            self._frozen_recent.append((w[0], self._norm(w[2])))
        if fresh:
            self._full.extend(t for _s, _e, t in fresh)
            self._full_words.extend(fresh)
        self._sentence_buffer.extend(fresh)
        self._commit_sentence()
        self.state.full_text = " ".join(self._full)
        self.state.full_words = list(self._full_words)
        self.state.partial = ""
        self.state.level = float(getattr(capture, "level", 0.0) or 0.0)
        self._sync_translation()
        return self.state

    # -- file streaming (test/demo path)
    def stream_file(self, path: str, on_update: Callable[[LiveState], None] | None = None,
                    speed: float = 0.0) -> LiveState:
        """Feed a real file through the live pipeline as if it were a device.

        Reads with ffmpeg's own demuxer (via faster-whisper's decode helper),
        walks WINDOW_SECONDS at a time, and decodes every step — the same
        code path live audio takes, without a sound card.
        """
        from faster_whisper.audio import decode_audio

        # sampling_rate forces a 16 kHz resample; output is mono by default
        audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        capture = MicCapture(device_id="unused")
        capture.write_frames(np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32))  # priming silence

        step = int(STRIDE_SECONDS * SAMPLE_RATE)
        window = int(WINDOW_SECONDS * SAMPLE_RATE)
        pos = 0
        while pos < len(audio):
            chunk = audio[pos:pos + step]
            capture.write_frames(chunk)
            pos += step
            if capture.total_frames >= window or pos >= len(audio):
                self.decode_once(capture)
                if on_update:
                    on_update(self.state)
                if speed > 0:
                    time.sleep(step / SAMPLE_RATE / speed)
        # The file ended: commit the tail sentence a live session would keep
        # decoding as the user keeps talking.
        self.flush_once(capture)
        if on_update:
            on_update(self.state)
        return self.state


def decode_ring_once(ring: np.ndarray, model_size: str | None = None) -> list[dict]:
    """One-shot helper for tests: transcribe a plain float32 array."""
    engine = LiveEngine(model_size=model_size or "tiny.en")
    fake = MicCapture(device_id="unused")
    fake.write_frames(ring)
    engine.decode_once(fake)
    return engine.state.segments


# transcript export - the same segmentation + writers the batch jobs use

def _session_blocks(full_words: list[tuple]) -> list[Block]:
    """Group a live session's frozen words into caption blocks."""
    words = [
        Word(text=text, start=start, end=end)
        for start, end, text in full_words
        if end is not None and start is not None and end >= start and text
    ]
    return segment_words(words)


def transcript_to_srt(full_words: list[tuple]) -> str:
    """A live session as a standard .srt (grouped like batch captions)."""
    return srt.to_srt(_session_blocks(full_words))


def transcript_to_vtt(full_words: list[tuple]) -> str:
    """A live session as a standard .vtt."""
    return srt.to_vtt(_session_blocks(full_words))


def save_transcript(full_words: list[tuple], path: str) -> str:
    """Write a live transcript; the extension picks the format.

    ``.srt`` and ``.vtt`` get real grouped, timed captions identical in style
    to batch output; anything else is written as plain text. Returns the path.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".srt":
        text = transcript_to_srt(full_words)
    elif suffix == ".vtt":
        text = transcript_to_vtt(full_words)
    else:
        text = " ".join(t for _s, _e, t in full_words) + "\n"
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path
