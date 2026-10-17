"""Live captioning core (Phase 3): capture, ring buffer, sliding decode.

Design for the 6GB laptop:
  - capture runs on its own thread via SoundCard (WASAPI): microphones and
    output-device loopback (what the system is playing) both feed the same
    16 kHz mono ring buffer;
  - a decode timer re-transcribes the last WINDOW_SECONDS every STRIDE_SECONDS
    with the small live model, backing off to IDLE_STRIDE_SECONDS while nobody
    is talking - but that wait is cut short the moment the microphone hears the
    speaker return (the capture's 50 ms meter against the engine's gate), so a
    pause costs the resumed words no more than the decode that follows it;
  - text stabilizes as the window slides past it: words whose start time falls
    before the "committed" frontier are frozen (no more flicker), and only the
    tail of the window is re-written every stride. When the speaker stops, the
    window is frozen at once instead of holding the tail until they talk again.

Two latencies come out of this, and they are not the same number. A word is
on screen about a second after it is spoken (the newest word the model reports
sits that far behind the window edge), and it settles into the saved
transcript ~3 s later, because COMMIT_MARGIN is the room the sliding window
gets to revise a word before it is frozen. Both are measured on real audio in
the README's "On real audio" section.

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
SPEECH_BLOCK_SECONDS = 0.05  # resolution of the speech/silence envelope
PAUSE_SECONDS = 1.0     # no speech in the newest second = the utterance ended
FREEZE_BAND_SECONDS = 3.0  # a word first seen later than this is a re-write
# The first decode after a pause *can* read this much audio instead of the whole
# WINDOW_SECONDS: twelve seconds of mostly silence that already contained a
# finished utterance, against the few seconds the words just starting need. It is
# off by default because it did not help - see `resume_window_seconds` and the
# README's "On real audio".
RESUME_WINDOW_SECONDS = 4.0
# ...and the cadence, not just the window. Waking the loop on sound ends the idle
# nap, but the model will not report a word with ~0.1 s of audio behind it, so the
# words a returning speaker says are still caught by whatever decode comes next -
# a whole stride later. For the first moments of the resumed utterance the loop
# comes back sooner instead (`resume_stride_seconds`), and the short window above
# is what makes those decodes cheap enough to pay for: a resume decode costs 42%
# of a full one. 0.5 s is also the fastest the loop can run at all: `ui.overlay.
# stride_wait` never waits less than that, so the burst is "decode as fast as this
# machine allows while the resumed words are still arriving". RESUME_BURST_SECONDS
# is how much audio it covers - long enough to reach the ~1.0-1.2 s after which a
# word can be reported, short enough to hand back to the normal stride.
#
# This one ships *on* (`resume_stride_seconds`), unlike the window above, because
# it was measured to pay: three passes per configuration on recording 1, pass
# order rotating (18 pauses, small.en), put the first post-pause caption 0.37 s
# earlier than the wake alone and the loop's own wait 0.40 s earlier (medians
# 0.30 / 0.37 s; 12 of 18 pauses faster; every one of the three repeats beat
# every repeat of the loop without it) against a 0.20 s repeat-to-repeat floor,
# for ~2 extra decodes per pause - which is what the 42% window above is for.
RESUME_STRIDE_SECONDS = 0.5
RESUME_BURST_SECONDS = 2.0

AudioCallback = Callable[[np.ndarray], None]  # float32 mono frames


def speech_levels(audio: np.ndarray) -> np.ndarray:
    """Per-block RMS of *audio*, at the speech envelope's resolution.

    One definition of "how loud is this moment" for everything that asks: the
    engine's speech/silence envelope, the capture's wake detector and the
    audit scripts that reason about when somebody was talking. Blocks that do
    not fill SPEECH_BLOCK_SECONDS at the end are dropped, so callers see one
    level per whole block and can move between the two with the block size.
    """
    block = max(1, int(SPEECH_BLOCK_SECONDS * SAMPLE_RATE))
    usable = len(audio) - (len(audio) % block)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    blocks = np.asarray(audio[:usable], dtype=np.float32).reshape(-1, block)
    return np.sqrt(np.mean(np.square(blocks), axis=1))


def resume_instant(levels: np.ndarray, gate: float,
                   min_quiet: float = PAUSE_SECONDS) -> float | None:
    """Where speech resumed after a gap, in window seconds - or None.

    The engine's older view of a pause was "the last window ended in quiet",
    which misses the case the live loop actually creates: during a pause it
    waits IDLE_STRIDE_SECONDS between decodes, so a short pause can begin *and*
    end between two of them and never appear as a quiet tail at all. The window
    still holds the evidence - speech, then `min_quiet` of quiet, then speech
    again - so the first decode to read it can tell that a speaker has just come
    back, however the loop happened to be scheduled.

    The gap is `min_quiet` rather than "any quiet" on purpose: a speaker drawing
    breath mid-sentence is not a resume, and the whole point is to know when
    somebody *stopped* and started again. So the search walks back over the runs
    until it finds a quiet stretch that qualifies, rather than stopping at the
    first one: real audio has 50 ms dips inside the speech that comes back, and
    taking the nearest dip instead of the silence before it would answer "nobody
    came back" for exactly the window that shows somebody did.
    """
    above = levels >= gate
    if not len(above) or not above[-1]:
        return None              # not ending in speech: nobody has come back yet
    min_blocks = max(1, int(round(min_quiet / SPEECH_BLOCK_SECONDS)))
    index = len(above) - 1
    while index >= 0:
        if above[index]:
            index -= 1
            continue
        end_of_quiet = index
        while index >= 0 and not above[index]:
            index -= 1           # back over this quiet stretch
        if end_of_quiet - index >= min_blocks:
            return (end_of_quiet + 1) * SPEECH_BLOCK_SECONDS
    return None                  # no stretch of quiet long enough to be a pause


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
        self._sound = 0.0  # RMS of the newest speech block; feeds the wake
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

    @property
    def sound_level(self) -> float:
        """RMS of the newest speech-resolution block — what a wake hears.

        Read by the live loop while a pause is being waited out: speech at or
        above the engine's gate means the speaker came back and the wait is
        over. The whole-block `level` answers badly there: a syllable starting
        in the last 20 ms of a 100 ms block is averaged away by the silence
        before it, so the wake would be a whole block late - or, on a quiet
        speaker, never.
        """
        with self._lock:
            return self._sound

    def _update_meter(self, mono: np.ndarray) -> None:
        """Update both meters from the newest block (call under the lock)."""
        self._level = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
        tail = mono[len(mono) - int(SPEECH_BLOCK_SECONDS * SAMPLE_RATE):]
        levels = speech_levels(tail)
        self._sound = float(levels[-1]) if len(levels) else self._level

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
                        self._update_meter(mono)
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
                self._update_meter(mono)


# stabilized live text

@dataclass
class LiveState:
    """What the overlay renders: frozen text plus the changing tail."""
    committed: str = ""
    partial: str = ""
    full_text: str = ""  # every frozen word since the session started (uncapped)
    full_words: list[tuple] = field(default_factory=list)  # (start, end, text) seconds
    level: float = 0.0   # input RMS 0..~1, for the overlay's level meter
    window_end: float = 0.0  # absolute seconds of the newest frame decoded
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
                 resume_window_seconds: float | None = None,
                 resume_stride_seconds: float | None = RESUME_STRIDE_SECONDS,
                 translate_to: str | None = None, translator=None,
                 on_translation_error: Callable[[str], None] | None = None):
        self.model_size = model_size or config.LIVE_MODEL_SIZE
        self.beam_size = int(beam_size)
        self.window_seconds = float(window_seconds)
        # A shorter window for the first decode after a pause (None = never, the
        # default). Measured on both real recordings, in two pass orders: it does
        # not bring the first caption after a pause any earlier - the wake above
        # already did that - and it was 0.05-0.12 s slower if anything, because
        # the model wants the preceding audio to commit to a tail word. It costs
        # about a third of a full decode, so it is the knob a shorter post-resume
        # stride would spend (BACKLOG); nothing ships it on its own.
        self.resume_window_seconds = resume_window_seconds or None
        # The cadence for the first RESUME_BURST_SECONDS of audio after a pause.
        # It ships on; `resume_stride_seconds=None` (or 0) is the loop every
        # earlier measurement ran on, and the audit's `before`/`wake` passes. The
        # window above is what makes it affordable: a burst decode reads the few
        # seconds the resumed words need, not twelve of finished speech.
        self.resume_stride_seconds = (
            float(resume_stride_seconds) if resume_stride_seconds else None
        )
        self._burst_until: float = 0.0   # absolute ring-seconds; see decode_once
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
        # What audio may cut the wait after a decode short (the live loop in
        # ui.overlay reads this); decode_once sets it every window. None means
        # "wait out the full stride": see where it is assigned for why.
        self.wake_level: float | None = None
        self._full: list[str] = []         # every frozen word, uncapped (text only)
        self._full_words: list[tuple] = []  # frozen (start, end, text), seconds
        self._frontier: float = 0.0        # absolute ring-seconds: words starting before this freeze
        self._frozen_recent: list[tuple] = []  # (abs_start, norm_text) for dup matching
        self.late_drops = 0            # words first reported too late to still be speech
        self._speech_recently = True   # decode until proven silent
        self._decoded_until: float = 0.0  # ring-seconds the model has seen up to
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

    def _freeze(self, words: list[tuple], frontier: float,
                floor: float | None = None) -> list[tuple]:
        """Freeze the words that have just cleared the commit margin.

        A word has to be *new* as well as old enough. Anything further back
        than FREEZE_BAND_SECONDS behind the frontier was spoken too long ago
        to be arriving for the first time: the model has listened again and
        changed its mind about audio whose caption is already written. On a
        real lecture that second-guessing was the whole tail of the latency
        distribution - one spoken word came back as "a", "signal", "of" and
        "the" at the same timestamp over ten seconds, each arriving late
        enough to carry a 10 s caption age and each frozen as extra text.
        Ignoring the re-writes caps the age near the margin and stops the
        transcript growing a word per re-think.

        *floor* overrides the band — see `decode_once`, which widens it to
        cover whatever the frontier skipped over since the last decode.
        """
        if floor is None:
            floor = frontier - FREEZE_BAND_SECONDS
        fresh: list[tuple] = []
        for w in words:
            if self._is_dup(w):
                continue  # already captioned; every window re-proposes these
            if w[0] >= frontier:
                continue  # still inside the margin: it may yet be revised
            if w[0] < floor:
                self.late_drops += 1  # never seen before its window closed
                continue
            fresh.append(w)
        fresh = self._strip_repeat_run(fresh)
        for w in fresh:
            self._frozen_recent.append((w[0], self._norm(w[2])))
        # keep the matching list bounded: the window only ever re-proposes
        # words from the last WINDOW_SECONDS, so a slice of that is plenty
        if len(self._frozen_recent) > 400:
            del self._frozen_recent[:200]
        return fresh

    def _bursting(self, now: float) -> bool:
        """Is this decode inside the short cadence right after a resume?"""
        return now < self._burst_until

    def _short_window(self, now: float) -> float | None:
        """The shorter window this decode may read, or None for the full one.

        Two callers want it, and both are about the first moments after a pause.
        `resume_window_seconds` is the older one, applying to the single decode
        after a pause; measured on real audio it does not help on its own (see
        the constructor), so it only engages when something asks for it.
        """
        if self._bursting(now):
            return self.resume_window_seconds or RESUME_WINDOW_SECONDS
        if self.resume_window_seconds is not None and not self._speech_recently:
            return self.resume_window_seconds
        return None

    def _window_for(self, capture: MicCapture) -> float:
        """How much audio this decode should read (WINDOW_SECONDS, or less).

        A shorter window is the price of a shorter cadence: the post-resume burst
        reads a few seconds for each of its decodes - the resumed words and the
        utterance they continue, not twelve seconds of speech the captions
        already hold - which is what a 0.5 s stride is paid for with. Which
        decodes get one is `_short_window`'s business, and that is where the
        standalone ``resume_window_seconds`` (off by default) is explained.

        Either way it is never less than what no decode has looked at yet: the
        freeze band accepts a first-time word only within FREEZE_BAND_SECONDS of
        the margin, so a window that skipped audio a slow loop had not reached
        would be speech nobody ever writes down. A pause longer than the resume
        window therefore gets the full window back, and so does a loop that fell
        behind its stride.
        """
        now = capture.total_frames / SAMPLE_RATE
        short = self._short_window(now)
        if short is None:
            return self.window_seconds
        unseen = max(0.0, now - self._decoded_until)
        return min(self.window_seconds, max(short, unseen))

    def _speech_envelope(self, audio: np.ndarray) -> tuple[float | None, float, float | None]:
        """How this window is arranged: (last speech end, peak, resume instant).

        The whole-window RMS cannot answer any of the three questions the live
        loop asks. A quiet speaker is averaged away by the silence around them
        (one second of speech spread over twelve averages below the gate, so
        their words are never decoded at all), while a window full of room tone
        can sit above it. A 50 ms envelope sees all of it: *peak* is the loudest
        block, so a window with any speech in it passes; *last speech* is where
        the speaker stopped; and *resume* is where they started again after a
        pause, which is what a decode needs to know that the words in front of
        it are somebody coming back rather than somebody who never left.
        """
        levels = speech_levels(audio)
        if not len(levels):
            return (None, 0.0, None)
        block = max(1, int(SPEECH_BLOCK_SECONDS * SAMPLE_RATE))
        peak = float(levels.max())
        above = np.nonzero(levels >= self.silence_rms)[0]
        if not len(above):
            return (None, peak, None)
        return (float(above[-1] + 1) * block / SAMPLE_RATE, peak,
                resume_instant(levels, self.silence_rms))

    def decode_once(self, capture: MicCapture) -> LiveState:
        """Transcribe the current window and update the stabilized state."""
        audio = capture.read_tail(self._window_for(capture))
        if len(audio) < SAMPLE_RATE * 0.6:  # nothing meaningful yet
            return self.state
        window_start_t = (capture.total_frames - len(audio)) / SAMPLE_RATE
        window_len = len(audio) / SAMPLE_RATE
        window_end_t = window_start_t + window_len
        self.state.window_end = window_end_t

        last_speech, peak, resumed_at = self._speech_envelope(audio)
        quiet_tail = None if last_speech is None else window_len - last_speech
        # "Still talking" means speech in the newest PAUSE_SECONDS of audio.
        speaking_now = quiet_tail is not None and quiet_tail < PAUSE_SECONDS
        self.last_window_rms = peak
        self.suggested_stride = (
            STRIDE_SECONDS if speaking_now else IDLE_STRIDE_SECONDS
        )
        # This window holds a speaker coming back (`resumed_at`), so the loop
        # should not wait out the stride it chose while they were away: for the
        # first moments of the resumed utterance it decodes on the short cadence
        # instead. The burst is anchored to *when they came back*, not to now,
        # because the pause can begin and end between two decodes - the loop
        # backs off to IDLE_STRIDE_SECONDS while nobody talks, so the decode that
        # notices the return may already be half a second late, and it must not
        # buy itself a fresh two seconds of short strides every time it looks.
        # Is it worth the decodes? The wake ends the nap, but the model will not
        # report a word with ~0.1 s of audio behind it, so the resumed words are
        # caught by whatever decode comes next, and without this that is a stride
        # later (see the README's "On real audio").
        if (self.resume_stride_seconds is not None and speaking_now
                and resumed_at is not None):
            self._burst_until = window_start_t + resumed_at + RESUME_BURST_SECONDS
        if window_end_t < self._burst_until:
            self.suggested_stride = self.resume_stride_seconds
        # ...and what may cut that wait short. While the speaker is flowing,
        # nothing: the stride is already short, sound is on every block, and
        # waking on it would decode back to back and never let the GPU go.
        # Once they have stopped, the loop is deliberately waiting
        # IDLE_STRIDE_SECONDS on the assumption that there is nothing to
        # transcribe - and speech at the gate is that assumption being wrong.
        # The engine's own gate is the right threshold to listen for, since a
        # window carrying less than it is not decoded at all. And a wake cannot
        # loop on itself: the block that woke the loop is inside the next
        # window, which makes speaking_now true, which disarms the wake and
        # restores the active stride until the speaker really stops again.
        self.wake_level = None if speaking_now else self.silence_rms

        # Nothing above the gate anywhere in the window: there is no speech
        # to transcribe. Decoding it anyway is how Whisper's fluent filler
        # ("Thank you for watching.") gets invented on near-silence, and the
        # commit step would freeze it - so the whole failure class is removed
        # by not decoding. A full window of quiet also ends the speech run.
        if peak < self.silence_rms:
            self._speech_recently = False
            return self.state
        # The speaker is paused and their words are already frozen: stay quiet
        # (no GPU on the pause) until the envelope hears them again.
        if not speaking_now and not self._speech_recently:
            return self.state
        # Past both gates: this window holds speech (or ends an utterance), and
        # the model is about to see everything up to this window's edge.
        self._speech_recently = speaking_now
        self._decoded_until = window_end_t

        # The speaker just stopped. Everything they said is already final -
        # no further audio can revise it - so this window is frozen whole
        # instead of holding its tail words until speech resumes. Held, they
        # would freeze one pause later (however long the speaker thinks) even
        # though the band accepts them, and it is this flush that keeps a
        # paused tail at the ~3s an uninterrupted word gets.
        # Freezing to the last sound (not to the window edge) is safe either
        # way: no word can start after the speaker stopped, and the word ends
        # at least PAUSE_SECONDS before the window edge, so nothing is cut.
        flushing = not speaking_now
        frontier = (
            window_start_t + (last_speech or 0.0) if flushing
            else window_end_t - COMMIT_MARGIN
        )
        # The band, but never narrower than one decode's worth of audio. A
        # flush jumps the frontier by the whole pause, and a slow decode jumps
        # it by more than the stride; the words in that gap were pending when
        # the last decode ran, so refusing them would drop speech rather than
        # a re-write. (Nothing here is stale: the last decode would have
        # frozen it.)
        floor = min(frontier - FREEZE_BAND_SECONDS, self._frontier)
        if frontier > self._frontier:
            self._frontier = frontier

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

        frozen = self._freeze(words, self._frontier, floor=floor)
        if frozen:
            self._full.extend(t for _s, _e, t in frozen)
            self._full_words.extend(frozen)
        self.state.full_text = " ".join(self._full)
        self.state.full_words = list(self._full_words)
        # Display: whole sentences at once. Frozen-but-unfinished words sit
        # in _sentence_buffer and render as part of the partial tail; only
        # a completed sentence (or the end of the utterance) commits to the
        # green text.
        self._sentence_buffer.extend(frozen)
        last_text = frozen[-1][2] if frozen else ""
        if self._ends_sentence(last_text) or flushing:
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

        Reads with ffmpeg's own demuxer (via faster-whisper's decode helper)
        and decodes every stride — the same code path live audio takes,
        without a sound card. The first decodes see the short window a real
        session starts with (a few seconds of ring, not WINDOW_SECONDS):
        waiting for the window to fill would put the first words ~11 s out.
        """
        from faster_whisper.audio import decode_audio

        # sampling_rate forces a 16 kHz resample; output is mono by default
        audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        capture = MicCapture(device_id="unused")
        capture.write_frames(np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32))  # priming silence

        step = int(STRIDE_SECONDS * SAMPLE_RATE)
        pos = 0
        while pos < len(audio):
            chunk = audio[pos:pos + step]
            capture.write_frames(chunk)
            pos += step
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
    """Group a live session's frozen words into caption blocks.

    Words arrive in *decode* order, not time order. Two things break that
    order on real audio: the sliding window re-proposes the same word with a
    jittered start, so a later decode can freeze it after a word spoken after
    it, and overlapping window estimates can claim overlapping spans. A
    player walks cues in the order they were written, so those reversals came
    out as cues whose end precedes their start.

    Sorting by time and pulling each end back to the next word's start makes
    the timeline monotonic without dropping words; measured on a 4-minute
    lecture this took a saved live transcript from 59 reversed and 10
    overlapping cues out of 124 to none.
    """
    usable = sorted(
        (
            (start, end, text)
            for start, end, text in full_words
            if end is not None and start is not None and end >= start and text
        ),
        key=lambda w: (w[0], w[1], w[2]),
    )
    words = []
    for index, (start, end, text) in enumerate(usable):
        if index + 1 < len(usable):
            # never let this word run past the next one: subtitles do not overlap
            end = min(end, max(start, usable[index + 1][0]))
        words.append(Word(text=text, start=start, end=end))
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
