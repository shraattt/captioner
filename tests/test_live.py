"""Tests for the live captioning core (Phase 3).

No audio hardware is touched: the ring buffer is driven with synthetic
frames, and the dedup logic is exercised purely. The full decode path runs
through `LiveEngine.stream_file` on a real speech fixture (tiny.en).
"""
import types
from pathlib import Path

import numpy as np
import pytest

from captioner import config

_ROOT = Path(config.__file__).resolve().parents[1]
from captioner.live import (
    COMMIT_MARGIN,
    IDLE_STRIDE_SECONDS,
    PAUSE_SECONDS,
    RESUME_BURST_SECONDS,
    RESUME_STRIDE_SECONDS,
    RESUME_WINDOW_SECONDS,
    SAMPLE_RATE,
    SILENCE_RMS,
    SPEECH_BLOCK_SECONDS,
    STRIDE_SECONDS,
    WINDOW_SECONDS,
    LiveEngine,
    LiveState,
    MicCapture,
)

SR = SAMPLE_RATE
RESUME_GAP = PAUSE_SECONDS + 0.5  # long enough to be a pause, not a breath


def tone(seconds: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ring buffer

def test_ring_read_tail_returns_newest_frames_in_order():
    cap = MicCapture(device_id="unused")
    cap.write_frames(np.ones(16_000, dtype=np.float32))
    cap.write_frames(np.full(16_000, 0.5, dtype=np.float32))
    tail = cap.read_tail(1.0)
    assert len(tail) == 16_000
    assert tail[0] == 0.5  # newest chunk, not the older one


def test_ring_wraps_without_corruption():
    cap = MicCapture(device_id="unused", ring_seconds=1.0)
    cap.write_frames(np.full(32_000, 2.0, dtype=np.float32))  # twice the ring
    tail = cap.read_tail(0.5)
    assert len(tail) == 8_000
    assert np.all(tail == 2.0)
    assert cap.total_frames == 32_000  # accounting survives the wrap


def test_read_tail_more_than_buffered_returns_what_exists():
    cap = MicCapture(device_id="unused")
    cap.write_frames(tone(0.3))
    assert len(cap.read_tail(10.0)) == 4_800


def test_read_tail_copies_are_independent():
    cap = MicCapture(device_id="unused")
    cap.write_frames(np.ones(1_000, dtype=np.float32))
    out = cap.read_tail(1.0)
    out[:] = 0
    assert cap.read_tail(1.0)[0] == 1.0


def test_the_meter_hears_a_syllable_the_whole_block_rms_averages_away():
    """Late in a capture block, speech must still register as sound.

    A 100 ms block whose last 50 ms is speech has half the energy of the
    speech, so its block RMS sits below the gate: a loop waking on that
    meter would be a block late at best, and on a quiet speaker never.
    """
    cap = MicCapture(device_id="unused")
    block = np.zeros(1_600, dtype=np.float32)
    block[-800:] = 0.005  # 50 ms of soft speech at the very end
    cap.write_frames(block)
    assert cap.level < SILENCE_RMS          # what the overlay meter shows
    assert cap.sound_level >= SILENCE_RMS   # what the wake listens for
    assert cap.sound_level == pytest.approx(0.005, rel=0.01)


def test_the_sound_meter_follows_the_newest_frames():
    cap = MicCapture(device_id="unused")
    cap.write_frames(np.full(1_600, 0.2, dtype=np.float32))
    loud = cap.sound_level
    cap.write_frames(np.zeros(1_600, dtype=np.float32))  # the speaker stopped
    assert loud > SILENCE_RMS
    assert cap.sound_level == 0.0


# engine: window bookkeeping (no model needed)

def test_engine_skips_decode_until_audible():
    engine = LiveEngine(model_size="tiny.en")
    cap = MicCapture(device_id="unused")
    cap.write_frames(np.zeros(int(0.2 * SR), dtype=np.float32))  # too short
    state = engine.decode_once(cap)
    assert state.text == ""


def test_constants_match_design():
    # the sliding window is a design promise the overlay's latency depends on
    assert WINDOW_SECONDS == pytest.approx(12.0)
    assert STRIDE_SECONDS == pytest.approx(1.5)
    assert COMMIT_MARGIN == pytest.approx(3.0)


def test_commit_deduplicates_across_windows():
    """Overlapping windows must freeze each word exactly once."""
    engine = LiveEngine(model_size="tiny.en")
    frozen_texts = []
    for words in (
        [(0.0, 0.5, "hello"), (0.6, 1.0, "world")],
        [(0.6, 1.0, "world"), (1.4, 2.0, "again")],
        [(1.4, 2.0, "again"), (2.4, 3.0, "once")],
    ):
        fresh = engine._freeze(words, frontier=3.0)
        frozen_texts.extend(t for _s, _e, t in fresh)
    assert frozen_texts == ["hello", "world", "again", "once"]
    # re-freezing at the same frontier yields nothing new
    assert engine._freeze([(0.0, 0.5, "hello")], frontier=3.0) == []


def test_commit_dedup_tolerates_timestamp_jitter():
    """Same word with slightly moved timestamps must not freeze twice.

    Real decode windows re-propose words with start times that drift by a
    few hundred ms; the old exact-key dedup froze them all again.
    """
    engine = LiveEngine(model_size="tiny.en")
    engine._freeze([(0.0, 0.5, "Hello")], frontier=3.0)
    # jittered + recased + re-punctuated proposal of the same word
    fresh = engine._freeze([(0.25, 0.8, "hello,")], frontier=3.0)
    assert fresh == []
    # but the same word spoken again later is NOT a duplicate
    later = engine._freeze([(8.0, 8.5, "hello")], frontier=9.0)
    assert [t for _s, _e, t in later] == ["hello"]


def test_engine_beam_and_window_settings():
    engine = LiveEngine(model_size="tiny.en", beam_size=4, window_seconds=15.0)
    assert engine.beam_size == 4
    assert engine.window_seconds == 15.0


def test_silent_window_skips_decode_without_recent_speech():
    """Silence gate: a quiet window is not decoded after a long pause.

    This is the anti-hallucination gate — Whisper invents fluent filler on
    near-silence, and the commit step would freeze it.
    """
    engine = LiveEngine(model_size="tiny.en")
    engine._speech_recently = False  # e.g. 10s of quiet already observed
    calls = []

    class FakeCap:
        total_frames = int(SR * 12)
        level = 0.0

        def read_tail(self, seconds):
            return np.zeros(int(seconds * SR), dtype=np.float32)

    state = engine.decode_once(FakeCap())
    assert state.text == ""
    assert not calls


def test_pause_threshold_clears_whispers_and_breaths():
    """PAUSE_SECONDS has to beat the VAD's own silence threshold.

    Whisper is told to split segments after 350 ms of silence, so a shorter
    threshold would call an ordinary mid-sentence breath the end of an
    utterance.
    """
    assert PAUSE_SECONDS >= 0.5
    assert 0 < SPEECH_BLOCK_SECONDS <= 0.2


# the speech envelope (what "quiet" and "still talking" actually mean)

class _WindowCap:
    """A capture whose window is exactly the audio the test hands it."""

    def __init__(self, window: np.ndarray):
        self._window = np.asarray(window, dtype=np.float32)
        self.total_frames = len(self._window)
        self.level = float(np.sqrt(np.mean(np.square(self._window[-1600:]))))

    def read_tail(self, seconds):
        return self._window[-int(seconds * SR):]


class _WordModel:
    """Stands in for Whisper: reports the words the test chose, counts calls."""

    def __init__(self, words):
        self.words = words
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        segment = types.SimpleNamespace(
            text=" ".join(w[2] for w in self.words),
            avg_logprob=-0.2,
            no_speech_prob=0.1,
            end=max((w[1] for w in self.words), default=0.0),
            words=[types.SimpleNamespace(word=w[2], start=w[0], end=w[1])
                   for w in self.words],
        )
        return [segment], types.SimpleNamespace(duration=len(audio) / SR)


def speech_then_quiet(speech_seconds: float, total_seconds: float = 12.0,
                      amplitude: float = 0.2) -> np.ndarray:
    """Someone talking, then a pause that runs to the end of the window."""
    speech = np.full(int(speech_seconds * SR), amplitude, dtype=np.float32)
    quiet = np.zeros(int(total_seconds * SR) - len(speech), dtype=np.float32)
    return np.concatenate([speech, quiet])


def speech_gap_speech(gap_seconds: float, resumed_seconds: float = 0.5,
                      total_seconds: float = 12.0,
                      amplitude: float = 0.2) -> np.ndarray:
    """One window holding the whole story: talking, a pause, talking again.

    This is what the live loop actually reads at a resume. It waits
    IDLE_STRIDE_SECONDS between decodes while nobody talks, so a short pause can
    begin and end between two of them - the window has the evidence even when
    no decode ever caught the quiet as a quiet *tail*.
    """
    before = total_seconds - gap_seconds - resumed_seconds
    return np.concatenate([
        np.full(int(before * SR), amplitude, dtype=np.float32),
        np.zeros(int(gap_seconds * SR), dtype=np.float32),
        np.full(int(resumed_seconds * SR), amplitude, dtype=np.float32),
    ])


def test_envelope_hears_a_quiet_speaker_the_window_rms_averages_away():
    """One second of soft speech in a quiet window is still speech.

    The whole-window RMS put this below the gate, so those words were never
    decoded at all - the gate was answering "is this window loud?" when the
    question is "is the speaker in it?".
    """
    audio = np.concatenate([
        np.zeros(11 * SR, dtype=np.float32),
        np.full(SR, 0.006, dtype=np.float32),
    ])
    engine = LiveEngine(model_size="tiny.en")
    last_speech, peak, resumed_at = engine._speech_envelope(audio)
    assert float(np.sqrt(np.mean(np.square(audio)))) < SILENCE_RMS  # old answer
    assert peak >= SILENCE_RMS                                     # new answer
    assert last_speech == pytest.approx(12.0, abs=0.1)
    # the long silence is still a pause as far as the window is concerned, and the
    # speech that ends it is where a decode would start its short cadence
    assert resumed_at == pytest.approx(11.0, abs=0.1)


def test_envelope_reports_no_speech_in_a_silent_window():
    engine = LiveEngine(model_size="tiny.en")
    last_speech, peak, resumed_at = engine._speech_envelope(
        np.zeros(12 * SR, dtype=np.float32))
    assert last_speech is None and peak < SILENCE_RMS and resumed_at is None


def test_the_envelope_finds_where_a_pause_ended():
    """The gap and the speech after it, read off the same 50 ms blocks.

    This is the fact a decode needs to know it is looking at somebody coming
    back: the live loop can sleep through a whole short pause, so the window's
    own shape is the only evidence there is.
    """
    engine = LiveEngine(model_size="tiny.en")
    window = speech_gap_speech(1.4, resumed_seconds=0.6, total_seconds=12.0)
    _last, _peak, resumed_at = engine._speech_envelope(window)
    assert resumed_at == pytest.approx(12.0 - 0.6, abs=0.1)
    # a window that does not end in speech has nobody newly back in it
    _last, _peak, none_yet = engine._speech_envelope(speech_then_quiet(5.0))
    assert none_yet is None


def test_a_window_with_no_speech_is_never_decoded(monkeypatch):
    """A fresh session in a silent room must not invent captions."""
    from captioner import live as live_mod

    model = _WordModel([(0.0, 0.5, "Thank")])
    monkeypatch.setattr(live_mod.transcribe, "get_model", lambda *a, **kw: model)
    engine = LiveEngine(model_size="tiny.en")
    state = engine.decode_once(_WindowCap(np.zeros(12 * SR, dtype=np.float32)))
    assert model.calls == 0, "a silent window reached the model"
    assert state.text == ""
    assert engine._speech_recently is False


def test_pause_flush_freezes_the_tail_instead_of_waiting_for_speech(monkeypatch):
    """A pause is an ending, not a delay.

    The gate used to shut after a couple of seconds of quiet - one stride
    before a word could clear COMMIT_MARGIN - so the words the speaker left
    behind waited for them to start talking again. Measured on a 4-minute
    lecture, that wait was the 10.6 s p95 caption age; the words are final as
    soon as the speaking stops, so they are frozen with the window.
    """
    from captioner import live as live_mod

    words = [(3.0, 3.4, "the"), (3.5, 4.0, "smaller"), (4.1, 4.9, "element.")]
    monkeypatch.setattr(live_mod.transcribe, "get_model", lambda *a, **kw: _WordModel(words))
    engine = LiveEngine(model_size="tiny.en")
    # five seconds of speech, then the speaker stops for the rest of the window
    state = engine.decode_once(_WindowCap(speech_then_quiet(5.0)))
    assert [t for _s, _e, t in state.full_words] == ["the", "smaller", "element."]
    assert state.committed == "the smaller element."  # the line commits too
    assert state.partial == ""                        # nothing left in flight
    assert engine._speech_recently is False            # the pause shut the gate


def test_a_pause_costs_one_decode_then_nothing(monkeypatch):
    """The pause is decoded once (to freeze it), then the GPU rests."""
    from captioner import live as live_mod

    model = _WordModel([(3.0, 3.4, "done.")])
    monkeypatch.setattr(live_mod.transcribe, "get_model", lambda *a, **kw: model)
    engine = LiveEngine(model_size="tiny.en")
    capture = _WindowCap(speech_then_quiet(5.0))
    engine.decode_once(capture)
    assert model.calls == 1
    engine.decode_once(capture)  # still the same pause
    engine.decode_once(capture)
    assert model.calls == 1, "the pause kept decoding"


def test_the_commit_margin_still_holds_words_back_while_talking(monkeypatch):
    """The pause flush must not shorten the margin during speech.

    While the speaker keeps going, the sliding window is still allowed to
    revise the newest words, so only what cleared COMMIT_MARGIN freezes.
    """
    from captioner import live as live_mod

    words = [(7.0, 7.4, "the"), (11.0, 11.4, "tail")]
    monkeypatch.setattr(live_mod.transcribe, "get_model", lambda *a, **kw: _WordModel(words))
    engine = LiveEngine(model_size="tiny.en")
    state = engine.decode_once(_WindowCap(speech_then_quiet(12.0)))
    assert [t for _s, _e, t in state.full_words] == ["the"]  # 7.0 < 12-3
    assert "tail" in state.partial                            # still revisable
    assert engine._speech_recently is True


def test_a_late_re_write_of_already_frozen_audio_is_refused():
    """The model's second thoughts about old audio are not new caption text.

    One spoken word came back as four different words at the same timestamp
    over ten seconds of a real lecture; each late arrival cleared the commit
    margin, so each was frozen - the transcript grew a word per re-think and
    the last of them carried a 10 s age.
    """
    engine = LiveEngine(model_size="tiny.en")
    late = engine._freeze([(1.0, 1.4, "the")], frontier=9.0)  # 8s old: a re-write
    assert late == [] and engine.late_drops == 1
    # the same word arriving when it is actually due is of course kept
    fresh = engine._freeze([(7.0, 7.4, "the")], frontier=9.0)
    assert [t for _s, _e, t in fresh] == ["the"]
    assert engine.late_drops == 1


def test_the_band_never_eats_the_session_ending(monkeypatch):
    """At session end the whole window is frozen, however old it looks.

    The band exists to stop the model re-writing captions that are already
    on screen; when the session stops there is no next window to freeze these
    words in, so dropping them would lose speech outright.
    """
    from captioner import live as live_mod

    words = [(1.0, 1.4, "goodbye"), (2.0, 2.6, "everyone.")]  # old, not late
    monkeypatch.setattr(live_mod.transcribe, "get_model", lambda *a, **kw: _WordModel(words))
    engine = LiveEngine(model_size="tiny.en")
    state = engine.flush_once(_WindowCap(speech_then_quiet(6.0)))
    assert [t for _s, _e, t in state.full_words] == ["goodbye", "everyone."]
    assert state.committed == "goodbye everyone."


def test_the_wake_is_armed_only_while_the_speaker_has_stopped(monkeypatch):
    """While speech flows, nothing may cut the stride short.

    Sound arrives on every block during speech, so a wake armed there would
    decode back to back and never let the GPU go. The pause is the only time
    the loop is waiting on the assumption that there is nothing to transcribe
    — and that is the one assumption a returning speaker can falsify.
    """
    from captioner import live as live_mod

    model = _WordModel([(11.0, 11.4, "tail")])
    monkeypatch.setattr(live_mod.transcribe, "get_model", lambda *a, **kw: model)
    engine = LiveEngine(model_size="tiny.en")
    engine.decode_once(_WindowCap(speech_then_quiet(12.0)))
    assert engine.wake_level is None                    # still talking
    assert engine.suggested_stride == pytest.approx(STRIDE_SECONDS)
    # one window later the speaker has stopped, and the loop backs off
    engine.decode_once(_WindowCap(speech_then_quiet(5.0)))
    assert engine.wake_level == pytest.approx(SILENCE_RMS)
    assert engine.suggested_stride == pytest.approx(IDLE_STRIDE_SECONDS)


def test_a_full_window_of_quiet_still_leaves_the_wake_armed(monkeypatch):
    """The idle stride after a long pause is the same wait to interrupt."""
    from captioner import live as live_mod

    model = _WordModel([(0.0, 0.4, "Thank")])
    monkeypatch.setattr(live_mod.transcribe, "get_model", lambda *a, **kw: model)
    engine = LiveEngine(model_size="tiny.en")
    state = engine.decode_once(_WindowCap(np.zeros(12 * SR, dtype=np.float32)))
    assert model.calls == 0 and state.text == ""
    assert engine.wake_level == pytest.approx(SILENCE_RMS)


class _SpyCap(_WindowCap):
    """A window capture that remembers how much audio each decode asked for."""

    def __init__(self, window: np.ndarray):
        super().__init__(window)
        self.asked: list[float] = []

    def read_tail(self, seconds):
        self.asked.append(seconds)
        return super().read_tail(seconds)


def test_the_short_resume_window_asks_for_less_after_a_pause(monkeypatch):
    """Opt-in: a resume decodes a few seconds, not twelve of finished speech.

    Off by default because measuring it showed no gain (see RESUME_WINDOW_SECONDS
    and the README); what this pins down is that when it *is* asked for, it
    applies only to that first decode.
    """
    from captioner import live as live_mod

    monkeypatch.setattr(live_mod.transcribe, "get_model",
                        lambda *a, **kw: _WordModel([(11.0, 11.4, "tail.")]))
    engine = LiveEngine(model_size="tiny.en",
                        resume_window_seconds=RESUME_WINDOW_SECONDS)
    while_talking = _SpyCap(speech_then_quiet(12.0))
    engine.decode_once(while_talking)
    assert while_talking.asked == [pytest.approx(WINDOW_SECONDS)]
    # the same engine one window later: the speaker stopped (the pause flush)
    engine.decode_once(_SpyCap(speech_then_quiet(5.0)))
    assert engine._speech_recently is False
    resuming = _SpyCap(speech_then_quiet(12.0))
    engine.decode_once(resuming)
    assert resuming.asked == [pytest.approx(RESUME_WINDOW_SECONDS)]
    # ...and the decode after that is back on the full window
    after = _SpyCap(speech_then_quiet(12.0))
    engine.decode_once(after)
    assert after.asked == [pytest.approx(WINDOW_SECONDS)]


def test_the_resume_window_never_skips_what_no_decode_has_seen(monkeypatch):
    """The short window is a floor, not a rule.

    The freeze band only accepts a first-time word within FREEZE_BAND_SECONDS of
    the margin, so a resume window shorter than the audio a slow loop had not
    reached yet would be speech nobody ever writes down. Whatever is unseen wins,
    up to the full window.
    """
    from captioner import live as live_mod

    monkeypatch.setattr(live_mod.transcribe, "get_model", lambda *a, **kw: _WordModel([]))
    engine = LiveEngine(model_size="tiny.en",
                        resume_window_seconds=RESUME_WINDOW_SECONDS)
    engine._speech_recently = False

    # speech then quiet, so the window is still a pause and the early return
    # leaves _decoded_until where the test put it
    def window_when(decoded_until: float, now: float) -> float:
        engine._decoded_until = decoded_until
        cap = _SpyCap(speech_then_quiet(5.0))
        cap.total_frames = int(now * SR)
        engine.decode_once(cap)
        return cap.asked[0]

    assert window_when(now=20.0, decoded_until=19.4) == pytest.approx(RESUME_WINDOW_SECONDS)
    assert window_when(now=20.0, decoded_until=14.0) == pytest.approx(6.0)
    # a pause longer than the window (or a loop far behind): the full window
    assert window_when(now=40.0, decoded_until=18.0) == pytest.approx(WINDOW_SECONDS)


def test_the_resume_window_is_off_by_default():
    """Every decode reads the full window unless a caller asks otherwise.

    A shorter resume window was measured on two real recordings, in two pass
    orders, and did not bring the first caption after a pause any earlier - so
    the default is the window every earlier measurement ran on.
    """
    engine = LiveEngine(model_size="tiny.en")
    assert engine.resume_window_seconds is None
    engine._speech_recently = False
    cap = _SpyCap(speech_then_quiet(12.0))
    engine.decode_once(cap)
    assert cap.asked == [pytest.approx(WINDOW_SECONDS)]


def test_a_resume_decodes_on_a_short_cadence_for_the_first_moments(monkeypatch):
    """The wake ends the nap; the burst is what can actually catch the words.

    Waking the loop on sound starts a decode the moment the speaker is back,
    but the model will not report a word with ~0.1 s of audio behind it, so
    those words land on whatever decode comes next - a whole stride later. The
    burst comes back sooner, and reads the short window that makes it cheap.
    """
    from captioner import live as live_mod

    monkeypatch.setattr(live_mod.transcribe, "get_model",
                        lambda *a, **kw: _WordModel([(11.0, 11.4, "tail.")]))
    engine = LiveEngine(model_size="tiny.en",
                        resume_stride_seconds=RESUME_STRIDE_SECONDS)
    # while the speaker is flowing: the normal stride, the full window
    engine.decode_once(_WindowCap(speech_then_quiet(12.0)))     # ring time 12.0 s
    assert engine.suggested_stride == pytest.approx(STRIDE_SECONDS)
    # they stop: the loop backs off and arms the wake
    engine.decode_once(_WindowCap(speech_then_quiet(5.0)))
    assert engine.suggested_stride == pytest.approx(IDLE_STRIDE_SECONDS)
    assert engine.wake_level is not None

    resume_at = 12.1  # absolute seconds: where the speaker said something again

    def after(resumed_for: float) -> _SpyCap:
        """A decode *resumed_for* seconds into the resumed utterance."""
        cap = _SpyCap(speech_gap_speech(RESUME_GAP, resumed_for))
        cap.total_frames = int((resume_at + resumed_for) * SR)
        engine.decode_once(cap)
        return cap

    # the window reads as "speech, a pause, speech again": short cadence from
    # here, until RESUME_BURST_SECONDS after they came back. This first decode
    # still reads the whole window - the engine cannot know what it is looking at
    # until it has looked - so what the burst makes cheap is what follows it.
    resumed = after(0.5)
    assert engine.suggested_stride == pytest.approx(RESUME_STRIDE_SECONDS)
    assert resumed.asked == [pytest.approx(WINDOW_SECONDS)]
    for step in (1.0, 1.6):
        cap = after(step)
        assert cap.asked == [pytest.approx(RESUME_WINDOW_SECONDS)]
        assert engine.suggested_stride == pytest.approx(RESUME_STRIDE_SECONDS)
    # ...and once the burst has covered its audio, stride and window are back
    later = after(RESUME_BURST_SECONDS + 0.2)
    assert later.asked == [pytest.approx(WINDOW_SECONDS)]
    assert engine.suggested_stride == pytest.approx(STRIDE_SECONDS)


def test_the_resume_burst_can_be_switched_off(monkeypatch):
    """The burst ships on, and `None` is the loop it is measured against.

    It earned the default by measurement (see RESUME_STRIDE_SECONDS), so turning
    it off has to be possible for the same reason: the audit's `before` and
    `wake` passes are the comparison it won, and they run without it.
    """
    from captioner import live as live_mod

    monkeypatch.setattr(live_mod.transcribe, "get_model",
                        lambda *a, **kw: _WordModel([(11.0, 11.4, "tail.")]))
    assert LiveEngine(model_size="tiny.en").resume_stride_seconds == pytest.approx(
        RESUME_STRIDE_SECONDS)
    engine = LiveEngine(model_size="tiny.en", resume_stride_seconds=None)
    assert engine.resume_stride_seconds is None
    engine.decode_once(_WindowCap(speech_then_quiet(12.0)))
    engine.decode_once(_WindowCap(speech_then_quiet(5.0)))      # the pause
    assert engine.wake_level is not None                        # wake armed
    resumed = _SpyCap(speech_gap_speech(RESUME_GAP, 0.5))        # and they are back
    resumed.total_frames = int(13.2 * SR)
    engine.decode_once(resumed)
    assert engine.suggested_stride == pytest.approx(STRIDE_SECONDS)
    assert resumed.asked == [pytest.approx(WINDOW_SECONDS)]
    # ...and the decode after it, which a burst would have pulled forward
    later = _SpyCap(speech_gap_speech(RESUME_GAP, 1.0))
    later.total_frames = int(13.7 * SR)
    engine.decode_once(later)
    assert later.asked == [pytest.approx(WINDOW_SECONDS)]
    assert engine.suggested_stride == pytest.approx(STRIDE_SECONDS)


def test_speech_that_never_paused_never_bursts(monkeypatch):
    """A burst while the speaker flows would decode back to back for nothing.

    A resume is a *gap* followed by speech; speech with no gap in it is a
    speaker who never left, and the loop must keep leaving the GPU alone between
    its decodes. A breath mid-sentence (here half of PAUSE_SECONDS) is not a gap
    either - if it counted as one, every other decode would be a burst.
    """
    from captioner import live as live_mod

    monkeypatch.setattr(live_mod.transcribe, "get_model",
                        lambda *a, **kw: _WordModel([(10.0, 10.4, "and")]))
    engine = LiveEngine(model_size="tiny.en",
                        resume_stride_seconds=RESUME_STRIDE_SECONDS)
    windows = [speech_then_quiet(12.0),                            # never stopped
               speech_gap_speech(PAUSE_SECONDS / 2, 0.5)]          # a breath
    for now, window in zip((12.0, 13.0), windows, strict=False):
        cap = _SpyCap(window)
        cap.total_frames = int(now * SR)
        engine.decode_once(cap)
        assert cap.asked == [pytest.approx(WINDOW_SECONDS)]
        assert engine.suggested_stride == pytest.approx(STRIDE_SECONDS)


def test_the_wake_listens_at_this_session_s_gate(monkeypatch):
    """A hot mic's calibrated gate is the level the loop listens for.

    The threshold has to be the gate that decides whether a window is decoded
    at all: anything below it would wake the loop only to return without a
    decode, and anything above it would miss a quiet speaker's return.
    """
    from captioner import live as live_mod

    monkeypatch.setattr(live_mod.transcribe, "get_model",
                        lambda *a, **kw: _WordModel([(3.0, 3.4, "done.")]))
    engine = LiveEngine(model_size="tiny.en")
    engine.calibrate(_AmbientCap(rms=0.012), seconds=0.5)
    engine.decode_once(_WindowCap(speech_then_quiet(5.0)))
    assert engine.wake_level == pytest.approx(engine.silence_rms)
    assert engine.wake_level > SILENCE_RMS


def test_speech_resuming_reopens_the_gate(monkeypatch):
    from captioner import live as live_mod

    model = _WordModel([(7.0, 7.4, "again.")])
    monkeypatch.setattr(live_mod.transcribe, "get_model", lambda *a, **kw: model)
    engine = LiveEngine(model_size="tiny.en")
    engine._speech_recently = False  # a pause already shut the gate
    state = engine.decode_once(_WindowCap(speech_then_quiet(12.0)))
    assert model.calls == 1
    assert engine._speech_recently is True
    assert state.full_words


def test_sentence_buffer_commits_whole_sentences():
    """Frozen words reach the green text only at sentence ends."""
    engine = LiveEngine(model_size="tiny.en")
    engine._freeze([(0.0, 0.3, "Hello"), (0.4, 0.7, "there,")], frontier=3.0)
    engine._sentence_buffer.extend(engine._freeze([], 3.0))
    # simulate the decode path: buffer then commit on sentence end
    engine._sentence_buffer.extend([(0.0, 0.3, "Hello"), (0.4, 0.7, "there,")])
    engine._commit_sentence()
    assert engine.state.committed == "Hello there,"


def test_ends_sentence_detection():
    assert LiveEngine._ends_sentence("dog.")
    assert LiveEngine._ends_sentence("really?")
    assert LiveEngine._ends_sentence("wow!")
    assert not LiveEngine._ends_sentence("hello")
    assert not LiveEngine._ends_sentence("")


def test_repeat_run_stripped_at_tail():
    """TV-audio re-proposals (drifted stamps, same phrase) are dropped."""
    e = LiveEngine(model_size="tiny.en")
    e._frozen_recent = [(0.5, "we"), (0.9, "ll"), (1.2, "dive"), (1.6, "in")]
    fresh = [(5.9, 6.2, "We"), (6.3, 6.5, "'ll"), (6.7, 7.0, "dive"), (7.1, 7.4, "in.")]
    assert e._strip_repeat_run(fresh) == []


def test_repeat_run_stripped_midstream_new_words_kept():
    e = LiveEngine(model_size="tiny.en")
    e._frozen_recent = [(0.5, "we"), (0.9, "ll"), (1.2, "dive"), (1.6, "in")]
    fresh = [(5.9, 6.2, "We"), (6.3, 6.5, "'ll"), (6.7, 7.0, "dive"),
             (7.1, 7.4, "in."), (7.5, 7.9, "now"), (8.0, 8.4, "really.")]
    kept = [t for _s, _e, t in e._strip_repeat_run(fresh)]
    assert kept == ["now", "really."]


def test_short_and_genuine_repeats_kept():
    """Two-word echoes are real speech; new words never match anything."""
    e = LiveEngine(model_size="tiny.en")
    e._frozen_recent = [(0.5, "yeah"), (0.9, "man")]
    assert len(e._strip_repeat_run([(5.0, 5.2, "Yeah"), (5.3, 5.6, "man")])) == 2
    e2 = LiveEngine(model_size="tiny.en")
    e2._frozen_recent = [(0.5, "one"), (0.9, "two")]
    kept = e2._strip_repeat_run([(1, 1.3, "Three"), (2, 2.3, "Four.")])
    assert [t for _s, _e, t in kept] == ["Three", "Four."]


def test_commit_guard_skips_identical_sentence():
    """Last-resort: a sentence identical to the just-committed one is skipped."""
    e = LiveEngine(model_size="tiny.en")
    e._sentence_buffer.extend([(0, 0.3, "Hello"), (0.3, 0.6, "there"), (0.6, 0.9, "friend.")])
    e._commit_sentence()
    e._sentence_buffer.extend([(1, 1.3, "Hello"), (1.3, 1.6, "there"), (1.6, 1.9, "friend.")])
    e._commit_sentence()
    assert e.state.committed == "Hello there friend."


# per-session noise-floor calibration

class _AmbientCap:
    """Fake capture whose ambient level is settable; used for calibration."""

    def __init__(self, rms: float):
        self.rms = rms
        self.total_frames = int(SR * 12)

    def read_tail(self, seconds):
        n = int(seconds * SR)
        block = np.ones(n, dtype=np.float32) * self.rms
        return block


def test_calibration_raises_gate_on_hot_mic():
    """A hot mic (room tone above the default gate) gets a higher gate."""
    e = LiveEngine(model_size="tiny.en")
    assert e.silence_rms == pytest.approx(SILENCE_RMS)
    result = e.calibrate(_AmbientCap(rms=0.012), seconds=0.5)
    assert result["calibrated"] and result["floor"] == pytest.approx(0.012)
    # floor*2.5 = 0.03 clamps at GATE_MAX_RMS; either way above the default
    assert e.silence_rms > SILENCE_RMS
    assert e.silence_rms <= LiveEngine.GATE_MAX_RMS + 1e-6


def test_calibration_keeps_default_on_quiet_mic():
    """A quiet mic must not get a gate below the safe default."""
    e = LiveEngine(model_size="tiny.en")
    e.calibrate(_AmbientCap(rms=0.001), seconds=0.5)
    assert e.silence_rms == pytest.approx(SILENCE_RMS)  # clamped up to default
    assert e.min_logprob == pytest.approx(LiveEngine.MIN_SEGMENT_LOGPROB)


def test_calibration_tightens_confidence_on_noisy_mic():
    e = LiveEngine(model_size="tiny.en")
    e.calibrate(_AmbientCap(rms=0.015), seconds=0.5)
    assert e.min_logprob < LiveEngine.MIN_SEGMENT_LOGPROB  # stricter


def test_calibration_floor_ignores_speech_spikes():
    """The 25th-percentile floor ignores occasional loud blocks."""
    levels = [0.002] * 18 + [0.2] * 2  # mostly quiet, two loud spikes
    floor = float(np.percentile(sorted(levels), 25))
    assert floor == pytest.approx(0.002)  # spikes did not poison the floor


def test_committed_text_is_capped():
    engine = LiveEngine(model_size="tiny.en")
    many = [(float(i), float(i) + 0.4, f"w{i}") for i in range(400)]
    engine._append_committed([t for _s, _e, t in many])
    assert len(engine.state.committed) <= engine.COMMITTED_KEEP_CHARS
    assert engine.state.committed.split()[-1] == "w399"  # keeps the newest tail


def test_live_state_text_joins_committed_and_partial():
    s = LiveState(committed="hello world", partial="and more")
    assert s.text == "hello world and more"
    assert LiveState(committed="only").text == "only"
    assert LiveState(partial="only").text == "only"
    assert LiveState().text == ""


# transcript export (live SRT/VTT)

WORDS = [
    (0.0, 0.4, "Hello"), (0.45, 0.9, "and"), (0.95, 1.4, "welcome."),
    (4.0, 4.5, "Second"), (4.55, 5.0, "sentence"),
]


def test_transcript_to_srt_groups_and_numbers_like_batch():
    from captioner.live import transcript_to_srt

    text = transcript_to_srt(WORDS)
    assert text.startswith("1\n00:00:00,000")  # files get platform endings on write
    assert "-->" in text
    assert "Hello and welcome." in text  # same segmentation as batch captions
    assert "Second sentence" in text


def test_transcript_to_vtt_has_the_header():
    from captioner.live import transcript_to_vtt

    text = transcript_to_vtt(WORDS)
    assert text.startswith("WEBVTT")
    assert "00:00:04.000" in text


def test_save_transcript_picks_format_by_extension(tmp_path):
    from captioner.live import save_transcript

    srt_path = save_transcript(WORDS, str(tmp_path / "live.srt"))
    assert (tmp_path / "live.srt").read_text(encoding="utf-8").startswith("1")
    assert str(srt_path).endswith(".srt")

    txt_path = save_transcript(WORDS, str(tmp_path / "notes.txt"))
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8").startswith("Hello and welcome")
    assert str(txt_path).endswith(".txt")
    # atomic write: no .tmp leftovers
    assert not list(tmp_path.glob("*.tmp"))


def test_transcript_export_skips_bad_words():
    from captioner.live import transcript_to_srt

    dirty = [(0.0, 0.4, "Good"), (5.0, 4.0, "backwards"), (1.0, 1.5, "")]
    text = transcript_to_srt(dirty)
    assert "Good" in text
    assert "backwards" not in text  # end < start is dropped, not fatal


def test_transcript_export_orders_words_that_arrived_out_of_order():
    """A live session freezes words in decode order, not time order.

    Overlapping windows re-propose words with jittered starts, so a later
    decode can freeze a word spoken earlier, and two estimates of one word
    can claim overlapping spans. A 4-minute real lecture produced 59 reversed
    and 10 overlapping cues out of 124 before the export sorted and clamped
    them; a player walks cues in written order, so those read as text jumping
    backwards.
    """
    from captioner.live import transcript_to_srt
    from captioner.srt import parse_srt

    jittered = [
        (4.00, 4.40, "second"),   # frozen first, spoken last
        (1.00, 1.60, "the"),
        (1.55, 3.80, "first"),    # its end runs past the next word's start
        (4.35, 4.80, "line"),
    ]
    blocks = parse_srt(transcript_to_srt(jittered))
    assert [b.text for b in blocks] == ["the first second line"]
    _assert_cues_walk_forward(blocks)

    overlapping = [(1.00, 3.00, "hello"), (1.10, 2.90, "hello"),
                   (3.05, 3.40, "there")]
    _assert_cues_walk_forward(parse_srt(transcript_to_srt(overlapping)))


def _assert_cues_walk_forward(blocks):
    assert blocks, "no cues written"
    assert all(b.end > b.start for b in blocks), [(b.start, b.end, b.text) for b in blocks]
    for earlier, later in zip(blocks, blocks[1:], strict=False):
        assert later.start >= earlier.end, (earlier.text, later.text)


# full decode path on real speech (tiny.en, no hardware)

@pytest.fixture()
def smoke_audio():
    audio_path = _ROOT / "data" / "uploads" / "smoke_speech.wav"
    if not audio_path.exists():
        pytest.skip("smoke fixture missing")
    return str(audio_path)


def test_stream_file_produces_stabilized_state(smoke_audio, monkeypatch):
    """A real speech file must come out the other end with real words."""
    monkeypatch.setattr(config, "LIVE_MODEL_SIZE", "tiny.en")
    engine = LiveEngine(model_size="tiny.en")
    updates = []
    state = engine.stream_file(str(smoke_audio), on_update=updates.append)
    text = state.text
    assert text, "live pipeline produced no text for real speech"
    assert len(updates) >= 2, "expected several sliding-window updates"
    assert state.segments and all(w["text"] for w in state.segments)
    # everything frozen must still be part of what the state renders
    committed_words = state.committed.split()
    assert all(w in text for w in committed_words[-3:])


def test_stream_file_is_repeatable(smoke_audio, monkeypatch):
    """A fresh engine on the same audio reaches the same committed words."""
    monkeypatch.setattr(config, "LIVE_MODEL_SIZE", "tiny.en")
    a = LiveEngine(model_size="tiny.en").stream_file(str(smoke_audio))
    b = LiveEngine(model_size="tiny.en").stream_file(str(smoke_audio))
    assert a.committed.split()[-3:] == b.committed.split()[-3:]
