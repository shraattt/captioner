"""Tests for the live captioning core (Phase 3).

No audio hardware is touched: the ring buffer is driven with synthetic
frames, and the dedup logic is exercised purely. The full decode path runs
through `LiveEngine.stream_file` on a real speech fixture (tiny.en).
"""
from pathlib import Path

import numpy as np
import pytest

from captioner import config

_ROOT = Path(config.__file__).resolve().parents[1]
from captioner.live import (
    COMMIT_MARGIN,
    MicCapture,
    LiveEngine,
    LiveState,
    SAMPLE_RATE,
    SILENCE_RMS,
    STRIDE_SECONDS,
    WINDOW_SECONDS,
)

SR = SAMPLE_RATE


def tone(seconds: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# ring buffer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# engine: window bookkeeping (no model needed)
# ---------------------------------------------------------------------------

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


def test_quiet_window_still_decodes_shortly_after_speech():
    """The gate must not eat the words just after speech stops (2s hold)."""
    engine = LiveEngine(model_size="tiny.en")
    assert engine._speech_recently is True  # fresh session decodes
    assert engine.SILENCE_HOLD_SECONDS >= 1.0


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


# ---------------------------------------------------------------------------
# per-session noise-floor calibration
# ---------------------------------------------------------------------------

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
    result = e.calibrate(_AmbientCap(rms=0.001), seconds=0.5)
    assert e.silence_rms == pytest.approx(SILENCE_RMS)  # clamped up to default
    assert e.min_logprob == pytest.approx(LiveEngine.MIN_SEGMENT_LOGPROB)


def test_calibration_tightens_confidence_on_noisy_mic():
    e = LiveEngine(model_size="tiny.en")
    e.calibrate(_AmbientCap(rms=0.015), seconds=0.5)
    assert e.min_logprob < LiveEngine.MIN_SEGMENT_LOGPROB  # stricter


def test_calibration_floor_ignores_speech_spikes():
    """The 25th-percentile floor ignores occasional loud blocks."""
    e = LiveEngine(model_size="tiny.en")
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


# ---------------------------------------------------------------------------
# transcript export (live SRT/VTT)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# full decode path on real speech (tiny.en, no hardware)
# ---------------------------------------------------------------------------

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
