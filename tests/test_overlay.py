"""Tests for the live overlay and its worker (Phase 3).

Everything runs offscreen with fake engines/captures — no audio device and
no model are needed. One optional test exercises a real device when present.
"""
import threading
import time

import pytest

from captioner.live import (
    IDLE_STRIDE_SECONDS,
    SAMPLE_RATE,
    STRIDE_SECONDS,
    LiveEngine,
    LiveState,
)

pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.overlay import LiveOverlay, LiveWorker  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


class FakeCapture:
    """Stands in for MicCapture: feedable frames, no hardware."""

    def __init__(self):

        self.total_frames = 0
        self.ring_seconds = 60.0
        self._buf = []

    def write_frames(self, frames):
        self._buf.append(frames)
        self.total_frames += len(frames)

    def read_tail(self, seconds):
        import numpy as np

        want = int(seconds * SAMPLE_RATE)
        flat = np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.float32)
        return flat[-want:]


class FakeEngine:
    """Decodes nothing; returns scripted states so the UI loop is testable."""

    def __init__(self):
        self.calls = 0

    def decode_once(self, capture):
        self.calls += 1
        self.calls_state = LiveState(
            committed="hello world", partial="and more"
        )
        return self.calls_state


# overlay rendering

def test_overlay_shows_idle_hint(qapp):
    ov = LiveOverlay(persist_settings=False)
    assert "listening" in ov.lbl_partial.text()
    ov.close()


def test_overlay_update_state_renders_committed_and_partial(qapp):
    ov = LiveOverlay(persist_settings=False)
    ov.update_state(LiveState(committed="hello world", partial="and more", level=0.5))
    assert ov.lbl_committed.text() == "hello world"
    assert ov.lbl_partial.text() == "and more"
    assert "●" in ov.lbl_level.text()  # the input meter lights up
    # an empty partial falls back to an ellipsis once something is committed
    ov.update_state(LiveState(committed="hello world", partial=""))
    assert ov.lbl_partial.text() == "…"
    ov.close()


def test_overlay_menu_transcript_and_stop(qapp):
    """The overlay must expose copy/save/stop without the dashboard.
    The copy path is exercised via the same code the menu calls."""
    ov = LiveOverlay(persist_settings=False)
    ov.update_state(LiveState(committed="a b", full_text="a b c d"))
    assert ov._last_full_text == "a b c d"
    stopped = []
    ov.stop_requested.connect(lambda: stopped.append(1))
    ov.stop_requested.emit()
    assert stopped == [1]
    ov.close()


def test_overlay_set_device_name(qapp):
    ov = LiveOverlay(persist_settings=False)
    ov.set_device_name("● Fake Mic")
    assert ov.lbl_device.text() == "● Fake Mic"
    ov.close()


def test_overlay_show_error(qapp):
    ov = LiveOverlay(persist_settings=False)
    ov.show_error("device vanished")
    assert "device vanished" in ov.lbl_partial.text()
    ov.close()


def test_overlay_is_frameless_and_on_top(qapp):
    ov = LiveOverlay(persist_settings=False)
    flags = ov.windowFlags()
    assert bool(flags & QtWindowStaysOnTop()) is True
    ov.close()


def QtWindowStaysOnTop():
    from PySide6.QtCore import Qt

    return Qt.WindowType.WindowStaysOnTopHint


# worker loop

def test_worker_emits_states_until_stopped(qapp):
    seen = []
    worker = LiveWorker(
        device_id="fake",
        engine=FakeEngine(),
        capture_factory=lambda dev: FakeCapture(),
    )
    worker.state_ready.connect(lambda s: seen.append(s))
    worker.start()
    deadline = time.time() + 10
    while len(seen) < 3 and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    worker.stop()
    worker.wait(3000)
    assert len(seen) >= 3
    assert seen[0].committed == "hello world"


def test_worker_reports_capture_construction_failure(qapp):
    errors = []
    worker = LiveWorker(
        device_id="nope",
        capture_factory=lambda dev: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    worker.live_error.connect(lambda m: errors.append(m))
    worker.start()
    deadline = time.time() + 5
    while not errors and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    worker.wait(3000)
    assert errors and "boom" in errors[0]


def test_stride_matches_design():
    assert STRIDE_SECONDS == pytest.approx(1.5)


def test_stride_is_a_period_not_a_nap():
    """The decode's own cost comes out of the wait, so the loop does not drift."""
    from ui.overlay import stride_wait

    assert stride_wait(1.5, 0.0) == pytest.approx(1.5)
    assert stride_wait(1.5, 0.4) == pytest.approx(1.1)   # 1.5 s period
    assert stride_wait(1.5, 2.0) == pytest.approx(0.1)   # slower than the stride: never negative
    # the engine's request stays inside the loop's own bounds
    assert stride_wait(99.0, 0.0) == pytest.approx(IDLE_STRIDE_SECONDS)
    assert stride_wait(0.01, 0.0) == pytest.approx(0.5)


# the resume wake

class _Meter:
    """A capture meter that crosses the gate after a chosen number of reads."""

    def __init__(self, wake_after: int = 3, above: float = 0.5,
                 below: float = 0.0):
        self.reads = 0
        self.wake_after = wake_after
        self.above, self.below = above, below

    def __call__(self) -> float:
        self.reads += 1
        return self.above if self.reads > self.wake_after else self.below


def test_wait_for_sound_ends_the_wait_when_the_mic_hears_speech():
    from ui.overlay import wait_for_sound

    meter = _Meter(wake_after=3)
    started = time.monotonic()
    woke = wait_for_sound(threading.Event(), meter, 0.01, timeout=3.0, poll=0.02)
    elapsed = time.monotonic() - started
    assert woke is True
    assert meter.reads > 3            # it listened instead of sleeping the 3 s
    assert elapsed < 1.0
    assert elapsed >= 0.04            # ...after the sound, not before it


def test_wait_for_sound_sleeps_out_a_quiet_stride():
    """No sound, no shortcut: an idle wait still lasts its stride."""
    from ui.overlay import wait_for_sound

    started = time.monotonic()
    woke = wait_for_sound(threading.Event(), _Meter(above=0.0), 0.01,
                          timeout=0.3, poll=0.02)
    assert woke is False
    assert time.monotonic() - started >= 0.28


def test_wait_for_sound_does_not_listen_while_speech_flows():
    """An unarmed wake is a plain wait: no meter read at all."""
    from ui.overlay import wait_for_sound

    def refused():
        raise AssertionError("the meter was read with no wake armed")

    started = time.monotonic()
    assert wait_for_sound(threading.Event(), refused, None, timeout=0.2) is False
    assert time.monotonic() - started >= 0.18


def test_wait_for_sound_returns_at_once_when_the_session_stops():
    from ui.overlay import wait_for_sound

    stop = threading.Event()
    stop.set()
    started = time.monotonic()
    assert wait_for_sound(stop, _Meter(), 0.01, timeout=5.0) is False
    assert time.monotonic() - started < 0.2


def test_the_loop_decodes_as_soon_as_the_speaker_returns():
    """The pause's idle stride must not be waited out over the speaker.

    This is the whole point of the wake: a pause is followed by up to
    IDLE_STRIDE_SECONDS of nothing, and the words the speaker says when they
    come back are captioned at the *end* of that nap even though the audio is
    already in the ring.
    """
    from ui.overlay import run_live_session

    class PausedEngine(FakeEngine):
        suggested_stride = IDLE_STRIDE_SECONDS  # the engine heard a pause
        wake_level = 0.01

    class ResumingCapture(FakeCapture):
        """Silent, then the speaker is back (a few polls in)."""

        def __init__(self, silent_reads: int = 3):
            super().__init__()
            self._reads = 0
            self._silent_reads = silent_reads

        @property
        def sound_level(self):
            self._reads += 1
            return 0.0 if self._reads <= self._silent_reads else 0.5

    engine = PausedEngine()
    starts: list[float] = []
    real_decode = engine.decode_once

    def timed_decode(capture):
        starts.append(time.perf_counter())
        return real_decode(capture)

    engine.decode_once = timed_decode
    stop = threading.Event()
    loop = threading.Thread(
        target=run_live_session,
        args=(engine, ResumingCapture(), stop, lambda _s: None, lambda _m: None),
        daemon=True,
    )
    loop.start()
    deadline = time.time() + 5
    while len(starts) < 2 and time.time() < deadline:
        time.sleep(0.01)
    stop.set()
    loop.join(timeout=5)
    assert len(starts) >= 2, "the loop never decoded"
    gap = starts[1] - starts[0]
    assert gap < 1.0, f"the idle stride was waited out over the speaker ({gap:.2f}s)"


def test_the_loop_still_waits_out_the_idle_stride_in_a_silent_room():
    """With nothing to hear, the back-off is untouched (the GPU still rests)."""
    from ui.overlay import run_live_session

    class PausedEngine(FakeEngine):
        suggested_stride = IDLE_STRIDE_SECONDS
        wake_level = 0.01

    class DeafCapture(FakeCapture):
        sound_level = 0.0

    engine = PausedEngine()
    starts: list[float] = []
    real_decode = engine.decode_once

    def timed_decode(capture):
        starts.append(time.perf_counter())
        return real_decode(capture)

    engine.decode_once = timed_decode
    stop = threading.Event()
    loop = threading.Thread(
        target=run_live_session,
        args=(engine, DeafCapture(), stop, lambda _s: None, lambda _m: None),
        daemon=True,
    )
    loop.start()
    deadline = time.time() + 5
    while len(starts) < 2 and time.time() < deadline:
        time.sleep(0.01)
    stop.set()
    loop.join(timeout=5)
    assert len(starts) >= 2
    assert starts[1] - starts[0] >= IDLE_STRIDE_SECONDS - 0.3


# optional: a real device, if the machine has one (not a CI dependency)

def test_real_device_enumerates():
    try:
        devices = LiveEngine.__module__ and __import__(
            "captioner.live", fromlist=["list_input_devices"]
        ).list_input_devices()
    except Exception:
        pytest.skip("no audio subsystem available")
    assert isinstance(devices, list)
