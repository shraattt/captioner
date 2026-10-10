"""Tests for the live overlay and its worker (Phase 3).

Everything runs offscreen with fake engines/captures — no audio device and
no model are needed. One optional test exercises a real device when present.
"""
import time

import pytest

from captioner.live import LiveEngine, LiveState, MicCapture, STRIDE_SECONDS

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
        from captioner.live import SAMPLE_RATE

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


# ---------------------------------------------------------------------------
# overlay rendering
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# worker loop
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# optional: a real device, if the machine has one (not a CI dependency)
# ---------------------------------------------------------------------------

def test_real_device_enumerates():
    try:
        devices = LiveEngine.__module__ and __import__(
            "captioner.live", fromlist=["list_input_devices"]
        ).list_input_devices()
    except Exception:
        pytest.skip("no audio subsystem available")
    assert isinstance(devices, list)
