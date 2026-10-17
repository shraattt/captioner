"""Tests for live translation.

No model and no audio device: the translator and the ASR model are both
fakes, so what is verified is the wiring (which text gets translated, when,
and what the UI shows) rather than the quality of anybody's translation.
"""
import time
import types

import numpy as np
import pytest

from captioner import config
from captioner.live import SAMPLE_RATE, LiveEngine, LiveState, TranslationRunner

pytest.importorskip("PySide6.QtWidgets")
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.overlay import LiveOverlay, TranscriptWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class FakeTranslator:
    """Any MT backend, minus the model."""

    def __init__(self, prefix="tr", available=True, boom=None):
        self.prefix = prefix
        self._available = available
        self._boom = boom
        self.seen = []

    def available(self):
        return (True, "ready") if self._available else (False, "weights missing")

    def translate_many(self, texts, target):
        if self._boom is not None:
            raise self._boom
        self.seen.extend(texts)
        return [f"{self.prefix}[{target}]: {text}" for text in texts]


class FakeCapture:
    """A capture device that hands back the same loud window every time."""

    def __init__(self, seconds=12.0, amplitude=0.2):
        self.total_frames = int(seconds * SAMPLE_RATE)
        self._window = np.full(self.total_frames, amplitude, dtype=np.float32)
        self.level = amplitude

    def read_tail(self, seconds):
        return self._window[-int(seconds * SAMPLE_RATE):]

    def write_frames(self, frames):
        self.total_frames += len(frames)


class FakeModel:
    """Stands in for a loaded WhisperModel, returning one finished sentence.

    The word sits a few seconds back in the 12 s window, where a real session
    would freeze it: just inside the commit margin and well inside the band
    the engine accepts first-time words from.
    """

    def __init__(self, text="Hello there."):
        self.text = text

    def transcribe(self, audio, **kwargs):
        segment = types.SimpleNamespace(
            text=self.text,
            avg_logprob=-0.2,
            no_speech_prob=0.1,
            end=8.0,
            words=[types.SimpleNamespace(word=self.text, start=7.0, end=7.5)],
        )
        return [segment], types.SimpleNamespace(duration=1.0)


def wait_for(predicate, timeout=3.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


#the runner

def test_runner_translates_submitted_sentences_in_order():
    runner = TranslationRunner("hi", translator=FakeTranslator())
    try:
        runner.submit("First one.")
        runner.submit("Second one.")
        assert wait_for(lambda: len(runner.snapshot()[1]) == 2)
        latest, pairs = runner.snapshot()
        assert latest == "tr[hi]: Second one."
        assert pairs == [("First one.", "tr[hi]: First one."),
                         ("Second one.", "tr[hi]: Second one.")]
    finally:
        runner.stop()


def test_runner_ignores_empty_text():
    translator = FakeTranslator()
    runner = TranslationRunner("hi", translator=translator)
    try:
        runner.submit("   ")
        time.sleep(0.3)
        assert translator.seen == [] and runner.snapshot() == ("", [])
    finally:
        runner.stop()


def test_runner_reports_an_unavailable_backend_once_and_disables_itself():
    errors = []
    translator = FakeTranslator(available=False)
    runner = TranslationRunner("hi", translator=translator, on_error=errors.append)
    try:
        runner.submit("One.")
        assert wait_for(lambda: len(errors) == 1)
        runner.submit("Two.")
        time.sleep(0.3)
        assert len(errors) == 1  # reported once, not per sentence
        assert translator.seen == []  # nothing was sent to a dead backend
        assert runner.snapshot() == ("", [])
    finally:
        runner.stop()


def test_runner_reports_a_failing_translation():
    errors = []
    runner = TranslationRunner("hi", translator=FakeTranslator(boom=RuntimeError("no gpu")),
                               on_error=errors.append)
    try:
        runner.submit("One.")
        assert wait_for(lambda: errors and "no gpu" in errors[0])
    finally:
        runner.stop()


#the engine

def test_translation_is_off_unless_requested():
    engine = LiveEngine(model_size="tiny.en")
    assert engine.translate_to == ""
    assert engine._translation is None
    engine.close()  # must be safe with nothing running


def test_engine_translates_each_finished_sentence(monkeypatch):
    from captioner import live as live_mod

    monkeypatch.setattr(live_mod.transcribe, "get_model", lambda *a, **kw: FakeModel())
    monkeypatch.setattr(live_mod.transcribe, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(live_mod.transcribe, "resolve_compute_type", lambda device: "int8")
    translator = FakeTranslator()
    engine = LiveEngine(model_size="tiny.en", translate_to="hi", translator=translator)
    capture = FakeCapture()
    try:
        engine.decode_once(capture)
        assert wait_for(lambda: engine._translation.snapshot()[0] != "")
        engine._sync_translation()
        assert translator.seen == ["Hello there."]
        assert engine.state.translation == "tr[hi]: Hello there."
        assert engine.state.translated_pairs == [("Hello there.", "tr[hi]: Hello there.")]
        # The caption text itself is untouched by translation.
        assert "Hello there." in engine.state.committed
    finally:
        engine.close()


def test_engine_survives_a_translation_backend_that_cannot_start(monkeypatch):
    from captioner import live as live_mod

    monkeypatch.setattr(live_mod.transcribe, "get_model", lambda *a, **kw: FakeModel())
    monkeypatch.setattr(live_mod.transcribe, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(live_mod.transcribe, "resolve_compute_type", lambda device: "int8")
    errors = []
    engine = LiveEngine(model_size="tiny.en", translate_to="hi",
                        translator=FakeTranslator(available=False),
                        on_translation_error=errors.append)
    capture = FakeCapture()
    try:
        engine.decode_once(capture)
        assert wait_for(lambda: errors)
        engine._sync_translation()
        assert engine.state.translation == ""
        assert engine.state.committed  # captions kept working
    finally:
        engine.close()


def test_live_state_defaults_keep_translation_off():
    state = LiveState(committed="hello")
    assert state.translation == "" and state.translated_pairs == []
    assert state.text == "hello"


def test_config_can_turn_translation_on_for_headless_sessions(monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRANSLATE_TO", "es")
    engine = LiveEngine(model_size="tiny.en", translator=FakeTranslator())
    try:
        assert engine.translate_to == "es"
        assert engine._translation is not None
    finally:
        engine.close()


#the UI

def test_overlay_shows_the_translation_only_when_there_is_one(qapp):
    ov = LiveOverlay(persist_settings=False)
    ov.update_state(LiveState(committed="Hello there.", translation="namaste"))
    assert ov.lbl_translation.text() == "namaste"
    assert ov.lbl_translation.isHidden() is False

    ov.update_state(LiveState(committed="Hello there.", translation=""))
    assert ov.lbl_translation.text() == ""
    assert ov.lbl_translation.isHidden() is True


def test_overlay_idle_hides_the_translation_line(qapp):
    ov = LiveOverlay(persist_settings=False)
    ov.update_state(LiveState(committed="Hi.", translation="namaste"))
    ov.show_idle()
    assert ov.lbl_translation.text() == ""
    assert ov.lbl_translation.isHidden() is True


def test_transcript_window_reads_as_a_bilingual_transcript(qapp):
    window = TranscriptWindow(persist_settings=False)
    window.update_state(LiveState(
        full_text="Hello there. More.",
        partial="Still typing",
        translated_pairs=[("Hello there.", "namaste"), ("More.", "aur")],
    ))
    text = window.view.toPlainText()
    assert "Hello there.\n    namaste" in text
    assert "More.\n    aur" in text
    assert text.rstrip().endswith("Still typing")


def test_transcript_window_is_unchanged_without_translation(qapp):
    window = TranscriptWindow(persist_settings=False)
    window.update_state(LiveState(full_text="Hello there.", partial="More"))
    assert window.view.toPlainText() == "Hello there. More"
