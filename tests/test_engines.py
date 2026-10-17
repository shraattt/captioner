"""Tests for the pluggable engine registry.

Three layers:

* The registry itself: default resolution, the environment override, unknown
  names, and probing a backend that explodes.
* The API backends, driven through an injected transport so no network is
  touched: request shape, response parsing, error mapping.
* The wiring: the worker transcribes through whatever the registry selects,
  and the local backends still delegate to the original code.
"""
import json

import pytest

from captioner import engines, worker
from captioner.engines import registry as reg
from captioner.engines.local import LocalM2M100Engine, LocalWhisperEngine
from captioner.engines.openai_compat import (
    OpenAIChatTranslator,
    OpenAIWhisperEngine,
    _parse_numbered,
    words_from_response,
)
from captioner.segmenter import Word


class FakeTransport:
    """Records requests and replays canned (status, body) responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.timeouts = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if not self.responses:
            raise AssertionError("transport called more times than expected")
        status, body = self.responses.pop(0)
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        return status, body

    def payloads(self):
        return [json.loads(r.data.decode("utf-8")) for r in self.requests]


def verbose_response(*words):
    return {"text": " ".join(w[2] for w in words),
            "words": [{"word": w[2], "start": w[0], "end": w[1]} for w in words]}


class FakeEngine:
    """A registered backend that never touches a model."""

    name = "fake-asr"

    def __init__(self):
        self.calls = []

    def describe(self):
        return "a fake engine for tests"

    def available(self):
        return True, "always"

    def transcribe_words(self, audio_path, model_size=None, on_progress=None,
                         on_stage=None, should_cancel=None, language=None,
                         on_language=None, **_):
        self.calls.append((audio_path, model_size))
        return [Word(text="fake", start=0.0, end=0.5)]


#the registry

def test_local_backends_are_the_defaults():
    assert engines.resolve_name(engines.ASR) == "faster-whisper"
    assert engines.resolve_name(engines.MT) == "m2m100"


def test_environment_selects_the_engine(monkeypatch):
    monkeypatch.setenv("CAPTIONER_ASR_ENGINE", "openai")
    monkeypatch.setenv("CAPTIONER_MT_ENGINE", "openai")
    assert isinstance(engines.asr(), OpenAIWhisperEngine)
    assert isinstance(engines.mt(), OpenAIChatTranslator)


def test_an_explicit_name_beats_the_environment(monkeypatch):
    monkeypatch.setenv("CAPTIONER_ASR_ENGINE", "openai")
    assert engines.resolve_name(engines.ASR, "faster-whisper") == "faster-whisper"


def test_unknown_engine_lists_what_exists():
    with pytest.raises(reg.UnknownEngine) as excinfo:
        engines.asr("does-not-exist")
    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert "faster-whisper" in message and "openai" in message


def test_a_new_backend_can_be_registered(monkeypatch):
    engines.register(engines.ASR, "fake-asr-test", FakeEngine, describe="test backend")
    monkeypatch.setenv("CAPTIONER_ASR_ENGINE", "fake-asr-test")
    assert isinstance(engines.asr(), FakeEngine)
    assert engines.describe_asr()["fake-asr-test"] == "test backend"
    # The default stays first in the listing so the UI shows it as the default.
    assert engines.asr_names()[0] == "faster-whisper"


def test_describe_lists_both_shipped_backends():
    asr = engines.describe_asr()
    mt = engines.describe_mt()
    assert asr["faster-whisper"] and asr["openai"]
    assert mt["m2m100"] and mt["openai"]


def test_probe_reports_a_broken_backend_instead_of_raising():
    def boom():
        raise ValueError("bad configuration")

    engines.register(engines.ASR, "fake-broken-test", boom)
    report = engines.probe_asr()
    usable, reason = report["fake-broken-test"]
    assert usable is False
    assert "bad configuration" in reason


def test_unknown_kind_is_rejected():
    with pytest.raises(KeyError):
        engines.register("vision", "x", FakeEngine)


#the worker uses whatever the registry selects

def test_worker_transcribes_through_the_selected_engine(monkeypatch):
    fake = FakeEngine()
    engines.register(engines.ASR, "fake-worker-test", lambda: fake)
    monkeypatch.setenv("CAPTIONER_ASR_ENGINE", "fake-worker-test")
    words = worker.transcribe_words("clip.wav", "tiny.en")
    assert [w.text for w in words] == ["fake"]
    assert fake.calls == [("clip.wav", "tiny.en")]


def test_worker_still_accepts_a_patched_transcribe_words(monkeypatch):
    monkeypatch.setattr(worker, "transcribe_words", lambda *a, **kw: [Word("x", 0.0, 1.0)])
    assert worker.transcribe_words("x.wav")[0].text == "x"


#the local backends

def test_local_asr_engine_delegates_to_transcribe(monkeypatch):
    from captioner import transcribe

    seen = {}

    def fake(path, model_size=None, on_progress=None, on_stage=None,
             should_cancel=None, language=None, on_language=None, **_):
        seen.update(path=path, model_size=model_size, on_stage=on_stage)
        return [Word(text="hi", start=0.0, end=1.0)]

    monkeypatch.setattr(transcribe, "transcribe_words", fake)
    engine = LocalWhisperEngine()
    assert engine.transcribe_words("a.wav", "small.en", on_stage="stage")[0].text == "hi"
    assert seen == {"path": "a.wav", "model_size": "small.en", "on_stage": "stage"}


def test_local_mt_engine_uses_the_injected_translator():
    seen = []

    class FakeTranslator:
        def available(self):
            return True

        def translate_many(self, texts, target, source=None):
            seen.append((list(texts), target, source))
            return [f"{target}:{t}" for t in texts]

    engine = LocalM2M100Engine(translator=FakeTranslator())
    assert engine.translate_many(["one", "two"], "hi") == ["hi:one", "hi:two"]
    assert engine.available() == (True, "ready")
    # The source language must reach the model: dropping it here is what made
    # every transcript get read as English whatever it really was.
    assert engine.translate_many(["నమస్తే"], "fr", "te") == ["fr:నమస్తే"]
    assert seen == [(["one", "two"], "hi", None), (["నమస్తే"], "fr", "te")]


#the API ASR backend

def test_api_engine_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CAPTIONER_API_KEY", raising=False)
    usable, reason = OpenAIWhisperEngine().available()
    assert usable is False
    assert "OPENAI_API_KEY" in reason


def test_api_engine_uploads_the_file_and_parses_words(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"ID3fake")
    transport = FakeTransport((200, verbose_response((0.0, 0.5, "Hello"), (0.6, 1.1, "world"))))
    engine = OpenAIWhisperEngine(transport=transport)

    words = engine.transcribe_words(str(clip), "whisper-1")

    assert [(w.text, w.start, w.end) for w in words] == [
        ("Hello", 0.0, 0.5), ("world", 0.6, 1.1)
    ]
    request = transport.requests[0]
    assert request.full_url.endswith("/audio/transcriptions")
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("Content-type", "").startswith("multipart/form-data; boundary=")
    body = request.data
    assert b'name="model"' in body and b"whisper-1" in body
    assert b'filename="clip.mp3"' in body
    assert b"timestamp_granularities" in body


def test_api_engine_reports_the_api_error_message(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "bad-key")
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"x")
    transport = FakeTransport((401, {"error": {"message": "Incorrect API key provided"}}))
    with pytest.raises(engines.EngineError) as excinfo:
        OpenAIWhisperEngine(transport=transport).transcribe_words(str(clip))
    assert "401" in str(excinfo.value)
    assert "Incorrect API key provided" in str(excinfo.value)


def test_api_engine_refuses_a_file_over_the_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clip = tmp_path / "big.mp3"
    clip.write_bytes(b"x" * 64)
    engine = OpenAIWhisperEngine(transport=FakeTransport())
    monkeypatch.setattr(OpenAIWhisperEngine, "MAX_UPLOAD_BYTES", 8)
    with pytest.raises(engines.EngineError) as excinfo:
        engine.transcribe_words(str(clip))
    assert "the API limit is" in str(excinfo.value)


def test_api_engine_rejects_a_missing_file(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(engines.EngineError):
        OpenAIWhisperEngine(transport=FakeTransport()).transcribe_words("nope.mp3")


def test_api_engine_reports_a_network_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"x")

    def dead(request, timeout):
        raise OSError("connection refused")

    with pytest.raises(engines.EngineError) as excinfo:
        OpenAIWhisperEngine(transport=dead).transcribe_words(str(clip))
    assert "connection refused" in str(excinfo.value)


def test_segment_only_responses_are_apportioned():
    words = words_from_response(
        {"segments": [{"start": 0.0, "end": 3.0, "text": "one two three"}]}
    )
    assert [w.text for w in words] == ["one", "two", "three"]
    assert words[0].start == 0.0
    assert words[-1].end == pytest.approx(3.0)
    assert all(a.end <= b.start + 1e-9 for a, b in zip(words, words[1:], strict=False))


def test_a_response_without_timings_is_an_error():
    with pytest.raises(engines.EngineError) as excinfo:
        words_from_response({"text": "hello there"})
    assert "no word or segment timings" in str(excinfo.value)


#the API translation backend

def test_chat_translator_translates_a_numbered_batch(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transport = FakeTransport((200, {"choices": [{"message": {"content": "1. नमस्ते\n2. दुनिया"}}]}))
    translator = OpenAIChatTranslator(transport=transport)

    assert translator.translate_many(["Hello", "world"], "hi") == ["नमस्ते", "दुनिया"]
    payload = transport.payloads()[0]
    assert payload["model"] == "gpt-4o-mini"
    assert payload["temperature"] == 0
    assert "Hindi" in payload["messages"][0]["content"]
    assert payload["messages"][1]["content"] == "1. Hello\n2. world"


def test_chat_translator_retries_line_by_line_when_the_batch_loses_a_line(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transport = FakeTransport(
        (200, {"choices": [{"message": {"content": "1. one"}}]}),          # line 2 missing
        (200, {"choices": [{"message": {"content": "1. uno"}}]}),
        (200, {"choices": [{"message": {"content": "1. dos"}}]}),
    )
    assert OpenAIChatTranslator(transport=transport).translate_many(["one", "two"], "es") == [
        "uno", "dos"
    ]
    assert len(transport.requests) == 3


def test_chat_translator_handles_an_empty_input(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transport = FakeTransport()
    assert OpenAIChatTranslator(transport=transport).translate_many([], "hi") == []
    assert transport.requests == []


def test_parse_numbered_ignores_prose_and_keeps_order():
    assert _parse_numbered("Sure!\n1. a\n2) b\n", 2) == ["a", "b"]
    assert _parse_numbered("1. a", 2) is None
    assert _parse_numbered("", 1) is None
