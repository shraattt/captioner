"""Tests for Phase 2 translation (captioner/translate.py) and the SRT parser.

The translator is injected everywhere, so nothing here downloads NLLB.
"""
import json

import pytest

from captioner import srt
from captioner import translate as T
from captioner.segmenter import Block, Word


def blocks():
    return [
        Block(words=[Word(text="Hello there.", start=0.0, end=1.5)]),
        Block(words=[Word(text="How are you?", start=1.5, end=3.0)]),
    ]


class FakeTranslator(T.Translator):
    """Records every call so caching can be asserted."""

    def __init__(self, mapping=None, cache_path=None):
        self.calls = []
        mapping = mapping or {}
        super().__init__(
            cache_path=cache_path,
            translate_fn=lambda texts, target: [
                mapping.get(t, f"[{target}]{t}") for t in texts
            ],
        )
        # the injection hook: keep recording without losing the fn
        original = self._translate_fn

        def recording(texts, target):
            self.calls.append((list(texts), target))
            return original(texts, target)

        self._translate_fn = recording


def test_translate_blocks_keeps_timing_and_order(tmp_path):
    translator = FakeTranslator(cache_path=tmp_path / "c.json")
    out = T.translate_blocks(blocks(), "hi", translator=translator)

    assert [b.text for b in out] == ["[hi]Hello there.", "[hi]How are you?"]
    assert [(b.start, b.end) for b in out] == [(0.0, 1.5), (1.5, 3.0)]
    assert translator.calls == [(["Hello there.", "How are you?"], "hi")]


def test_translate_blocks_preserves_speaker_labels(tmp_path):
    translator = FakeTranslator(cache_path=tmp_path / "c.json")
    diarized = [
        Block(words=[Word(text="[SPEAKER_00]: welcome back", start=0.0, end=1.0,
                          speaker="SPEAKER_00")]),
    ]
    out = T.translate_blocks(diarized, "hi", translator=translator)
    assert out[0].text == "[SPEAKER_00]: [hi]welcome back"
    assert translator.calls[0][0] == ["welcome back"]  # label never sent
    assert out[0].words[0].speaker == "SPEAKER_00"


def test_translation_is_cached_on_disk(tmp_path):
    cache = tmp_path / "c.json"
    first = FakeTranslator(cache_path=cache)
    T.translate_blocks(blocks(), "hi", translator=first)
    assert cache.exists()

    second = FakeTranslator(cache_path=cache)
    out = T.translate_blocks(blocks(), "hi", translator=second)
    assert [b.text for b in out] == ["[hi]Hello there.", "[hi]How are you?"]
    assert second.calls == []  # served entirely from the cache

    stored = json.loads(cache.read_text(encoding="utf-8"))
    assert len(stored) == 2


def test_cache_is_keyed_per_language(tmp_path):
    cache = tmp_path / "c.json"
    translator = FakeTranslator(cache_path=cache)
    T.translate_blocks(blocks(), "hi", translator=translator)
    T.translate_blocks(blocks(), "ta", translator=translator)
    # Two languages, two distinct entries, and the second pass did hit the model.
    assert len(json.loads(cache.read_text(encoding="utf-8"))) == 4


def test_long_text_is_chunked_and_rejoined(tmp_path):
    seen = []

    def fn(texts, target):
        seen.extend(texts)
        return [f"{target}:{len(t)}" for t in texts]

    translator = T.Translator(cache_path=tmp_path / "c.json", translate_fn=fn)
    long_text = " ".join(f"word{i}" for i in range(200))  # > MAX_CHARS
    out = translator.translate_many([long_text], "hi")
    assert len(seen) > 1
    assert all(len(chunk) <= T.MAX_CHARS for chunk in seen)
    assert out[0].startswith("hi:") and " " in out[0]


def test_unknown_language_is_rejected_before_any_work(tmp_path):
    translator = FakeTranslator(cache_path=tmp_path / "c.json")
    with pytest.raises(ValueError):
        translator.translate_many(["hello"], "klingon")
    assert translator.calls == []


def test_language_table_is_consistent():
    for code, name in T.LANGUAGES.items():
        assert len(code) in (2, 3), code
        assert name
    assert T.language_code("hi") == "hi"
    assert T.language_name("hi") == "Hindi"
    assert T.language_name("xx") == "xx"
    with pytest.raises(ValueError):
        T.language_code("te")  # M2M100 has no Telugu: absent, not silently broken


def test_translate_subtitles_writes_both_formats(tmp_path):
    job_dir = tmp_path / "job_1"
    job_dir.mkdir()
    (job_dir / "captions.srt").write_text(srt.to_srt(blocks()), encoding="utf-8")

    paths = T.translate_subtitles(
        job_dir, "hi", translator=FakeTranslator(cache_path=tmp_path / "c.json")
    )

    assert [p.name for p in paths] == ["captions.hi.srt", "captions.hi.vtt"]
    text = (job_dir / "captions.hi.srt").read_text(encoding="utf-8")
    assert "[hi]Hello there." in text
    assert "00:00:00,000 --> 00:00:01,500" in text
    vtt = (job_dir / "captions.hi.vtt").read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT")
    # the original artifact is untouched
    assert (job_dir / "captions.srt").read_text(encoding="utf-8") == srt.to_srt(blocks())


def test_translate_subtitles_requires_existing_captions(tmp_path):
    with pytest.raises(FileNotFoundError):
        T.translate_subtitles(tmp_path / "job_9", "hi",
                              translator=FakeTranslator(cache_path=tmp_path / "c.json"))


#the parser that makes re-translation possible
def test_parse_srt_round_trips():
    original = srt.to_srt(blocks())
    parsed = srt.parse_srt(original)
    assert [b.text for b in parsed] == ["Hello there.", "How are you?"]
    assert [(b.start, b.end) for b in parsed] == [(0.0, 1.5), (1.5, 3.0)]
    assert srt.to_srt(parsed) == original


def test_parse_srt_tolerates_messy_input():
    messy = (
        "WEBVTT\r\n\r\n"
        "00:00:01.000 --> 00:00:02.000\r\none\r\ntwo\r\n\r\n"
        "not a cue\r\n\r\n"
        "3\r\n00:00:03,000 --> 00:00:04,000\r\nthree"
    )
    parsed = srt.parse_srt(messy)
    assert [(b.text, b.start, b.end) for b in parsed] == [
        ("one two", 1.0, 2.0),
        ("three", 3.0, 4.0),
    ]


def test_decoding_defaults_guard_against_repetition(tmp_path):
    """Greedy decoding looped on short lines; the defaults must not be greedy."""
    translator = T.Translator(cache_path=tmp_path / "c.json",
                              translate_fn=lambda texts, target: list(texts))
    assert translator.num_beams >= 2
    assert translator.no_repeat_ngram >= 2


def test_model_override_does_not_reuse_the_default_cache(tmp_path):
    cache = tmp_path / "c.json"
    default = FakeTranslator(cache_path=cache)
    T.translate_blocks(blocks(), "hi", translator=default)

    other = FakeTranslator(cache_path=cache)
    other.model_name = "facebook/m2m100_1.2B"
    other.calls.clear()
    T.translate_blocks(blocks(), "hi", translator=other)
    assert other.calls  # a different model must re-translate, not read the cache


def test_model_can_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(T.MODEL_ENV, "facebook/m2m100_1.2B")
    translator = T.Translator(cache_path=tmp_path / "c.json",
                              translate_fn=lambda texts, target: list(texts))
    assert translator.model_name == "facebook/m2m100_1.2B"
    monkeypatch.setenv(T.MODEL_ENV, "")
    assert T.Translator(cache_path=tmp_path / "c.json").model_name == T.MODEL_NAME
