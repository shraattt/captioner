"""Tests for the language tables and the spoken-language plumbing.

Three things are worth pinning down, because they are easy to break silently:

* the generated tables still match what the installed engines accept, so the
  UI can never offer a language that would fail mid-job;
* a bad or unknown code is rejected at the queue step, with the region-tag and
  case leniency spelled out;
* the translator is told what it is reading, in its own language set, and its
  cache cannot confuse two readings of the same text.
"""
import pytest

from captioner import languages, translate
from captioner.languages import (
    SPOKEN_LANGUAGES,
    TRANSLATION_LANGUAGES,
    choices,
    spoken_language,
    spoken_name,
    translation_language,
    translation_name,
)
from captioner.segmenter import Block, Word

#the tables

def test_the_spoken_table_matches_the_installed_recogniser():
    """Whisper's language set is the authority for what can be listened for."""
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    assert set(SPOKEN_LANGUAGES) == set(_LANGUAGE_CODES)
    assert len(SPOKEN_LANGUAGES) == 100


def test_the_translation_table_matches_the_installed_tokenizer():
    from transformers.models.m2m_100.tokenization_m2m_100 import (
        FAIRSEQ_LANGUAGE_CODES,
    )

    assert set(TRANSLATION_LANGUAGES) == set(FAIRSEQ_LANGUAGE_CODES["m2m100"])
    assert len(TRANSLATION_LANGUAGES) == 100


def test_every_entry_has_a_name_that_is_not_the_code():
    for table in (SPOKEN_LANGUAGES, TRANSLATION_LANGUAGES):
        for code, name in table.items():
            assert len(code) in (2, 3), code
            assert name and name != code, code
            assert name[0].isupper(), (code, name)


def test_the_two_sets_are_not_the_same_set():
    """The API and the UI both rely on this being true, not assumed."""
    assert "te" in SPOKEN_LANGUAGES and "te" not in TRANSLATION_LANGUAGES
    assert "jv" in TRANSLATION_LANGUAGES and "jv" not in SPOKEN_LANGUAGES


def test_choices_are_sorted_by_display_name():
    pairs = choices("spoken")
    assert len(pairs) == len(SPOKEN_LANGUAGES)
    assert [name for _code, name in pairs] == sorted(
        (name for _code, name in pairs), key=str.casefold
    )
    assert choices("translation")[0][:2] != ("", "")


def test_names_fall_back_to_the_code():
    assert spoken_name("hi") == "Hindi"
    assert translation_name("te") == "te"  # M2M100 has no Telugu
    assert spoken_name("") == "" and translation_name(None) == ""


#validation

def test_empty_means_detect_and_no_translation():
    assert spoken_language(None) is None
    assert spoken_language("") is None
    assert spoken_language("   ") is None
    assert translation_language(None) is None


def test_codes_are_normalised_not_guessed():
    assert spoken_language("HI") == "hi"
    assert spoken_language(" en-GB ") == "en"  # a region tag resolves to its base
    assert translation_language("pt_BR") == "pt"


def test_an_unknown_code_is_rejected_with_an_example():
    with pytest.raises(ValueError) as excinfo:
        spoken_language("klingon")
    message = str(excinfo.value)
    assert "klingon" in message and "en" in message

    with pytest.raises(ValueError):
        translation_language("te")  # recognised, but the translator has no model


def test_translate_module_keeps_its_public_names():
    assert translate.LANGUAGES is TRANSLATION_LANGUAGES
    assert translate.language_code("hi") == "hi"
    assert translate.language_name("hi") == "Hindi"
    with pytest.raises(ValueError):
        translate.language_code("klingon")


#source-aware translation

class _Recorder(translate.Translator):
    """A translator that records the batch and the language it was read as."""

    def __init__(self, **kwargs):
        self.batches = []
        super().__init__(cache_path=kwargs.pop("cache_path", None),
                         translate_fn=lambda texts, target: [f"[{target}]{t}" for t in texts],
                         **kwargs)

    def _run(self, texts, target, source=None):
        self.batches.append((list(texts), target, source))
        return super()._run(texts, target, source)


def _blocks():
    return [Block(words=[Word(text="नमस्ते", start=0.0, end=1.0)])]


def test_translate_blocks_passes_the_spoken_language(tmp_path):
    translator = _Recorder(cache_path=tmp_path / "c.json")
    translate.translate_blocks(_blocks(), "fr", translator=translator, source="hi")
    assert translator.batches == [(["नमस्ते"], "fr", "hi")]


def test_english_is_the_default_when_nothing_is_known(tmp_path):
    translator = _Recorder(cache_path=tmp_path / "c.json")
    translate.translate_blocks(_blocks(), "fr", translator=translator)
    assert translator.batches == [(["नमस्ते"], "fr", "en")]


def test_a_language_the_translator_lacks_is_reported_not_defaulted(tmp_path):
    translator = _Recorder(cache_path=tmp_path / "c.json")
    with pytest.raises(ValueError) as excinfo:
        translate.translate_blocks(_blocks(), "fr", translator=translator, source="te")
    assert "te" in str(excinfo.value)
    assert translator.batches == []  # nothing was sent to the model


def test_translate_many_survives_a_backend_without_a_source_argument():
    """A third-party backend written before this existed must keep working."""
    class Old:
        def __init__(self):
            self.calls = []

        def translate_many(self, texts, target):
            self.calls.append((list(texts), target))
            return [f"[{target}]{t}" for t in texts]

    old = Old()
    out = translate.translate_blocks(_blocks(), "fr", translator=old, source="hi")
    assert old.calls == [(["नमस्ते"], "fr")]
    assert out[0].text == "[fr]नमस्ते"


def test_the_cache_separates_two_readings_of_the_same_text(tmp_path):
    """Same text, same target, different source: two different answers."""
    cache = tmp_path / "c.json"
    translator = translate.Translator(
        cache_path=cache, translate_fn=lambda texts, target: [f"{target}:{t}" for t in texts]
    )
    assert translator.cache_key("fr", "नमस्ते", None, None) == \
        translator.cache_key("fr", "नमस्ते", None, "en")
    assert translator.cache_key("fr", "नमस्ते", None, "hi") != \
        translator.cache_key("fr", "नमस्ते", None, None)
    # And the model override still namespaces on top of the source.
    assert translator.cache_key("fr", "x", "other/model", "hi") != \
        translator.cache_key("fr", "x", "other/model", "en")


def test_translate_subtitles_accepts_a_source_language(tmp_path):
    from captioner import srt

    job_dir = tmp_path / "job_1"
    job_dir.mkdir()
    (job_dir / "captions.srt").write_text(
        srt.to_srt(_blocks()), encoding="utf-8"
    )

    translator = _Recorder(cache_path=tmp_path / "c.json")
    paths = translate.translate_subtitles(
        job_dir, "fr", translator=translator, source="hi"
    )

    assert [p.name for p in paths] == ["captions.fr.srt", "captions.fr.vtt"]
    assert translator.batches == [(["नमस्ते"], "fr", "hi")]
    assert "[fr]नमस्ते" in (job_dir / "captions.fr.srt").read_text(encoding="utf-8")


def test_languages_module_reexports_the_tables():
    assert languages.SPOKEN_LANGUAGES is SPOKEN_LANGUAGES
    assert languages.TRANSLATION_LANGUAGES is TRANSLATION_LANGUAGES
