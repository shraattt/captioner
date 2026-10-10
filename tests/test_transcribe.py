"""Transcriber unit tests — a fake model stands in for faster-whisper.

These cover the parts of the wrapper that are ours: dropping blank word
tokens, mapping word timestamps, passing the right decode options, and
reporting progress as segments stream out. No weights, no GPU, no audio.
"""
from captioner import transcribe


class _Word:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _Segment:
    def __init__(self, end, words):
        self.end, self.words = end, words


class _Info:
    duration = 10.0


class _FakeModel:
    def __init__(self, segments):
        self._segments = segments
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        return iter(self._segments), _Info()


def _patch_model(monkeypatch, model):
    monkeypatch.setattr(transcribe, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(transcribe, "resolve_compute_type", lambda device: "int8")
    monkeypatch.setattr(transcribe, "get_model", lambda *args, **kwargs: model)


def test_word_extraction_and_progress(monkeypatch):
    model = _FakeModel(
        [
            _Segment(4.0, [_Word(" Hello", 0.0, 0.2), _Word(" world.", 0.3, 0.5)]),
            _Segment(10.0, [_Word("", 9.0, 9.1), _Word(" Bye.", 9.5, 9.8)]),
        ]
    )
    _patch_model(monkeypatch, model)

    fractions = []
    words = transcribe.transcribe_words("x.wav", "tiny.en", on_progress=fractions.append)

    assert [w.text for w in words] == ["Hello", "world.", "Bye."]  # blank dropped
    assert [w.start for w in words] == [0.0, 0.3, 9.5]
    assert fractions == [0.4, 1.0]  # segment end / media duration
    assert model.kwargs["word_timestamps"] is True
    assert model.kwargs["vad_filter"] is True
    assert model.kwargs["language"] == "en"


def test_progress_is_optional(monkeypatch):
    model = _FakeModel([_Segment(1.0, [_Word("Hi", 0.0, 0.1)])])
    _patch_model(monkeypatch, model)

    assert len(transcribe.transcribe_words("x.wav", "tiny.en")) == 1


def test_progress_falls_back_to_model_default(monkeypatch):
    """No model_size given -> the configured batch model is used."""
    model = _FakeModel([_Segment(1.0, [_Word("Hi", 0.0, 0.1)])])
    seen = {}
    monkeypatch.setattr(transcribe, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(transcribe, "resolve_compute_type", lambda device: "int8")

    def fake_get_model(model_size, device, compute_type):
        seen["model_size"] = model_size
        return model

    monkeypatch.setattr(transcribe, "get_model", fake_get_model)
    transcribe.transcribe_words("x.wav")

    from captioner import config

    assert seen["model_size"] == config.BATCH_MODEL_SIZE


#model references (Phase 5: fine-tuned local checkpoints)
def test_size_names_map_to_their_repos():
    assert transcribe.model_ref("tiny.en") == "Systran/faster-whisper-tiny.en"
    assert transcribe.model_ref("distil-large-v3") == "Systran/faster-distil-whisper-large-v3"


def test_explicit_repo_ids_pass_through():
    assert transcribe.model_ref("Systran/faster-whisper-base.en") == \
        "Systran/faster-whisper-base.en"


def test_local_directories_are_used_as_is(tmp_path):
    """A converted fine-tune must not be mangled into a repo name."""
    directory = tmp_path / "base_en_noisy_ct2"
    directory.mkdir()
    (directory / "model.bin").write_bytes(b"x")
    assert transcribe.model_ref(str(directory)) == str(directory)
    assert transcribe.model_is_cached(str(directory)) is True


def test_incomplete_local_directory_is_not_cached(tmp_path):
    """Half-converted output must not look loadable."""
    directory = tmp_path / "half_ct2"
    directory.mkdir()
    assert transcribe.model_is_cached(str(directory)) is False
    assert transcribe._model_cache_dir(str(directory)) is None


def test_get_model_loads_a_local_checkpoint(tmp_path, monkeypatch):
    directory = tmp_path / "tiny_en_noisy_ct2"
    directory.mkdir()
    (directory / "model.bin").write_bytes(b"x")
    seen = {}

    class Fake:
        def __init__(self, ref, device, compute_type):
            seen.update(ref=ref, device=device, compute_type=compute_type)

    monkeypatch.setattr(transcribe, "WhisperModel", Fake)
    transcribe._model_cache.clear()
    try:
        transcribe.get_model(str(directory), "cpu", "int8")
    finally:
        transcribe._model_cache.clear()
    assert seen["ref"] == str(directory)
