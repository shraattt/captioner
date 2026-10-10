"""Tests for the model registry (captioner/models.py) — Phase 4/5 wiring."""
import json

import pytest

from captioner import config, models


def make_ct2(root, name, meta=None, with_bin=True):
    """A converted model dir, plus (optionally) the training metadata.

    Mirrors what the pipeline produces: ``data/ft/<name>_ct2`` from the
    converter and ``ft_meta.json`` written by training.
    """
    directory = root / f"{name}_ct2"
    directory.mkdir(parents=True)
    if with_bin:
        (directory / "model.bin").write_bytes(b"x")
    if meta is not None:
        (directory / "ft_meta.json").write_text(json.dumps(meta))
    return directory


def test_builtin_labels_resolve_to_themselves():
    for size in models.BUILTIN_SIZES:
        assert models.resolve(size) == size


def test_empty_label_falls_back_to_the_configured_live_model():
    assert models.resolve("") == config.LIVE_MODEL_SIZE
    assert models.resolve(None) == config.LIVE_MODEL_SIZE


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        models.resolve("definitely-not-a-model")


def test_an_explicit_path_beats_the_registry(tmp_path):
    target = tmp_path / "somewhere" / "model_dir"
    target.mkdir(parents=True)
    assert models.resolve(str(target)) == str(target)


def test_finetuned_models_are_discovered(tmp_path):
    make_ct2(tmp_path, "base_en_noisy",
             meta={"base_model": "openai/whisper-base.en", "train_hours": 4.0,
                   "eval_wer": 12.5})
    make_ct2(tmp_path, "half_converted", with_bin=False)  # incomplete -> skipped

    found = models.find_finetuned(tmp_path)
    assert [entry["label"] for entry in found] == ["base_en_noisy"]
    assert found[0]["meta"]["eval_wer"] == 12.5


def test_metadata_is_also_read_from_the_training_dir(tmp_path):
    from captioner import models as M

    (tmp_path / "base_en_noisy").mkdir()
    (tmp_path / "base_en_noisy" / "ft_meta.json").write_text(
        json.dumps({"base_model": "openai/whisper-base.en"})
    )
    directory = make_ct2(tmp_path, "base_en_noisy")  # meta only in the sibling
    assert models._meta_for(directory)["base_model"] == "openai/whisper-base.en"
    assert M.FINETUNED_SUFFIX == "_ct2"


def test_resolve_finds_a_converted_checkpoint(tmp_path, monkeypatch):
    directory = make_ct2(tmp_path, "tiny_en_noisy")
    monkeypatch.setattr(models, "finetuned_root", lambda: tmp_path)
    assert models.resolve("tiny_en_noisy") == str(directory)


def test_live_and_batch_choices_include_finetuned(tmp_path, monkeypatch):
    make_ct2(tmp_path, "base_en_noisy", meta={"base_model": "openai/whisper-base.en"})
    monkeypatch.setattr(models, "finetuned_root", lambda: tmp_path)

    live = models.live_choices()
    batch = models.batch_choices()
    assert live[: len(models.LIVE_BUILTIN)] == list(models.LIVE_BUILTIN)
    assert "base_en_noisy" in live and "base_en_noisy" in batch
    assert "distil-large-v3" in batch
    # tiny.en stays the first (default) live choice
    assert live[0] == "tiny.en"


def test_describe_mentions_the_training_run(tmp_path, monkeypatch):
    make_ct2(tmp_path, "base_en_noisy",
             meta={"base_model": "openai/whisper-base.en", "train_hours": 4.0,
                   "eval_wer": 9.25})
    monkeypatch.setattr(models, "finetuned_root", lambda: tmp_path)
    text = models.describe("base_en_noisy")
    assert "openai/whisper-base.en" in text
    assert "4.0 h" in text
    assert "9.2" in text
    assert models.describe("tiny.en") == "tiny.en"


def test_missing_root_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "finetuned_root", lambda: tmp_path / "nope")
    assert models.find_finetuned() == []
    assert models.live_choices() == list(models.LIVE_BUILTIN)
