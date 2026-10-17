"""Tests for captioner.features: what this installation can actually do.

The interesting case is the one this machine is not in: a packaged slim build
with no PyTorch. Availability is decided by `importlib.util.find_spec`, which is
a single seam, so the missing side is simulated by patching it rather than by
uninstalling 4 GB of wheels.
"""
import pytest

from captioner import features


@pytest.fixture
def without_torch(monkeypatch):
    """A slim build: torch, transformers, pyannote and friends are absent."""
    light = {"numpy", "faster_whisper", "soundcard"}
    monkeypatch.setattr(features, "present",
                        lambda module: module in light)


@pytest.fixture
def with_everything(monkeypatch):
    monkeypatch.setattr(features, "present", lambda _module: True)


def test_every_feature_is_named_and_explained():
    for extra in features.EXTRAS:
        assert extra.key and extra.label and extra.modules and extra.needs
        assert extra.fallback, f"{extra.key} has no fallback for the message"
        # The key is what other modules look the feature up by.
        assert features.find(extra.key) is extra


def test_a_slim_build_knows_what_it_is_missing(without_torch):
    keys = {extra.key for extra in features.unavailable()}
    assert keys == {"translate_local", "speakers", "align", "finetune"}
    assert features.available("translate_local") is False
    assert features.missing("speakers") == ("torch", "pyannote")


def test_a_complete_install_is_missing_nothing(with_everything):
    assert features.unavailable() == ()
    assert features.available("align") is True
    assert features.missing("finetune") == ()


def test_the_label_carries_the_reason(without_torch):
    """The point of the whole module: a switch says it is unavailable in place."""
    label = features.label("align")
    assert label.startswith("Align word timings")   # still findable/searchable
    assert "not in this build" in label


def test_the_label_is_untouched_when_the_feature_is_here(with_everything):
    assert features.label("align") == features.find("align").label
    assert "not in this build" not in features.label("align")


def test_a_tooltip_is_left_alone_when_the_feature_is_here(with_everything):
    advertised = "Re-time every word before the captions are grouped."
    assert features.tooltip("align", advertised) == advertised


def test_a_tooltip_says_why_it_is_greyed_out(without_torch):
    advertised = "Re-time every word before the captions are grouped."
    absent = features.tooltip("align", advertised)
    assert "missing" in absent and "torch" in absent
    assert advertised not in absent          # no promise it cannot keep
    assert features.find("align").needs in absent


def test_a_refusal_names_what_to_install_and_what_still_works(without_torch):
    message = features.refusal("speakers")
    assert "Identify speakers" in message
    assert "torch" in message and "pyannote" in message
    assert "pyannote.audio" in message          # how to get it back
    assert "Captions are still written" in message   # what happens instead


def test_the_refusal_is_a_runtime_error_a_job_can_report(without_torch):
    """The worker stores the exception's text in the job, so it must be a plain
    RuntimeError with the reason in it - not an ImportError with no context."""
    with pytest.raises(RuntimeError, match="not available in this installation"):
        raise RuntimeError(features.refusal("align"))


def test_summary_of_a_complete_install_says_so(with_everything):
    everything = features.summary()
    assert "Everything is available" in everything
    assert "Not included" not in everything


def test_summary_of_a_slim_build_lists_what_is_missing(without_torch):
    missing = features.summary()
    assert missing.startswith("Not included in this installation:")
    assert "Align word timings" in missing
    assert "greyed-out" in missing               # tells the user where to look


def test_the_report_lists_both_sides(without_torch):
    report = features.report()
    assert "Captioner build:" in report
    assert "Available:" in report
    assert "Not in this installation:" in report
    assert "missing: torch, transformers" in report


def test_a_broken_package_counts_as_absent(monkeypatch):
    """find_spec can raise for a half-installed package; that is not an error."""
    def explode(_module):
        raise ImportError("parent package is broken")

    monkeypatch.setattr(features.importlib.util, "find_spec", explode)
    assert features.present("anything") is False


def test_the_real_machine_is_described_correctly():
    """Whatever this machine has, `present` must agree with importing it."""
    import importlib.util

    for module in ("numpy", "torch"):
        assert features.present(module) == (importlib.util.find_spec(module) is not None)
