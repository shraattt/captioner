"""What this installation can actually do, as opposed to what it offers.

Two kinds of build ship from this repository:

* the **source checkout**, where every optional feature is installed;
* the **packaged slim build** (`scripts/package.py`), which leaves PyTorch out on
  purpose - it is 4.2 GB, and only local translation, speaker labels, word
  alignment and fine-tuning ever reach it.

Nothing here loads a model: `importlib.util.find_spec` only asks whether a module
is installed, which is a filesystem lookup. The window asks this module before it
offers a switch, so a feature that cannot run is *greyed out and explained*
rather than failing twenty minutes into a job; the job modules ask it when a
feature is reached anyway (the API and the CLI can enable one without any
window), and turn a bare `ImportError` into a sentence naming what is missing.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class Extra:
    """An optional feature, what it needs, and what still works without it."""

    key: str                      # the name the rest of the app looks it up by
    label: str                    # what a user sees
    modules: tuple[str, ...]      # every one of these has to be importable
    needs: str                    # what to install to get it
    fallback: str = ""            # what keeps working meanwhile


# Top-level module names only: checking a submodule would import its parent,
# which is the heavy import this module exists to avoid.
EXTRAS: tuple[Extra, ...] = (
    Extra(
        key="translate_local",
        label="Translate captions on this machine (M2M100)",
        modules=("torch", "transformers", "sentencepiece"),
        needs="PyTorch, transformers and sentencepiece - the full build "
              "(scripts/package.py --full) or `pip install -r requirements.txt`",
        fallback="Translating through an OpenAI-compatible API still works: "
                 "set CAPTIONER_MT_ENGINE=openai and CAPTIONER_API_BASE_URL.",
    ),
    Extra(
        key="speakers",
        label="Identify speakers",
        modules=("torch", "pyannote"),
        needs="PyTorch and pyannote.audio - the full build or "
              "`pip install pyannote.audio`",
        fallback="Captions are still written; only the [SPEAKER_00] labels are missing.",
    ),
    Extra(
        key="align",
        label="Align word timings",
        modules=("torch", "transformers"),
        needs="PyTorch and transformers - the full build or "
              "`pip install torch transformers`",
        fallback="Whisper's own word timings are used instead, which is what "
                 "runs when alignment is off.",
    ),
    Extra(
        key="finetune",
        label="Fine-tune a model",
        modules=("torch", "datasets", "accelerate"),
        needs="PyTorch, datasets and accelerate (the development environment)",
        fallback="The built-in models and converted checkpoints still work.",
    ),
)

_BY_KEY = {extra.key: extra for extra in EXTRAS}


def find(key: str) -> Extra:
    """Look up one feature by the short name the rest of the app uses."""
    return _BY_KEY[key]


def present(module: str) -> bool:
    """Is `module` importable here? Never imports it, so it is cheap."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # a broken parent package counts as absent
        return False


def missing(extra: Extra | str) -> tuple[str, ...]:
    """The modules of `extra` that are not installed."""
    extra = _BY_KEY[extra] if isinstance(extra, str) else extra
    return tuple(module for module in extra.modules if not present(module))


def available(extra: Extra | str) -> bool:
    """Can this installation run `extra` right now?"""
    return not missing(extra)


def unavailable() -> tuple[Extra, ...]:
    """Everything this installation cannot do, in the order above."""
    return tuple(extra for extra in EXTRAS if missing(extra))


def label(extra: Extra | str) -> str:
    """The switch's text: honest about not being available here.

    Kept as the label rather than a separate message so the reason is visible in
    a menu, in a screenshot and to a screen reader without hovering anything.
    """
    extra = _BY_KEY[extra] if isinstance(extra, str) else extra
    return extra.label if available(extra) else f"{extra.label} - not in this build"


def tooltip(extra: Extra | str, when_available: str) -> str:
    """A control's tooltip: its own explanation, or why it is greyed out."""
    extra = _BY_KEY[extra] if isinstance(extra, str) else extra
    if available(extra):
        return when_available
    return (f"Not included in this installation: {', '.join(missing(extra))} "
            f"is missing.\n{extra.fallback}\nInstall {extra.needs} to turn this on.")


def refusal(extra: Extra | str) -> str:
    """The sentence a *job* gets when it needs something this build lacks.

    Raised as a RuntimeError by the job modules; the queue stores it, so the
    reason ends up in the job's details instead of in a log nobody reads.
    """
    extra = _BY_KEY[extra] if isinstance(extra, str) else extra
    missing_names = ", ".join(missing(extra)) or "a required package"
    return (f"{extra.label} is not available in this installation: "
            f"{missing_names} is not installed. {extra.fallback} "
            f"To enable it, install {extra.needs}.")


def summary() -> str:
    """One plain sentence for an About box: what this build carries."""
    absent = unavailable()
    if not absent:
        return ("Everything is available in this installation: local translation, "
                "speaker labels, word alignment and fine-tuning.")
    names = ", ".join(extra.label for extra in absent)
    return (f"Not included in this installation: {names}. Everything else works, "
            "and the tooltip on a greyed-out option says what to install to get "
            "it back.")


def report() -> str:
    """A short, honest summary of this build, for logs and support questions."""
    lines = [f"Captioner build: {_kind()}", ""]
    lines.append("Available:")
    for extra in EXTRAS:
        if available(extra):
            lines.append(f"  yes  {extra.label}")
    absent = unavailable()
    if absent:
        lines.append("")
        lines.append("Not in this installation:")
        for extra in absent:
            lines.append(f"  no   {extra.label}")
            lines.append(f"         missing: {', '.join(missing(extra))}")
            if extra.fallback:
                lines.append(f"         instead: {extra.fallback}")
    return "\n".join(lines)


def _kind() -> str:
    """Which of the two builds this is, as far as anything can tell from here."""
    from . import config

    where = "packaged build" if config.frozen() else "source checkout"
    return f"{where}, data in {config.DATA_DIR}"
