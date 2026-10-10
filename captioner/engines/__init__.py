"""Pluggable speech-recognition and translation backends.

Importing this package registers the shipped backends, so the rest of the app
only ever names one and asks for it:

    from captioner import engines
    engine = engines.asr()                       # default, or CAPTIONER_ASR_ENGINE
    words = engine.transcribe_words(path, model_size="small.en")

    translator = engines.mt()                    # default, or CAPTIONER_MT_ENGINE
    lines = translator.translate_many(["hello"], "hi")

Interfaces (deliberately small, both duck-typed):

ASR backend
    name                          short id
    describe()                    one line for the UI
    available() -> (bool, str)    usable right now? if not, why not
    transcribe_words(path, *, model_size=None, on_progress=None,
                     on_stage=None, should_cancel=None) -> list[Word]

MT backend
    name / describe() / available()
    translate_many(texts, target) -> list[str]

Shipped backends:
    asr  faster-whisper   local CTranslate2 models, the default
    asr  openai           the audio/transcriptions API, or any compatible
                          server (faster-whisper-server, LM Studio, Groq)
    mt   m2m100           local M2M100 through transformers, the default
    mt   openai           chat completions, or any compatible endpoint
"""
# Importing the backend modules is what registers them.
from . import local, openai_compat  # noqa: F401  (import for side effect)
from .registry import (
    ASR,
    KINDS,
    MT,
    EngineError,
    UnknownEngine,
    create,
    describe_all,
    names,
    probe,
    register,
    resolve_name,
)

__all__ = [
    "ASR",
    "MT",
    "KINDS",
    "EngineError",
    "UnknownEngine",
    "asr",
    "mt",
    "asr_names",
    "mt_names",
    "describe_asr",
    "describe_mt",
    "probe_asr",
    "probe_mt",
    "register",
    "create",
    "names",
    "resolve_name",
    "local",
    "openai_compat",
]


def asr(name: str = None):
    """The speech-recognition backend to use (default: local faster-whisper)."""
    return create(ASR, name)


def mt(name: str = None):
    """The translation backend to use (default: local M2M100)."""
    return create(MT, name)


def asr_names():
    return names(ASR)


def mt_names():
    return names(MT)


def describe_asr():
    return describe_all(ASR)


def describe_mt():
    return describe_all(MT)


def probe_asr():
    return probe(ASR)


def probe_mt():
    return probe(MT)
