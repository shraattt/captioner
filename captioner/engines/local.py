"""The local backends: faster-whisper for ASR, M2M100 for translation.

Both wrap code that already existed and keep working the same way; the point
of the wrappers is that callers no longer import ``transcribe`` or
``translate`` directly. Everything heavier than stdlib is imported inside a
method, so importing this module costs nothing and a machine that only ever
uses an API backend never loads torch.
"""
from collections.abc import Sequence


class LocalWhisperEngine:
    """faster-whisper (CTranslate2) running on this machine.

    Lookups go through the module rather than from-imports: the project's own
    tests replace ``captioner.transcribe.get_model`` with a fake, and that has
    to keep working through the registry.
    """

    name = "faster-whisper"

    def describe(self) -> str:
        return "local faster-whisper (CTranslate2), GPU or CPU"

    def available(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - report, never raise from a probe
            return False, f"faster-whisper not installed ({type(exc).__name__})"
        from .. import transcribe

        device = transcribe.resolve_device()
        return True, f"ready on {device}"

    def transcribe_words(
        self,
        audio_path: str,
        model_size: str = None,
        on_progress=None,
        on_stage=None,
        should_cancel=None,
        language: str | None = None,
        on_language=None,
    ):
        """Timed words for a media file (see ``transcribe.transcribe_words``)."""
        from .. import transcribe

        return transcribe.transcribe_words(
            audio_path,
            model_size,
            on_progress=on_progress,
            on_stage=on_stage,
            should_cancel=should_cancel,
            language=language,
            on_language=on_language,
        )

    def model_is_cached(self, model_size: str) -> bool:
        from .. import transcribe

        return transcribe.model_is_cached(model_size)


class LocalM2M100Engine:
    """M2M100 through transformers, with the project's on-disk cache."""

    name = "m2m100"

    def describe(self) -> str:
        return "local M2M100 (transformers), on-disk cache"

    def __init__(self, translator=None):
        self._translator = translator

    def _translator_or_default(self):
        if self._translator is None:
            from ..translate import Translator

            self._translator = Translator()
        return self._translator

    def available(self) -> tuple[bool, str]:
        try:
            from ..translate import Translator

            translator = self._translator or Translator()
            if translator.available():
                return True, "ready"
            size = "~1.8 GB"
            return False, f"weights not downloaded yet ({size} on first use)"
        except Exception as exc:  # noqa: BLE001 - report, never raise from a probe
            return False, f"{type(exc).__name__}: {exc}"

    def translate_many(self, texts: Sequence[str], target: str,
                       source: str | None = None) -> list[str]:
        """Translate into ``target``, reading ``source`` (``None`` = English).

        The source language has to reach the underlying model: M2M100 is not a
        language detector, so dropping this argument would silently translate
        every transcript as if it were English.
        """
        return self._translator_or_default().translate_many(list(texts), target, source)

    def languages(self) -> dict[str, str]:
        """Target codes this backend offers, with display names."""
        from ..translate import LANGUAGES

        return dict(LANGUAGES)


def _register() -> None:
    from .registry import ASR, MT, register

    register(ASR, "faster-whisper", LocalWhisperEngine,
             describe="local faster-whisper (CTranslate2), GPU or CPU", default=True)
    register(MT, "m2m100", LocalM2M100Engine,
             describe="local M2M100 (transformers), on-disk cache", default=True)


_register()
