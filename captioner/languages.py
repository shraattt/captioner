"""Which languages the app can listen to, and which it can translate into.

Two different questions with two different answers:

* **spoken** - what the recogniser can be told to listen for. This is Whisper's
  language set (100 codes), so it includes Telugu, Cantonese and Hawaiian.
* **translation** - what the translator can write. This is M2M100's set, also
  100 codes, but *not the same 100*: M2M100 has Javanese as ``jv`` where Whisper
  says ``jw``, and it has no Telugu at all.

Keeping them apart is what lets the app say "I can transcribe this Telugu
recording but I cannot translate it yet" instead of failing halfway through.
The tables themselves live in :mod:`captioner.language_data`, generated from the
upstream sources by ``scripts/gen_language_tables.py``.
"""
from .language_data import SPOKEN_LANGUAGES, TRANSLATION_LANGUAGES

__all__ = [
    "SPOKEN_LANGUAGES",
    "TRANSLATION_LANGUAGES",
    "choices",
    "spoken_language",
    "spoken_name",
    "translation_language",
    "translation_name",
]

# What to suggest in an error message: the languages an English-speaking user is
# most likely to be reaching for, so a typo gets a useful nudge.
_EXAMPLES = ("en", "hi", "es", "zh", "te")


def _table(kind: str) -> dict[str, str]:
    if kind == "spoken":
        return SPOKEN_LANGUAGES
    return TRANSLATION_LANGUAGES


def choices(kind: str = "spoken") -> list[tuple[str, str]]:
    """``[(code, name), ...]`` sorted by name - what a picker should show."""
    table = _table(kind)
    return sorted(table.items(), key=lambda kv: (kv[1].casefold(), kv[0]))


def spoken_name(code: str) -> str:
    """Display name for a spoken-language code (falls back to the code)."""
    return SPOKEN_LANGUAGES.get(code or "", code or "")


def translation_name(code: str) -> str:
    """Display name for a translation target (falls back to the code)."""
    return TRANSLATION_LANGUAGES.get(code or "", code or "")


def _normalise(code: str | None) -> str | None:
    """Trim/lowercase a code, mapping ``None`` and ``""`` to ``None``."""
    if code is None:
        return None
    text = str(code).strip().lower().replace("_", "-")
    return text or None


def _validate(kind: str, code: str | None, label: str) -> str | None:
    text = _normalise(code)
    if text is None:
        return None
    table = _table(kind)
    if text not in table:
        # A region tag ("en-GB") or a near-miss ("jp") should not be a dead end:
        # the base code is what both engines actually accept.
        base = text.split("-")[0]
        if base in table:
            return base
        raise ValueError(
            f"unknown {label} {code!r}; use an ISO code such as "
            f"{', '.join(repr(c) for c in _EXAMPLES)} "
            f"(see the language list for all {len(table)})"
        )
    return text


def spoken_language(code: str | None) -> str | None:
    """Validate a spoken-language code; ``None``/empty means "detect it".

    Accepts a region tag (``en-GB`` resolves to ``en``) and raises
    :class:`ValueError` for anything the recogniser would not understand, so a
    typo fails at the queue step rather than mid-transcription.
    """
    return _validate("spoken", code, "spoken language")


def translation_language(code: str | None) -> str | None:
    """Validate a translation target; ``None``/empty means "no translation"."""
    return _validate("translation", code, "target language")
