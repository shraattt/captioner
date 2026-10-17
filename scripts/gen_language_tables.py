"""Regenerate ``captioner/language_data.py`` from the upstream sources.

Two tables, two engines, two upstreams:

* the **spoken** table is Whisper's own ``LANGUAGES`` map (fetched from
  ``openai/whisper``), filtered to the codes the installed faster-whisper
  accepts, so it can never offer a language the engine would reject;
* the **translation** table is M2M100's published language list (fetched from
  the model card on the Hub), filtered to the codes the installed transformers
  build actually knows.

Run it when either engine is upgraded::

    python scripts/gen_language_tables.py            # rewrite the data module
    python scripts/gen_language_tables.py --check    # exit 1 if it is stale

``--check`` is the CI-friendly form: it does not need the network to be useful
day to day (the committed file is the source of truth for tests), but it will
tell you when an engine upgrade has moved the goalposts.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = PROJECT_ROOT / "captioner" / "language_data.py"

WHISPER_URL = "https://raw.githubusercontent.com/openai/whisper/main/whisper/tokenizer.py"
M2M_URL = "https://huggingface.co/facebook/m2m100_418M/raw/main/README.md"

# Upstream names as published, corrected by hand where the source is a typo or
# an ISO synonym nobody uses. Keep this list short and justified: every entry is
# a deliberate divergence from the source of record.
NAME_OVERRIDES = {
    "el": "Greek",          # the M2M100 card spells it "Greeek"
    "or": "Odia",           # ISO 639-3 renamed Oriya in 2008; the card predates it
    "pa": "Punjabi",        # card: "Panjabi; Punjabi"
    "ca": "Catalan",        # card: "Catalan; Valencian"
    "gd": "Scottish Gaelic",  # card: "Gaelic; Scottish Gaelic"
    "ht": "Haitian Creole",  # card: "Haitian; Haitian Creole"
    "ro": "Romanian",       # card: "Romanian; Moldavian; Moldovan"
    "nl": "Dutch",          # card: "Dutch; Flemish"
    "ps": "Pashto",         # card: "Pushto; Pashto"
    "si": "Sinhala",        # card: "Sinhala; Sinhalese"
    "oc": "Occitan",        # card: "Occitan (post 1500)"
    "lb": "Luxembourgish",  # card: "Luxembourgish; Letzeburgesch"
}

_M2M_ENTRY = re.compile(r"([^(),]+?)\s*\(([a-z]{2,3})\)")


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - fixed https URLs
        return response.read().decode("utf-8")


def _local_spoken_codes() -> set[str]:
    """Codes the installed faster-whisper will actually accept."""
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    return set(_LANGUAGE_CODES)


def _local_translation_codes() -> set[str]:
    """Codes the installed transformers M2M100 tokenizer knows (best effort).

    ``FAIRSEQ_LANGUAGE_CODES`` is a dict of code sets - "m2m100" is the model we
    ship; "wmt21" is the shared task's slightly different set.
    """
    try:
        from transformers.models.m2m_100.tokenization_m2m_100 import (
            FAIRSEQ_LANGUAGE_CODES,
        )
    except Exception:  # noqa: BLE001 - transformers is optional at generation time
        return set()
    return set(FAIRSEQ_LANGUAGE_CODES["m2m100"])


def _title(name: str) -> str:
    return " ".join(part[:1].upper() + part[1:] for part in name.split())


def spoken_names(tokenizer_source: str, keep: set[str]) -> dict[str, str]:
    """Whisper's code -> name map, filtered to the codes we can ask for."""
    start = tokenizer_source.index("LANGUAGES = {")
    end = tokenizer_source.index("}", start) + 1
    table = ast.literal_eval(tokenizer_source[start + len("LANGUAGES = "):end])
    out = {}
    for code, name in table.items():
        if code in keep:
            out[code] = NAME_OVERRIDES.get(code) or _title(name)
    return dict(sorted(out.items(), key=lambda kv: (kv[1], kv[0])))


def translation_names(card: str, keep: set[str]) -> dict[str, str]:
    """M2M100's published language list, filtered to the tokenizer's own codes."""
    marker = card.index("## Languages covered")
    section = card[marker:]
    nxt = section.find("\n## ", 1)
    if nxt != -1:
        section = section[:nxt]
    out = {}
    for name, code in _M2M_ENTRY.findall(section):
        if code in keep:
            out[code] = NAME_OVERRIDES.get(code) or _title(name.strip())
    return dict(sorted(out.items(), key=lambda kv: (kv[1], kv[0])))


def render(spoken: dict[str, str], translation: dict[str, str]) -> str:
    def table(name: str, data: dict[str, str], indent: str = "    ") -> str:
        lines = [f"{name}: dict[str, str] = {{"]
        lines += [f'{indent}"{code}": "{text}",' for code, text in data.items()]
        lines.append("}")
        return "\n".join(lines)

    return f'''"""Language tables, generated from the upstream sources - do not edit by hand.

Regenerate with ``python scripts/gen_language_tables.py`` (see that script for
the two upstreams and the hand-checked corrections).

``SPOKEN_LANGUAGES`` is what the speech recogniser can be told to listen for;
``TRANSLATION_LANGUAGES`` is what the translator can write. They are different
sets on purpose - M2M100 has no Telugu, Whisper does - so a recording can be
transcribed from a language it cannot be translated into, and the UI says so.
"""

# Provenance, for the curious and for a future upgrade check.
SPOKEN_SOURCE = "{WHISPER_URL}"
TRANSLATION_SOURCE = "{M2M_URL}"

{table("SPOKEN_LANGUAGES", spoken)}

{table("TRANSLATION_LANGUAGES", translation)}
'''


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="fail instead of writing when the file is out of date")
    args = parser.parse_args(argv)

    spoken_keep = _local_spoken_codes() | set(NAME_OVERRIDES)
    translation_keep = _local_translation_codes() or set(NAME_OVERRIDES)

    spoken = spoken_names(_fetch(WHISPER_URL), spoken_keep)
    translation = translation_names(_fetch(M2M_URL), translation_keep | set(NAME_OVERRIDES))
    for label, table, expected in (("spoken", spoken, _local_spoken_codes()),
                                   ("translation", translation, _local_translation_codes())):
        missing = sorted(expected - set(table))
        if missing:
            print(f"warning: no name for {label} code(s): {', '.join(missing)}", file=sys.stderr)
    # A shared code should read the same in both pickers; Whisper's spelling wins
    # because it is the one the transcription side shows first.
    for code, name in spoken.items():
        if code in translation:
            translation[code] = name

    text = render(spoken, translation)
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            diff = difflib.unified_diff(current.splitlines(), text.splitlines(),
                                        "committed", "regenerated", lineterm="")
            print("\n".join(list(diff)[:40]))
            print(f"\n{TARGET.name} is out of date; re-run without --check")
            return 1
        print(f"{TARGET.name} is up to date ({len(spoken)} spoken, "
              f"{len(translation)} translation)")
        return 0

    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(PROJECT_ROOT)}: {len(spoken)} spoken, "
          f"{len(translation)} translation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
