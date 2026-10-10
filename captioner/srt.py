"""SRT and WebVTT subtitle writers.

Both formats share the same shape (index/timestamp line, text, blank line);
only the timestamp style differs: SRT uses comma milliseconds and a numeric
index, VTT uses dot milliseconds and a WEBVTT header.
"""
from typing import List, Sequence, Tuple

from .segmenter import Block


def _fmt(seconds: float, ms_sep: str) -> str:
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:  # rounding carry, e.g. 0.9996s
        secs += 1
        ms = 0
    if secs == 60:
        mins += 1
        secs = 0
    if mins == 60:
        hrs += 1
        mins = 0
    return f"{hrs:02d}:{mins:02d}:{secs:02d}{ms_sep}{ms:03d}"


def fmt_ms(ms: int, ms_sep: str = ",") -> str:
    """Format integer milliseconds as HH:MM:SS<sep>mmm.

    Public wrapper around _fmt so the UI can render timestamps exactly the
    way the subtitle files do (single source of formatting truth).
    """
    return _fmt(ms / 1000.0, ms_sep)


def parse_ms(text: str) -> int:
    """Parse a subtitle timestamp back to milliseconds (inverse of fmt_ms).

    Accepts both the SRT comma and the WebVTT dot millisecond separator, with
    or without an hours field. Raises ValueError on anything else, so callers
    get a message they can show the user instead of a silent zero.
    """
    raw = text.strip().replace(",", ".")
    if not raw:
        raise ValueError("empty timestamp")
    parts = raw.split(":")
    if len(parts) == 2:  # MM:SS.mmm
        parts.insert(0, "0")
    if len(parts) != 3:
        raise ValueError(f"bad timestamp {text.strip()!r}")
    try:
        hours, minutes = int(parts[0]), int(parts[1])
        seconds = float(parts[2])
    except ValueError as exc:
        raise ValueError(f"bad timestamp {text.strip()!r}") from exc
    if hours < 0 or minutes < 0 or seconds < 0:
        raise ValueError(f"negative timestamp {text.strip()!r}")
    return int(round((hours * 3600 + minutes * 60 + seconds) * 1000))


def parse_range(line: str) -> "tuple[int, int]":
    """Parse an ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` line into (start, end) ms."""
    if "-->" not in line:
        raise ValueError(f"missing '-->' in {line.strip()!r}")
    left, right = line.split("-->", 1)
    return parse_ms(left), parse_ms(right)


def _render(blocks: Sequence[Block], sep: str, numbered: bool, header: str,
            speaker_names: bool = False) -> str:
    lines: List[str] = []
    if header:
        lines.append(header)
        lines.append("")
    for i, block in enumerate(blocks, start=1):
        if numbered:
            lines.append(str(i))
        lines.append(
            f"{_fmt(block.start, sep)} --> {_fmt(block.end, sep)}"
        )
        text = block.text
        if speaker_names and block.words:
            speaker = getattr(block.words[0], "speaker", None)
            if speaker:
                text = f"[{speaker}]: {text}"
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def parse_srt(text: str) -> List[Block]:
    """Parse SRT (or WebVTT) text back into blocks — inverse of :func:`to_srt`.

    Tolerates CRLF line endings, a missing trailing blank line, and either
    millisecond separator, so a subtitle file touched by another editor still
    loads. Multi-line captions are joined with a space, and any speaker label
    stays part of the text (it is a label, not a separate field).
    """
    blocks: List[Block] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for chunk in normalized.split("\n\n"):
        lines = [line.strip() for line in chunk.split("\n") if line.strip()]
        if lines and lines[0].isdigit():
            lines = lines[1:]  # drop the sequence number
        if not lines or "-->" not in lines[0]:
            continue  # header, or trailing junk
        try:
            start_ms, end_ms = parse_range(lines[0])
        except ValueError:
            continue
        body = " ".join(lines[1:]).strip()
        if not body:
            continue
        blocks.append(Block.from_text(body, start_ms / 1000.0, end_ms / 1000.0))
    return blocks


def to_srt(blocks: Sequence[Block], speaker_names: bool = False) -> str:
    return _render(blocks, sep=",", numbered=True, header="", speaker_names=speaker_names)


def to_vtt(blocks: Sequence[Block], speaker_names: bool = False) -> str:
    return _render(blocks, sep=".", numbered=False, header="WEBVTT", speaker_names=speaker_names)
