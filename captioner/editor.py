"""Caption text editing: parse the preview pane, validate, re-export.

The preview pane is the editing surface, so its text has to be readable back.
This module owns both directions in one place — ``render`` produces the exact
text the pane shows (plus the block map playback highlighting needs) and
``parse_preview`` reads it back.

Only the *text* of a caption may change. Numbering and timestamps are compared
against the database and must match exactly, so a pasted-in block of subtitles
from somewhere else is rejected with a message the user can act on instead of
silently corrupting the output files.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import config, srt
from .segmenter import Block

# Time range inside one caption's timestamp line: "00:00:01,000 --> 00:00:02,500"
RANGE_SEP = "-->"


class CaptionParseError(ValueError):
    """The edited preview text cannot be read back as caption blocks."""


@dataclass
class CaptionDraft:
    """One caption as the editor sees it: identity, window, and free text."""

    seq: int
    start_ms: int
    end_ms: int
    text: str


@dataclass
class EditResult:
    """What an edit did, for the status bar."""

    changed: int
    total: int
    srt_path: Path
    vtt_path: Path


def drafts_from_rows(rows) -> list[CaptionDraft]:
    """Build drafts from ``captions`` table rows (sqlite3.Row or dict-like)."""
    return [
        CaptionDraft(
            seq=int(r["seq"]),
            start_ms=int(r["start_ms"]),
            end_ms=int(r["end_ms"]),
            text=(r["text"] or "").strip(),
        )
        for r in rows
    ]


def render(captions: Sequence[CaptionDraft]) -> tuple[str, list[tuple[int, int, int]]]:
    """Render captions as the preview pane shows them, plus a highlight map.

    Returns ``(text, spans)`` where each span is ``(start_ms, end_ms,
    block_number)`` and ``block_number`` is the document block holding that
    caption's body line — the block playback highlighting scrolls to and tints.
    """
    chunks: list[str] = []
    spans: list[tuple[int, int, int]] = []
    block_no = 0
    for caption in captions:
        text = (caption.text or "").strip()
        chunks.append(
            f"{caption.seq}\n"
            f"{srt.fmt_ms(caption.start_ms)} {RANGE_SEP} {srt.fmt_ms(caption.end_ms)}\n"
            f"{text}\n"
        )
        # block numbers: the caption number, the timestamp line, then the body
        spans.append((caption.start_ms, caption.end_ms, block_no + 2))
        block_no += text.count("\n") + 4  # 3 lines + the blank separator line
    return "\n".join(chunks), spans


def parse_preview(text: str) -> list[CaptionDraft]:
    """Parse the preview pane's text back into caption drafts.

    Raises :class:`CaptionParseError` with a caption number in the message for
    anything unreadable, so the UI can point at the problem.
    """
    lines = text.split("\n")
    drafts: list[CaptionDraft] = []
    i, total = 0, len(lines)

    while i < total:
        while i < total and not lines[i].strip():
            i += 1  # blank separator between captions
        if i >= total:
            break

        index_line = lines[i].strip()
        if not index_line.isdigit():
            raise CaptionParseError(
                f"expected a caption number, found {index_line!r} — "
                "revert and edit only the caption text"
            )
        seq = int(index_line)
        i += 1

        if i >= total:
            raise CaptionParseError(f"caption {seq} is missing its timestamp line")
        try:
            start_ms, end_ms = srt.parse_range(lines[i])
        except ValueError as exc:
            raise CaptionParseError(f"caption {seq}: {exc}") from exc
        i += 1

        body: list[str] = []
        while i < total and lines[i].strip():
            body.append(lines[i].strip())
            i += 1
        if not body:
            raise CaptionParseError(
                f"caption {seq} has no text — type a line, or revert"
            )

        drafts.append(
            CaptionDraft(seq=seq, start_ms=start_ms, end_ms=end_ms, text="\n".join(body))
        )

    if not drafts:
        raise CaptionParseError("no captions found in the editor")
    return drafts


def load_drafts(conn, job_id: int) -> list[CaptionDraft]:
    """Read a job's captions from the database as drafts."""
    rows = conn.execute(
        "SELECT seq, start_ms, end_ms, text FROM captions WHERE job_id = ? ORDER BY seq",
        (job_id,),
    ).fetchall()
    return drafts_from_rows(rows)


def apply_edits(conn, job_id: int, drafts: Sequence[CaptionDraft]) -> EditResult:
    """Save edited caption text and rewrite the job's SRT and VTT files.

    Validation happens before anything is written: the draft set must match the
    stored captions one-for-one with identical timings, and no caption may be
    empty. On success the ``captions`` table is updated and both subtitle files
    are replaced atomically.
    """
    stored_rows = conn.execute(
        "SELECT seq, start_ms, end_ms, text FROM captions WHERE job_id = ? ORDER BY seq",
        (job_id,),
    ).fetchall()
    if not stored_rows:
        raise CaptionParseError("this job has no captions to edit")

    stored = {int(r["seq"]): r for r in stored_rows}
    if [d.seq for d in drafts] != [int(r["seq"]) for r in stored_rows]:
        raise CaptionParseError(
            "caption numbers were changed — revert, then edit only the caption text"
        )

    changed = 0
    updates = []
    for draft in drafts:
        row = stored[draft.seq]
        if (draft.start_ms, draft.end_ms) != (int(row["start_ms"]), int(row["end_ms"])):
            raise CaptionParseError(
                f"caption {draft.seq}: timestamps must stay as they are — "
                "only the text can be edited"
            )
        text = draft.text.strip()
        if not text:
            raise CaptionParseError(f"caption {draft.seq} is empty — every caption needs text")
        if draft.end_ms <= draft.start_ms:
            raise CaptionParseError(
                f"caption {draft.seq}: the end time is not after the start time"
            )
        if text != (row["text"] or "").strip():
            changed += 1
        updates.append((text, job_id, draft.seq))

    with conn:  # one transaction: either every caption text lands, or none does
        conn.executemany(
            "UPDATE captions SET text = ? WHERE job_id = ? AND seq = ?", updates
        )

    out_dir = Path(config.OUTPUT_DIR) / f"job_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = [
        Block.from_text(d.text.strip(), d.start_ms / 1000.0, d.end_ms / 1000.0)
        for d in drafts
    ]
    srt_path = out_dir / "captions.srt"
    vtt_path = out_dir / "captions.vtt"
    write_atomic(srt_path, srt.to_srt(blocks))
    write_atomic(vtt_path, srt.to_vtt(blocks))
    return EditResult(changed=changed, total=len(drafts), srt_path=srt_path, vtt_path=vtt_path)


def write_atomic(path: Path, text: str) -> None:
    """Replace a subtitle file in one step.

    Media players read these files while they exist, so a half-written SRT
    would show up as a broken subtitle track; writing a temp file and
    renaming it means readers only ever see the complete version.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
