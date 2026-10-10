"""Small display helpers: durations, ages, stage labels, media discovery."""

from datetime import datetime
from pathlib import Path

from .constants import MEDIA_EXTS


def _elapsed(started: str | None, finished: str | None = None) -> str:
    """Human-readable duration between two DB timestamps (to now if unfinished)."""
    if not started:
        return ""
    try:
        t0 = datetime.strptime(started, "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(finished, "%Y-%m-%d %H:%M:%S") if finished else datetime.now()
    except ValueError:
        return ""
    secs = max(0, int((t1 - t0).total_seconds()))
    if secs < 60:
        return f"{secs}s"
    mins, secs = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins:02d}m"


def _fmt_clock(ms: int) -> str:
    """Format milliseconds as m:ss, or h:mm:ss once past an hour."""
    total = max(0, int(ms)) // 1000
    mins, secs = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def _human_age(seconds: float) -> str:
    """Compact 'how long ago' text for the staleness hint."""
    secs = max(0, int(seconds))
    if secs < 60:
        return f"{secs}s"
    mins, secs = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins:02d}m"


def _stage_label(stage: str | None, progress: float | None) -> str:
    """What a running job is doing right now, with a percentage when known."""
    percent = f"{int(round((progress or 0.0) * 100))}%" if progress is not None else ""
    if stage == "downloading":
        return (f"downloading model {percent}").strip()
    if stage == "loading":
        return "loading model"
    if stage == "transcribing":
        return (f"transcribing {percent}").strip()
    return f"processing {percent}".strip()


def _expand_media(paths) -> list:
    """Expand a mix of files and folders into media files (recursively).

    Folders are the natural unit for a lecture series or a pile of voice
    notes — dropping one should not mean hand-selecting every file inside.
    """
    found = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(
                sorted(
                    child
                    for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() in MEDIA_EXTS
                )
            )
    return found
