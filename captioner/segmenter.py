"""Caption segmentation: turn word-level timestamps into subtitle blocks.

This is a constrained greedy grouping problem. Whisper gives us every word
with start/end times; captions need short blocks (a line or two, a few
seconds each) that break at natural points. The greedy rule set:

  before appending the next word, close the current block when ANY of these holds:
    1. block text + word would exceed max_chars   (readability — hard cap)
    2. block duration + word would exceed max_seconds (dwell time — hard cap)
    3. gap since the last word   >= max_gap       (speaker paused — clean break)

  after appending a word, close the block when:
    4. word ends with sentence punctuation AND block >= min_chars
       AND the next word is not already pause-separated

Because checks 1–2 run *before* the append, emitted blocks never exceed the
limits; a single word longer than a limit still keeps its own block (we
never split mid-word).
"""
from dataclasses import dataclass
from typing import List, Optional

from . import config

SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？")


@dataclass
class Word:
    """One recognized word with its timing, in seconds.

    ``speaker`` is set by Phase 2 diarization (None in plain transcribes).
    """
    text: str
    start: float
    end: float
    speaker: Optional[str] = None


@dataclass
class Block:
    """A caption block: grouped words + display window, in seconds."""
    words: List[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @classmethod
    def from_text(cls, text: str, start: float, end: float) -> "Block":
        """Build a block from already-final text and a display window.

        Used when re-exporting subtitles after a caption edit: the user's text
        has no word timings to group, but the writers only need text + start +
        end. Line breaks inside ``text`` are preserved verbatim.
        """
        return cls(words=[Word(text=text.strip(), start=start, end=end)])


def _ends_sentence(word: Word) -> bool:
    return word.text.endswith(SENTENCE_ENDINGS)


def segment_words(
    words: List[Word],
    max_chars: int = config.SEGMENT_MAX_CHARS,
    max_seconds: float = config.SEGMENT_MAX_SECONDS,
    max_gap: float = config.SEGMENT_MAX_GAP,
    min_chars: int = config.SEGMENT_MIN_CHARS,
) -> List[Block]:
    """Group timed words into caption blocks (greedy, O(n)).

    Never splits mid-word; ensures start <= end and non-empty text on
    every emitted block. Words with None/invalid timing are skipped.
    """
    blocks: List[Block] = []
    current: List[Word] = []

    def flush() -> None:
        if current:
            blocks.append(Block(words=list(current)))
            current.clear()

    for i, word in enumerate(words):
        # Skip garbage timing defensively.
        if word.end is None or word.start is None or word.end < word.start:
            continue

        if current and (word.start - current[-1].end) >= max_gap:
            flush()  # rule 3: pause lands *between* blocks

        if current:
            # Rules 1-2 as hard caps: close BEFORE the word that would breach.
            new_len = len(" ".join(w.text for w in current)) + 1 + len(word.text)
            new_dur = word.end - current[0].start
            if new_len > max_chars or new_dur > max_seconds:
                flush()

        current.append(word)

        # Rule 4: sentence punctuation, but only if the next word isn't
        # already pause-separated (that case flushes on its own next loop).
        if _ends_sentence(word) and len(" ".join(w.text for w in current)) >= min_chars:
            next_word = words[i + 1] if i + 1 < len(words) else None
            if next_word is None or (next_word.start - word.end) < max_gap:
                flush()

    flush()
    return [b for b in blocks if b.text]
