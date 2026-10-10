"""Unit tests for the caption segmentation engine."""
import pytest

from captioner.segmenter import Block, Word, segment_words


def w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def block_texts(words, **kwargs):
    return [b.text for b in segment_words(words, **kwargs)]


def test_empty_input_gives_no_blocks():
    assert segment_words([]) == []


def test_short_sentence_single_block():
    words = [w("Hello", 0.0, 0.4), w("world.", 0.5, 0.9)]
    assert block_texts(words) == ["Hello world."]


def test_breaks_on_char_limit():
    long_text = "a" * 10
    words = [w(long_text, 0.0, 0.2) for _ in range(10)]  # 10x10 = 100 chars
    blocks = segment_words(words, max_chars=40)
    assert all(len(b.text) <= 40 for b in blocks)  # hard cap never breached
    assert len(blocks) == 4  # 3+3+3+1 words (4 words join to 43 chars > 40)


def test_breaks_on_time_limit():
    words = [w(f"word{i}", i * 1.0, i * 1.0 + 0.9) for i in range(10)]
    blocks = segment_words(words, max_seconds=2.0)
    for b in blocks:
        assert b.duration <= 2.0 + 1e-6  # hard cap never breached
    assert len(blocks) == 5  # pairs: (0,1) (2,3) (4,5) (6,7) (8,9)


def test_breaks_at_long_pause():
    words = [w("Before", 0.0, 0.5), w("pause", 0.6, 0.9),
             w("After", 5.0, 5.4), w("pause.", 5.5, 5.9)]
    assert block_texts(words) == ["Before pause", "After pause."]


def test_breaks_on_sentence_punctuation():
    words = [w("One", 0.0, 0.3), w("two.", 0.4, 0.6),
             w("Three", 0.7, 0.9), w("four.", 1.0, 1.2)]
    assert block_texts(words, min_chars=6) == ["One two.", "Three four."]


def test_punctuation_break_respects_min_chars():
    # Each sentence is < min_chars=12, so no break until the pause/time rule.
    words = [w("Hi.", 0.0, 0.2), w("OK.", 0.25, 0.4), w("Yes.", 0.45, 0.6)]
    assert block_texts(words, min_chars=12) == ["Hi. OK. Yes."]


def test_invalid_words_are_skipped():
    words = [w("Good", 0.0, 0.4), w("bad", 2.0, 1.0), w("words.", 0.5, 0.9)]
    blocks = block_texts(words)
    assert blocks == ["Good words."]


def test_words_carry_timing_through():
    words = [w("A", 1.0, 1.2), w("b.", 1.3, 1.5)]
    blocks = segment_words(words)
    assert blocks[0].start == pytest.approx(1.0)
    assert blocks[0].end == pytest.approx(1.5)


def test_never_splits_mid_word():
    text = "w" * 30
    blocks = segment_words([w(text, 0.0, 1.0)], max_chars=10)
    assert [b.text for b in blocks] == [text]
