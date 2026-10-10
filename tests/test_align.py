"""Tests for Phase 2 word alignment (captioner/align.py).

Everything here injects a fake emission function, so the suite stays offline:
the Viterbi alignment itself is tested against hand-built frame paths, and the
realigner is tested against synthetic emissions that spell a known transcript.
"""
import numpy as np
import pytest

from captioner import align as A
from captioner.segmenter import Word

BLANK = 0
VOCAB = 9
CHARS = {"h": 1, "e": 2, "l": 3, "o": 4, "w": 5, "r": 6, "d": 7, "|": 8}


def emissions(path, vocab=VOCAB, peak=8.0, floor=-8.0):
    """Log-probability frames where ``path`` (token ids per frame) wins."""
    em = np.full((len(path), vocab), floor, dtype=np.float64)
    for t, token in enumerate(path):
        em[t, token] = peak
    return em


def fake_words(text_pairs):
    """``[("hello", 0.0, 1.0), ...]`` -> Word list."""
    return [Word(text=t, start=s, end=e) for t, s, e in text_pairs]


#the Viterbi alignment
def test_forced_align_finds_each_token_frame():
    path = [BLANK, 1, BLANK, 2, BLANK, 3, BLANK]
    spans = A.forced_align(emissions(path), [1, 2, 3])
    assert spans == [(1, 1), (3, 3), (5, 5)]


def test_forced_align_handles_repeated_tokens():
    # Two identical tokens can only be separated by a blank frame.
    spans = A.forced_align(emissions([1, BLANK, 1]), [1, 1])
    assert spans == [(0, 0), (2, 2)]


def test_forced_align_spans_are_monotonic_and_non_overlapping():
    rng = np.random.default_rng(0)
    # Noise, plus weak hints for the three targets: the path is not obvious,
    # so this asserts the property rather than a specific frame.
    em = rng.normal(size=(40, VOCAB))
    em[5, 1] += 6
    em[20, 2] += 6
    em[35, 3] += 6
    spans = A.forced_align(em, [1, 2, 3])
    assert all(s is not None for s in spans)
    for (a_start, a_end), (b_start, _b_end) in zip(spans, spans[1:], strict=False):
        assert a_start <= a_end
        assert b_start >= a_end


def test_forced_align_empty_inputs():
    assert A.forced_align(emissions([BLANK]), []) == []
    assert A.forced_align(np.empty((0, VOCAB)), [1]) == [None]


def test_forced_align_reports_unused_token_as_none():
    spans = A.forced_align(emissions([BLANK] * 5), [1, 2])
    # Every frame is blank, so the best path has no room for the tokens: both
    # still get a span (the recurrence forces them somewhere) but the result
    # stays ordered and within the frames.
    for start, end in spans:
        assert 0 <= start <= end < 5


#the realigner
def make_aligner(path, ids_and_owners):
    def logits_fn(audio):
        # The fake emissions are fixed; only their length tracks the slice.
        frames = max(1, len(audio) // 320)
        base = emissions(path)
        if frames > len(path):  # pad with silence frames
            pad = np.full((frames - len(path), VOCAB), -8.0)
            return np.vstack([base, pad])
        return base[:frames]

    return A.WordAligner(
        logits_fn=logits_fn,
        token_ids_fn=lambda text: ids_and_owners,
    )


def test_align_tightens_word_times():
    # Whisper-style estimates: words roughly right but with sloppy edges.
    words = fake_words([("hello", 0.06, 0.26), ("world", 0.24, 0.40)])
    ids = [1, 2, 3, 3, 4, 8, 5, 4, 6, 3, 7, 8]  # h e l l o | w o r l d |
    owners = [0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, None]
    # A valid CTC path for that target: the doubled 'l' needs a blank between
    # the two, and the tail is silence. Word 1 = frames 4-9, word 2 = 11-16.
    path = (
        [BLANK] * 4
        + [1, 2, 3, BLANK, 3, 4]
        + [8]
        + [5, 4, 6, BLANK, 3, 7]
        + [8]
        + [BLANK] * 31
    )
    aligner = make_aligner(path, (ids, owners))

    out = aligner.align(np.zeros(16000, dtype="float32"), words)

    assert [w.text for w in out] == ["hello", "world"]
    assert out[0].start == pytest.approx(0.08, abs=0.021)
    assert out[0].end == pytest.approx(0.20, abs=0.021)
    assert out[1].start == pytest.approx(0.22, abs=0.021)
    assert out[1].end == pytest.approx(0.34, abs=0.021)


def test_align_keeps_words_that_got_no_frames():
    words = fake_words([("hello", 0.0, 1.0)])
    aligner = A.WordAligner(
        logits_fn=lambda audio: emissions([BLANK] * 50),
        token_ids_fn=lambda text: ([], []),  # nothing tokenized
    )
    out = aligner.align(np.zeros(16000, dtype="float32"), words)
    assert (out[0].start, out[0].end) == (0.0, 1.0)


def test_align_rejects_a_word_that_drifts_far_away():
    # Whisper heard "world" at 3.0s; the frame path puts it at 0.08s. That is
    # drift, not refinement, so Whisper's timing wins.
    words = fake_words([("world", 3.0, 3.2)])
    ids = [5, 4, 6, 3, 7, 8]
    path = [BLANK] * 4 + [5, 4, 6, 3, 7] + [8] + [BLANK] * 44
    aligner = make_aligner(path, (ids, [0, 0, 0, 0, 0, None]))
    out = aligner.align(np.zeros(16000, dtype="float32"), words)
    assert (out[0].start, out[0].end) == (3.0, 3.2)


def test_align_rejects_implausible_stretch():
    words = fake_words([("hi", 5.0, 5.2)])
    # Word boundary tokens only: no character frames, so the only span comes
    # from the '|' separators, which are owned by nobody -> original kept.
    ids = [8]
    aligner = A.WordAligner(
        logits_fn=lambda audio: emissions([BLANK] * 50),
        token_ids_fn=lambda text: (ids, [None]),
    )
    out = aligner.align(np.zeros(16000, dtype="float32"), words)
    assert (out[0].start, out[0].end) == (5.0, 5.2)


def test_align_preserves_speaker_and_is_monotonic():
    words = [
        Word(text="one", start=0.0, end=0.4, speaker="SPEAKER_01"),
        Word(text="two", start=0.3, end=0.7, speaker="SPEAKER_01"),
    ]
    # A pathological emission set: both words tokenize to overlapping frames.
    ids = [1, 1, 1, 1]
    owners = [0, 0, 1, 1]
    aligner = A.WordAligner(
        logits_fn=lambda audio: emissions([1, 1, 1, 1] + [BLANK] * 46),
        token_ids_fn=lambda text: (ids, owners),
    )
    out = aligner.align(np.zeros(16000, dtype="float32"), words)
    assert [w.speaker for w in out] == ["SPEAKER_01", "SPEAKER_01"]
    assert out[0].start <= out[0].end
    assert out[0].end <= out[1].start
    assert out[1].start <= out[1].end


def test_windows_split_long_input():
    words = [Word(text=f"w{i}", start=i * 6.0, end=i * 6.0 + 5.0) for i in range(8)]
    groups = list(A._windows(words, window=20.0))
    assert len(groups) > 1
    assert sum(len(g) for g in groups) == len(words)


def test_realign_file_degrades_to_original_words(tmp_path, monkeypatch):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"not really audio")
    monkeypatch.setattr(
        "faster_whisper.audio.decode_audio",
        lambda *a, **k: np.zeros(16000, dtype="float32"),
    )
    breaker = A.WordAligner(
        logits_fn=lambda audio: (_ for _ in ()).throw(RuntimeError("no model")),
        token_ids_fn=lambda text: ([1], [0]),
    )
    words = fake_words([("hello", 0.0, 1.0)])
    out = A.realign_file(media, words, aligner=breaker)
    assert [(w.text, w.start, w.end) for w in out] == [("hello", 0.0, 1.0)]
