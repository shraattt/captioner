"""Phase 2 diarization logic tests (no pyannote/audio files needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from captioner.diarize import SpeakerTurn, assign_speakers, speaker_blocks
from captioner.segmenter import Word


def test_assign_speakers_picks_max_overlap():
    turns = [
        SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
        SpeakerTurn(2.0, 4.0, "SPEAKER_01"),
    ]
    words = [
        Word(text="hello", start=0.2, end=0.8),
        Word(text="there", start=2.2, end=2.8),
        Word(text="friend", start=3.5, end=4.2),
    ]
    tagged = assign_speakers(words, turns)
    assert [w.speaker for w in tagged] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]


def test_assign_speakers_gap_word_gets_nearest_turn():
    """A word in a gap inherits the nearest turn's label (never None).

    pyannote boundaries never align perfectly with word ends; an unassigned
    word would needlessly split caption blocks.
    """
    turns = [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
        SpeakerTurn(8.0, 9.0, "SPEAKER_01"),
    ]
    words = [
        Word(text="near", start=2.0, end=2.5),   # closer to SPEAKER_00's end
        Word(text="far", start=7.8, end=8.1),    # overlapping SPEAKER_01's start
    ]
    tagged = assign_speakers(words, turns)
    assert [w.speaker for w in tagged] == ["SPEAKER_00", "SPEAKER_01"]
    assert assign_speakers(words, [])[0].speaker is None


def test_speaker_blocks_split_on_speaker_change():
    words = [
        Word(text="one", start=0.0, end=0.4, speaker="SPEAKER_00"),
        Word(text="two", start=0.5, end=0.9, speaker="SPEAKER_00"),
        Word(text="three", start=1.0, end=1.4, speaker="SPEAKER_01"),
    ]
    turns = [SpeakerTurn(0.0, 0.95, "SPEAKER_00"), SpeakerTurn(0.95, 2.0, "SPEAKER_01")]
    blocks = speaker_blocks(words, turns)
    speakers = [b.words[0].speaker for b in blocks]
    assert speakers == ["SPEAKER_00", "SPEAKER_01"]


def test_plain_words_still_work():
    """Words without speakers (diarize off) segment exactly as before."""
    from captioner.segmenter import segment_words

    words = [Word(text="hello", start=0.0, end=0.5), Word(text="world", start=0.6, end=1.0)]
    blocks = segment_words(words)
    assert blocks[0].text == "hello world"
    assert all(w.speaker is None for w in blocks[0].words)
