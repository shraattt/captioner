"""Unit tests for SRT/VTT writers."""
import pytest

from captioner.segmenter import Block, Word
from captioner.srt import fmt_ms, parse_ms, parse_range, to_srt, to_vtt


def make_blocks():
    b1 = Block([Word("Hello", 0.0, 0.5), Word("world.", 0.6, 1.0)])
    b2 = Block([Word("Second", 2.0, 2.4), Word("line.", 2.5, 2.99999)])
    return [b1, b2]


def test_srt_structure_and_numbering():
    out = to_srt(make_blocks())
    lines = out.strip().splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:01,000"
    assert lines[2] == "Hello world."
    assert lines[4] == "2"
    # 2.99999s rounds to the whole second 3.000 (rounding-carry path);
    # block starts at its first word (2.0s).
    assert lines[5] == "00:00:02,000 --> 00:00:03,000"


def test_vtt_header_and_dots():
    out = to_vtt(make_blocks())
    lines = out.strip().splitlines()
    assert lines[0] == "WEBVTT"
    assert lines[1] == ""  # blank line after the header
    assert lines[2] == "00:00:00.000 --> 00:00:01.000"
    assert "-->" in out
    assert "," not in out.split("WEBVTT")[1]


def test_zero_and_negative_clamped():
    b = Block([Word("Edge", -1.0, 0.0)])
    assert "00:00:00,000 --> 00:00:00,000" in to_srt([b])


def test_hour_rollover():
    b = Block([Word("Long", 3599.9996, 3599.9996)])
    out = to_srt([b])
    # 3599.9996 rounds to 01:00:00,000 with no negative or overflow.
    assert "01:00:00,000 --> 01:00:00,000" in out


def test_fmt_ms_helper_matches_writer_output():
    assert fmt_ms(0) == "00:00:00,000"
    assert fmt_ms(999) == "00:00:00,999"
    assert fmt_ms(3_661_500, ".") == "01:01:01.500"


@pytest.mark.parametrize("ms", [0, 999, 1_000, 61_500, 3_661_500, 3_599_999, 36_000_000])
def test_parse_ms_is_the_inverse_of_fmt_ms(ms):
    assert parse_ms(fmt_ms(ms)) == ms
    # WebVTT spells the same instant with a dot separator
    assert parse_ms(fmt_ms(ms, ".")) == ms


def test_parse_ms_accepts_stray_whitespace():
    assert parse_ms("  00:00:02,500  ") == 2500


def test_parse_ms_handles_a_missing_hours_field():
    assert parse_ms("02:03.250") == 123_250


@pytest.mark.parametrize("bad", ["", "nonsense", "12:ab", "1:2:3:4", "00:-1,000", "aa:bb:cc"])
def test_parse_ms_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_ms(bad)


def test_parse_range_splits_a_timestamp_line():
    assert parse_range("00:00:01,250 --> 00:00:02,999") == (1250, 2999)


def test_parse_range_requires_the_arrow():
    with pytest.raises(ValueError):
        parse_range("00:00:01,250 00:00:02,999")


def test_written_srt_can_be_parsed_back():
    out = to_srt(make_blocks())
    lines = [line for line in out.splitlines() if "-->" in line]
    assert [parse_range(line) for line in lines] == [(0, 1000), (2000, 3000)]
