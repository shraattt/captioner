"""Dependency-compatibility guards.

faster-whisper 1.2.1 calls ``av.open(..., metadata_errors="ignore")``, an
argument PyAV removed in 19.0.0. An environment that resolves av>=19 still
imports fine and fails only when a real file is decoded — a mystifying
TypeError at the bottom of the stack. This test fails fast, with the reason,
if the pin in requirements.txt is ever lost.
"""
import wave

import av


def test_pyav_accepts_metadata_errors(tmp_path):
    """Exercise the exact call faster-whisper makes when decoding a file."""
    wav = tmp_path / "tone.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)  # 0.1s of silence

    with av.open(str(wav), mode="r", metadata_errors="ignore") as container:
        assert len(container.streams) == 1
