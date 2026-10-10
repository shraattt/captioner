"""Captioner — local-first AI captioning (batch + live overlay).

Phase 1 implements the batch pipeline: file drop -> SQLite job queue ->
worker process -> Whisper transcription -> caption segmentation -> SRT/VTT.
"""
__version__ = "0.1.0"
