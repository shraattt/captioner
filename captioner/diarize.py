"""Phase 2: speaker diarization ("who said what") via pyannote.audio.

``diarize(path)`` returns a list of ``(start, end, label)`` speaker turns for
an audio/video file. ``assign_speakers`` folds those turns into the timed
words from faster-whisper, producing speaker-tagged caption blocks for the
batch output (``[SPEAKER_00]: text`` lines in SRT/VTT).

pyannote needs the Hugging Face token (``hf_token()`` from config) on first
run: the speaker-segmentation model is gated, and accepting its terms once
per account is enough — the weights are then cached locally.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from . import config
from .segmenter import Block, Word, segment_words

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    start: float
    end: float
    label: str


def _token() -> str:
    token = config.hf_token()
    if not token:
        raise RuntimeError(
            "diarization needs a Hugging Face token in .env (HF_TOKEN=...). "
            "See .env.example."
        )
    return token


_pipeline_cache = None  # one pipeline per process; model loads are expensive


def _load_pipeline():
    """Build the classic speaker-diarization-3.1 pipeline on pyannote >= 4.

    pyannote 4 redirects the legacy ``speaker-diarization-3.1`` model id to
    the newer ``speaker-diarization-community-1`` repo, which is gated
    differently (a token accepted for 3.1 gets a 403) — and its constructor
    eagerly downloads a PLDA file from that repo even when the 3.1-style
    AgglomerativeClustering (which never uses PLDA) is selected.

    So: build the pipeline explicitly from the 3.1 recipe —
    ``segmentation-3.0`` + the ungated wespeaker embedding + Agglomerative
    clustering — with an uninitialized PLDA placeholder to satisfy the
    constructor, and the official 3.1 hyperparameters. Only repositories the
    token can already access are ever downloaded.
    """
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    from pyannote.audio.core.plda import PLDA
    from pyannote.audio.pipelines import SpeakerDiarization

    placeholder = PLDA.__new__(PLDA)  # never used by AgglomerativeClustering
    pipeline = SpeakerDiarization(
        segmentation="pyannote/segmentation-3.0",
        embedding="pyannote/wespeaker-voxceleb-resnet34-LM",
        clustering="AgglomerativeClustering",
        plda=placeholder,
        legacy=True,  # return a plain Annotation (turns), not DiarizeOutput
        token=_token(),
    )
    # hyperparameters straight from speaker-diarization-3.1's config.yaml
    pipeline.instantiate(
        {
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 12,
                "threshold": 0.7045654963945799,
            },
            "segmentation": {"min_duration_off": 0.0},
        }
    )
    _pipeline_cache = pipeline
    return pipeline


def _audio_input(path: str) -> dict:
    """A file as a pyannote waveform dict: {'waveform': (1,T), 'sample_rate'}.

    pyannote >= 4 reads files through torchcodec, whose Windows DLLs are
    unreliable; the dict form sidesteps it. Decoding uses faster-whisper's
    ffmpeg helper — exactly what batch transcription reads with, so any
    format the queue accepts works for diarization too.
    """
    import torch
    from faster_whisper.audio import decode_audio

    audio = decode_audio(path, sampling_rate=16000)
    waveform = torch.from_numpy(audio.astype("float32")).unsqueeze(0)  # (1, T)
    return {"waveform": waveform, "sample_rate": 16000}


def diarize(path: str, num_speakers: Optional[int] = None) -> List[SpeakerTurn]:
    """Speaker turns for *path*; downloads models on first ever use."""
    pipeline = _load_pipeline()
    try:  # keep diarization on the same GPU as Whisper when there is one
        import torch

        device = torch.device("cuda" if _torch_cuda() else "cpu")
        pipeline.to(device)
    except Exception as exc:  # pragma: no cover - CPU-only fallback
        logger.warning("could not move diarization to GPU: %s", exc)

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    annotation = pipeline(_audio_input(path), **kwargs)
    turns = [
        SpeakerTurn(start=float(turn.start), end=float(turn.end), label=str(label))
        for turn, _track, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: (t.start, t.end))
    logger.info("diarized %s: %d turns", path, len(turns))
    return turns


def _torch_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def assign_speakers(words: List[Word], turns: List[SpeakerTurn]) -> List[Word]:
    """Give each word the label of the speaker covering most of it.

    A word falling in a gap between turns (pyannote boundaries never align
    perfectly with word ends) gets the nearest turn's label — a word sits
    next to the speaker who was speaking around it, never unassigned.
    """
    out: List[Word] = []
    for w in words:
        best_label, best_overlap = None, 0.0
        for t in turns:
            overlap = min(w.end, t.end) - max(w.start, t.start)
            if overlap > best_overlap:
                best_overlap, best_label = overlap, t.label
        if best_label is None and turns:
            nearest = min(
                turns,
                key=lambda t: min(abs(w.start - t.end), abs(t.start - w.end)),
            )
            best_label = nearest.label
        out.append(
            Word(text=w.text, start=w.start, end=w.end, speaker=best_label)
        )
    return out


def _blocks_with_speakers(words: List[Word]) -> List[Block]:
    """Segment into blocks, never mixing two speakers in one caption block."""
    blocks: List[Block] = []
    run: List[Word] = []
    for w in words:
        if run and getattr(run[-1], "speaker", None) != getattr(w, "speaker", None):
            blocks.extend(segment_words(run))
            run = []
        run.append(w)
    if run:
        blocks.extend(segment_words(run))
    return blocks


def speaker_blocks(words: List[Word], turns: List[SpeakerTurn]) -> List[Block]:
    """Diarized caption blocks: split on speakers, then on time/length."""
    return _blocks_with_speakers(assign_speakers(words, turns))


def format_block_text(block: Block) -> str:
    """``[SPEAKER_00]: words…`` prefix for a diarized block."""
    speaker = getattr(block.words[0], "speaker", None) if block.words else None
    text = " ".join(w.text for w in block.words)
    return f"[{speaker}]: {text}" if speaker else text
