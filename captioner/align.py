"""Phase 2 — WhisperX-style word alignment (wav2vec2 CTC forced alignment).

Whisper's own word timestamps come from cross-attention heuristics: good enough
for subtitles, but boundaries smear — a word often inherits the pause in front
of it, and quick words overlap. WhisperX's trick is to keep Whisper's
*transcript* and re-time it against a character CTC acoustic model, which knows
much more precisely where each word starts and stops.

This module implements that idea without the whisperx dependency:

  Whisper words + audio
        -> slice audio into ~20 s windows
        -> tokenize the known transcript to characters ('|' = the wav2vec2
           word boundary)
        -> Viterbi-align those targets to the model's emission frames
           (``forced_align`` below: the standard CTC recurrence with blank
           insertion and the two-step skip transition)
        -> each word's span = min/max frame over its own characters

Frames are 20 ms (wav2vec2's conv stride), so timings are ~10x more precise
than Whisper's. Degenerate alignments (a word that collapses to nothing, or
that would run back past its neighbour) fall back to Whisper's original times,
so turning alignment on can never make timings *worse* than not using it.

The model (facebook/wav2vec2-base-960h, ~380 MB) is English-only and loads on
first use; ``model_is_cached`` lets the UI say so before the wait.
"""
import logging
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from .segmenter import Word

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SECONDS = 320 / SAMPLE_RATE  # wav2vec2 conv stride -> 20 ms frames
ALIGN_MODEL = "facebook/wav2vec2-base-960h"
WINDOW_SECONDS = 20.0
MARGIN_SECONDS = 0.5  # audio kept either side of a window's words
MAX_SHIFT_SECONDS = 0.6  # reject alignments that move a word further than this
NEG = -1e30


def _log_softmax(x):
    import numpy as np

    shifted = x - x.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def forced_align(
    log_probs, tokens: Sequence[int], blank_id: int = 0
) -> List[Optional[Tuple[int, int]]]:
    """Align ``tokens`` (non-blank ids, in order) to emission frames.

    ``log_probs`` is a (frames, vocab) array of log probabilities. Returns one
    ``(start_frame, end_frame)`` per token (inclusive), or ``None`` when the
    best path spends no frame on that token.

    The DP state sequence is ``blank, t1, blank, t2, blank, …`` (length
    2L+1). A state may come from itself, its predecessor, or — for a token
    state whose token differs from the one two states back — from two states
    back (the standard CTC skip that lets two identical frames collapse into
    one emission).
    """
    import numpy as np

    n_tokens = len(tokens)
    if log_probs.shape[0] == 0 or n_tokens == 0:
        return [None] * n_tokens

    state_token = np.full(2 * n_tokens + 1, blank_id, dtype=np.int64)
    for i, tok in enumerate(tokens):
        state_token[2 * i + 1] = tok

    n_states = len(state_token)
    prev = np.full(n_states, NEG, dtype=np.float64)
    prev[0] = log_probs[0, state_token[0]]
    if n_states > 1:
        prev[1] = log_probs[0, state_token[1]]

    skip_ok = np.zeros(n_states, dtype=bool)
    if n_states > 2:
        skip_ok[2:] = (state_token[2:] != blank_id) & (
            state_token[2:] != state_token[:-2]
        )

    back = np.zeros((log_probs.shape[0], n_states), dtype=np.int32)
    idx = np.arange(n_states)
    # argmax over the three candidates selects a *move*; store the state that
    # move came from so the traceback can walk states directly.
    predecessors = np.vstack([idx, np.maximum(idx - 1, 0), np.maximum(idx - 2, 0)])
    for t in range(1, log_probs.shape[0]):
        step1 = np.concatenate(([NEG], prev[:-1]))
        if n_states > 2:
            step2 = np.concatenate(([NEG, NEG], prev[:-2]))
            step2 = np.where(skip_ok, step2, NEG)
        else:
            step2 = np.full(n_states, NEG)
        candidates = np.vstack([prev, step1, step2])
        choice = np.argmax(candidates, axis=0)
        prev = log_probs[t, state_token] + candidates[choice, idx]
        back[t] = predecessors[choice, idx]

    # A path may end on the final blank or on the last token.
    end = n_states - 1
    if n_states > 1 and prev[n_states - 2] > prev[n_states - 1]:
        end = n_states - 2

    states = np.empty(log_probs.shape[0], dtype=np.int32)
    s = end
    for t in range(log_probs.shape[0] - 1, -1, -1):
        states[t] = s
        s = int(back[t, s])

    spans: List[Optional[Tuple[int, int]]] = []
    for i in range(n_tokens):
        frames = np.nonzero(states == 2 * i + 1)[0]
        spans.append(
            None if frames.size == 0 else (int(frames[0]), int(frames[-1]))
        )
    return spans


class WordAligner:
    """Re-times known words against a wav2vec2 CTC model.

    ``logits_fn`` and ``token_ids_fn`` are the injection points tests use:
    supply them and no model is ever downloaded or loaded.
    """

    def __init__(
        self,
        model_name: str = ALIGN_MODEL,
        device: Optional[str] = None,
        logits_fn: Optional[Callable] = None,
        token_ids_fn: Optional[Callable] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._logits_fn = logits_fn
        self._token_ids_fn = token_ids_fn
        self._model = None
        self._tokenizer = None

    # --- model plumbing -------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None or self._logits_fn is not None:
            return
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading alignment model %s on %s...", self.model_name, self.device)
        processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self._tokenizer = processor.tokenizer
        self._model = (
            Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device).eval()
        )

    def log_probs(self, audio) -> "object":
        """Frame-level log probabilities for a float32 16 kHz slice."""
        if self._logits_fn is not None:
            return self._logits_fn(audio)
        self._ensure_model()
        import torch

        with torch.no_grad():
            tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
            logits = self._model(tensor).logits[0].float().cpu().numpy()
        return _log_softmax(logits)

    def _blank_id(self) -> int:
        if self._token_ids_fn is not None:
            return getattr(self, "blank_id", 0)
        self._ensure_model()
        return int(getattr(self._tokenizer, "pad_token_id", 0) or 0)

    def token_ids(self, text: str):
        """Character ids for ``text`` plus which word (or None) each belongs to."""
        if self._token_ids_fn is not None:
            return self._token_ids_fn(text)
        self._ensure_model()
        ids: List[int] = []
        owners: List[Optional[int]] = []
        for word_index, word in enumerate(text.split()):
            for char in word.lower():
                if char.isalnum() or char == "'":
                    ids.append(int(self._tokenizer.convert_tokens_to_ids(char)))
                    owners.append(word_index)
            ids.append(int(self._tokenizer.convert_tokens_to_ids("|")))
            owners.append(None)
        return ids, owners

    # --- the alignment itself -------------------------------------------
    def align(
        self,
        audio,
        words: Sequence[Word],
        window: float = WINDOW_SECONDS,
        margin: float = MARGIN_SECONDS,
        max_shift: float = MAX_SHIFT_SECONDS,
    ) -> List[Word]:
        """Return ``words`` with tightened start/end times (text/speaker kept)."""
        import numpy as np

        words = [w for w in words if w.text and w.start is not None and w.end is not None]
        if not words:
            return []

        blank = self._blank_id()
        refined: List[Word] = []
        for group in _windows(words, window):
            start_seconds = max(0.0, group[0].start - margin)
            end_seconds = group[-1].end + margin
            slice_audio = np.asarray(
                audio[int(start_seconds * SAMPLE_RATE): int(end_seconds * SAMPLE_RATE)],
                dtype="float32",
            )
            if slice_audio.size < SAMPLE_RATE // 10:  # under 100 ms: nothing to trust
                refined.extend(group)
                continue
            refined.extend(
                self._align_group(slice_audio, group, start_seconds, blank, max_shift)
            )
        return _monotonic(refined)


    def _align_group(self, audio, group: Sequence[Word], offset: float, blank: int,
                     max_shift: float = MAX_SHIFT_SECONDS):
        text = " ".join(w.text for w in group)
        ids, owners = self.token_ids(text)
        if not ids:
            return list(group)
        log_probs = self.log_probs(audio)
        if log_probs.shape[1] <= max(ids):
            # Unknown characters can map to an id outside this model's vocab.
            ids = [i if i < log_probs.shape[1] else blank for i in ids]
        spans = forced_align(log_probs, ids, blank_id=blank)

        per_word: dict = {}
        for span, owner in zip(spans, owners):
            if owner is None or span is None:
                continue
            low, high = per_word.get(owner, (span[0], span[1]))
            per_word[owner] = (min(low, span[0]), max(high, span[1]))

        out: List[Word] = []
        for i, word in enumerate(group):
            span = per_word.get(i)
            if span is None:
                out.append(word)  # nothing aligned: keep Whisper's timing
                continue
            start = offset + span[0] * FRAME_SECONDS
            end = offset + (span[1] + 1) * FRAME_SECONDS
            end = max(end, start + FRAME_SECONDS)
            # Two sanity checks, both falling back to Whisper's timing. The
            # aligner's job is to TIGHTEN boundaries, never to move a word to
            # a different part of the audio:
            #   1. a span inflated past ~4x Whisper's duration is a bad word
            #   2. a boundary that jumps more than ``max_shift`` seconds away
            #      from Whisper's estimate means the frame path drifted (common
            #      on repetitive audio, where identical phrases repeat)
            whisper_len = max(word.end - word.start, FRAME_SECONDS)
            drifted = (
                abs(start - word.start) > max_shift
                and abs(end - word.end) > max_shift
            )
            if end - start > whisper_len * 4 + 0.5 or drifted:
                out.append(word)
                continue
            out.append(replace(word, start=round(start, 3), end=round(end, 3)))
        return out


def _windows(words: Sequence[Word], window: float):
    """Split words into time-bounded groups (a Viterbi pass per group)."""
    group: List[Word] = []
    for word in words:
        if group and word.end - group[0].start > window:
            yield group
            group = []
        group.append(word)
    if group:
        yield group


def _monotonic(words: List[Word]) -> List[Word]:
    """Keep refined timings ordered and non-overlapping.

    Forced alignment is monotonic by construction, but falls back to original
    times per word, and a mix of refined-original spans can cross. A word that
    would start before its predecessor ends is nudged to the predecessor's end
    (and given a minimum span) rather than left inverted.
    """
    out: List[Word] = []
    prev_end = -1.0
    for word in words:
        start, end = word.start, word.end
        if start < prev_end:
            start = prev_end
        if end <= start:
            end = start + FRAME_SECONDS
        out.append(replace(word, start=round(start, 3), end=round(end, 3)))
        prev_end = end
    return out


@lru_cache(maxsize=2)
def get_aligner(model_name: str = ALIGN_MODEL, device: Optional[str] = None) -> WordAligner:
    """Cached aligner, so a batch of jobs loads the model once."""
    return WordAligner(model_name=model_name, device=device)


def model_is_cached(model_name: str = ALIGN_MODEL) -> bool:
    """True when the alignment model is already in the local HF cache."""
    try:
        from huggingface_hub import snapshot_download
    except Exception:  # pragma: no cover - hub always present with transformers
        return False
    try:
        snapshot_download(model_name, local_files_only=True)
        return True
    except Exception:
        return False


def realign_file(
    audio_path: "str | Path",
    words: Sequence[Word],
    aligner: Optional[WordAligner] = None,
    on_stage: Optional[Callable] = None,
) -> List[Word]:
    """Decode ``audio_path`` and re-time ``words`` against it.

    Returns the original list unchanged when the alignment model cannot be
    loaded, so a missing/broken download degrades to Whisper timings instead of
    failing the job.
    """
    from faster_whisper.audio import decode_audio

    aligner = aligner or get_aligner()
    try:
        if on_stage is not None:
            on_stage("aligning", None)
        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        return aligner.align(audio, words)
    except Exception as exc:  # noqa: BLE001 - optional pass, never fatal
        logger.warning("word alignment skipped (%s: %s)", type(exc).__name__, exc)
        return list(words)
