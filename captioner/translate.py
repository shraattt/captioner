"""Phase 2 — optional subtitle translation, cached and lazy.

Nothing is downloaded until a translation is actually requested: the model
(facebook/m2m100_418M, ~1.8 GB) loads on first use and stays resident, and every
translated string is cached on disk keyed by (target language, source text).
Translating a finished job, or re-translating one you already did, then costs
nothing.

Why M2M100 and not NLLB: NLLB was the plan, but **transformers 5 removed NLLB's
text model** (and the task ``translation`` pipeline with it — only the tokenizer
survives). M2M100 is the multilingual encoder-decoder NLLB was built on top of:
100 languages, same family of language codes, same encoder-decoder shape, and
still supported. Swapping models is a one-line change of ``MODEL_NAME`` if NLLB
comes back.

Translated captions keep the original timing and speaker labels — only the words
change. Diarized blocks keep their ``[SPEAKER_00]:`` prefix verbatim: machine-
translating a speaker tag is pure noise.
"""
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .segmenter import Block, Word

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/m2m100_418M"
SOURCE_LANG = "en"
MAX_CHARS = 400  # model limit is 512 tokens; leave headroom for expansion

# Decoding knobs. Greedy decoding repeats itself on short subtitle lines (a
# real Tamil run produced "ஹலோ ஹலோ ஹலோ ஹலோ" for "Hello and welcome."), so the
# default is beam search with an n-gram block. Both are adjustable per
# Translator and overridable for a quick experiment.
BEAM_SIZE = 4
NO_REPEAT_NGRAM = 3

# Roughly 3.5x the parameters of the default, and the quality lever when a
# language comes out badly: CAPTIONER_TRANSLATE_MODEL=facebook/m2m100_1.2B
MODEL_ENV = "CAPTIONER_TRANSLATE_MODEL"

# Target-language codes accepted by M2M100's tokenizer, with display names.
# (M2M100's language set is 100 codes; this is the shortlist the UI offers —
# notably it has no Telugu, so "te" is absent rather than silently broken.)
LANGUAGES: Dict[str, str] = {
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "zh": "Chinese (simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "tr": "Turkish",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "fa": "Persian",
    "he": "Hebrew",
    "sw": "Swahili",
    "ta": "Tamil",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "ne": "Nepali",
    "si": "Sinhala",
}

SPEAKER_PREFIX = re.compile(r"^(\[[^\]]+\]:\s*)")


def language_name(code: str) -> str:
    """Display name for a language key (falls back to the key itself)."""
    return LANGUAGES.get(code, code)


def language_code(code: str) -> str:
    """Validate a target language key, returning M2M100's code for it."""
    if code not in LANGUAGES:
        raise ValueError(
            f"unknown target language {code!r}; choose from {', '.join(sorted(LANGUAGES))}"
        )
    return code


def default_cache_path() -> Path:
    from . import config

    return Path(config.PROJECT_ROOT) / "data" / "cache" / "translations.json"


def _chunks(text: str, limit: int = MAX_CHARS) -> List[str]:
    """Split ``text`` on word boundaries so no chunk exceeds ``limit`` chars."""
    if len(text) <= limit:
        return [text]
    out, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > limit:
            out.append(current)
            current = ""
        current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out


class Translator:
    """M2M100 translation with an on-disk cache.

    ``translate_fn`` is the test seam: give it ``(texts, target_code) ->
    list[str]`` and no model is ever loaded or downloaded.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_path: Optional[Path] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        num_beams: int = BEAM_SIZE,
        no_repeat_ngram: int = NO_REPEAT_NGRAM,
        translate_fn: Optional[Callable[[Sequence[str], str], List[str]]] = None,
    ):
        self.model_name = model_name or os.environ.get(MODEL_ENV) or MODEL_NAME
        self.num_beams = num_beams
        self.no_repeat_ngram = no_repeat_ngram
        self.cache_path = Path(cache_path) if cache_path else default_cache_path()
        self.device = device
        self.batch_size = batch_size
        self._translate_fn = translate_fn
        self._model = None
        self._tokenizer = None
        self._cache: Optional[Dict[str, str]] = None

    # --- cache -----------------------------------------------------------
    def _load_cache(self) -> Dict[str, str]:
        if self._cache is None:
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache = {}
        return self._cache

    @staticmethod
    def cache_key(target: str, text: str, model: Optional[str] = None) -> str:
        """Cache identity for one line.

        The default model keeps the original key shape so caches written before
        model overrides existed stay valid; a different model is namespaced
        into the key, because its output is not interchangeable.
        """
        tag = target if model in (None, MODEL_NAME) else f"{model}|{target}"
        return hashlib.sha1(f"{tag}\x1f{text}".encode("utf-8")).hexdigest()

    def save_cache(self) -> None:
        """Write the cache atomically (temp file + replace), like the editor."""
        if not self._cache:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.cache_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._cache, handle, ensure_ascii=False, indent=0)
            os.replace(tmp, self.cache_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # --- model -----------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None or self._translate_fn is not None:
            return
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading translation model %s on %s...", self.model_name, self.device)
        self._tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        # fp16 halves the VRAM (418M params ~0.9 GB) and is fine on GPU; CPU
        # has no fp16 kernels for this model, so keep fp32 there.
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = (
            M2M100ForConditionalGeneration.from_pretrained(self.model_name, dtype=dtype)
            .to(self.device)
            .eval()
        )

    def available(self) -> bool:
        """True when the model can be loaded without downloading."""
        if self._translate_fn is not None:
            return True
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(self.model_name, local_files_only=True)
            return True
        except Exception:
            return False

    def _run(self, texts: Sequence[str], target: str) -> List[str]:
        if self._translate_fn is not None:
            return list(self._translate_fn(texts, target))
        import torch

        self._ensure_model()
        code = language_code(target)
        self._tokenizer.src_lang = SOURCE_LANG
        bos = self._tokenizer.get_lang_id(code)
        out: List[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start: start + self.batch_size])
            encoded = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                generated = self._model.generate(
                    **encoded,
                    forced_bos_token_id=bos,
                    max_new_tokens=512,
                    num_beams=self.num_beams,
                    no_repeat_ngram_size=self.no_repeat_ngram,
                )
            out.extend(
                text.strip()
                for text in self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return out

    # --- public API ------------------------------------------------------
    def translate_many(self, texts: Sequence[str], target: str) -> List[str]:
        """Translate ``texts`` to ``target``, using the cache for repeats.

        The returned list matches the input order and length; only strings that
        were not already cached are sent to the model, in one batched pass.
        """
        texts = list(texts)
        language_code(target)  # validates the language before doing any work
        cache = self._load_cache()
        keys = [self.cache_key(target, text, self.model_name) for text in texts]

        missing = [i for i, key in enumerate(keys) if key not in cache]
        if missing:
            # Split over-long strings, then stitch each answer back together.
            flat: List[str] = []
            owner: List[int] = []
            for index in missing:
                pieces = _chunks(texts[index])
                flat.extend(pieces)
                owner.extend([index] * len(pieces))
            translated = self._run(flat, target)
            if len(translated) != len(flat):  # pragma: no cover - defensive
                raise RuntimeError(
                    f"translator returned {len(translated)} results for {len(flat)} inputs"
                )
            rebuilt: Dict[int, List[str]] = {}
            for piece, index in zip(translated, owner):
                rebuilt.setdefault(index, []).append(piece)
            for index, parts in rebuilt.items():
                cache[keys[index]] = " ".join(p.strip() for p in parts if p.strip())
            self.save_cache()

        return [cache[key] for key in keys]

    def translate_block(self, block: Block, target: str) -> Block:
        """One caption block, timing and speaker label preserved."""
        translated = self.translate_many([block.text], target)[0]
        speaker = block.words[0].speaker if block.words else None
        return replace(
            block,
            words=[Word(text=translated, start=block.start, end=block.end, speaker=speaker)],
        )


def translate_blocks(
    blocks: Sequence[Block],
    target: str,
    translator: Optional[Translator] = None,
) -> List[Block]:
    """Translate every block in one batch, keeping speaker labels verbatim."""
    translator = translator or Translator()
    prefixes = [SPEAKER_PREFIX.match(block.text) for block in blocks]
    bare = [
        block.text if match is None else block.text[match.end():]
        for block, match in zip(blocks, prefixes)
    ]
    translated = translator.translate_many(bare, target)

    out: List[Block] = []
    for block, match, text in zip(blocks, prefixes, translated):
        if match is not None:
            text = match.group(1) + text
        speaker = block.words[0].speaker if block.words else None
        out.append(
            replace(
                block,
                words=[Word(text=text.strip(), start=block.start,
                            end=block.end, speaker=speaker)],
            )
        )
    return out


def translate_subtitles(
    out_dir: "str | Path",
    target: str,
    translator: Optional[Translator] = None,
    speaker_names: bool = False,
) -> List[Path]:
    """Write ``captions.<lang>.srt`` / ``.vtt`` next to the original captions.

    Reads the existing ``captions.srt``, so translation is a pure
    post-processing step over the artifact the batch already produced: it never
    needs the queue, the source media, or a re-transcribe.
    """
    from . import srt

    out_dir = Path(out_dir)
    source = out_dir / "captions.srt"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found — transcribe the job first")

    blocks = srt.parse_srt(source.read_text(encoding="utf-8"))
    translated = translate_blocks(blocks, target, translator=translator)

    srt_path = out_dir / f"captions.{target}.srt"
    vtt_path = out_dir / f"captions.{target}.vtt"
    srt_path.write_text(srt.to_srt(translated, speaker_names=speaker_names), encoding="utf-8")
    vtt_path.write_text(srt.to_vtt(translated, speaker_names=speaker_names), encoding="utf-8")
    logger.info("translated captions -> %s, %s", srt_path.name, vtt_path.name)
    return [srt_path, vtt_path]
