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
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from . import features
from .languages import (
    TRANSLATION_LANGUAGES,
    translation_language,
    translation_name,
)
from .segmenter import Block, Word

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/m2m100_418M"
# The language M2M100 reads when nothing else is known. Pass ``source=`` to
# translate from anything else: the recogniser reports the spoken language, so
# a Hindi transcript translated into French is now tagged ``hi`` rather than
# run through the English default. On the short clean lines we measured, the
# output happened to be identical either way - M2M100's encoder is genuinely
# multilingual - but the tag is what the model is designed around, and it is
# what keeps a cache shared between two readings of the same text honest.
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
# The full set (100 codes) comes from the model card - see captioner.languages.
# It is *not* the same 100 as the recogniser's: M2M100 has no Telugu, so a
# Telugu recording can be transcribed but not translated, and the UI says so.
LANGUAGES: dict[str, str] = TRANSLATION_LANGUAGES

SPEAKER_PREFIX = re.compile(r"^(\[[^\]]+\]:\s*)")


def language_name(code: str) -> str:
    """Display name for a language key (falls back to the key itself)."""
    return translation_name(code)


def language_code(code: str) -> str:
    """Validate a target language key, returning M2M100's code for it."""
    resolved = translation_language(code)
    if resolved is None:
        raise ValueError(
            f"unknown target language {code!r}; choose from "
            f"{', '.join(sorted(LANGUAGES))}"
        )
    return resolved


def default_cache_path() -> Path:
    """Where translated lines are cached: beside the other job data.

    Deliberately `config.DATA_DIR` and not a path beside the code: from a
    checkout the two are the same folder, but a packaged build keeps its data
    in the user's profile and treats its own program folder as read-only, so a
    cache written next to the code would fail to save on every translated line.
    """
    from . import config

    return config.DATA_DIR / "cache" / "translations.json"


def _chunks(text: str, limit: int = MAX_CHARS) -> list[str]:
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
        model_name: str | None = None,
        cache_path: Path | None = None,
        device: str | None = None,
        batch_size: int = 16,
        num_beams: int = BEAM_SIZE,
        no_repeat_ngram: int = NO_REPEAT_NGRAM,
        translate_fn: Callable[[Sequence[str], str], list[str]] | None = None,
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
        self._cache: dict[str, str] | None = None

    #cache
    def _load_cache(self) -> dict[str, str]:
        if self._cache is None:
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache = {}
        return self._cache

    @staticmethod
    def cache_key(target: str, text: str, model: str | None = None,
                  source: str | None = None) -> str:
        """Cache identity for one line.

        The default model and source language keep the original key shape so
        caches written before either could be overridden stay valid; anything
        else is namespaced into the key, because its output is not
        interchangeable.
        """
        tag = target if model in (None, MODEL_NAME) else f"{model}|{target}"
        if source and source != SOURCE_LANG:
            tag = f"{source}>{tag}"
        return hashlib.sha1(f"{tag}\x1f{text}".encode()).hexdigest()

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

    #model
    def _ensure_model(self) -> None:
        if self._model is not None or self._translate_fn is not None:
            return
        try:
            import torch
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        except ImportError as exc:  # a packaged slim build has no torch at all
            raise RuntimeError(features.refusal("translate_local")) from exc

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

    def _run(self, texts: Sequence[str], target: str, source: str | None = None) -> list[str]:
        if self._translate_fn is not None:
            return list(self._translate_fn(texts, target))
        import torch

        self._ensure_model()
        code = language_code(target)
        # Tell the tokenizer what it is reading. M2M100 is not a detector: it
        # has no way to notice that the English tag is wrong, so the caller's
        # answer has to be right.
        self._tokenizer.src_lang = _source_language(source)
        bos = self._tokenizer.get_lang_id(code)
        out: list[str] = []
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

    #public API
    def translate_many(self, texts: Sequence[str], target: str,
                       source: str | None = None) -> list[str]:
        """Translate ``texts`` to ``target``, using the cache for repeats.

        The returned list matches the input order and length; only strings that
        were not already cached are sent to the model, in one batched pass.
        ``source`` is the language being read (default English); translating a
        Hindi transcript into French needs it, and the worker passes whatever
        the recogniser heard.
        """
        texts = list(texts)
        language_code(target)  # validates the language before doing any work
        source = _source_language(source)
        cache = self._load_cache()
        keys = [self.cache_key(target, text, self.model_name, source) for text in texts]

        missing = [i for i, key in enumerate(keys) if key not in cache]
        if missing:
            # Split over-long strings, then stitch each answer back together.
            flat: list[str] = []
            owner: list[int] = []
            for index in missing:
                pieces = _chunks(texts[index])
                flat.extend(pieces)
                owner.extend([index] * len(pieces))
            translated = self._run(flat, target, source)
            if len(translated) != len(flat):  # pragma: no cover - defensive
                raise RuntimeError(
                    f"translator returned {len(translated)} results for {len(flat)} inputs"
                )
            rebuilt: dict[int, list[str]] = {}
            for piece, index in zip(translated, owner, strict=True):
                rebuilt.setdefault(index, []).append(piece)
            for index, parts in rebuilt.items():
                cache[keys[index]] = " ".join(p.strip() for p in parts if p.strip())
            self.save_cache()

        return [cache[key] for key in keys]

    def translate_block(self, block: Block, target: str,
                        source: str | None = None) -> Block:
        """One caption block, timing and speaker label preserved."""
        translated = self.translate_many([block.text], target, source)[0]
        speaker = block.words[0].speaker if block.words else None
        return replace(
            block,
            words=[Word(text=translated, start=block.start, end=block.end, speaker=speaker)],
        )


def _source_language(source: str | None) -> str:
    """The code to tag the input with, validated against M2M100's set.

    Nothing else in the app should have to know that the translator reads a
    different (smaller, differently spelled) language set than the recogniser;
    a Whisper code that M2M100 lacks is reported, not silently defaulted to
    English and quietly mistranslated.
    """
    if source is None or not str(source).strip():
        return SOURCE_LANG
    try:
        return translation_language(source) or SOURCE_LANG
    except ValueError:
        wanted = str(source).strip().lower()
        from .languages import SPOKEN_LANGUAGES

        hint = ""
        if wanted in SPOKEN_LANGUAGES:
            hint = (
                f" - {SPOKEN_LANGUAGES[wanted]} can be transcribed, but the "
                "translator has no model for it"
            )
        raise ValueError(
            f"the translator cannot read {source!r}{hint}; set the recording's "
            "language to one it knows, or leave it unset"
        ) from None


def default_translator():
    """The configured translation backend.

    Local M2M100 unless ``CAPTIONER_MT_ENGINE`` names another one (see
    ``captioner.engines``). Any object with ``translate_many(texts, target)``
    works, so a backend never has to be a ``Translator`` subclass.
    """
    from .engines import mt

    return mt()


def translate_blocks(
    blocks: Sequence[Block],
    target: str,
    translator=None,
    source: str | None = None,
) -> list[Block]:
    """Translate every block in one batch, keeping speaker labels verbatim.

    ``source`` is the spoken language of the transcript (``None`` = English).
    """
    translator = translator or default_translator()
    # Validate once here, so every backend gets a code M2M100's family knows or
    # an error naming the problem - never a silent English default.
    source_code = _source_language(source)
    prefixes = [SPEAKER_PREFIX.match(block.text) for block in blocks]
    bare = [
        block.text if match is None else block.text[match.end():]
        for block, match in zip(blocks, prefixes, strict=True)
    ]
    try:
        translated = translator.translate_many(bare, target, source_code)
    except TypeError:
        # A backend written before the source language existed. It still works,
        # but it will assume English, so say so rather than let a wrong reading
        # pass for a right one.
        logger.warning(
            "%s does not take a source language; its input is assumed to be %s",
            type(translator).__name__, SOURCE_LANG,
        )
        translated = translator.translate_many(bare, target)

    out: list[Block] = []
    for block, match, text in zip(blocks, prefixes, translated, strict=True):
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
    translator=None,
    speaker_names: bool = False,
    source: str | None = None,
) -> list[Path]:
    """Write ``captions.<lang>.srt`` / ``.vtt`` next to the original captions.

    Reads the existing ``captions.srt``, so translation is a pure
    post-processing step over the artifact the batch already produced: it never
    needs the queue, the source media, or a re-transcribe.

    ``source`` is the transcript's spoken language; omit it for English.
    """
    from . import srt

    out_dir = Path(out_dir)
    caption_file = out_dir / "captions.srt"
    if not caption_file.exists():
        raise FileNotFoundError(f"{caption_file} not found — transcribe the job first")

    blocks = srt.parse_srt(caption_file.read_text(encoding="utf-8"))
    translated = translate_blocks(blocks, target, translator=translator, source=source)

    srt_path = out_dir / f"captions.{target}.srt"
    vtt_path = out_dir / f"captions.{target}.vtt"
    srt_path.write_text(srt.to_srt(translated, speaker_names=speaker_names), encoding="utf-8")
    vtt_path.write_text(srt.to_vtt(translated, speaker_names=speaker_names), encoding="utf-8")
    logger.info("translated captions -> %s, %s", srt_path.name, vtt_path.name)
    return [srt_path, vtt_path]
