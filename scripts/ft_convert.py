"""Phase 5 — convert a fine-tuned HF checkpoint to CTranslate2 for live use.

The live pipeline loads models through faster-whisper, which needs CTranslate2
format. Wraps ct2 converters with the same device/dtype choices the batch code
uses, then optionally benchmarks the converted model end-to-end through
transcribe_words — the same path live captions will run.

Usage:
  .venv/Scripts/python scripts/ft_convert.py \
      --src data/ft/base_en_noisy --dst data/ft/base_en_noisy_ct2
  .venv/Scripts/python scripts/ft_convert.py --src data/ft/base_en_noisy \
      --dst data/ft/base_en_noisy_ct2 --bench data/uploads/smoke_speech.wav
"""
import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("ft_convert")


def convert(src: Path, dst: Path, quantization: str = "float16") -> None:
    import ctranslate2
    import transformers as tr

    if dst.exists():
        logger.warning("removing existing %s", dst)
        shutil.rmtree(dst)

    # HF Whisper checkpoints carry the tokenizer alongside the weights;
    # ct2's converter reads both from the same directory.
    converter = ctranslate2.converters.TransformersConverter(str(src))
    converter.convert(str(dst), quantization=quantization)
    # faster-whisper expects the tokenizer files in the model dir.
    tok = tr.WhisperTokenizer.from_pretrained(str(src))
    tok.save_pretrained(str(dst))
    # Carry the training metadata across so the model registry can describe
    # this model (base checkpoint, data size, measured WER) without guessing.
    meta = src / "ft_meta.json"
    if meta.exists():
        shutil.copy2(meta, dst / "ft_meta.json")
    logger.info("converted %s -> %s (%s)", src, dst, quantization)


def bench(model_dir: Path, audio_path: Path) -> None:
    """End-to-end smoke: load through faster-whisper exactly like live does."""
    from captioner import transcribe

    t0 = time.perf_counter()
    words = transcribe.transcribe_words(str(audio_path), model_size=str(model_dir))
    dt = time.perf_counter() - t0
    text = " ".join(w.text for w in words)
    logger.info("benchmark: %.2fs, %d words: %s", dt, len(words), text[:120])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="fine-tuned HF checkpoint dir")
    ap.add_argument("--dst", type=Path, required=True, help="output CT2 dir")
    ap.add_argument("--quantization", default="float16",
                    help="float16 for GPU live use; int8 for CPU fallback")
    ap.add_argument("--bench", type=Path, default=None,
                    help="audio file to smoke-test the converted model through transcribe_words")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.src.is_dir():
        raise SystemExit(f"{args.src} is not a directory (train first: scripts/ft_train.py)")
    convert(args.src, args.dst, args.quantization)
    if args.bench:
        if not args.bench.exists():
            raise SystemExit(f"{args.bench} does not exist")
        bench(args.dst, args.bench)


if __name__ == "__main__":
    main()
