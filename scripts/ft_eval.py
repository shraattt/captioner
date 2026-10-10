"""Phase 4/5 — WER evaluation: base vs fine-tuned on the same held-out clips.

Compares checkpoints on the noisy eval split built by ft_build_dataset.py and
prints a WER table. Each checkpoint is either

  * a Hugging Face Whisper dir (data/ft/base_en_noisy) — run through
    transformers, or
  * a CTranslate2 dir / faster-whisper size (Systran/faster-whisper-base.en,
    data/ft/base_en_noisy_ct2) — run through faster-whisper, the live path.

Reading the table:
  * baseline row = how bad noisy audio is before the effort
  * fine-tuned row = what the effort bought
  * the Phase 4 gate is still REAL-audio WER — scripts/wer_check.py with a
    recording and a reference you trust. These synthetic-noise numbers only
    rank checkpoints against each other.

Usage:
  .venv/Scripts/python scripts/ft_eval.py \
      --checkpoints Systran/faster-whisper-base.en data/ft/base_en_noisy_ct2
  .venv/Scripts/python scripts/ft_eval.py --checkpoints data/ft/base_en_noisy \
      --limit 32 --device cpu
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("ft_eval")


def compute_wer(refs, hyps) -> float:
    import jiwer

    norm = lambda xs: [x.strip().lower() for x in xs]  # noqa: E731
    return float(jiwer.wer(norm(refs), norm(hyps)))


def is_hf_dir(path: str) -> bool:
    """A transformers checkpoint: has config.json but no CT2 model.bin."""
    p = Path(path)
    return p.is_dir() and (p / "config.json").exists() and not (p / "model.bin").exists()


def transcribe_faster_whisper(model: str, audios, device: str) -> list:
    # Use the app's own loader: it preloads the CUDA DLLs ctranslate2 needs on
    # Windows, attaches the HF token, and caches the model.
    from captioner import transcribe as T

    compute = "float16" if device == "cuda" else "int8"
    wm = T.get_model(model, device, compute)
    out = []
    for arr in audios:
        segments, _info = wm.transcribe(
            arr, language="en", beam_size=5, condition_on_previous_text=False,
            vad_filter=False,
        )
        out.append(" ".join(s.text for s in segments).strip())
    return out


def transcribe_transformers(model_dir: str, audios, device: str) -> list:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_dir, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    dtype = torch.float16 if device == "cuda" else torch.float32
    model.to(dtype)

    # English-only checkpoints reject language=/task= outright (v5 raises for
    # them), so only the multilingual ones get the prompt flags.
    multilingual = bool(getattr(model.generation_config, "is_multilingual", False))
    extra = {"language": "en", "task": "transcribe"} if multilingual else {}

    out = []
    with torch.no_grad():
        for arr in audios:
            feats = processor.feature_extractor(
                arr, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device, dtype)
            ids = model.generate(feats, max_new_tokens=224, **extra)
            out.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="HF dir, CT2 dir, or faster-whisper size — in comparison order")
    ap.add_argument("--data-dir", type=Path, default=Path("data") / "ft")
    ap.add_argument("--limit", type=int, default=0, help="eval on first N clips only")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import numpy as np
    from datasets import load_from_disk

    eval_dir = args.data_dir / "ft_eval"
    if not eval_dir.is_dir():
        raise SystemExit(f"{eval_dir} missing — run scripts/ft_build_dataset.py first")
    eval_ds = load_from_disk(str(eval_dir))
    if args.limit:
        eval_ds = eval_ds.select(range(min(args.limit, len(eval_ds))))
    refs = list(eval_ds["text"])
    audios = [np.asarray(a, dtype=np.float32) for a in eval_ds["audio"]]
    logger.info("eval clips: %d (%.1f min noisy audio)",
                len(refs), sum(len(a) for a in audios) / 16000 / 60)

    total_audio = sum(len(a) for a in audios) / 16000
    print(f"{'checkpoint':<40} {'WER':>7} {'kind':>7} {'wall':>8} {'RTF':>6}")
    print("-" * 72)
    scores = {}
    for ckpt in args.checkpoints:
        started = time.perf_counter()
        if is_hf_dir(ckpt):
            hyps = transcribe_transformers(ckpt, audios, args.device)
            kind = "hf"
        else:  # CT2 dir, repo id, or a plain size name
            hyps = transcribe_faster_whisper(ckpt, audios, args.device)
            kind = "ct2/fw"
        wall = time.perf_counter() - started
        score = compute_wer(refs, hyps)
        scores[ckpt] = score
        # RTF = audio seconds handled per wall second (higher is faster);
        # live captioning only needs ~1.0, batch wants much more.
        rtf = total_audio / wall if wall else 0.0
        print(f"{ckpt:<40} {score:>6.1%} {kind:>7} {wall:>7.1f}s {rtf:>5.1f}x")

    if len(scores) > 1:
        first = next(iter(scores))
        for name, score in list(scores.items())[1:]:
            delta = scores[first] - score
            verdict = "better" if delta > 0.005 else ("worse" if delta < -0.005 else "same")
            print(f"{name} vs {first}: {delta*100:+.1f} pp ({verdict})")
    print("\nReal-material gate: python scripts/wer_check.py <audio> <reference>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
