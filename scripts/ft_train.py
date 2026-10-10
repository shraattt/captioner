"""Phase 4 — fine-tune Whisper for noisy real-world audio.

Fine-tunes ``openai/whisper-base.en`` (or tiny.en) with HF's Seq2SeqTrainer:
bf16 weights + gradient accumulation + gradient checkpointing keep it inside
the 6 GB RTX 4050. The dataset comes from scripts/ft_build_dataset.py.

Nothing here runs in the pytest suite — it needs the real GPU and hours of
wall clock, so every heavy import happens inside functions and ``--max-steps``
gives a minutes-long smoke run that proves the whole path works.

Usage:
  # smoke run: 30 steps, ~1-2 min, proves featurizing + training + saving
  .venv/Scripts/python scripts/ft_train.py --max-steps 30 --epochs 1

  # real run
  .venv/Scripts/python scripts/ft_train.py --model openai/whisper-base.en \
      --epochs 3 --output data/ft/base_en_noisy
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("ft_train")

SAMPLE_RATE = 16000


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/whisper-base.en")
    ap.add_argument("--data-dir", default=str(Path("data") / "ft"))
    ap.add_argument("--output", default=None, help="defaults to data/ft/<model-name>_noisy")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-gpu", type=int, default=8, help="micro-batch per step")
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-steps", type=int, default=-1, help="override for a short smoke run")
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-limit", type=int, default=64,
                    help="cap eval clips so generation stays cheap (0 = all)")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="train the decoder only (less VRAM; less noise robustness)")
    ap.add_argument("--no-eval", action="store_true", help="skip eval during training")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import numpy as np
    import torch
    from datasets import load_from_disk
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    data_dir = Path(args.data_dir)
    for name in ("ft_train", "ft_eval"):
        if not (data_dir / name).is_dir():
            raise SystemExit(
                f"{data_dir / name} missing — run scripts/ft_build_dataset.py first"
            )
    train_ds = load_from_disk(str(data_dir / "ft_train"))
    eval_ds = None if args.no_eval else load_from_disk(str(data_dir / "ft_eval"))
    if eval_ds is not None and args.eval_limit:
        eval_ds = eval_ds.select(range(min(args.eval_limit, len(eval_ds))))
    logger.info("train %d clips, eval %d", len(train_ds), 0 if eval_ds is None else len(eval_ds))

    if not torch.cuda.is_available():
        raise SystemExit("Phase 4 training needs the GPU (cuda); CPU would take days.")

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    logger.info("device %s, dtype %s",
                torch.cuda.get_device_name(0), compute_dtype)

    processor = WhisperProcessor.from_pretrained(args.model, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32  # keep master weights fp32; autocast/bf16 handles compute
    )
    # Decoder prompt handling. English-only checkpoints ("*.en") carry a
    # forced no-timestamps token and have no language tokens at all; setting
    # generation_config.language on one sends v5's generate() down a
    # multilingual path that looks for lang_to_id and crashes. So only the
    # multilingual checkpoints get language/task, and everyone keeps a sane
    # decoder prompt.
    model.config.forced_decoder_ids = None  # the processor supplies the prompt
    model.config.suppress_tokens = []
    generation = model.generation_config
    if getattr(generation, "is_multilingual", False):
        generation.language = "en"
        generation.task = "transcribe"
        generation.forced_decoder_ids = None
    else:
        logger.info("English-only checkpoint: keeping its no-timestamps decoder prompt")

    if args.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("params %.1fM total, %.1fM trainable", total / 1e6, trainable / 1e6)

    def featurize(batch):
        feats = processor.feature_extractor(
            batch["audio"], sampling_rate=SAMPLE_RATE, return_tensors="np"
        )
        features = np.asarray(feats.input_features)
        if features.ndim == 4:  # the extractor keeps a mono channel axis
            features = features[:, 0]
        batch["input_features"] = list(features)
        batch = processor.tokenizer(batch["text"], truncation=True, max_length=448)
        # keep the key name the trainer expects
        batch["labels"] = batch.pop("input_ids")
        return batch

    # batched=True matters: the feature extractor shapes its output differently
    # for a lone array than for a list of them, and per-example calls leave a
    # leading axis that reaches conv1d as an extra dimension.
    keep = {"input_features", "labels"}
    train_ds = train_ds.map(
        featurize, batched=True, batch_size=8,
        remove_columns=[c for c in train_ds.column_names if c not in keep],
        desc="featurizing train",
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            featurize, batched=True, batch_size=8,
            remove_columns=[c for c in eval_ds.column_names if c not in keep],
            desc="featurizing eval",
        )

    pad_id = processor.tokenizer.pad_token_id
    decoder_start = model.config.decoder_start_token_id

    def collate(features):
        max_len = max(len(f["labels"]) for f in features)
        max_len = min(max_len, 448)
        inputs = torch.tensor(np.stack([f["input_features"] for f in features]),
                              dtype=torch.float32)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            lab = list(f["labels"])[:max_len]
            labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        return {"input_features": inputs, "labels": labels}

    pad_id = pad_id or 50256
    assert decoder_start is not None

    import jiwer

    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, pad_id)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        score = jiwer.wer([s.strip().lower() for s in label_str],
                          [s.strip().lower() for s in pred_str])
        return {"wer": 100.0 * score}

    if args.output is None:
        name = args.model.split("/")[-1].replace(".", "_")
        args.output = str(data_dir / f"{name}_noisy")

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_gpu,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=20,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=not args.freeze_encoder,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps,
        predict_with_generate=eval_ds is not None,
        generation_max_length=224,
        dataloader_num_workers=0,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        compute_metrics=compute_metrics if eval_ds is not None else None,
        processing_class=processor,
    )

    t0 = time.perf_counter()
    trainer.train()
    logger.info("training wall clock: %.1f min", (time.perf_counter() - t0) / 60)
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)

    # The last log entry is the training summary, so scan backwards for the
    # final eval number rather than reading the tail blindly.
    eval_wer = next(
        (entry["eval_wer"] for entry in reversed(trainer.state.log_history)
         if "eval_wer" in entry),
        None,
    )
    meta = {
        "base_model": args.model,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_gpu": args.batch_gpu,
        "accum": args.accum,
        "effective_batch": args.batch_gpu * args.accum,
        "lr": args.lr,
        "freeze_encoder": args.freeze_encoder,
        "dtype": str(compute_dtype).replace("torch.", ""),
        "train_clips": len(train_ds),
        "eval_clips": 0 if eval_ds is None else len(eval_ds),
        "eval_wer": eval_wer,
    }
    (Path(args.output) / "ft_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("checkpoint saved to %s (%s)", args.output, json.dumps(meta))


if __name__ == "__main__":
    main()
