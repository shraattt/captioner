"""Phase 4 — build the fine-tuning dataset.

Pairs LibriSpeech (clean, real read speech) with synthetic noise so the model
learns to transcribe under imperfect conditions — the "noisy/distant voices"
profile that live captioning actually sees (mics, TV loopback). Noise comes
from audiomentations' procedural bank (Gaussian SNR, background noise, gain,
dropout, room) by default; optionally add real MUSAN recordings.

The split is **streamed**: only the parquet shards needed for the requested
number of clips are pulled (a few hundred MB), instead of the ~6 GB that
addressing ``train.clean.100`` non-streaming downloads up front. Clips are
stored as plain float32 arrays + uppercase text, so training never touches
librosa again.

Usage:
  .venv/Scripts/python scripts/ft_build_dataset.py --hours 4 --noise snr,background
  .venv/Scripts/python scripts/ft_build_dataset.py --hours 4 \
      --musan-path D:/datasets/musan   # adds real MUSAN noise clips
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from captioner import config  # noqa: E402

logger = logging.getLogger("ft_dataset")

FT_DIR = config.PROJECT_ROOT / "data" / "ft"
NOISE_TYPES = ["snr", "background", "gain", "dropout", "room", "mp3"]
SAMPLE_RATE = 16000


def build_noise_augmenter(types: list[str], musan_dir: Path | None, seed: int):
    """Audiomentations chain for the chosen noise profile."""
    import numpy as np
    from audiomentations import (
        AddBackgroundNoise,
        AddGaussianNoise,
        AddGaussianSNR,
        Compose,
        Gain,
        Mp3Compression,
        TimeMask,
    )

    transforms = []
    if "snr" in types:
        transforms.append(AddGaussianSNR(min_snr_db=5.0, max_snr_db=25.0, p=0.8))
    if "background" in types:
        if musan_dir is None:
            # Procedural stand-in for MUSAN until the user provides real clips.
            transforms.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6))
        else:
            transforms.append(
                AddBackgroundNoise(
                    sounds_path=str(musan_dir),
                    min_snr_db=5.0,
                    max_snr_db=20.0,
                    p=0.7,
                )
            )
    if "gain" in types:
        transforms.append(Gain(min_gain_db=-6.0, max_gain_db=0.0, p=0.5))
    if "dropout" in types:
        # audiomentations >= 0.41 dropped TimeDropout in favour of TimeMask
        # (fraction-of-clip bands): packet loss / clipped-mic dropouts.
        transforms.append(TimeMask(min_band_part=0.005, max_band_part=0.05, p=0.2))
    if "room" in types:
        # RoomSimulator is the one transform with an optional dependency
        # (pyroomacoustics). Skipping it silently would quietly change the
        # training distribution, so say so.
        try:
            from audiomentations import RoomSimulator

            transforms.append(RoomSimulator(p=0.3))
        except Exception as exc:  # pragma: no cover - optional transform
            logger.warning("room simulation unavailable (%s); skipping", exc)
    if "mp3" in types:
        transforms.append(Mp3Compression(min_bitrate=48, max_bitrate=128, p=0.3))

    assert np is not None
    if not transforms:
        raise SystemExit("no noise types selected — nothing to augment with")
    return Compose(transforms, p=1.0)


def collect_split(hours: float, split: str, noise_types: list[str],
                  musan_dir: Path | None, seed: int, buffer_size: int = 300):
    """Stream a LibriSpeech split, augment it, return (dataset, seconds, info).

    ``split`` is "train" or "eval". Streaming means the download stops as soon
    as enough clips are gathered, instead of pulling the whole split.

    Audio arrives as raw FLAC bytes (``Audio(decode=False)``) and is decoded
    here with soundfile: datasets' own decoder wants torchcodec, which this
    project uninstalled on purpose (its Windows DLLs fail to load), and the
    batch pipeline never needs it. A few hundred MB of shards gives hours of
    speech, so we never touch the 6 GB a full non-streaming download costs.
    """
    import io

    import datasets
    import librosa
    import soundfile as sf
    from datasets import load_dataset

    subset = "train.clean.100" if split == "train" else "validation.clean"
    logger.info("streaming LibriSpeech %s (seed=%d)", subset, seed)
    stream = load_dataset("openslr/librispeech_asr", "all", split=subset, streaming=True)
    stream = stream.cast_column("audio", datasets.Audio(decode=False))
    # Shuffling with a wide buffer pulls from several shards at once, which
    # matters: a single shard is one chapter (one reader), so taking clips in
    # order would train on almost one voice.
    stream = stream.shuffle(seed=seed, buffer_size=buffer_size)
    # LibriSpeech clean-100 clips average ~12 s.
    wanted = max(1, int(hours * 3600 / 12))

    augmenter = build_noise_augmenter(noise_types, musan_dir, seed)
    random.seed(seed)
    audios: list = []
    texts: list[str] = []
    speakers: set = set()
    seconds = 0.0
    t0 = time.perf_counter()
    for row in stream:
        if len(audios) >= wanted:
            break
        arr, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if arr.ndim > 1:  # LibriSpeech is mono, but be safe
            arr = arr.mean(axis=1)
        if sr != SAMPLE_RATE:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
        arr = augmenter(arr.astype("float32"), sample_rate=SAMPLE_RATE)
        audios.append(arr)
        texts.append(row["text"].upper())
        speakers.add(row.get("speaker_id"))
        seconds += len(arr) / SAMPLE_RATE
        if len(audios) % 50 == 0:
            rate = len(audios) / max(time.perf_counter() - t0, 1e-6)
            logger.info("  %d/%d clips (%.1f min audio, %.1f clips/s, %d speakers)",
                        len(audios), wanted, seconds / 60, rate, len(speakers))

    features = datasets.Features({
        "audio": datasets.Sequence(datasets.Value("float32")),
        "text": datasets.Value("string"),
    })
    ds = datasets.Dataset.from_dict({"audio": audios, "text": texts}, features=features)
    info = {"clips": len(ds), "hours": round(seconds / 3600, 3), "speakers": len(speakers)}
    logger.info("%s: %d clips, %.2f h, %d speakers",
                split, len(ds), seconds / 3600, len(speakers))
    return ds, seconds, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=4.0,
                    help="hours of noisy speech to synthesize (default 4)")
    ap.add_argument("--eval-hours", type=float, default=0.5,
                    help="hours for the held-out eval split (default 0.5)")
    ap.add_argument("--noise", type=str, default="snr,background,gain",
                    help=f"comma list from {NOISE_TYPES}")
    ap.add_argument("--musan-path", type=str, default=None,
                    help="path to a local MUSAN download (real noise clips)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=("both", "train", "eval"), default="both",
                    help="build one split only. Useful on Windows: the HF "
                         "streaming HTTP session can go bad after a very long "
                         "pass ('not a socket'), and a fresh process per split "
                         "avoids re-downloading the first split to recover")
    ap.add_argument("--force", action="store_true",
                    help="rebuild a split that already exists (otherwise the "
                         "existing one is kept: a quick smoke run must not "
                         "silently replace the hours of data behind a model "
                         "you already trained)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # huggingface_hub logs every ranged HTTP request at INFO; the download is
    # long enough that those lines would bury the real progress messages.
    for noisy in ("httpx", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    FT_DIR.mkdir(parents=True, exist_ok=True)

    # The dataset is public, but an authenticated request gets a much higher
    # rate limit — reuse the project .env token (same one diarization needs).
    token = config.hf_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
    else:
        logger.info("no HF token in .env — downloading anonymously (slower)")

    musan = Path(args.musan_path) if args.musan_path else None
    if musan and not musan.is_dir():
        raise SystemExit(f"--musan-path {musan} is not a directory")

    types = [t.strip() for t in args.noise.split(",") if t.strip()]
    unknown = [t for t in types if t not in NOISE_TYPES]
    if unknown:
        raise SystemExit(f"unknown noise types {unknown}; choose from {NOISE_TYPES}")

    manifest = {}
    manifest_path = FT_DIR / "dataset_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}

    existing = {
        name for name in ("ft_train", "ft_eval") if (FT_DIR / name).is_dir()
    }
    wanted = {
        "both": {"ft_train", "ft_eval"},
        "train": {"ft_train"},
        "eval": {"ft_eval"},
    }[args.only]
    clobber = sorted(existing & wanted)
    if clobber and not args.force:
        raise SystemExit(
            f"{', '.join(clobber)} already exist in {FT_DIR} — pass --force to "
            "rebuild (existing splits and the model trained on them are kept "
            "otherwise)"
        )

    if args.only in ("both", "train"):
        train, train_seconds, train_info = collect_split(
            args.hours, "train", types, musan, args.seed
        )
        out = FT_DIR / "ft_train"
        train.save_to_disk(str(out))
        logger.info("saved ft_train -> %s", out)
        manifest.update({
            "train_clips": len(train),
            "train_hours": round(train_seconds / 3600, 3),
            "train_speakers": train_info["speakers"],
        })

    if args.only in ("both", "eval"):
        eval_ds, eval_seconds, eval_info = collect_split(
            args.eval_hours, "eval", ["snr", "background"], musan, args.seed + 1
        )
        out = FT_DIR / "ft_eval"
        eval_ds.save_to_disk(str(out))
        logger.info("saved ft_eval -> %s", out)
        manifest.update({
            "eval_clips": len(eval_ds),
            "eval_hours": round(eval_seconds / 3600, 3),
            "eval_speakers": eval_info["speakers"],
        })

    manifest.update({
        "noise": types,
        "musan": str(musan) if musan else None,
        "seed": args.seed,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
