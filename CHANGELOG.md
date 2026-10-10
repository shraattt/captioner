# Changelog

Versions follow the phases the project was built in. Numbers in the tables are
measured on the development machine (RTX 4050 laptop, 6 GB VRAM, i5-12450HX,
16 GB RAM) — they are floors for a real lecturer's audio, not promises.

## 0.6.0 — Phase 4 + Phase 5: fine-tuned models, and the rest of Phase 2

### Added
- **Word alignment (Phase 2).** `captioner/align.py` re-times Whisper's words
  against `facebook/wav2vec2-base-960h` with a numpy Viterbi CTC forced
  alignment over 20 ms frames. Runs before segmentation, so caption blocks are
  built from tight timings. Conservative by design: spans that are implausible
  (>4× the word's length, or drifted >0.6 s from Whisper's estimate) or a model
  that will not load both fall back to Whisper's timings. Measured on the 13.9 s
  fixture: 31/31 words retimed, no overlapping pairs.
- **Subtitle translation (Phase 2).** `captioner/translate.py` writes
  `captions.<lang>.srt`/`.vtt` beside the originals with the same timings and
  speaker labels. Lazy model loading, per-line on-disk cache, and beam-4
  decoding with an n-gram repetition block (greedy decoding looped on short
  subtitle lines — a real Tamil run produced "ஹலோ ஹலோ ஹலோ ஹலோ" for "Hello and
  welcome." before this). The model is overridable per call or via
  `CAPTIONER_TRANSLATE_MODEL`.
- **Phase 4 fine-tuning.** `scripts/ft_build_dataset.py` (noisy LibriSpeech via
  audiomentations, streamed), `scripts/ft_train.py` (Seq2SeqTrainer, bf16 +
  gradient accumulation + checkpointing for 6 GB), `scripts/ft_eval.py`
  (WER + realtime-factor table across any mix of HF and CT2 checkpoints).
- **Phase 5 conversion + live swap.** `scripts/ft_convert.py` converts the
  checkpoint to CTranslate2 and smoke-tests it through the real pipeline;
  `captioner/models.py` discovers `data/ft/*_ct2` automatically and offers it
  in **Live → Live model** alongside the built-in sizes (`tiny.en` stays the
  default).
- **Per-job options.** `align`, `translate_to` and `diarize` live on the job row
  (migrated in place), so changing a checkbox can never reconfigure a job that
  is already queued. Exposed in the **Options** popover and as CLI flags.
- **CLI:** `add --diarize --align --translate-to LANG`, `translate <job> --to
  LANG` (post-hoc, no re-transcribe), `languages`.
- `pyproject.toml` (metadata + pytest/ruff config) and this changelog.

### Measured (100 held-out noisy clips)
| model | WER | realtime factor |
| --- | --- | --- |
| `tiny.en` (live default) | 21.8% | 21.7× |
| `base.en` | 17.6% | 19.1× |
| `small.en` | 16.3% | 13.9× |
| fine-tuned `base_en_noisy` (HF) | **10.4%** | 24.9× |
| fine-tuned `base_en_noisy` (CT2 fp16) | **12.8%** | 44.3× |

Training 3 epochs over 4.25 h of noisy LibriSpeech took 5.6 minutes; the
CTranslate2 build is 2× faster than the checkpoint it came from and costs
~2.4 WER points.

### Fixed
- `transcribe.get_model` mangled a local checkpoint path into a bogus repo name
  (`Systran/faster-whisper-D:/path/...`), which made converted models
  unloadable — the Phase 5 blocker.
- The UI screenshot tool seeded its demo jobs into the **real** queue, leaving
  fake rows in the user's dashboard; it now uses a sandbox database.
- `datasets` streaming needs `torchcodec` for its own audio decoding, which this
  project deliberately uninstalls (broken Windows DLLs) — the dataset builder
  now streams with `Audio(decode=False)` and decodes through soundfile.
- `run_captioner.bat` had a blank line between every statement.
- The dataset builder silently overwrote an existing split (a smoke run could
  replace hours of data behind an already-trained model); it now refuses
  without `--force`.

## 0.5.0 — Phase 3 hardening and Phase 2 diarization
- pyannote 4.0.7 diarization behind the **Identify speakers** option:
  `[SPEAKER_00]:` labels in SRT/VTT and the editable preview; caption blocks
  never mix speakers. Gating and torchcodec both bypassed.
- Live accuracy fix: overlapping-window dedup froze duplicated words (WER ~300%
  on fixtures); text+time-tolerance dedup brought it to 2.3% (tiny.en, beam 1).
- Silence energy gate and no-speech/confidence filters: 40 s of continuous fan
  noise now yields zero hallucinated words.
- Sentence-coherent display (frozen buffer moves whole sentences), live
  transcript window, overlay appearance controls, live model picker.
- Noise-floor auto-calibration per live session.
- App icon, About dialog, file logging, crash dialog; silent VBS launcher.
- Stress tests: 40 live sessions with no leak; batch + live sharing 6 GB GPU.

## 0.4.0 — Phase 3: live caption overlay
- Capture (SoundCard/WASAPI mic + system loopback) into a 16 kHz ring buffer,
  12 s sliding window re-decoded every 1.5 s, commit-frontier stabilization.
- Draggable always-on-top overlay; global **Ctrl+Alt+L** hotkey; in-app
  **Ctrl+Alt+Y** fallback.

## 0.3.0 — Phase 1 polish
- Model pre-download with byte progress, cancel, heartbeat + stale-job recovery,
  caption editing with atomic SRT/VTT re-export, job context menus.
- `scripts/bench_1h.py` and the first measured throughput table (tiny.en 18.4×,
  medium 4.1× realtime).

## 0.1.0–0.2.0 — Phase 1: batch captioning
- SQLite-backed queue and separate worker process, faster-whisper transcription
  with word timestamps, constrained-greedy caption segmenter, SRT/VTT writers,
  PySide6 dashboard with table + preview + audio transport.
