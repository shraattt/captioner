# Changelog

Versions follow the phases the project was built in. Numbers in the tables are
measured on the development machine (RTX 4050 laptop, 6 GB VRAM, i5-12450HX,
16 GB RAM) — they are floors for a real lecturer's audio, not promises.

## Unreleased

### Added
- **Cancellation works everywhere, and across processes.** Stopping a job used
  to be an in-process event owned by the dashboard: the API and the CLI had no
  way to stop anything, and a worker started in its own console
  (`run_worker.bat`) could not be reached at all. The request is now a
  `cancel_requested` column that the worker holding the job polls, so
  `POST /api/jobs/{id}/cancel`, `python cli.py cancel <id>`, the dashboard's
  button and a Cancel button in the browser UI all stop a job wherever it is
  running. A queued job is called off outright (200); a running one stops at the
  next segment boundary (202, and `cancel_requested: true` in the JSON).
- **Live queue updates in the browser UI.** `GET /api/events` streams a small
  snapshot whenever the queue changes. The page used to re-read five endpoints
  every two seconds; it now opens one stream and keeps a slow poll only as a
  fallback. The stream is bounded and says goodbye before closing, so a request
  thread cannot be orphaned by a page that goes away.
- **Content de-duplication, using the `file_hash` column that until now nothing
  wrote or read.** Queuing, watching or adding a file whose content is already
  queued, running or done reuses that job instead of spending another full
  transcription on it: the API answers `200` with `duplicate: true` and the job
  it already has, the CLI says which job covers the file, and the dashboard
  reports what it skipped. The fingerprint is the file's size plus both ends
  (not a full-file read, which would mean hashing a two-hour lecture before it
  could be queued). Failed and canceled jobs do not count as duplicates - those
  are exactly the ones worth retrying - and the watch scanner follows the same
  rule, so a file whose attempt errored is picked up again.
- `GET /favicon.ico` serves the app icon; a browser tab is no longer unbranded.
- Security headers: `X-Content-Type-Options: nosniff` and `Referrer-Policy`
  everywhere, plus a `Content-Security-Policy` on the HTML page (its own origin
  only, no inline anything).
- Tests for the CLI, which had none: `cancel` and the `(stopping)` status hint.

### Changed
- **A new app icon.** `scripts/make_icon.py` now draws the mark at every size it
  ships (16, 20, 24, 32, 40, 48, 64, 128, 256) instead of embedding one 256px
  image for Windows to blur down to the size it actually draws, and it takes
  its colors from `ui/dashboard/theme.py`, so the icon is violet like the app
  rather than the green it had been. Detail is dropped as the size falls: the
  waveform goes below 48px and 16-24px is a plain violet tile with two text
  lines, because the full mark is mush that small.

### Added
- **The browser UI has a favicon.** `GET /favicon.ico` serves `captioner.ico`
  itself, so the tab is branded without a second copy of the binary in the tree.
- `pillow` in `requirements-dev.txt`: the icon script uses it to assemble the
  multi-size `.ico` (nothing at runtime depends on it).

## 0.7.0 — Phase 6: reach (engines, HTTP API, automation, live translation)

### Added
- **Pluggable engines (`captioner/engines/`).** Recognition and translation are
  now chosen from a registry instead of being imported directly:
  `faster-whisper` and `m2m100` stay the local defaults, and `openai` backends
  speak the `audio/transcriptions` and `chat/completions` wire protocols so any
  compatible server works (OpenAI, Groq, LM Studio, faster-whisper-server,
  vLLM). Selection is `CAPTIONER_ASR_ENGINE` / `CAPTIONER_MT_ENGINE`; every
  backend reports `available()` with a reason, and `probe()` is exposed through
  the API and the status bar. The API backends take an injected HTTP transport,
  so their request building, response parsing and error mapping are tested with
  no network at all.
- **HTTP API + browser UI (`captioner/api.py`, `ui/web/`).** A stdlib-only JSON
  API over the queue: health, engines, models, languages, jobs (list, create,
  detail, captions as json/srt/vtt/text, retry, delete, translate) plus an
  OpenAPI 3.0 description and a small browser page. `python cli.py api` runs it
  standalone; **Tools → Toggle web API + UI** runs the same server inside the
  dashboard. Loopback by default, and it refuses to bind elsewhere without a
  token, because it can queue local files and return their transcripts.
- **Watch folders + webhooks (`captioner/automation.py`).** One scanner feeds
  both the dashboard's watch toggle and the new `python cli.py watch`, with
  `CAPTIONER_WATCH_DIRS` for extra folders, idempotent queueing, and a
  two-second settling window so a half-copied download is never captioned.
  Settled jobs POST a JSON event to `CAPTIONER_WEBHOOK_URL` (optional shared
  secret header); delivery is best-effort and can never change a job's outcome.
- **Live translation.** The overlay can translate live captions as sentences
  finish, on a background thread with the same on-disk cache the batch path
  uses, chosen from **Live → Live translation** (or
  `CAPTIONER_LIVE_TRANSLATE_TO` headless). The transcript window reads as a
  bilingual transcript. A backend that cannot start is reported once and
  disables itself; captions keep flowing in the source language.

### Changed
- **The dashboard is a package (`ui/dashboard/`).** The 2,000-line single
  module became `window.py` plus mixins for jobs, preview, live and worker
  control, with `theme.py`, `formatting.py`, `widgets.py`, `services.py` and
  `constants.py` beside them - the largest module is now ~620 lines and each
  file owns one concern. `from ui.dashboard import Dashboard` and
  `PlaybackControls` still work; the app is launched with
  `python -m ui.dashboard` (the `.bat`, `.vbs` and shortcut launchers were
  updated).
- **Comments are plain ASCII.** Decorative comment banners are gone and
  typographic dashes were normalized inside comments (UI strings keep their own
  glyphs - those are product design, not code clutter).
- **Lint-clean.** `ruff check .` passes with no suppressions beyond two
  documented per-file `E402` exemptions for the files that deliberately put the
  project root on `sys.path`. Along the way this fixed a handful of real
  issues: a lambda in the hotkey path that captured loop variables, an
  undefined `SAMPLE_RATE` in a test helper, a duplicated status-color dict, a
  dead `audio_secs` computation, and a version check that could never be false.
- One version source: `captioner.__version__` feeds the dashboard and the API.

### Fixed
- `python cli.py languages` crashed on every call (the shortlist became a
  `code -> name` map and the unpacking was never updated).
- A first-run-only bug in `run_captioner.bat` (a stray blank line between every
  statement) and LF line endings that `cmd.exe` misparses.

### Verification
- 325 tests pass (324 passed, 1 skipped), including the new engine, API,
  automation and live translation suites. The webhook tests talk to a real
  loopback HTTP server; the API tests drive a real server through a real HTTP
  client (routing, status codes, token auth, cross-origin writes, path
  traversal, oversize and malformed bodies, queueing end to end).

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
