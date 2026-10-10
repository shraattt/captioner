# Captioner — backlog & reminders

Things deliberately deferred, with the trigger for picking each up.

## Option A — Consolidate on real usage (NEXT, when the user is around)
- [ ] The user runs a **real live session** (mic or system audio) and reports
      what annoys them; fix those first.
- [ ] **The Phase 4 gate is still unmeasured on real audio.** Everything below
      is synthetic-noise LibriSpeech; the number that decides whether the
      fine-tune was worth it is on the user's own recordings:
      `python scripts\wer_check.py <recording> <reference.txt> --live --live-model base_en_noisy`
      (compare against `--live-model tiny.en` on the same file).
- [ ] If real WER is still bad, try: MUSAN noise instead of procedural
      (`--musan-path`), 20 h instead of 4 h (`--hours 20`), or `small.en` as
      the fine-tune base (`--model openai/whisper-small.en`).
- [ ] Decide what to expand based on what real usage reveals.

## Option B — Alignment tuning (only if real audio shows drift)
- [ ] `align.MAX_SHIFT_SECONDS = 0.6` is the guard that rejects an alignment
      that moves a word too far from Whisper's estimate. On long, repetitive
      audio a genuine word can sit further out than that and get skipped; if
      the user reports stale timing on a specific file, raise it and re-check
      `scripts\wer_check.py`.

## Option C — Translation quality
- [ ] `captioner/translate.py::LANGUAGES` is a 27-language shortlist of
      M2M100's 100 codes. Add entries as needed; M2M100 has **no Telugu**, so
      `te` is deliberately absent rather than silently broken.
- [ ] **418M quality is uneven by language** (measured, not guessed): Hindi
      comes out clean ("नमस्ते और स्वागत है।"), Tamil is weaker on longer
      lines. When a language reads badly, try the bigger model without a code
      change:
      `python cli.py translate <job> --to ta --model facebook/m2m100_1.2B`
      (or set `CAPTIONER_TRANSLATE_MODEL`). Overrides are namespaced in the
      cache, so a new model re-translates instead of reading the old lines.
- [ ] Greedy decoding looped on short lines (a Tamil run gave
      "ஹலோ ஹலோ ஹலோ ஹலோ" for "Hello and welcome."). Fixed with beams=4 +
      `no_repeat_ngram_size=3` (both are `Translator` knobs); the user's cache
      was rebuilt so the old bad lines are gone. If a language drifts into
      repetition anyway, raise `NO_REPEAT_NGRAM`.

## Done since the user last checked (this cycle)
- **Phase 2 — word alignment shipped.** `captioner/align.py` re-times Whisper's
  words against `facebook/wav2vec2-base-960h` with a numpy Viterbi CTC forced
  alignment (~60 lines, no whisperx dependency) over 20 ms frames. Runs before
  segmentation, so caption blocks are built from tight timings. Conservative by
  design: implausible spans (>4× length or >0.6 s drift) fall back to Whisper,
  and a model that will not load degrades to Whisper timings instead of failing
  the job. Measured on the 13.9 s fixture: 31/31 words retimed, 0 overlaps.
- **Phase 2 — translation shipped, on M2M100 not NLLB.** **transformers 5
  removed NLLB's text model** (and the task `translation` pipeline with it), so
  the plan's NLLB became `facebook/m2m100_418M` — same encoder-decoder family
  NLLB was built on, 100 languages, one constant to change if NLLB returns.
  Lazy (loads on first use, ~1.8 GB), cached per (language, line) in
  `data/cache/translations.json`, speaker labels preserved verbatim, and a
  failed translation cannot throw away a finished transcript. Decoding is
  beam-4 with an n-gram repetition block (greedy looped on short subtitle
  lines), and the model is overridable per call / via
  `CAPTIONER_TRANSLATE_MODEL`.
- **Phase 4 — fine-tuning shipped and measured.** 4.25 h / 1200 clips of noisy
  LibriSpeech (20 speakers; audiomentations SNR + background + gain + time-mask)
  via streaming — a few hundred MB instead of the 6 GB full split. Training
  `openai/whisper-base.en`: 3 epochs in **5.6 minutes** (~1.2 s/step, batch 8 +
  accum 2, bf16, gradient checkpointing). On 100 held-out noisy clips:
  `tiny.en` 21.8% → `base.en` 17.6% → `small.en` 16.3% → **fine-tuned 10.4%
  (HF fp32/BF16) / 12.8% (CT2 fp16)**. Quantization costs ~2.4 points (measured).
- **Phase 5 — CTranslate2 conversion + live swap shipped.** `scripts/ft_convert.py`
  converts, copies `ft_meta.json` across, and smoke-tests through
  `transcribe_words` (the live path). `captioner/models.py` auto-discovers
  `data/ft/*_ct2` and offers it in **Live → Live model**; `tiny.en` stays the
  default. `transcribe.get_model` now resolves local directories and explicit
  repo ids (it used to mangle a path into a bogus repo name — that was the
  Phase 5 blocker).
- **UI:** the Options popover gained **Align word timings (wav2vec2)** and a
  **Translate captions** picker; the status line reports what a queued batch
  was set up with. Job options are stored per row (`align`, `translate_to`
  columns, migrated in place), so changing a checkbox never re-configures a job
  that is already queued.
- **CLI:** `add --diarize --align --translate-to LANG`, `translate <job_id>
  --to LANG` (post-hoc, no re-transcribe), `languages`.
- Tests: **245 pass, 1 skip** (was 203) — the new suites are
  `tests/test_align.py` (CTC alignment against hand-built frame paths),
  `tests/test_translate.py` (+ the new SRT parser), `tests/test_models.py`
  (registry), plus pipeline-level checks that `align`/`translate_to` reach the
  worker and that a translation failure leaves the job `done`.

## Known quirks
- This Windows session drops *synthesized* keystrokes (policy), so the
  hotkey keystroke test skips; press Ctrl+Alt+L by hand once to confirm.
- pyannote 4.x would redirect to its new gated community-1 model; we pin the
  classic 3.1 recipe explicitly. If speakers stop working after a pyannote
  upgrade, check `captioner/diarize.py::_load_pipeline`.
- **torchcodec must stay uninstalled** (its Windows DLLs do not load). That
  also means `datasets`' own audio decoding is unavailable, so
  `ft_build_dataset.py` streams with `Audio(decode=False)` and decodes FLAC
  through soundfile itself. If a dataset script ever fails with
  "to support decoding audio data, please install 'torchcodec'", that is why.
- **One-shot dataset builds can die at the very end on Windows** with
  `WinError 10038 ... not a socket` (the HF streaming HTTP session goes bad
  after a long pass). The splits that finished are saved and the manifest is
  written, so recovery is `--only eval` / `--only train` in a fresh process
  rather than starting over.
- `spellcheck`-style review of translated output is manual; nothing validates
  translation *quality* automatically.
