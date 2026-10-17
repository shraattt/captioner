# Captioner — backlog & reminders

Things deliberately deferred, with the trigger for picking each up.

## Option A — Consolidate on real usage (NEXT, when the user is around)
- [ ] The user runs a **real live session** (mic or system audio) and reports
      what annoys them; fix those first.
- [x] **First real-audio measurement, 2026-10-14** — a 3 min 59 s screen
      recording of a student presenting "merge sort versus quick sort", run
      through `scripts/bench_real_audio.py --live-models tiny.en,small.en`.
      Batch `distil-large-v3`: 12.5 s wall (19.1x real time). Live decode cost
      0.26 s (tiny.en) / 0.52 s (small.en) per 12 s window against a 1.5 s
      stride, so the live loop has headroom on this GPU. Caption age in audio
      time: 3.43 s / 3.20 s median, 10.8 s / 10.6 s p95 — read at the time as a
      pause rather than the GPU. The 2026-10-15 follow-up found the real cause
      (the model re-writing the same instant of audio, each re-write frozen as
      extra text) and fixed it; see the entry further down and the README's "On
      real audio" section.
- [x] **`small.en` is now the live default**, on two recordings rather than one.
      Against the batch transcript: `tiny.en` disagreed on 27.5% / 23.1% of words
      (12.2% / 8.7% of characters), `small.en` on 14.7% / 13.8% (6.4% / 5.7%).
      Decode p95 1.07 s / 1.32 s against the 1.5 s stride, so the slower model
      still fits on the 6 GB card. `tiny.en` remains in **Live → Live model**.
      Both recordings are the same session in the same room, so the *ranking* is
      solid and the *sizes* are not: another speaker, mic or language is still
      unmeasured.
- [ ] **A real WER number still does not exist.** The script measures agreement
      between two models, not accuracy, because no ground truth was available.
      Either type a reference for one short file, or use
      `python scripts\wer_check.py <recording> <reference.txt> --live
      --live-model base_en_noisy` (compare against `--live-model tiny.en` on the
      same file). This is also the answer to whether the fine-tune was worth it.
- [ ] **Does the batch path drop speech?** The batch transcript has 425 words
      for 238.7 s (107 words/min) while live reports 460-469 (116-118/min; it
      reported 538-615 before the freeze band stopped duplicate re-writes
      inflating it). On recording 2 the two paths agree - the batch has 294
      words in 185.3 s (95 words/min) against live's 309/314 (100-102) - so the
      gap is not a constant: either the live models invent ~9% of their words on
      recording 1's material, or `distil-large-v3` with `vad_filter=True` drops
      segments there. A ground-truth reference on that file settles which, and
      the answer decides whether the batch defaults are the problem.
- [ ] **Duplicate words survive in live text.** The export clamps times but not
      text: a second jittered estimate that passes `_is_dup`'s 0.7 s tolerance
      is still written, visible in the transcript overlay as a stammer. The
      freeze band removed the *late* ones - all 159/79 refusals on recording 1
      and 84/49 on recording 2 were second estimates of a moment already
      captioned - which took adjacent same-word pairs from 2/7 to 0/6 on
      recording 1. What is left is a variant that arrives *inside* the band, so
      the fix to weigh is still a text-aware dedup at the freeze boundary (it
      must not eat a genuinely repeated word).
- [ ] **A live transcript stops early.** The saved transcript's last cue ends
      at ~215 s of a 239 s recording (76-79% timeline coverage). Confirm the
      tail is silence rather than dropped audio, and if it is dropped, why.
- [x] **Caption age p95 10.6 s is fixed - and it was never the pause.** The
      guess above (the gate holds a word through a quiet stretch) was wrong: the
      ages came from the model re-writing the *same instant* of audio over and
      over, each re-write arriving late enough to carry a large age and being
      frozen as extra text. `_freeze` now refuses a first-time word further than
      `FREEZE_BAND_SECONDS` behind the margin, widened to cover whatever a pause
      flush or a slow decode skipped. Measured: p95 10.80 → 4.58 s (tiny.en) /
      10.56 → 4.31 s (small.en), median 3.43 → 3.02, disagreement with batch
      59.5% → 27.5% of words, worst opening word 11.4 → 5.2 s, and coverage
      79.4% → 73.1% as duplicates stopped covering gaps.
- [x] **A pause's idle stride no longer outlives the pause - and all three ways
      to turn that into a faster caption have now been measured, the third one
      shipping.** While
      nobody talks the loop waits `IDLE_STRIDE_SECONDS` (3 s) so the GPU rests,
      which also meant a speaker returning 0.2 s into that wait waited the
      remaining 2.8 s though their words were already in the ring. The capture
      keeps a 50 ms meter (`MicCapture.sound_level`, the gate's own resolution)
      and the wait ends on it (`ui.overlay.wait_for_sound`, with the loop moved
      to `ui.overlay.run_live_session` so it can be timed without Qt). Three
      passes per configuration with the pass order rotating (recording 1,
      `small.en`, 18 pauses) put the loop's own share of the wait 0.16 s lower on
      average (median 0.17 s, sd 0.44 s, 95% CI ±0.20 s, 12 pauses faster and 6
      slower, sign-test p=0.24) - inside the 0.12-0.21 s drift the *same*
      configuration shows between repeats, so the repeats do not settle it. Six
      earlier single pairs said 0.2 s and put the tail 0.7-1.9 s better; the
      repeats do not reproduce that tail (of the three pauses the old loop waited
      longest on: 0.65 s faster, 0.76 s faster, 0.65 s slower). The average is
      small because the loop was never the gate: the model will not report a
      word with ~0.1 s of audio behind it, so the words are caught by the next
      decode at ~1.6 s (the stride) either way. **A 4 s resume window was tried
      and does not help** (`short` pass, `LiveEngine.resume_window_seconds`):
      −0.03 s of loop wait and −0.04 s of caption over the repeats (4 pauses
      faster, 14 slower), so it ships off by default. What it does buy is a
      cheaper decode - 42% of a full one (0.25 s against 0.59 s,
      `live_resume_audit.py --window-cost`) - and that is what the third lever
      needed: **decoding on a shorter stride for the first moments after a
      resume** is what shipped (`burst` pass, `LiveEngine.resume_stride_seconds`)
      - 0.5 s for 2 s of resumed audio, reading that cheap window. It took
      **0.40 s off the loop's own wait and 0.37 s off the caption** against the
      wake alone (median 0.37 / 0.30 s, 12 of 18 pauses faster, 6 slower; the
      caption 13 of 18), against a 0.20 s repeat-to-repeat floor, with every one
      of the three repeats better than every repeat without it. It also needed
      `live.resume_instant`: the wake's own signal (a window that *ended* in
      quiet) never fires for the short pauses this recording is made of, because
      the loop sleeps through them - the pause has to be found inside the window.
      Cost: ~2 extra decodes per pause, 0.24 s each (193 per 4-minute pass
      against 154).
- [x] **The metric is now measured repeatedly, and it turns out the repeats do
      not settle it either.** `live_resume_audit.py --repeats N` (default 3)
      replays each configuration N times with the order rotating, averages the
      repeats per pause, and reports the paired difference with sd, a 95%
      interval and an exact sign test, beside the repeat-to-repeat spread of each
      configuration's own mean. On recording 1 the wake's loop-wait effect is
      0.16 s with that spread at 0.21 s and the sign test at p=0.24: three
      repeats per configuration are not enough to resolve it, and the same
      applied to the 4 s resume window. It *did* resolve the post-resume cadence
      (0.40 s against that same floor - see the item above), which is the
      argument for reporting the floor beside every number rather than trusting
      a difference because it is positive. The floor the numbers now report is also
      what kills a whole class of previous claims - a single run's mean moved
      more than the effect it was being used to measure. What would settle it:
      more repeats, more recordings, and pauses that do not arrive on top of a
      machine whose decode cost moves 2× in one sitting.
- [ ] **The wake's own latency is the capture block size.** Speech is only
      visible to the meter once a whole 100 ms block containing it is written,
      so ~0.1 s of every post-pause wait is the buffer, not the loop
      (`MicCapture(blocksize_ms=...)`). Worth revisiting only if post-pause
      captions still annoy after the stride change above.
- [ ] **`COMMIT_MARGIN` is where the remaining ~3 s lives, and lowering it is
      not worth a code change yet.** The stride (1.5 s), not the margin, sets
      the granularity: with a margin of 2 s the first freeze lands on the same
      grid, so the median would fall by ~0.4 s for 1 s less revision room
      before a word is frozen. Revisit only if the settling time turns out to
      annoy in real use, and then decide it on a reference transcript.
- [ ] **The device loop's cadence is a promise the measurements lean on.**
      `bench_real_audio.py` replays one stride per decode; the app runs at
      decode + wait, which the new `stride_wait` trims to a real 1.5 s period.
      If a future GPU is slower than the stride, the loop can only fall behind -
      check `engine.suggested_stride` and the decode p95 in the benchmark before
      raising `WINDOW_SECONDS` or the stride.
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

## Option D — Shipping it (packaging leftovers)

- [x] **One file that installs the whole app** (`scripts/package.py`,
      `scripts/installer.py`): frozen folder + setup `.exe`, shortcuts, `.env`
      template, data folder, `--portable` / `--silent` / `--uninstall`.
- [x] **The full build no longer carries the CUDA runtime twice.** torch/lib
      already ships cuBLAS, cuDNN and nvrtc, so the nvidia wheels were 0.9 GiB
      of duplicate DLLs in the 5.7 GiB folder; a full build leaves them out
      now and installs 4.8 GiB from a 3.0 GiB setup. A packaged GPU job with
      word alignment and French translation was re-run afterwards to prove
      nothing was lost.
- [x] **`--target` and `--uninstall` refuse a folder that is not Captioner** -
      a drive root, the home folder, a file, or a non-empty folder without
      `Captioner.exe`/`uninstall.cmd`. The shipped setup was pointed at a decoy
      folder with one file in it: both commands refused and the file survived.
- [x] **The app opens on a title card** (CAPTIONER / project by sharat) and
      fades into the window; `--no-splash` or `CAPTIONER_NO_SPLASH=1` skips it.
- [ ] **The wizard window itself is only exercised by hand.** The install
      *steps* have tests (26 of them, no PowerShell involved); the tkinter
      window around them has none, because a GUI test harness would be a
      dependency this project does not have. Before the next release: run it on
      a laptop that has never had Python, and try Cancel/Close mid-install.
- [ ] **Models are still downloaded on the target machine** (~1.5 GB on first
      launch, and the machine must have a network once). An option to bundle the
      weights into the setup would make an air-gapped install possible; it also
      roughly doubles the file, which is why it is not the default.
- [x] **`About this build` exists now, in the three places it was needed.**
      `captioner/features.py` answers what is installed; the *Options* popover
      greys out what this build cannot run (reason in the label, fix in the
      tooltip, preference corrected at startup), a job that reaches the feature
      anyway gets a sentence instead of a traceback, and `Help → About` lists
      what is here, and the setup window now says what it is about to install
      (stamped by the build: slim/full, cuda/cpu, and the three features a slim
      build leaves out). What is still missing is a *link*: someone who
      installed the slim build because it was the file they had, and now wants
      speaker labels, has to find the full build themselves - the note says what
      is missing but not where to get it.
- [ ] **No code signing**, so SmartScreen warns once per download. A certificate
      costs money per year; documented instead, with the folder build as the
      alternative that never triggers the dialog.
- [ ] **Per-user install only.** `%LOCALAPPDATA%\Programs` needs no elevation,
      which is the right default for a laptop; an IT-managed machine that wants
      `Program Files` would need an elevation path the installer deliberately
      does not have.
- [ ] **No update mechanism.** A new version means running a new setup (it
      replaces the folder and keeps the data). Anything more (checking a URL,
      in-place patch) is a service this project does not want to run.

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
