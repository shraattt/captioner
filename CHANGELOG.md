# Changelog

Versions follow the phases the project was built in. Numbers in the tables are
measured on the development machine (RTX 4050 laptop, 6 GB VRAM, i5-12450HX,
16 GB RAM) — they are floors for a real lecturer's audio, not promises.

## Unreleased

### Added
- **The app opens on a title card.** *CAPTIONER* over *project by sharat* - the
  name the user asked for - in the same violet-glass language as the window,
  with the icon above it, a slow violet sweep at the bottom and the version in
  the corner. It covers the moment the dashboard spends reading the queue (the
  window is built while the card is up), stays a 1.1 s minimum so it reads as
  an opening rather than a flicker, then fades over 240 ms and only *then*
  shows the window - never both at once. `--no-splash`, `CAPTIONER_NO_SPLASH=1`
  and `--self-test` skip it, and Help -> About carries the same credit. Checked
  in a real event loop rather than by building the widget: 2.5 s after `main()`
  under Qt's offscreen platform the card was gone and the Dashboard was visible
  and closed cleanly; the frozen build was started and left running for 10 s
  with no startup error and a created queue, which is what the splash path does
  before the window appears.
- **The full build ships as one file too, and it is measured rather than
  estimated.** `--full` had only been described before; building it here took
  610 s and produced a **5.7 GiB** folder (6,076,335,467 bytes) carrying torch
  2.11+cu128, transformers, pyannote and sentencepiece, with `version.txt`
  stamped `full`/`cuda`. Packing that folder into the one-file setup then died
  at the very end with `struct.error: 'I' format requires 0 <= number <=
  4294967295` - PyInstaller's one-file archive addresses its contents with
  32-bit offsets, so it cannot carry 4 GiB, however the files are arranged.
  `scripts/package.py` now compresses a payload that big into the portable
  `.zip` (3,721,194,486 bytes in ~13 min) and embeds that single archive,
  telling PyInstaller to store it instead of deflating it a second time; the
  setup then builds in 61 s and comes out **3,809,148,020 bytes** (4.8 GiB /
  3,179,586,136 bytes after the duplicate-CUDA fix below). The archive
  is reused as the portable download, and a payload older than the folder is
  recompressed rather than trusted. `scripts/installer.py` unpacks it at
  install time, dropping the archive's `Captioner/` level; a compressed payload
  installs, replaces an older install and uninstalls like any other, which is
  now tested. On the shipped file: `--version` answers *"Captioner 0.7.0, 4.8
  GiB to install"* and the full-build note; a silent install from a *relative*
  `--target` now resolves it (it used to write that relative path into `.env`,
  which the app would have read from its own folder) and writes 4.8 GiB in
  about two minutes; the installed copy passes `--self-test` and captions a real file
  with `--align --translate-to fr`; `--uninstall` removed all of it. First run
  with an empty Hugging Face cache downloaded 2.2 GB - Whisper `tiny.en` 75 MB,
  wav2vec2 360 MB, M2M100 1.9 GB - and took 20 minutes for a 14 s clip; the
  same job from cache took 31 s. Models are not in the installer, and the
  README now prices the first job instead of implying one download.
- **A packaged build can run one job without opening a window.**
  `Captioner.exe --job lecture.mp3 --align --translate-to fr` queues that file,
  claims *that* job (`worker.claim_job`, new, so an older queued file cannot be
  run instead), runs exactly the pipeline the dashboard runs and prints the
  caption files - or the sentence a failed job got. Exit codes are 0/1/2
  (done/failed/bad arguments), and the run is logged to `logs/captioner.log`.
  It is also how a build is proven: a slim build refuses `--align` in the same
  words a job would, and the full one aligns. Two things only running it could
  show: a windowed build has no console, so the mode attaches to the caller's
  one (`AttachConsole`, or the null device) - and after a cold full-build
  translation the process sat for minutes tearing down its CUDA context, so a
  finished job now flushes and leaves the process immediately. The hub's HTTP
  client was also logging every HEAD request into that console (17 KB before
  the first word); library request logging is now warning-level.
- **It can now leave this machine as one file.** `scripts/package.py` freezes the
  app with PyInstaller and `scripts/installer.py` turns the frozen folder into
  **a single setup `.exe`** that needs no Python, no pip, no administrator rights
  and no network on the other laptop:

  ```powershell
  .venv\Scripts\python scripts\package.py --installer        # ~1.5 GB
  ```

  The setup copies `Captioner` into `%LOCALAPPDATA%\Programs\Captioner`, adds
  Start Menu (app, *data folder*, **Uninstall**) and Desktop shortcuts, writes a
  commented `.env` template, creates the data folder and offers to start the app;
  `--silent`, `--target`, `--portable`, `--uninstall --purge-data` and
  `--version` are all there for unattended use, and a silent run also writes
  `%LOCALAPPDATA%\Captioner\logs\setup.log`, because a windowed build has no
  console and nobody is watching an unattended install. Shortcuts go through
  `WScript.Shell` and are **verified on disk** before they count; a link that
  cannot be made is a warning, never a failed install. `--portable` keeps jobs,
  captions, settings and uninstaller inside the install folder (the *Portable*
  checkbox in the window) for a USB stick.

  **Slim by default is a measurement, not a preference.** Torch is 4.2 GB
  installed and is only reached by local translation, speaker labels, word
  alignment and fine-tuning, so the default build leaves it out (installer
  ~1.5 GB, of which 740 MB is the CUDA runtime: cuBLAS and nvrtc travel with the
  app so a machine with a driver but no CUDA Toolkit still gets the GPU).
  `--full` installs all of it - 4.8 GiB, shipped as a ~3.0 GiB setup. `--no-cuda`
  drops the CUDA wheels for a CPU-only machine. `--zip` adds a portable archive, `--smoke` starts the built
  app with `--self-test`, `--smoke-setup` installs and uninstalls the built setup
  into `build/` and `--specs-only` writes the recipes without building them.
  Measured here (RTX 4050 laptop, warm PyInstaller cache): the app folder builds
  in 70-76 s and comes out 1.3 GiB; the one-file setup takes another 245 s and
  comes out 1.4 GiB (1,491,505,188 bytes); `--smoke` starts the frozen app
  offscreen and gets *"dashboard self-test OK: window built, rendered, and closed
  cleanly"* back from it; `--smoke-setup` installs and uninstalls the built setup
  into `build/` and reports OK; and `Captioner-Setup-0.7.0.exe --version` answers
  *"Captioner 0.7.0, 1.3 GiB to install"* - so the payload, the size and the
  version in the shipped file are all read from the thing that was actually
  built. What running the artifacts for real then caught, none of which the unit
  tests could:
  * excluding `pip`/`wheel`/`setuptools` kills the build outright (PyInstaller
    imports them itself, and excluding an already-imported module raises), and
    the setup spec needed the payload passed by keyword;
  * **the wizard's four checkboxes passed a `BooleanVar` as their parent frame**,
    so the window could not be built at all. The stale setup's fallback did what
    it says - *"the setup window could not open (...); installing without it"* -
    and installed the app anyway, which is how the real install (Start Menu
    entries, Desktop icon, `.env` template, data folder, `uninstall.cmd`) got
    checked end to end in `%LOCALAPPDATA%`. The wiring is fixed, a test now
    builds the whole wizard window with the event loop stubbed out (skipped
    where there is no display), and the stale install was removed with the
    uninstaller it had written;
  * the generated `uninstall.cmd` worked when run the way its Start Menu entry
    runs it (app folder and shortcuts gone, data kept and named on screen) but
    printed *"The batch file cannot be found."* after deleting itself, and left
    an empty Start Menu folder behind. It now deletes itself from a second copy
    of `cmd` and removes the folder it owned; both were re-verified by running
    it, not by reading it.
- **The setup says what it is about to install, before it installs it.** The
  build now stamps `version.txt` with three lines - version, `slim`/`full`, and
  `cuda`/`cpu` - and the setup window (and `--version`) reads them back: *"Slim
  build, with GPU support (CUDA included). Transcribing, live captions, editing,
  the queue and the API are all included. Local translation, speaker labels and
  word alignment are not - they need PyTorch, which this build leaves out (a full
  build installs 4.8 GiB)."* The two builds differ by 3.5 GiB and three features, and the
  full one is a different download rather than a checkbox, so the window is the
  last place where that choice can be stated. It also gained Alt-key mnemonics
  on its buttons (`&Install`, `&Cancel`, `&Browse...`), on top of the existing
  Enter/Escape bindings.
- **A build that cannot run a feature no longer pretends it can.** The slim
  installer leaves PyTorch out, and that is exactly the kind of thing a user
  used to discover twenty minutes into a job. `captioner/features.py` is now the
  one place that answers "is this installed?" - a `find_spec` lookup, so nothing
  is imported and no model is touched - and three things use it:
  * the Options popover greys out *Identify speakers*, *Align word timings* and
    *Translate captions* **with the reason in the label itself** ("… - not in
    this build"), so the limits are readable in a screenshot and to a screen
    reader, and puts the missing packages, the fix and what still works in the
    tooltip and the accessible description;
  * the preferences are corrected when the window opens, so an option
    remembered from a different installation cannot be inherited by a queue that
    cannot honour it;
  * a job that reaches the feature anyway - the API and the CLI can enable one
    with no window at all - fails with a `RuntimeError` naming what is missing,
    what to install and what happens instead, rather than an `ImportError`
    traceback in a log. `Help → About` now also says what this build carries.
- **An installed build keeps its data in the user's profile, not beside its own
  executable.** `captioner/config.py` decides this in one place: a checkout
  still writes to its own `data/` (every script and every line of documentation
  is unchanged), a packaged build uses `%LOCALAPPDATA%\Captioner\data`. Writing
  beside the executable would put a user's jobs in a folder that needs
  elevation and that an uninstall or a repair is allowed to delete. The same
  rule covers the `.env` (`secrets.env_paths()` reads one beside the executable,
  then the profile's, then the bundle's) and the translation cache, which used
  to be derived from the code folder and would have failed to save in a
  read-only install.
- **`--portable` really does move the data folder, and the load order was wrong.**
  The `.env` is now read at the *top* of `captioner.config`, before `DATA_DIR` is
  computed: it used to be read on first use, which was after the decision that
  `CAPTIONER_DATA_DIR` in it was meant to make. A test starts a fresh interpreter with
  the installer's portable `.env` and asserts where `DATA_DIR` lands. An empty
  `CAPTIONER_DATA_DIR=` in the environment no longer shadows the file's value
  either (that is how a launcher clears a variable).
- **A packaged build fails loudly instead of silently.** A windowed build has no
  console, so an import error in a bundle used to mean a blank screen and
  nothing else: `scripts/app_window.py` (the frozen entry point) writes the
  traceback to `%LOCALAPPDATA%\Captioner\logs\captioner-startup.log`, shows it in
  a message box and prints it, and `transcribe.preload_cuda_dlls()` looks for
  cuBLAS in the bundle's own `nvidia\cublas\bin` as well as in site-packages (a
  frozen app has no site-packages to search).
- **The first caption after a pause now comes back ~0.4 s sooner, by decoding on
  a shorter cadence for the first moments of the resumed utterance.** Waking the
  loop on sound ends the idle nap, but the model will not report a word with
  ~0.1 s of audio behind it, so the resumed words were still caught by the *next*
  decode - a whole stride later. For the first `RESUME_BURST_SECONDS` (2 s) of a
  resumed utterance the loop now decodes every `RESUME_STRIDE_SECONDS` (0.5 s,
  the fastest `ui.overlay.stride_wait` allows) while reading the 4 s window, so
  each extra decode costs 42% of a full one. `live.resume_instant` is what makes
  it work: the engine's old resume signal - a window that *ended* in quiet -
  misses the common case, because the loop sleeps 3 s between decodes during a
  pause and a short pause can begin and end between two of them. The window holds
  the whole story either way (speech, a pause, speech again), and 50 ms dips
  inside the returning speech are skipped rather than mistaken for the gap.
  Measured with `scripts/live_resume_audit.py --passes before,wake,burst
  --baseline wake` (recording 1, `small.en`, 18 pauses, three repeats each, pass
  order rotating): the loop's own wait fell **0.40 s against the wake alone**
  (median 0.37 s, 12 of 18 pauses faster; 6 gained ≥0.5 s against 3 that lost
  ≥0.5 s) and the caption **0.37 s** (median 0.30 s, 13 of 18 faster), against a
  0.20 s repeat-to-repeat floor - every repeat of the burst better than every
  repeat without it. Cost: ~2 extra decodes per pause (193 per 4-minute pass
  against 154, 0.24 s each). On by default;
  `LiveEngine(resume_stride_seconds=None)` is the loop it was measured against.
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

### Fixed
- **`--target` can no longer erase a folder that is not Captioner.** Installing
  replaces the target wholesale and uninstalling deletes it, so a mistyped path
  (`--target C:\Users\me`) used to be a way to lose files. `check_target` now
  refuses a drive root, the user's home folder or any parent of it, a path that
  is a file rather than a folder, and an existing folder that holds anything
  but `Captioner.exe` or `uninstall.cmd` - the same rule for install and for
  uninstall. Verified on the shipped `.exe`: both refused a folder with one
  unrelated file in it, exit 1, and the file survived. A `%` in the install
  path is escaped in `uninstall.cmd` now as well: `%` is legal in a Windows
  folder name and batch expands it, so the script could have removed a
  different folder than the one the user chose.
- **A pause no longer costs the rest of the idle stride — and a shorter decode
  at the resume was tried and does not help.** The decode loop waits
  `IDLE_STRIDE_SECONDS` (3 s) between windows while nobody is talking, to leave
  the GPU alone — and that wait was also a guess that there was nothing to
  transcribe: a speaker returning mid-nap falsified it while their first words sat
  in the ring, so the *wait* could outlive the pause by a whole stride. The
  capture now keeps a 50 ms meter beside the 100 ms one that drives the overlay
  (`MicCapture.sound_level`), and the loop waits on it
  (`ui.overlay.wait_for_sound`; the loop itself moved into
  `ui.overlay.run_live_session` so it runs without Qt and can be timed): a block
  at the engine's own silence gate ends the wait, which bounds that sleep at one
  50 ms poll. Measured with `scripts/live_resume_audit.py` — three passes per
  configuration, pass order rotating, on recording 1 (`small.en`, 18 pauses): the
  loop's own share of the wait before the first caption fell **0.16 s on average
  (median 0.17 s, sd 0.44 s, 95% CI ±0.20 s across pauses, 12 pauses faster and 6
  slower, sign-test p=0.24)** — against the *same* configuration's own mean
  moving 0.12–0.21 s between its repeats, so the effect is inside the drift and
  not settled by the repeats. Earlier single pairs put it at 0.2 s and found the
  tail consistently better (0.7–1.9 s off the unluckiest ~5% of pauses); repeating
  the measurement does not reproduce that tail — of these 18 pauses the three the
  old loop waited longest on came out 0.65 s faster, 0.76 s faster and 0.65 s
  slower — so that claim is corrected above rather than kept. The average is
  small because the loop was never what gated a post-pause caption: the model will
  not report a word with 0.1 s of audio behind it, so the resumed words land on the
  next 1.5 s stride either way. Decoding a 4 s window instead of 12 s at that
  moment — the obvious next thing to try — moved the loop's own wait by −0.03 s and
  the caption by −0.04 s on repeated passes (no effect), so
  `LiveEngine.resume_window_seconds` ships **off** by default. What it does buy
  is a cheaper decode (42% of a full one, 0.22–0.25 s against 0.54–0.59 s), and
  that is what the post-resume cadence added above spends.
- **The resume audit could not settle the effect it was measuring, so it now
  repeats each configuration and rotates the pass order.** One pass per
  configuration left the comparison carrying the machine's drift: two repeats of
  the *same* configuration differed by up to 0.21 s of mean loop wait, and the
  per-decode cost swung 2× across a 40-minute sitting (which also meant a pooled
  GPU-second total said more about when a pass ran than about the change).
  `--repeats` (default 3) replays each configuration that many times — which is
  what lets a drift over the sitting land on all of them equally — and rotates
  the order so that none of them always runs in another's wake. The report now
  gives the paired difference per pause over the repeats with sd, a 95% interval
  and an exact sign test — plus the repeat-to-repeat spread of each
  configuration's own mean, which is the floor an effect has to clear before the
  numbers mean anything. The pooling and the arithmetic moved into a `report()`
  that can be run against recorded passes, and results are saved with one entry
  per repeat (`per_repeat`, `rows`, `texts`, `order`).
- **The resume audit's warm-up and cost modes were measuring nothing.**
  `WhisperModel.transcribe` returns a *generator*: timing the call rather than
  consuming it reported 0.06 s for a 12 s window the live loop pays 0.5 s for, and
  every run printed `warm-up 0.0s` while the first measured pass still paid the
  model load. Both now consume the segments (`_decode_seconds`), and the numbers
  in the entry above come from runs that did.
- **The live caption age: the 10.8 s tail was the model re-writing itself, not
  the pause it looked like.** The sliding window froze *any* word whose start
  had cleared `COMMIT_MARGIN`, whenever it turned up — so a model that changed
  its mind about one instant of audio had every new guess frozen as extra text.
  On a 4-minute real lecture a single spoken word came back as "a", "signal",
  "of" and "the" over ten seconds, all four were written into the transcript
  (615 words against the batch model's 425), and the last re-write is what
  carried the 10.8 s age the README blamed on pauses. `_freeze` now accepts a
  word only inside `FREEZE_BAND_SECONDS` of the commit margin, and the floor
  never moves past the frontier the last decode left behind, so a pause flush or
  a slow GPU cannot turn the rule into lost speech. Same recording, same
  models: caption age p95 **10.80 → 4.58 s** (`tiny.en`) / **10.56 → 4.31 s**
  (`small.en`), median 3.43 → 3.02 s, words written 615 → 469, disagreement
  with the batch transcript 59.5% → 27.5% of words, worst opening word 11.4 →
  5.2 s. Coverage fell 79.4% → 73.1%: keeping the first estimate means the
  model's later reading of a word is dropped, and duplicate words had been
  papering over gaps.
- **The silence gate asked the wrong question.** It compared the whole 12 s
  window's RMS against the noise floor, which averages a quiet speaker away
  (one second of speech spread over twelve sits below the gate, so those words
  were never even decoded) and can sit above it on a window of room tone. It is
  now a 50 ms envelope: *peak* decides whether there is speech to decode at all,
  and *last speech* decides whether the speaker is still going. An utterance
  that stops now has its tail frozen by the pause itself instead of waiting for
  the speaker to start again, and a window with nothing above the gate is never
  decoded at all - a fresh session in a silent room used to decode its first
  window regardless.
- **The benchmark's file replay waited for a full 12 s window before its first
  decode, which no live session does.** It reported an ~11.5 s opening cost the
  app never had. `live.LiveEngine.stream_file` now decodes from the first stride
  like the app, and the opening it reports (5.2 s / 4.3 s worst word) is real.
- **The transcript a live session saves had reversed and overlapping cues.**
  Live words are frozen in *decode* order, not time order: overlapping windows
  re-propose a word with a jittered start, so a later decode can freeze it after
  a word that was spoken later (87 of 615 words on a 4-minute real lecture came
  out of order), and two estimates of one word can overlap. The export now sorts
  the timeline and pulls each word's end back to the next word's start, which
  took that session's saved `.srt` from 59 reversed and 10 overlapping cues out
  of 124 to none, without dropping a word. Found by the new real-audio
  benchmark; guarded by
  `tests/test_live.py::test_transcript_export_orders_words_that_arrived_out_of_order`.

### Changed
- **The full build no longer ships the CUDA runtime twice: 0.9 GiB less
  installed, 0.6 GB less to download.** torch/lib already carries cuBLAS,
  cuBLASLt, cuDNN and nvrtc, and a full build *also* packed
  `nvidia-cublas-cu12` and `cuda-nvrtc` beside them - the same DLLs in two
  folders, when `preload_cuda_dlls` would have used either (it looks in
  `nvidia/` first, then `torch/lib`). `app_datas` now leaves the wheels out
  when torch has them; a slim build still packs them, because it has no torch
  to borrow them from. Measured here: the installed folder went 6,076,335,467
  -> **5,117,865,752 bytes (4.8 GiB)**, the portable `.zip` 3,721,194,486 ->
  **3,091,725,066**, and the setup `.exe` 3,809,148,020 -> **3,179,586,136
  bytes (3.0 GiB)**, built in 58 s. The frozen artifact was then run, not
  imported: a 14 s clip with `--align --translate-to fr` finished in 34.7 s on
  the GPU and wrote all four caption files, and `--version` answers *"Captioner
  0.7.0, 4.8 GiB to install"*.
- **`small.en` is the live default** (`config.LIVE_MODEL_SIZE`), decided on two
  real recordings rather than one: it disagrees with the batch transcript on
  14.7% / 13.8% of words against `tiny.en`'s 27.5% / 23.1%, and its decode p95
  of 1.07 / 1.32 s still fits the 1.5 s stride. `tiny.en` remains in **Live →
  Live model** for a machine that cannot spare the ~0.5 s a window.
- **The overlay's decode stride is a period, not a nap after the decode.**
  `LiveWorker` slept the full stride *after* each decode, so the loop actually
  ran at decode + stride - about 1.8 s for `tiny.en`, 2.6 s for `small.en` -
  sliding the window further between refreshes and putting every caption that
  much later. The decode's own cost now comes out of the wait
  (`ui.overlay.stride_wait`).
- **A new app icon.**
 `scripts/make_icon.py` now draws the mark at every size it
  ships (16, 20, 24, 32, 40, 48, 64, 128, 256) instead of embedding one 256px
  image for Windows to blur down to the size it actually draws, and it takes
  its colors from `ui/dashboard/theme.py`, so the icon is violet like the app
  rather than the green it had been. Detail is dropped as the size falls: the
  waveform goes below 48px and 16-24px is a plain violet tile with two text
  lines, because the full mark is mush that small.

### Added
- **An MIT `LICENSE`.** The project said "all rights reserved" with no license
  file anywhere; it is now MIT (`sharat`), `pyproject.toml` says so too, and the
  README's license section points at the file instead of asking people to open
  an issue. Third-party terms are unchanged and still apply to what you
  download: Whisper's model weights, and `pyannote`'s gated diarization models
  behind a Hugging Face token you accept yourself.
- **`scripts/live_freeze_audit.py`: does refusing the model's re-writes throw
  speech away?** The freeze band trades duplicate words for the chance of
  losing a word the model reports late, so this replays a recording and
  classifies every refusal against what *was* captioned. On both recordings
  every refusal (159/79 and 84/49 for the two models) landed on a second that
  already had a caption word, or within 3 s of one, and none was speech that
  went uncaptioned anywhere near its time.
- **`scripts/bench_real_audio.py` reports word lag as well as caption age.**
  Two different latencies were being quoted as one number: the delay before a
  word reaches the screen (~1.2-1.9 s median, what the overlay's unstable tail
  gives you) and the delay before it settles into the transcript (~3 s median,
  what a saved caption file contains). The script measures both, in audio time.
- **`scripts/bench_real_audio.py`: the same claims, measured on your own
  recording.** Every benchmark in this file until now came from synthesized
  speech or a 13.9 s fixture. The new script replays a real file through both
  pipelines and reports batch wall clock, live decode cost per window, caption
  age *in audio time* (independent of how a file replay is paced), the live
  transcript's agreement with the batch transcript, and the health of the `.srt`
  a live session would save — plus a character-level agreement, because word
  agreement punishes a model twice for splitting a compound. `--live-models
tiny.en,small.en` compares live models in one run, `--reference` turns
  agreement into real WER when you have ground truth. Its first run earned its
  keep: on a 4-minute lecture it showed `tiny.en` disagreeing with the batch
  transcript on 59.5% of words against `small.en`'s 32.7% for twice the decode
  cost, and it found the reversed-cue bug in the "Fixed" section above.
- **The browser UI has a favicon.** `GET /favicon.ico` serves `captioner.ico`
  itself, so the tab is branded without a second copy of the binary in the tree.
- `pillow` in `requirements-dev.txt`: the icon script uses it to assemble the
  multi-size `.ico` (nothing at runtime depends on it).

### Added: languages, translation and accessibility
- **Every language each engine actually knows.** `captioner/language_data.py`
  (generated by the new `scripts/gen_language_tables.py` from Whisper's
  `LANGUAGES` map and the `facebook/m2m100_418M` model card, filtered against
  the *installed* `faster-whisper` and `transformers` builds) holds two tables:
  100 **spoken** languages the recogniser can be pointed at and 100
  **translation** targets. They are deliberately not the same 100 - M2M100
  spells Javanese `jv` where Whisper says `jw`, and has no Telugu at all - and
  `captioner/languages.py` keeps that difference visible instead of pretending.
  Both dashboard pickers and the browser UI now offer the complete sets sorted
  by name (the old translation list was a 27-entry shortlist), and
  `python cli.py languages` prints both, naming the languages that can be
  transcribed but not translated.
- **The spoken language is a per-job setting.** `--language hi` / `-l hi` on
  `add` and `watch`, a `Spoken language` picker in **Options**, and a `language`
  field on `POST /api/jobs`. Leaving it unset still means detect - and detection
  is now the default at the model level, because the transcriber used to pass
  `language="en"` unconditionally, which primes an English decoder and returns
  English captions for, say, a Hindi lecture with no error to show for it. Two
  new columns (`source_lang`, `detected_lang`) record what was asked for and
  what was heard; the details card, the browser UI's Language column and the
  job JSON all show both, so a wrong guess is visible rather than invisible.
- **Translate a finished job from the dashboard or the page.** **Job →
  Translate selection…** (`Ctrl+T`), a button under the caption editor, and a
  target picker in the browser UI's detail panel, all translated an existing
  `captions.srt` without re-transcribing. The page's old `window.prompt("which
  language code?")` is a real 100-entry select now.
- **Preferences that survive a restart (`ui/dashboard/preferences.py`).**
  Theme, text size, quiet mode, start-the-worker-with-the-window and the queue
  defaults (model, spoken language, translation target, speakers, alignment,
  watch folder, notifications) are stored with `QSettings` and applied at
  startup; the Options popover saves each choice as it changes. A settings file
  is hand-editable text, so every value is coerced on load and an impossible one
  falls back to the default instead of breaking the window.
- **Three themes, one scalable stylesheet (`ui/dashboard/theme.py`).** Dark (the
  violet-glass default), Light, High contrast (opaque surfaces, near-white text
  on black, yellow focus ring, 6px radii) and Follow Windows, all built from the
  same structure through a `Palette` dataclass. Text size scales every label,
  menu and button from one base number - `Ctrl+=` / `Ctrl+-` / `Ctrl+0`, a
  percentage slider in the dialog, or the same setting at startup.
- **Contrast is now checked, not eyeballed.** Every text colour in every theme
  is held to WCAG 2.1 (4.5:1 normal text, 7:1 in high contrast) by
  `tests/test_preferences.py`, and the served browser stylesheet is held to the
  same rule by `tests/test_api.py` (including `var()` aliases, which the high
  contrast block uses). Two real failures came out of writing that test: the
  dark theme's faint text - hints, timestamps, the status bar and table headers -
  measured **3.8:1** on the window background, and is now **5.8:1** (`#6d6d78`
  became `#8b8b95`). Button fills moved off the brand accent
  (`#8b5cf6` measured a marginal 4.6:1 under dark label text) onto a separate
  tint that measures **6.8:1**; the brand accent, and therefore the app icon,
  is untouched.
- **Keyboard and screen-reader support, as a feature rather than an accident.**
  Every control has an accessible name and description, the Tab order follows
  the window's layout instead of creation order, every focusable control shows a
  visible focus ring in all three themes, and **Help → Keyboard shortcuts**
  (`F1`) lists every shortcut from the same tuple the menus are built from, so
  the list cannot drift.
- **Quiet mode**, which keeps the status-bar note when the queue drains and
  drops the beep and the flashing taskbar entry - for anyone who set up the
  beep in the first place and now works next to other people.
- **The browser UI follows the operating system and can be made bigger.**
  `prefers-color-scheme: light` lightens the whole page, `prefers-contrast:
  more` hardens the borders and drops the shadows,
  `prefers-reduced-motion: reduce` turns transitions off; there is a skip link,
  the queue table has a caption and scoped headers, the toast is an `aria-live`
  region (assertive for errors, polite for progress), the row buttons say which
  job they act on ("cancel job 17 (talk.mp4)"), and a text-size control scales
  the page and remembers the choice in `localStorage`.

### Changed: the translator is told what it is reading
- **Source language.** `SOURCE_LANG = "en"` was
  applied to every transcript; now the pipeline passes what the user chose, else
  what the recogniser heard, else English. A Whisper-only code (Telugu) raises a
  message naming the problem instead of being quietly mistranslated, and the
  on-disk cache is keyed by source language as well as target, so two readings
  of the same text cannot collide. Measured on the development machine: three
  short Hindi lines, translated to French through `POST /api/jobs/{id}/translate`
  with `source=hi`, came back as "Aujourd'hui, nous allons parler de la
  traduction. / C'est un test." - and identical when the same text was tagged
  English, because M2M100's encoder is genuinely multilingual. The change is
  about being right about what the model is reading, not about a measured
  quality jump on short clean lines.
- `GET /api/languages` returns `spoken` and `translation` alongside the old
  `languages` key, so existing clients keep working.
- `Dashboard(...)` takes an optional `settings` object, so tests can point
  preferences at a temporary file instead of the real registry.

### Verification
- 409 tests pass (408 passed, 1 skipped: the Windows hotkey keystroke test this
  session policy blocks), `ruff check .` clean. The new suites are
  `tests/test_languages.py` (both tables against the installed engines, code
  validation, source-aware translation and cache keying) and
  `tests/test_preferences.py` (defaults, hand-edited settings, clamping, theme
  sheets, WCAG contrast for every theme, and the real window offscreen: options
  remembered between windows, saved appearance applied at startup, quiet mode
  silencing the beep while keeping the note, accessible names present).
- End-to-end on a running server: the browser page's queue form sent
  `language: hi` + `translate_to: bn` and the job came back with
  `source_lang: hi`, `source_lang_name: Hindi` and a Language column reading
  "Hindi → Bengali (bn)"; the text-size control moved the page from 14px to
  18.2px and back; a real Hindi transcript translated to French through the API
  wrote `captions.fr.srt` + `.vtt`. `python -m ui.dashboard --self-test` builds,
  renders and closes the window cleanly.

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
