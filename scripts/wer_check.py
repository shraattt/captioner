"""WER on YOUR audio: the Phase 4 gate.

Give it a media file plus a reference transcript (typed, or a .srt/.vtt you
trust) and it reports word-level WER for the batch pipeline — and, with
--live, the live pipeline too. This is the honest number that decides
whether fine-tuning (Phase 4) is worth it: synthetic fixtures said 2.3%,
real distant-mic audio will say more.

Reference file formats accepted:
  .txt   plain text, one pass over the whole file
  .srt/.vtt  only the text lines are used (timestamps ignored)

    python scripts/wer_check.py lecture.wav lecture_reference.txt
    python scripts/wer_check.py clip.wav ref.txt --live
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORD_RE = re.compile(r"[a-z']+")


def normalize(text: str):
    tokens = WORD_RE.findall(text.lower())
    return [t for t in tokens if t not in ("um", "uh", "erm")]


def wer(reference, hypothesis) -> tuple:
    """Returns (wer, substitutions, deletions, insertions)."""
    r, h = reference, hypothesis
    if not r:
        return (0.0, 0, 0, len(h)) if h else (0.0, 0, 0, 0)
    # full DP matrix so the error breakdown can be walked back reliably
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,                      # deletion
                d[i][j - 1] + 1,                      # insertion
                d[i - 1][j - 1] + (r[i - 1] != h[j - 1]),  # sub/match
            )
    i, j = len(r), len(h)
    sub = dele = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            if r[i - 1] != h[j - 1]:
                sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return (d[len(r)][len(h)] / len(r), sub, dele, ins)


def load_reference(path: Path) -> str:
    if path.suffix.lower() in (".srt", ".vtt"):
        lines = []
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            ls = block.strip().splitlines()
            if len(ls) >= 3:  # index, timestamp, text…
                lines.append(" ".join(ls[2:]))
        return " ".join(lines)
    return path.read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio")
    ap.add_argument("reference")
    ap.add_argument("--live", action="store_true",
                    help="also score the live pipeline (sliding window)")
    ap.add_argument("--model", default=None,
                    help="batch model (default: config.BATCH_MODEL_SIZE)")
    ap.add_argument("--live-model", default=None,
                    help="live model label or path, e.g. base_en_noisy "
                         "(default: config.LIVE_MODEL_SIZE)")
    args = ap.parse_args()

    audio, ref_path = Path(args.audio), Path(args.reference)
    if not audio.exists() or not ref_path.exists():
        print("audio or reference file missing")
        return 1
    ref = normalize(load_reference(ref_path))
    print(f"reference: {len(ref)} words from {ref_path.name}")

    from captioner import config, models, transcribe
    from captioner import live as L

    #batch
    words = transcribe.transcribe_words(str(audio), args.model)
    hyp = normalize(" ".join(w.text for w in words))
    score, sub, dele, ins = wer(ref, hyp)
    print(f"batch ({args.model or config.BATCH_MODEL_SIZE}): "
          f"WER {score*100:.1f}%  ({sub} sub, {dele} del, {ins} ins / {len(ref)} ref words)")

    #live
    if args.live:
        try:
            live_model = models.resolve(args.live_model or config.LIVE_MODEL_SIZE)
        except ValueError as exc:
            print(f"live model: {exc}")
            return 2
        engine = L.LiveEngine(model_size=live_model)
        engine.stream_file(str(audio))
        lh = normalize(" ".join(t for _s, _e, t in engine.state.full_words))
        lscore, lsub, ldel, lins = wer(ref, lh)
        print(f"live  ({args.live_model or config.LIVE_MODEL_SIZE}): "
              f"WER {lscore*100:.1f}%  ({lsub} sub, {ldel} del, {lins} ins / {len(ref)} ref words)")

    print("\nPhase 4 gate: fine-tuning is worth considering when real-audio "
          "WER is meaningfully worse than the ~2.3% synthetic baseline "
          "(say, >15%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
