"""What does the live freeze band refuse, on a recording you name?

Live captions refuse the model's *late* proposals: a word first reported more
than `live.FREEZE_BAND_SECONDS` behind the commit margin is a re-think about
audio whose caption is already written, and freezing it made the transcript
grow a word per re-think (and drag 10.8 s caption ages into the tail).

That policy trades one risk for another, so this measures the trade on real
audio: of the words the band refused, how many were speech we never captioned
anywhere near its time? A refusal that lands on a second that already has a
caption word is a re-write of that moment; a refusal with no caption word near
it is speech the band threw away.

    python scripts/live_freeze_audit.py "D:\\lectures\\merge-sort.mp4"
    python scripts/live_freeze_audit.py lecture.mp4 --model small.en

Run it through the live pipeline (no audio device needed) and read the last
line: on the two recordings in the README, every refusal - 159/79 on one,
84/49 on the other - landed on or within 3 s of a moment that *was* captioned,
and none was speech that went uncaptioned.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _key(text: str) -> str:
    """Case/punctuation-insensitive word, for matching across decodes."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--model", default=None,
                    help="live model (default: config.LIVE_MODEL_SIZE)")
    ap.add_argument("--near-seconds", type=float, default=1.0,
                    help="a refusal this close to a captioned word is a re-write (default 1.0)")
    ap.add_argument("--drift-seconds", type=float, default=3.0,
                    help="and this close counts as timing drift (default 3.0)")
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"audio file missing: {audio}")
        return 1

    from captioner import config, models
    from captioner import live as L

    try:
        model = models.resolve(args.model) if args.model else config.LIVE_MODEL_SIZE
    except ValueError as exc:
        print(f"live model: {exc}")
        return 2

    engine = L.LiveEngine(model_size=model)
    refused: dict[tuple, tuple] = {}
    real_freeze = engine._freeze
    band = L.FREEZE_BAND_SECONDS

    def audited_freeze(words, frontier, floor=None):
        limit = frontier - band if floor is None else floor
        for start, end, text in words:
            # the band only refuses words that are not already captioned: a
            # proposal the dedup would drop was never going to be written
            if not engine._is_dup((start, end, text)) and start < limit:
                refused.setdefault((_key(text), int(start * 2)), (start, end, text))
        return real_freeze(words, frontier, floor=floor)

    engine._freeze = audited_freeze
    print(f"replaying {audio.name} through the live pipeline ({model})...")
    engine.stream_file(str(audio))

    captioned = [tuple(w) for w in engine.state.full_words]
    rewrite = drift = lost = 0
    lost_examples: list[tuple] = []
    for start, _end, text in refused.values():
        if [c for c in captioned if abs(c[0] - start) <= args.near_seconds]:
            rewrite += 1
        elif [c for c in captioned if c[0] <= start and abs(c[0] - start) <= args.drift_seconds]:
            drift += 1
        else:
            lost += 1
            if len(lost_examples) < 10:
                lost_examples.append((round(float(start), 2), text))
    print(f"captioned {len(captioned)} words; refused {len(refused)} unique proposals")
    print(f"  re-writes of a moment already captioned: {rewrite}")
    print(f"  within {args.drift_seconds:.0f}s of a captioned word (timing drift): {drift}")
    print(f"  speech never captioned anywhere near it:  {lost}")
    for start, text in lost_examples:
        print(f"      {start:7.2f}s  {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
