"""Verify gated-model access before Phase 2 begins.

    python scripts/verify_hf_access.py

Checks, in order:
  1. a HF token is available (.env or environment)
  2. the token authenticates against the Hugging Face Hub
  3. the token can actually see the gated pyannote repos — this is the
     exact check that fails when the "agree to conditions" step was skipped
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from captioner.secrets import hf_token

GATED_REPOS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
]


def main() -> int:
    token = hf_token()
    if not token:
        print("FAIL: no HF token found.")
        print("Fix: copy .env.example to .env and set HF_TOKEN=hf_...")
        return 1
    print("OK: token found in environment/.env")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("FAIL: huggingface_hub not installed (pip install huggingface_hub)")
        return 1

    api = HfApi(token=token)
    try:
        who = api.whoami()
        print(f"OK: authenticated as '{who.get('name', '?')}'")
    except Exception as exc:
        print(f"FAIL: token rejected by Hugging Face: {exc}")
        return 1

    ok = True
    for repo in GATED_REPOS:
        try:
            api.model_info(repo)
            print(f"OK: access to {repo}")
        except Exception as exc:
            print(f"FAIL: cannot access {repo}")
            print(f"      -> {exc}")
            print("      -> log in at huggingface.co and click 'agree' on the")
            print("         model page conditions, then re-run this script")
            ok = False

    print("\nGated models reachable — Phase 2 is unblocked."
          if ok else "\nOne or more gated models still blocked.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
