"""One-shot environment check for this machine (RTX 4050 / CUDA 12 target).

Run from the project venv BEFORE Phase 1's first real job:
    python scripts/check_env.py

Verifies, in order: Python version, ctranslate2 + CUDA visibility,
faster-whisper import, and (optionally, with --model) an actual tiny-model
transcription on GPU.
"""
import platform
import sys
from pathlib import Path

# Make `captioner` importable when run as `python scripts/check_env.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    print(f"Python {platform.python_version()} on {platform.system()}")
    ok = True

    if sys.version_info < (3, 10):
        print("FAIL: Python 3.10+ recommended (3.11 best)")
        ok = False
    else:
        print("OK: Python version")

    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        if n > 0:
            print(f"OK: ctranslate2 sees {n} CUDA device(s)")
        else:
            print("WARN: no CUDA device visible — will fall back to CPU (slow)")
    except Exception as exc:
        print(f"FAIL: ctranslate2 import failed: {exc}")
        ok = False

    try:
        import faster_whisper  # noqa: F401
        print("OK: faster-whisper importable")
    except Exception as exc:
        print(f"FAIL: faster-whisper import failed: {exc}")
        ok = False

    if "--model" in sys.argv:
        try:
            from captioner import transcribe
            transcribe.preload_cuda_dlls()  # same path the app uses at runtime
            from faster_whisper import WhisperModel
            print("Downloading/loading tiny model for a 1s GPU test...")
            model = WhisperModel("tiny.en", device="cuda", compute_type="float16")
            import numpy as np
            silence = np.zeros(16000, dtype=np.float32)
            segments, info = model.transcribe(silence, language="en")
            list(segments)  # force generation
            print("OK: tiny.en ran a real inference on GPU")
        except Exception as exc:
            print(f"WARN: GPU inference test failed: {exc}")
            ok = False

    print("\nAll good." if ok else "\nProblems found — see lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
