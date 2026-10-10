# Root-level conftest ensures the project root is on sys.path so tests can
# `import captioner` regardless of where pytest is invoked from.
import os

# Tests must never claim a system-wide hotkey combination: registering one
# consumes it for every other program on the machine (and the real
# registration makes each Dashboard() construction measurably slower).
os.environ.setdefault("CAPTIONER_NO_GLOBAL_HOTKEY", "1")

# The HF hub warns about missing symlinks on Windows on every download;
# silence it in test output.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
