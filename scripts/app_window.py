"""The windowed entry point of a packaged build: the dashboard, or a dialog.

PyInstaller needs a plain script (it cannot freeze a package's ``__main__``,
because a relative import has no package to resolve against once the folder is
standing on its own), so this is the file the build points at. It stays tiny on
purpose - everything here exists to make a *failed start* visible, which is the
one thing a windowed build cannot otherwise do: with no console and no Python,
an import error would show an empty screen and nothing else.

    python scripts/app_window.py              # same as python -m ui.dashboard
    python scripts/app_window.py --self-test  # opens and closes itself
    python scripts/app_window.py --job file   # one job, no window (headless)

The build is expected to keep the project layout (`ui/`, `captioner/`,
`captioner.ico` at the top of the bundle), so `ui.dashboard` finds its own
resources exactly as it does from a checkout - see scripts/package.py.
"""
import ctypes
import logging
import os
import sys
import traceback
from pathlib import Path

if not getattr(sys, "frozen", False):
    # Run from the checkout (a plain `python scripts/app_window.py`).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _attach_console() -> None:
    """Give a windowed build the console it was started from, if there is one.

    A PyInstaller windowed executable has no stdout at all: every `print` would
    raise. ``--job`` is meant to be run from a command prompt, so attach to the
    caller's console and reopen the streams; started from a shortcut there is
    no console, and the streams point at the null device - the job still runs,
    its log still explains it, and printing can never be the thing that fails.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    attached = False
    if sys.platform == "win32":
        try:
            attached = bool(ctypes.windll.kernel32.AttachConsole(-1))
        except Exception:  # no console to attach to, or not allowed to
            attached = False
    stream = "CONOUT$" if attached else os.devnull
    try:
        if sys.stdout is None:
            sys.stdout = open(stream, "w", encoding="utf-8", buffering=1,
                              errors="replace")
        if sys.stderr is None:
            sys.stderr = open(stream, "w", encoding="utf-8", buffering=1,
                              errors="replace")
    except OSError:  # pragma: no cover - nothing sensible left to report to
        pass


def _flush() -> None:
    """Push out anything buffered before a batch run leaves the process."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except Exception:  # pragma: no cover - a closed console is not fatal
            pass
    logging.shutdown()


def _report(exc: BaseException) -> int:
    """Write the traceback where a user can find it, then say so on screen."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    where = "(no log file could be written)"
    try:
        from captioner import config

        log = config.DATA_DIR / "logs" / "captioner-startup.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text, encoding="utf-8")
        where = str(log)
    except Exception:  # the failure we are reporting may be in config itself
        print(text, file=sys.stderr)
    message = (
        f"Captioner could not start.\r\n\r\n{type(exc).__name__}: {exc}\r\n\r\n"
        f"Details were written to:\r\n{where}"
    )
    print(message, file=sys.stderr)
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.MessageBoxW(None, message, "Captioner", 0x10)
        except Exception:
            pass
    return 1


if __name__ == "__main__":
    # Before Qt, before the window: `--job` is the batch path, and it must work
    # on a machine where a window cannot open (a server, a scheduled task).
    if sys.argv[1:2] == ["--job"]:
        _attach_console()
        from captioner import headless

        try:
            code = headless.main(sys.argv[2:])
        except SystemExit:
            raise
        except BaseException:  # never a silent exit, never a dialog: see _attach_console
            traceback.print_exc()
            code = 1
        # The job is over when its files are written: the database is closed and
        # the report is out. Normal interpreter shutdown would now tear down a
        # multi-gigabyte CUDA context, which on a cold first run (model weights
        # downloaded in this process) was measured taking *minutes* on Windows.
        # A script waiting on this exit code should not wait for that.
        _flush()
        os._exit(code)
    try:
        from ui.dashboard import main

        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        # Broad on purpose: nothing should be able to end this process without
        # a word, and the alternative (a bare traceback into a void) is exactly
        # the failure mode this file exists to prevent.
        sys.exit(_report(exc))
