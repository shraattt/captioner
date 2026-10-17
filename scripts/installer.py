"""Install Captioner from the single-file setup: one .exe, no Python needed.

    Captioner-Setup-0.7.0.exe                 # the window: pick a folder, install
    Captioner-Setup-0.7.0.exe --silent        # unattended, all defaults
    Captioner-Setup-0.7.0.exe --target D:\\Captioner --portable
    Captioner-Setup-0.7.0.exe --uninstall     # remove the app, keep your data
    Captioner-Setup-0.7.0.exe --uninstall --purge-data   # remove everything

The file is built by scripts/package.py, which packs the finished app into this
program as a payload (``sys._MEIPASS/payload``) - or, when the folder is too big
for PyInstaller's 32-bit one-file archive (the full build: 4.8 GiB), as a
compressed ``sys._MEIPASS/payload.zip`` that is unpacked instead. Installing is
then four plain file operations, reported one at a time so the window can say
what it is doing:

 1. **Copy the app** to the install folder - by default
    ``%LOCALAPPDATA%\\Programs\\Captioner``, which is per-user and needs no
    administrator rights, so there is no UAC prompt anywhere in this program.
 2. **Shortcuts**: a Start Menu entry, an *Uninstall* entry and a *data folder*
    entry, plus a Desktop icon (skip with ``--no-shortcuts``).
 3. **A commented .env** in ``%LOCALAPPDATA%\\Captioner`` if there is not one
    already, so the Hugging Face token and the optional settings have an
    obvious home. An existing file is never touched.
 4. **The data folder** (``%LOCALAPPDATA%\\Captioner\\data``): the database,
    captions, logs and downloaded models. It outlives an uninstall - see
    ``--purge-data``.

``--portable`` is the exception to all of that: it writes the data folder
*inside* the install folder (next to ``Captioner.exe``) and tells the app to use
it, so a copy on a USB stick carries its own jobs, settings and captions.

Nothing here imports the app: the installer must work on a machine that has
never seen Python, and installing must never be able to fail because of a
half-loaded model. The only things it needs are `shutil`, `subprocess` and
`tkinter`, all of which the build carries.
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "Captioner"
EXE_NAME = f"{APP_NAME}.exe"
DEFAULT_FOLDER = "Captioner"

# Used when the payload carries no version file (a hand-made dist folder).
FALLBACK_VERSION = "0.0.0"

ENV_TEMPLATE = """\
# Captioner's settings file. Everything below is optional; delete the "#" in
# front of a line to switch it on. The app reads this file on every start.
#
# A free Hugging Face token (https://huggingface.co/settings/tokens). Only
# needed for speaker labels, whose models are gated - see the README's
# "Hugging Face token" step.
# HF_TOKEN=hf_your_token_here
#
# Where jobs, captions and downloaded models live. The default is this folder.
# CAPTIONER_DATA_DIR=D:\\Captioner\\data
#
# A second place to look for the CUDA DLLs (a CUDA Toolkit "bin" folder).
# CAPTIONER_CUDA_BIN=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin
#
# Live captions translated without the UI, e.g. hi for Hindi (empty = off).
# CAPTIONER_LIVE_TRANSLATE_TO=
#
# Serve the queue over HTTP. CAPTIONER_API_TOKEN is required by anything but
# loopback, and CAPTIONER_API_HOST=0.0.0.0 opens it to your network.
# CAPTIONER_API_PORT=8797
# CAPTIONER_API_TOKEN=
#
# Folders to watch for new media, separated by ";".
# CAPTIONER_WATCH_DIRS=
#
# POST every finished job's captions here (JSON, with CAPTIONER_WEBHOOK_SECRET
# as the X-Captioner-Secret header).
# CAPTIONER_WEBHOOK_URL=
# CAPTIONER_WEBHOOK_SECRET=
#
# Quiet mode: never claim the system-wide Ctrl+Alt+L for live captions.
# CAPTIONER_NO_GLOBAL_HOTKEY=1
"""

UNINSTALL_TEMPLATE = """\
@echo off
rem Written by Captioner Setup. Removes the application and its shortcuts.
rem Your jobs, captions and settings are kept: {data_dir}
setlocal
echo Removing {app} from "{target}" ...
if exist "{target}" rmdir /s /q "{target}"
for %%L in ({links}) do if exist "%%~L" del /q "%%~L"
if exist "{menu}" rmdir /q "{menu}" 2>nul
echo.
echo {app} removed. Your data is still in:
echo   {data_dir}
echo Delete that folder yourself if you want it gone too.
rem Delete this script from a *second* copy of cmd. Asking the running one to
rem remove the file it is reading makes it print "the batch file cannot be
rem found" afterwards - one confusing line, for no reason. Verified: the file
rem goes, and nothing is printed.
start "" /b %COMSPEC% /c del /q "%~f0"
"""


class InstallError(RuntimeError):
    """Something the user can act on: a locked file, a full disk, a bad path."""


# --------------------------------------------------------------------- paths


def local_app_data() -> Path:
    """``%LOCALAPPDATA%``, or a home-based fallback on a machine without it."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    return Path(base) if base else Path.home() / "AppData" / "Local"


def default_target() -> Path:
    """Where the app goes unless told otherwise: per-user, no admin rights."""
    return local_app_data() / "Programs" / DEFAULT_FOLDER


def data_dir() -> Path:
    """The app's own folder: `.env`, `data/`, the uninstaller.

    Deliberately *not* inside the install folder, so removing or replacing the
    app cannot take the user's transcripts with it.
    """
    return local_app_data() / APP_NAME


def payload_dir() -> Path:
    """The finished app that is packed into this program.

    Frozen (the shipped setup): beside this program's own code. Run from the
    checkout (someone testing the installer): ``dist/Captioner``, the folder
    scripts/package.py builds.
    """
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled) / "payload"
    return Path(__file__).resolve().parents[1] / "dist" / DEFAULT_FOLDER


def payload_archive() -> Path | None:
    """A compressed payload, when the folder was too big to embed file by file.

    PyInstaller's one-file archive addresses its contents with 32-bit offsets,
    so more than 4 GiB of payload cannot fit - the full build is 4.8 GiB and
    used to fail at the very end of a half-hour build. scripts/package.py
    compresses that folder into ``payload.zip`` instead; one compressed entry
    fits, and it is unpacked here. The zip holds the app's files at its root,
    so extracting it into the install folder *is* the copy.
    """
    bundled = getattr(sys, "_MEIPASS", None)
    if not bundled:
        return None
    candidate = Path(bundled) / "payload.zip"
    return candidate if candidate.exists() else None


def payload_lines(payload: Path, archive: Path | None = None) -> list[str]:
    """The build's own description: version, then build kind, then GPU or not."""
    version_file = payload / "version.txt"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").splitlines()
    if archive is not None:
        try:
            with zipfile.ZipFile(archive) as bundle:
                for info in bundle.infolist():
                    if _member_name(info.filename) == "version.txt":
                        return bundle.read(info).decode("utf-8").splitlines()
        except (OSError, zipfile.BadZipFile):
            return []
    return []


def payload_version(payload: Path, archive: Path | None = None) -> str:
    lines = payload_lines(payload, archive)
    return (lines[0].strip() if lines else "") or FALLBACK_VERSION


def payload_note(payload: Path, archive: Path | None = None) -> str:
    """What this build includes, in the words a user needs before installing.

    The two builds differ by 3.5 GiB and three features, and the `--full` one is
    the only way to get them: saying so in the setup window is the difference
    between a choice and a surprise discovered twenty minutes into a job (the
    app refuses the feature with the same reasoning - see captioner/features.py).
    """
    lines = [line.strip().lower() for line in payload_lines(payload, archive)]
    kind = lines[1] if len(lines) > 1 else ""
    gpu = "with GPU support (CUDA included)" if "cuda" in lines else "CPU only"
    if kind == "slim":
        return (f"Slim build, {gpu}. Transcribing, live captions, editing, the "
                "queue and the API are all included. Local translation, speaker "
                "labels and word alignment are not - they need PyTorch, which "
                "this build leaves out (a full build installs 4.8 GiB).")
    if kind == "full":
        return (f"Full build, {gpu}. Everything is included: transcribing, live "
                "captions, local translation, speaker labels, word alignment and "
                "fine-tuning.")
    return f"{gpu}. This build does not say what it includes."


def payload_bytes(payload: Path, archive: Path | None = None) -> int:
    """How much will land on disk, read from whatever carries the app."""
    if archive is not None:
        try:
            with zipfile.ZipFile(archive) as bundle:
                return sum(info.file_size for info in bundle.infolist())
        except (OSError, zipfile.BadZipFile):
            return 0
    total = 0
    for path in payload.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


# ----------------------------------------------------------------- shortcuts


@dataclass(frozen=True)
class Shortcut:
    """One .lnk to create, described before anything touches the disk."""

    link: Path
    target: Path
    args: str = ""
    description: str = ""
    workdir: Path | None = None


def _start_menu() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    return root / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def _desktop() -> Path:
    return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"


def shortcuts_for(target: Path, data: Path, app_dir: Path,
                  *, desktop: bool = True, start_menu: bool = True) -> list[Shortcut]:
    """Every shortcut an install makes, as data (so it can be tested).

    `app_dir` is where the app keeps its own files; its uninstall.cmd lives
    there, so the Uninstall entry keeps working after the program folder is
    deleted.
    """
    exe = target / EXE_NAME
    out: list[Shortcut] = []
    if start_menu:
        menu = _start_menu() / APP_NAME
        out.append(Shortcut(menu / f"{APP_NAME}.lnk", exe,
                            description="Caption files and live captions, offline"))
        out.append(Shortcut(menu / f"{APP_NAME} data folder.lnk", data,
                            description="Your jobs, captions, logs and settings"))
        uninstaller = app_dir / "uninstall.cmd"
        if uninstaller.exists():
            out.append(Shortcut(menu / f"Uninstall {APP_NAME}.lnk", uninstaller,
                                description=f"Remove {APP_NAME} (your data is kept)"))
    if desktop:
        out.append(Shortcut(_desktop() / f"{APP_NAME}.lnk", exe,
                            description="Caption files and live captions, offline"))
    return out


def powershell_shortcut(shortcut: Shortcut) -> bool:
    """Create one .lnk through the shell's own COM object.

    `WScript.Shell` is the only .lnk writer that already exists on Windows;
    writing the binary format by hand is a documented way to produce links that
    Explorer refuses to open. Returns False (with a warning) instead of raising:
    a missing shortcut must not fail an otherwise complete install.
    """
    def q(value) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    script = (
        "$s = New-Object -ComObject WScript.Shell; "
        f"$l = $s.CreateShortcut({q(shortcut.link)}); "
        f"$l.TargetPath = {q(shortcut.target)}; "
        f"$l.Arguments = {q(shortcut.args)}; "
        f"$l.WorkingDirectory = {q(shortcut.workdir or shortcut.target.parent)}; "
        f"$l.Description = {q(shortcut.description)}; "
        f"$l.IconLocation = {q(str(shortcut.target) + ',0')}; "
        "$l.Save()"
    )
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _warn(f"could not create {shortcut.link.name} ({exc})")
        return False
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip().splitlines()
        _warn(f"could not create {shortcut.link.name}"
              + (f": {detail[-1]}" if detail else ""))
        return False
    return True


def create_shortcuts(shortcuts: Iterable[Shortcut],
                     create: Callable[[Shortcut], bool] | None = None) -> list[Path]:
    """Make every shortcut that can be made; return the ones that exist now.

    `create` is the seam the tests use (passing the real one would launch
    PowerShell); None means the real `WScript.Shell` writer. A link only counts
    once it is on disk - a creator that claims success and writes nothing is a
    broken install nobody would notice until they looked for the icon.
    """
    creator = create or powershell_shortcut
    made: list[Path] = []
    for shortcut in shortcuts:
        try:
            shortcut.link.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _warn(f"could not create {shortcut.link.parent} ({exc})")
            continue
        if creator(shortcut) and shortcut.link.exists():
            made.append(shortcut.link)
    return made


def remove_shortcuts(shortcuts: Iterable[Shortcut]) -> list[Path]:
    """Delete the shortcuts an install made; missing ones are not an error."""
    removed: list[Path] = []
    for shortcut in shortcuts:
        try:
            if shortcut.link.exists():
                shortcut.link.unlink()
                removed.append(shortcut.link)
        except OSError as exc:
            _warn(f"could not remove {shortcut.link} ({exc})")
    # The Start Menu folder is ours alone; drop it once it is empty.
    for folder in {s.link.parent for s in shortcuts}:
        try:
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        except OSError:
            pass
    return removed


def _warn(message: str) -> None:
    """Say something went wrong without pretending the install failed."""
    print(f"warning: {message}", file=sys.stderr)


# -------------------------------------------------------------------- steps


def _member_name(name: str) -> str:
    """A zip member's path relative to the app folder.

    The payload is the portable archive, whose root is a ``Captioner/`` folder.
    The install folder *is* that folder, so the prefix is dropped; an archive
    built without one (the tests) works unchanged.
    """
    name = name.replace("\\", "/")
    prefix = APP_NAME + "/"
    return name[len(prefix):] if name.startswith(prefix) else name


def looks_like_an_install(folder: Path) -> bool:
    """Is this folder one of ours? The marker every install and uninstall leaves."""
    return (folder / EXE_NAME).is_file() or (folder / "uninstall.cmd").is_file()


def check_target(target: Path) -> Path:
    """Refuse a folder this program must not delete, and return it resolved.

    Installing replaces the whole folder (a stale module can otherwise shadow a
    new one), and uninstalling deletes it. Both are right for an install folder
    and catastrophic for anything else: ``--target C:\\`` or a mistyped
    ``--target C:\\Users\\me`` would erase files nobody asked to erase. The rule
    is the same for both paths: a drive root, the user's home or one of its
    parents is never a Captioner folder, and an existing folder that already
    holds other files is only replaced when it holds this app.
    """
    target = Path(target)
    try:
        resolved = target.resolve()
    except OSError as exc:  # a path Windows cannot even resolve
        raise InstallError(f"cannot use {target}: {exc}") from exc
    if resolved.parent == resolved:
        raise InstallError(
            f"{resolved} is a drive root, not a folder {APP_NAME} can be "
            "installed into - choose a folder of its own"
        )
    try:
        home = Path.home().resolve()
    except OSError:  # pragma: no cover - no profile, no home to protect
        home = None
    if home is not None and (resolved == home or home.is_relative_to(resolved)):
        raise InstallError(
            f"{resolved} is your home folder (or contains it), not a folder "
            f"{APP_NAME} can be installed into - choose a folder of its own"
        )
    if resolved.exists() and not resolved.is_dir():
        raise InstallError(f"{resolved} is a file, not a folder")
    if resolved.is_dir() and any(resolved.iterdir()) and not looks_like_an_install(resolved):
        raise InstallError(
            f"{resolved} already contains files that are not {APP_NAME}; "
            "refusing to replace it - choose an empty folder or a new one"
        )
    return resolved


def copy_app(payload: Path, target: Path, archive: Path | None = None) -> None:
    """Replace the program folder with this payload, keeping nothing stale.

    A file left behind by an older version can shadow a new one (a module that
    no longer exists, a Qt plugin), so the old folder is removed first - the
    only thing inside it is the app.

    ``archive`` is the compressed payload a full build carries; its members are
    the app's own files, so extracting it into the empty target is the copy.
    """
    target = check_target(target)
    if archive is None and not (payload / EXE_NAME).exists():
        raise InstallError(
            f"this setup has no application inside it (expected {payload / EXE_NAME})"
        )
    if archive is not None:
        try:
            with zipfile.ZipFile(archive) as bundle:
                if EXE_NAME not in {_member_name(n) for n in bundle.namelist()}:
                    raise InstallError(
                        f"this setup has no application inside it ({EXE_NAME} is "
                        f"missing from {archive.name})"
                    )
        except (OSError, zipfile.BadZipFile) as exc:
            raise InstallError(f"could not read the application archive: {exc}") from exc
    if target.exists():
        try:
            shutil.rmtree(target)
        except OSError as exc:
            raise InstallError(
                f"could not replace {target}: {exc}. Close {APP_NAME} and try again."
            ) from exc
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if archive is None:
            shutil.copytree(payload, target)
        else:
            with zipfile.ZipFile(archive) as bundle:
                for info in bundle.infolist():
                    name = _member_name(info.filename)
                    if not name or name.endswith("/"):
                        continue
                    dest = target / name
                    if not dest.resolve().is_relative_to(target.resolve()):
                        raise InstallError(
                            f"the application archive has an unsafe path: "
                            f"{info.filename}"
                        )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, dest.open("wb") as handle:
                        shutil.copyfileobj(source, handle)
    except (OSError, zipfile.BadZipFile) as exc:
        raise InstallError(f"could not write to {target}: {exc}") from exc


def write_file_once(path: Path, text: str) -> bool:
    """Write `text` only if `path` does not exist. True when it was written."""
    if path.exists():
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        _warn(f"could not write {path} ({exc})")
        return False
    return True


def prepare_data_dir(root: Path) -> None:
    r"""Create the folder tree the app expects to find on first start.

    `root` is the app's own folder (`%LOCALAPPDATA%\Captioner`), or the install
    folder for a portable copy: either way the app looks for `data/` inside it,
    and creating it here means the first launch never has to.
    """
    for sub in (root, root / "data" / "outputs", root / "data" / "logs"):
        try:
            sub.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _warn(f"could not create {sub} ({exc})")


def portable_env(target: Path) -> str:
    """A `.env` that keeps every byte of state inside the install folder."""
    return (
        "# Written by Captioner Setup --portable: this copy is self-contained,\n"
        "# so it can live on a USB stick and keep its own jobs and captions.\n"
        f"CAPTIONER_DATA_DIR={target / 'data'}\n"
    )


def _batch_path(path) -> str:
    """A path safe to sit inside a batch file's quoted string.

    ``%`` is a legal character in a Windows folder name and batch expands it:
    ``C:\\100% done`` would eat the rest of the line, and ``%TEMP%`` in a folder
    name would silently point at some other folder. Doubling it is the batch
    escape, and the script prints the path the user chose.
    """
    return str(path).replace("%", "%%")


def write_uninstaller(app_dir: Path, target: Path, links: Iterable[Path]) -> Path:
    r"""A one-click uninstall, next to the data so it survives its own delete.

    What it removes and what it keeps is the whole design: the program folder
    and the shortcuts go, `%LOCALAPPDATA%\Captioner` (database, captions,
    settings, models) stays, and the script says where it is so the next step is
    the user's decision rather than a surprise.
    """
    quoted = " ".join(f'"{_batch_path(link)}"' for link in links)
    text = UNINSTALL_TEMPLATE.format(
        app=APP_NAME, target=_batch_path(target), data_dir=_batch_path(app_dir),
        links=quoted or '""', menu=_batch_path(_start_menu() / APP_NAME),
    )
    path = app_dir / "uninstall.cmd"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ install


@dataclass
class Options:
    """One resolved request: everything the steps need, and nothing else."""

    payload: Path
    target: Path
    app_dir: Path
    shortcuts: bool
    desktop_icon: bool
    launch: bool
    portable: bool
    # A compressed payload, used instead of ``payload`` when the folder is too
    # big for PyInstaller's one-file archive (see payload_archive).
    archive: Path | None = None

    @property
    def state_dir(self) -> Path:
        """The folder that holds `.env`, `data/` and the uninstaller.

        The user's own folder normally; the install folder for a portable copy,
        which is what makes one carry its own jobs and settings.
        """
        return self.target if self.portable else self.app_dir


def steps(options: Options) -> list[tuple[str, Callable[[], None]]]:
    """The install, as named steps, so a window can report each one.

    Each callable takes no arguments and is expected to be quick: the whole
    thing is a file copy.
    """
    plan: list[tuple[str, Callable[[], None]]] = [
        (f"Copying the application to {options.target}",
         lambda: copy_app(options.payload, options.target, options.archive)),
        # Where `data/` goes: beside the app normally, inside it when portable.
        ("Preparing the data folder",
         lambda: prepare_data_dir(options.state_dir)),
    ]

    def make_shortcuts() -> None:
        # The Uninstall entry points at uninstall.cmd, so that file is written
        # first (empty), the shortcuts are made - the list then includes the
        # Uninstall entry - and it is rewritten with the links that really
        # landed, which is what a later `--uninstall` should try to delete.
        write_uninstaller(options.state_dir, options.target, [])
        wanted = shortcuts_for(options.target, options.state_dir, options.state_dir,
                               desktop=options.desktop_icon, start_menu=True)
        made = create_shortcuts(wanted)
        write_uninstaller(options.state_dir, options.target, made)
        if len(made) < len(wanted):
            _warn(f"{len(wanted) - len(made)} of {len(wanted)} shortcuts could not "
                  f"be created; start {options.target / EXE_NAME} directly")

    if options.shortcuts:
        plan.append(("Creating shortcuts", make_shortcuts))
    else:
        # Still an uninstaller to click, it just has no links to clean up - and
        # this is the only place one is written when shortcuts are skipped.
        plan.append((f"Writing the uninstaller to {options.state_dir}",
                     lambda: write_uninstaller(options.state_dir, options.target, [])))

    if options.portable:
        # The app looks for a .env beside its own executable first; that is what
        # makes a portable copy carry its state.
        plan.append(("Writing the portable settings file",
                     lambda: write_file_once(options.target / ".env",
                                             portable_env(options.target))))
    else:
        plan.append((f"Writing a settings template to {options.state_dir}",
                     lambda: write_file_once(options.state_dir / ".env", ENV_TEMPLATE)))
    return plan


def install(options: Options, *, say: Callable[[str], None] = print,
            progress: Callable[[int, int], None] | None = None) -> int:
    """Run the plan, reporting every step. Returns a process exit code."""
    plan = steps(options)
    for index, (label, action) in enumerate(plan, start=1):
        say(label)
        if progress:
            progress(index - 1, len(plan))
        try:
            action()
        except InstallError as exc:
            say(f"error: {exc}")
            return 1
        except OSError as exc:
            say(f"error: {exc}")
            return 1
    if progress:
        progress(len(plan), len(plan))
    say("")
    say(f"{APP_NAME} is installed in {options.target}")
    say(f"Your jobs, captions and settings live in {options.state_dir / 'data'}")
    return 0


def uninstall(target: Path, app_dir: Path, *, purge_data: bool = False,
              say: Callable[[str], None] = print) -> int:
    """Remove the app and its shortcuts; keep the data unless told otherwise.

    Deliberately conservative: transcripts are the one thing here that cannot be
    downloaded again, so `--purge-data` is the only way to lose them. The
    install folder is only removed when it is one (see check_target), so a
    mistyped ``--uninstall`` cannot take an unrelated folder with it.
    """
    try:
        target = check_target(target)
    except InstallError as exc:
        say(f"error: {exc}")
        return 1
    if target.exists():
        say(f"Removing {target}")
        try:
            shutil.rmtree(target)
        except OSError as exc:
            say(f"error: could not remove {target}: {exc}")
            say(f"Close {APP_NAME} and run this again.")
            return 1
    else:
        say(f"Nothing installed in {target}")

    # Every shortcut this install describes, including the Uninstall entry -
    # uninstall.cmd is still in place here, which is what puts that one in the
    # list. It is deleted below, once the links are gone.
    links = shortcuts_for(target, app_dir, app_dir, desktop=True, start_menu=True)
    removed = remove_shortcuts(links)
    say(f"Removed {len(removed)} shortcut(s)")

    if purge_data:
        say(f"Removing your data in {app_dir}")
        try:
            shutil.rmtree(app_dir)
        except OSError as exc:
            say(f"error: could not remove {app_dir}: {exc}")
            return 1
        say("Everything is gone.")
        return 0

    (app_dir / "uninstall.cmd").unlink(missing_ok=True)
    say(f"Your data is untouched: {app_dir}")
    say("Delete that folder too with --purge-data.")
    return 0


# ------------------------------------------------------------------------- ui


def resolve(args: argparse.Namespace) -> Options:
    # Absolute, always: a portable install writes its data path into `.env`, and
    # a relative one would be read relative to whatever folder the app starts
    # in (its own), putting the user's jobs somewhere they never chose.
    target = (Path(args.target).expanduser() if args.target
              else default_target()).resolve()
    archive = (Path(args.payload_zip).expanduser() if args.payload_zip
               else payload_archive())
    return Options(
        payload=Path(args.payload).expanduser() if args.payload else payload_dir(),
        target=target,
        app_dir=data_dir(),
        shortcuts=not args.no_shortcuts,
        desktop_icon=not args.no_desktop_icon,
        launch=not args.no_launch,
        portable=args.portable,
        archive=archive,
    )


def launch_app(target: Path) -> bool:
    """Start the installed app, detached, so this program can exit."""
    exe = target / EXE_NAME
    if not exe.exists():
        _warn(f"{exe} is missing, so it was not started")
        return False
    try:
        if sys.platform == "win32":
            subprocess.Popen([str(exe)], cwd=str(target), close_fds=True)
        else:  # a dev machine on another OS can still exercise the installer
            subprocess.Popen([str(exe)], cwd=str(target))
    except OSError as exc:
        _warn(f"could not start {exe} ({exc})")
        return False
    return True


def _run_wizard(options: Options, version: str) -> int:
    """The window: a folder to pick, two boxes, one button, live progress."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    result = {"code": 1}

    root = tk.Tk()
    root.title(f"{APP_NAME} {version} - Setup")
    root.minsize(560, 420)

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text=f"Install {APP_NAME} {version}",
              font=("Segoe UI", 14, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Captions your files and your live audio, offline, on this "
                          "computer's own GPU.\nNothing is uploaded and no account is "
                          "needed.").pack(anchor="w", pady=(4, 4))
    # What this build includes, before anything is copied: the slim and full
    # builds differ by 3.5 GiB and three features, and the option to get the full
    # one is *not* in the window - it is a different download.
    ttk.Label(frame, text=payload_note(options.payload), wraplength=520,
              justify="left").pack(anchor="w", pady=(0, 12))

    ttk.Label(frame, text="Install folder").pack(anchor="w")
    folder_row = ttk.Frame(frame)
    folder_row.pack(fill="x", pady=(2, 12))
    folder = tk.StringVar(value=str(options.target))
    entry = ttk.Entry(folder_row, textvariable=folder)
    entry.pack(side="left", fill="x", expand=True)

    def browse() -> None:
        chosen = filedialog.askdirectory(title="Where should Captioner go?",
                                         initialdir=str(options.target))
        if chosen:
            folder.set(chosen)

    ttk.Button(folder_row, text="&Browse...", command=browse).pack(side="left", padx=(8, 0))

    shortcuts = tk.BooleanVar(value=options.shortcuts)
    desktop = tk.BooleanVar(value=options.desktop_icon)
    portable = tk.BooleanVar(value=options.portable)
    launches = tk.BooleanVar(value=options.launch)
    ttk.Checkbutton(frame, text="Add to the Start Menu (with an Uninstall entry)",
                    variable=shortcuts).pack(anchor="w")
    ttk.Checkbutton(frame, text="Add a Desktop icon",
                    variable=desktop).pack(anchor="w")
    ttk.Checkbutton(frame, text="Portable: keep jobs and settings in the install "
                                "folder (for a USB stick)",
                    variable=portable).pack(anchor="w")
    ttk.Checkbutton(frame, text=f"Start {APP_NAME} when setup finishes",
                    variable=launches).pack(anchor="w")

    bar = ttk.Progressbar(frame, mode="determinate")
    bar.pack(fill="x", pady=(14, 6))
    log = tk.Text(frame, height=9, wrap="word", relief="flat")
    log.pack(fill="both", expand=True)
    log.configure(state="disabled")

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(10, 0))

    def say(message: str) -> None:
        log.configure(state="normal")
        log.insert("end", message + "\n")
        log.see("end")
        log.configure(state="disabled")

    def step_done(done: int, total: int) -> None:
        bar.configure(maximum=total, value=done)
        root.update_idletasks()

    # `&` gives each button an Alt-key mnemonic, so the whole window is usable
    # without a mouse (Enter installs, Escape closes - see the bindings below).
    install_button = ttk.Button(buttons, text="&Install")
    install_button.pack(side="right")
    cancel = ttk.Button(buttons, text="&Cancel", command=root.destroy)
    cancel.pack(side="right", padx=(0, 8))
    cancel.focus_set()

    def worker(chosen: Options) -> None:
        code = install(chosen, say=lambda m: root.after(0, say, m),
                       progress=lambda d, t: root.after(0, step_done, d, t))
        result["code"] = code
        root.after(0, finish, code, chosen)

    def finish(code: int, chosen: Options) -> None:
        install_button.configure(state="normal")
        cancel.configure(text="&Close")
        if code != 0:
            messagebox.showerror(f"{APP_NAME} Setup",
                                 "Setup could not finish. See the details above.")
            return
        if chosen.launch and launches.get():
            launch_app(chosen.target)
        messagebox.showinfo(
            f"{APP_NAME} is installed",
            f"{APP_NAME} is in {chosen.target}.\n\n"
            f"Your jobs and captions are kept in\n"
            f"{chosen.state_dir / 'data'}",
        )
        root.destroy()

    def start() -> None:
        options.target = Path(folder.get()).expanduser()
        options.shortcuts = shortcuts.get()
        options.desktop_icon = desktop.get()
        options.portable = portable.get()
        options.launch = launches.get()
        if not str(options.target).strip():
            messagebox.showerror(f"{APP_NAME} Setup", "Choose an install folder first.")
            return
        say(f"Installing {APP_NAME} {version} into {options.target}")
        install_button.configure(state="disabled")
        cancel.configure(text="&Cancel")
        threading.Thread(target=worker, args=(options,), daemon=True).start()

    install_button.configure(command=start)
    root.bind("<Return>", lambda _event: start())
    root.bind("<Escape>", lambda _event: root.destroy())
    entry.focus_set()
    root.mainloop()
    return int(result["code"])


# ------------------------------------------------------------------------ cli


def _payload_ok(payload: Path, archive: Path | None = None) -> bool:
    """Refuse to "install" an empty setup, loudly, before touching anything."""
    if archive is not None and archive.exists():
        return True
    if (payload / EXE_NAME).exists():
        return True
    print(f"error: this setup carries no application "
          f"({EXE_NAME} is missing from {payload})", file=sys.stderr)
    return False


def _log_to(app_dir: Path) -> Callable[[str], None]:
    """Say it on the console (when there is one) *and* in a log file.

    The log lives beside the app's own logs, so an unattended install can be
    audited afterwards on a machine nobody is sitting at.
    """
    log = app_dir / "logs" / "setup.log"
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    def say(message: str) -> None:
        print(message)
        try:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        except OSError:
            pass

    return say


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{APP_NAME}-Setup", description=__doc__.splitlines()[0])
    parser.add_argument("--silent", action="store_true",
                        help="install without a window (all defaults)")
    parser.add_argument("--target", metavar="DIR",
                        help=f"install folder (default {default_target()})")
    parser.add_argument("--portable", action="store_true",
                        help="keep jobs and settings in the install folder")
    parser.add_argument("--no-shortcuts", action="store_true",
                        help="create no Start Menu entries")
    parser.add_argument("--no-desktop-icon", action="store_true",
                        help="create no Desktop icon")
    parser.add_argument("--no-launch", action="store_true",
                        help=f"do not start {APP_NAME} when done")
    parser.add_argument("--uninstall", action="store_true",
                        help="remove the application (keeps your data)")
    parser.add_argument("--purge-data", action="store_true",
                        help="with --uninstall: remove the data folder as well")
    parser.add_argument("--payload", metavar="DIR",
                        help="the app to install (only for testing the installer)")
    parser.add_argument("--payload-zip", metavar="ZIP",
                        help="a compressed payload to install instead (only for "
                             "testing the installer)")
    parser.add_argument("--version", action="store_true",
                        help="print the version this setup carries and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    options = resolve(args)
    version = payload_version(options.payload, options.archive)

    if args.version:
        if not options.payload.exists() and options.archive is None:
            print(f"{APP_NAME} setup, no application payload found")
            return 1
        print(f"{APP_NAME} {version}, "
              f"{human_bytes(payload_bytes(options.payload, options.archive))} "
              "to install")
        print(payload_note(options.payload, options.archive))
        return 0

    if args.silent:
        # A windowed build has no console to print to, and nobody is watching an
        # unattended install - so the steps also go to a log file.
        say = _log_to(options.app_dir)
        if args.uninstall:
            return uninstall(options.target, options.app_dir,
                             purge_data=args.purge_data, say=say)
        if not _payload_ok(options.payload, options.archive):
            return 1
        return install(options, say=say)

    if args.uninstall:
        return uninstall(options.target, options.app_dir, purge_data=args.purge_data)

    if not _payload_ok(options.payload, options.archive):
        return 1

    try:
        return _run_wizard(options, version)
    except Exception as exc:  # a broken window must still be able to install
        print(f"warning: the setup window could not open ({exc}); "
              "installing without it", file=sys.stderr)
        return install(options)


if __name__ == "__main__":
    raise SystemExit(main())
