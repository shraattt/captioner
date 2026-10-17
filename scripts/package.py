"""Build the shippable Windows app: one folder, and then one .exe.

    python scripts/package.py                     # dist/Captioner/ (+ a size report)
    python scripts/package.py --installer         # dist/Captioner-Setup-<version>.exe
    python scripts/package.py --installer --zip   # ...and a portable .zip of the folder
    python scripts/package.py --full --installer  # + translation, speaker labels
    python scripts/package.py --no-cuda           # a small CPU-only build
    python scripts/package.py --smoke             # run the built exe, check it starts

Two artifacts, because they answer two different questions:

* ``dist/Captioner/`` - the app itself, `Captioner.exe` plus its dependencies.
  Copy that folder anywhere and it runs: no Python, no pip, no checkout.
* ``dist/Captioner-Setup-<version>.exe`` - **one file** to hand to another
  laptop. It carries the folder above as a payload, unpacks it, adds the
  shortcuts and writes the settings template (scripts/installer.py). That is
  the file to put on a USB stick; the folder is what it installs.

**Slim and full are a real choice, not a detail.** A slim build is batch
transcription plus live captions with translation *only* through an API engine:
torch (4.2 GB installed) is left out, and so are speaker labels, word alignment
and fine-tuning, which need it. A full build packs all of that in: 4.8 GiB
installed, shipped as a ~3.0 GiB setup because a payload that big has to be
compressed before PyInstaller can carry it (see RAW_PAYLOAD_LIMIT). Slim is the
default because the file has to travel.

The CUDA runtime is the opposite decision: cuBLAS is ~740 MB of the installer
and it is included by default, because without it every GPU job on a machine
that has a driver but no CUDA Toolkit fails with "cublas64_12.dll is not
found" - and a GPU is the whole point of the app. ``--no-cuda`` drops it (and
needs ``DEVICE=cpu``).

Everything is written under ``build/`` and ``dist/``, both gitignored.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "package"
DIST = ROOT / "dist"
APP_NAME = "Captioner"
APP_FOLDER = DIST / APP_NAME
SETUP_NAME = f"{APP_NAME}-Setup"

# What the app actually imports. Everything else that happens to be installed
# in the development environment (pip, pytest, the fine-tuning scripts'
# dataset stack) is noise in a shipped build.
APP_HIDDEN_IMPORTS = [
    "av", "ctranslate2", "faster_whisper", "huggingface_hub", "numpy",
    "onnxruntime", "psutil", "soundcard", "tokenizers",
]

# Left out of every build: development tools, test frameworks, and the parts of
# Qt nobody reaches. Worth ~100 MB, and more usefully it keeps a stray
# `import pytest` from turning into a mystery crash inside a windowed build.
# (Deliberately *not* on the list: pip, wheel and setuptools. PyInstaller's own
# build process imports them, and excluding a module that is already imported
# raises "Target module ... already imported as ExcludedModule(...)" - the build
# stops dead. They are a few MB; the accuracy is worth more.)
DEV_EXCLUDES = [
    "pytest", "IPython", "notebook", "jupyter", "matplotlib", "tkinter",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtDesigner", "PySide6.QtSql",
    "PySide6.QtTest", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtSensors", "PySide6.QtSerialPort",
]

# Left out of a slim build: the transformers stack (torch), which is reached
# only for local translation (`m2m100`), speaker labels and word alignment, plus
# the scientific stack those pull in.
SLIM_EXCLUDES = DEV_EXCLUDES + [
    "torch", "torchaudio", "torchvision", "transformers", "pyannote",
    "sentencepiece", "sacremoses", "datasets", "accelerate", "librosa",
    "jiwer", "audiomentations", "torchcodec", "torchgen",
    "scipy", "pandas", "sklearn",
]

# The CUDA wheels that have to travel with a build that has no torch to borrow
# them from: cuBLAS (ctranslate2's encoder needs it) and the runtime compiler
# cuBLAS links against. A full build carries both already, in torch/lib, and
# packing them twice was 0.8 GiB of duplicate DLLs (see app_datas).
CUDA_PACKAGES = ("cublas", "cuda_nvrtc")

# PyInstaller's one-file archive writes its table of contents with 32-bit
# offsets, so the archive cannot pass 4 GiB. A payload bigger than that (the
# full build is 4.8 GiB) fails at the very end of a half-hour build with
# "struct.error: 'I' format requires 0 <= number <= 4294967295". Above this
# limit the folder is compressed first and the setup unpacks the archive.
RAW_PAYLOAD_LIMIT = 3_900_000_000


def needs_archive(payload_bytes: int) -> bool:
    """Does this payload have to be compressed to fit in one setup .exe?"""
    return payload_bytes > RAW_PAYLOAD_LIMIT


def version() -> str:
    """The version in pyproject.toml - one place, so they cannot disagree."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def site_packages() -> Path:
    """Where the wheels this build packs in actually live."""
    import site

    for candidate in site.getsitepackages() + [site.getusersitepackages()]:
        path = Path(candidate)
        if (path / "faster_whisper").is_dir():
            return path
    return Path(sys.prefix) / "Lib" / "site-packages"


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def spec_text(*, entry: Path, name: str, onefile: bool, windowed: bool,
              datas: list[tuple[str, str]], binaries: list[tuple[str, str]],
              hidden: list[str], excludes: list[str], extra_args: list[str],
              icon: Path | None) -> str:
    """The PyInstaller recipe, written out so a build is reproducible by hand.

    Kept as text (and dropped in build/package/) rather than as a checked-in
    .spec: the recipe depends on which wheels this machine has, and a stale
    checked-in spec is how a build silently ships last month's dependencies.
    """
    lines = [
        "# Generated by scripts/package.py - do not edit; edit the script.",
        "from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs",
        "",
        f"datas = {datas!r}",
        f"binaries = {binaries!r}",
        "",
        "# The nvidia wheels are data, not importable packages: ctranslate2 loads",
        "# cuBLAS by absolute path at runtime (transcribe.preload_cuda_dlls).",
        "for _pkg in ('ctranslate2', 'faster_whisper'):",
        "    try:",
        "        binaries += collect_dynamic_libs(_pkg)",
        "        datas += collect_data_files(_pkg)",
        "    except Exception as _exc:",
        "        print('note: nothing to collect from', _pkg, _exc)",
        "",
        f"hiddenimports = {hidden!r}",
        f"excludes = {excludes!r}",
        "",
        f"a = Analysis([{str(entry)!r}], pathex=[{str(ROOT)!r}],",
        "             binaries=binaries, datas=datas,",
        "             hiddenimports=hiddenimports, excludes=excludes,",
        "             noarchive=False, optimize=0)",
        "pyz = PYZ(a.pure)",
    ]
    # One file packs the binaries *into* the executable and unpacks them into a
    # temporary folder at every start; one folder keeps them beside it. The
    # setup .exe is a one-file build (it is the thing you hand over), the app is
    # a one-folder build (it is what that setup unpacks, and it starts faster).
    if onefile:
        lines += [
            "exe = EXE(pyz, a.scripts, a.binaries, a.datas, [],",
            f"          name={name!r}, debug=False, strip=False, upx=False,",
            f"          console={not windowed!r}, disable_windowed_traceback=False,",
        ]
    else:
        lines += [
            "exe = EXE(pyz, a.scripts, [], exclude_binaries=True,",
            f"          name={name!r}, debug=False, strip=False, upx=False,",
            f"          console={not windowed!r}, disable_windowed_traceback=False,",
        ]
    lines += [
        "          bootloader_ignore_signals=False, argv_emulation=False,",
        "          target_arch=None, codesign_identity=None, entitlements_file=None,",
        f"          icon={str(icon)!r},",
        ")",
    ]
    if not onefile:
        lines += [
            "",
            f"coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name={name!r})",
        ]
    if extra_args:
        lines += ["", "# extra arguments from the command line", f"# {extra_args!r}"]
    return "\n".join(lines) + "\n"


def run_pyinstaller(spec: Path, *, label: str, clean: bool,
                    env: dict | None = None) -> None:
    """Run PyInstaller on a generated spec, with its output where we want it."""
    command = [
        sys.executable, "-m", "PyInstaller", str(spec),
        "--noconfirm", "--distpath", str(DIST), "--workpath", str(BUILD / "work"),
        "--log-level", "WARN",
    ]
    if clean:
        command.append("--clean")
    print(f"== {label}")
    print("   " + " ".join(command))
    started = time.monotonic()
    done = subprocess.run(command, cwd=str(ROOT), env=env)
    if done.returncode != 0:
        raise SystemExit(f"PyInstaller failed for {label} (exit {done.returncode})")
    print(f"   built in {time.monotonic() - started:.0f}s")


def app_datas(*, cuda: bool, torch_bundled: bool = False) -> list[tuple[str, str]]:
    """Files the app reads at runtime but never imports.

    The CUDA wheels are only packed when the build has no torch to borrow them
    from. A full build carries torch, and torch/lib already ships cublas,
    cublasLt, cudnn and nvrtc - packing nvidia-cublas-cu12 and cuda-nvrtc on top
    of that put the same 0.8 GiB of DLLs in the folder twice (transcribe.
    preload_cuda_dlls looks in both places, nvidia first, then torch/lib).
    """
    datas = [
        (str(ROOT / "captioner.ico"), "."),
        (str(ROOT / "ui" / "web"), "ui/web"),
        (str(ROOT / "LICENSE"), "."),
    ]
    if not cuda:
        return datas
    site = site_packages()
    if torch_bundled and (site / "torch" / "lib" / "cublas64_12.dll").is_file():
        print("note: torch already carries the CUDA runtime, so the nvidia "
              "wheels are not packed again (saves ~0.8 GiB installed)")
        return datas
    for package in CUDA_PACKAGES:
        source = site / "nvidia" / package / "bin"
        if source.is_dir():
            datas.append((str(source), f"nvidia/{package}/bin"))
        else:
            print(f"note: no {package} wheel installed; the build will need "
                  "CAPTIONER_CUDA_BIN or a CUDA Toolkit on the target machine")
    return datas


def app_spec(*, full: bool, cuda: bool) -> Path:
    """Write dist/Captioner's recipe; returns the spec file to hand PyInstaller."""
    excludes = sorted(set(DEV_EXCLUDES if full else SLIM_EXCLUDES))
    hidden = list(APP_HIDDEN_IMPORTS)
    if full:
        hidden += ["torch", "transformers", "pyannote.audio"]
    spec = BUILD / "app.spec"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(spec_text(
        entry=ROOT / "scripts" / "app_window.py",
        name=APP_NAME, onefile=False, windowed=True,
        datas=app_datas(cuda=cuda, torch_bundled=full), binaries=[], hidden=hidden,
        excludes=excludes,
        extra_args=[], icon=ROOT / "captioner.ico",
    ), encoding="utf-8")
    return spec


def build_app(*, full: bool, cuda: bool, clean: bool) -> Path:
    """Freeze the dashboard into dist/Captioner and stamp its version."""
    spec = app_spec(full=full, cuda=cuda)
    run_pyinstaller(spec, label=f"app ({'full' if full else 'slim'})", clean=clean)

    if not APP_FOLDER.is_dir():
        raise SystemExit(f"expected {APP_FOLDER} to exist after the build")
    # What this installation is, in three lines the installer reads back:
    # version, which build (slim/full), and whether CUDA is inside. Without the
    # second and third line the setup cannot tell a user what they are about to
    # install - and the two builds differ by 3.5 GiB and three features.
    kinds = [version(), "full" if full else "slim", "cuda" if cuda else "cpu"]
    (APP_FOLDER / "version.txt").write_text("\n".join(kinds) + "\n", encoding="utf-8")
    return APP_FOLDER


def setup_spec(*, payload: Path | None = None,
               payload_zip: Path | None = None) -> Path:
    """Write the setup .exe's recipe, which carries the app as a payload.

    A folder that fits is packed file by file (``payload/``). One that does not
    is packed as a single compressed ``payload.zip`` - the only way a >4 GiB
    full build can ship as one .exe (see RAW_PAYLOAD_LIMIT).
    """
    payload = payload or APP_FOLDER
    spec = BUILD / "setup.spec"
    spec.parent.mkdir(parents=True, exist_ok=True)
    if payload_zip is not None:
        # The file keeps its own name inside _MEIPASS, so the caller hands in a
        # payload named payload.zip - the name scripts/installer.py looks for.
        # (A dest of "payload.zip" would be read as a *folder* by PyInstaller.)
        datas = [(str(payload_zip), "."), (str(ROOT / "captioner.ico"), ".")]
    else:
        datas = [(str(payload), "payload"), (str(ROOT / "captioner.ico"), ".")]
    spec.write_text(spec_text(
        entry=ROOT / "scripts" / "installer.py",
        name=f"{SETUP_NAME}-{version()}", onefile=True, windowed=True,
        datas=datas,
        binaries=[],
        # tkinter is imported inside the wizard function, which PyInstaller does
        # follow - but this is the one artifact that must open on a laptop that
        # has never seen Python, so it is named explicitly rather than trusted.
        hidden=["tkinter", "tkinter.filedialog", "tkinter.messagebox", "tkinter.ttk"],
        excludes=[], extra_args=[], icon=ROOT / "captioner.ico",
    ), encoding="utf-8")
    return spec


def payload_archive(payload: Path) -> Path:
    """The portable .zip, reused as the one compressed entry a setup carries.

    scripts/installer.py strips the leading ``Captioner/`` folder when it
    unpacks, so the same archive serves both jobs: a portable copy someone
    unzips, and the payload of a setup too big to embed file by file.

    It is reused when it is newer than everything in the folder, so
    ``--only-installer`` after a previous build does not spend ten minutes
    compressing the same 4.8 GiB again.
    """
    target = DIST / f"{APP_NAME}-{version()}-portable.zip"
    newest = max((path.stat().st_mtime for path in payload.rglob("*")
                  if path.is_file()), default=0.0)
    if target.exists() and target.stat().st_mtime >= newest:
        print(f"== reusing {target.name} as the one-file payload "
              f"({human(target.stat().st_size)})")
        return target
    return zip_folder(payload)


def build_installer(*, clean: bool) -> Path:
    """Pack the built folder into one .exe that installs it (scripts/installer.py)."""
    payload = APP_FOLDER
    if not (payload / f"{APP_NAME}.exe").exists():
        raise SystemExit(f"{payload} is not built yet - run this script without "
                         "--installer first")
    name = f"{SETUP_NAME}-{version()}"
    size = folder_size(payload)
    archive = payload_archive(payload) if needs_archive(size) else None
    if archive is not None:
        print(f"note: {human(size)} does not fit in PyInstaller's 4 GiB one-file "
              f"archive; the setup carries a {human(archive.stat().st_size)} "
              "compressed payload and unpacks it at install time")
    if archive is not None:
        # PyInstaller keeps a data file's own name inside _MEIPASS, so the
        # payload is hard-linked (instant, same volume) to the name the
        # installer looks for. A copy is the fallback on another filesystem.
        embedded = BUILD / "payload.zip"
        embedded.unlink(missing_ok=True)
        try:
            os.link(archive, embedded)
        except OSError:
            shutil.copyfile(archive, embedded)
        archive = embedded
    spec = setup_spec(payload=payload, payload_zip=archive)
    env = None
    if archive is not None:
        # The payload is already deflated. PyInstaller would re-compress it at
        # zlib level 9 - minutes of work for a fraction of a percent - so tell
        # it to store the archive as it is. The .exe is the same size either
        # way; only the build time changes.
        env = dict(os.environ, PYINSTALLER_ZLIB_COMPRESSION_LEVEL="0")
    run_pyinstaller(spec, label="setup (one file)", clean=clean, env=env)
    exe = DIST / f"{name}.exe"
    if not exe.exists():
        raise SystemExit(f"expected {exe} after the build")
    return exe


def zip_folder(folder: Path) -> Path:
    """A portable .zip of the app folder, for someone who cannot run a .exe."""
    target = DIST / f"{APP_NAME}-{version()}-portable.zip"
    print(f"== zip {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, Path(APP_NAME) / path.relative_to(folder))
    return target


def smoke_test(exe: Path, *, timeout: int) -> int:
    """Start the packaged app and let it close itself (ui.dashboard --self-test).

    Run with no display, offscreen, and with the data folder redirected into
    build/ so a packaging check never touches the user's real queue.
    """
    data = BUILD / "smoke-data"
    print(f"== smoke test {exe.name}")
    # CAPTIONER_DATA_DIR moves the queue into build/, so a packaging check never
    # touches the real data folder; offscreen because a build machine may have
    # no display, and no display is also the harder case to get right.
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", CAPTIONER_NO_GLOBAL_HOTKEY="1",
               CAPTIONER_DATA_DIR=str(data))
    print("   " + str(exe) + " --self-test")
    done = subprocess.run([str(exe), "--self-test"], cwd=str(exe.parent), env=env,
                          timeout=timeout, capture_output=True, text=True)
    if done.stdout.strip():
        print("   stdout: " + done.stdout.strip().replace("\n", "\n   "))
    if done.stderr.strip():
        print("   stderr: " + done.stderr.strip().replace("\n", "\n   "))
    log = data / "logs" / "captioner.log"
    if done.returncode != 0 and log.exists():
        print("   app log tail:")
        print("\n".join(f"   {line}" for line in
                        log.read_text(encoding="utf-8").splitlines()[-15:]))
    return done.returncode


def report(paths: list[Path], *, cuda: bool, full: bool) -> None:
    print("\n== what was built")
    for path in paths:
        size = folder_size(path) if path.is_dir() else path.stat().st_size
        print(f"   {human(size):>10}  {path.relative_to(ROOT)}")
    print(f"\n   {'full' if full else 'slim'}{'' if cuda else ', no CUDA'}")
    if not full:
        print("   without: translation (local model), speaker labels, word alignment, "
              "fine-tuning")
    print("   the .exe is the one to hand over: it installs the folder above, "
          "writes the shortcuts and starts the app")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="include torch/transformers (translation, speaker labels, "
                             "fine-tuning): a much bigger file")
    parser.add_argument("--no-cuda", action="store_true",
                        help="leave out the CUDA wheels (~740 MB); CPU-only machines")
    parser.add_argument("--installer", action="store_true",
                        help="also build the single-file setup .exe")
    parser.add_argument("--zip", action="store_true",
                        help="also write a portable .zip of the app folder")
    parser.add_argument("--only-installer", action="store_true",
                        help="skip the folder build (reuse dist/Captioner as-is)")
    parser.add_argument("--specs-only", action="store_true",
                        help="write the PyInstaller recipes to build/package/ and stop")
    parser.add_argument("--clean", action="store_true",
                        help="throw away PyInstaller's cache first (slower, safer)")
    parser.add_argument("--smoke", action="store_true",
                        help="run the built app with --self-test afterwards")
    parser.add_argument("--smoke-setup", action="store_true",
                        help="with --installer: also install the built setup into a "
                             "throwaway folder and uninstall it again")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sys.platform != "win32":
        print(f"note: this builds a Windows app; on {sys.platform} it will produce "
              "a build for this OS instead", file=sys.stderr)

    if args.specs_only:
        # Writing a recipe needs no PyInstaller, so this works anywhere - it is
        # what the tests use to check that a build will ask for the right thing.
        print("wrote", app_spec(full=args.full, cuda=not args.no_cuda))
        print("wrote", setup_spec())
        return 0

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is required: python -m pip install -r requirements-dev.txt",
              file=sys.stderr)
        return 1

    built: list[Path] = []
    folder = APP_FOLDER
    if not args.only_installer:
        folder = build_app(full=args.full, cuda=not args.no_cuda, clean=args.clean)
        built.append(folder)
    exe = None
    if args.installer:
        exe = build_installer(clean=args.clean)
        built.append(exe)
    if args.zip:
        built.append(zip_folder(folder))
    if not built:
        print("nothing to do: pass --installer and/or --zip (a plain run builds only "
              f"{APP_FOLDER.relative_to(ROOT)}/ and stops there)")

    report(built or [folder], cuda=not args.no_cuda, full=args.full)

    code = 0
    if args.smoke:
        code = smoke_test(folder / f"{APP_NAME}.exe", timeout=180)
        print("smoke test:", "OK" if code == 0 else f"FAILED (exit {code})")
    if args.smoke_setup and exe is not None:
        sandbox = BUILD / "smoke-setup"
        print("== install/uninstall the built setup, into", sandbox)
        setup = [str(exe), "--silent", "--target", str(sandbox), "--no-shortcuts",
                 "--no-launch"]
        for command in (setup, setup + ["--uninstall"]):
            done = subprocess.run(command, timeout=900)
            if done.returncode != 0:
                code = done.returncode
                print("setup smoke test FAILED (exit", done.returncode, ")")
                break
        else:
            print("setup smoke test: OK (installed and removed",
                  sandbox.relative_to(ROOT), ")")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
