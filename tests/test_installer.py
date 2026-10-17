"""Tests for scripts/installer.py, the program the setup .exe unpacks.

Nothing here touches the real machine: the install target, the data folder and
the shortcut creator are all passed in, so the tests are the same on Windows,
Linux and CI. Shortcuts go through `WScript.Shell` (PowerShell), which a test
cannot and should not run - what is tested is *which* links an install wants and
what happens when one cannot be created.
"""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import installer


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A machine: an app to install, a place for it, and a fake shell."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))

    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / installer.EXE_NAME).write_bytes(b"MZ fake executable")
    (payload / "_internal").mkdir()
    (payload / "_internal" / "python313.dll").write_bytes(b"x" * 2048)
    (payload / "version.txt").write_text("9.9.9\n", encoding="utf-8")

    made: list = []

    def fake_create(shortcut) -> bool:
        made.append(shortcut)
        shortcut.link.write_text("link", encoding="utf-8")
        return True

    return {
        "payload": payload,
        "target": tmp_path / "Programs" / "Captioner",
        "data": tmp_path / "Local" / "Captioner",
        "create": fake_create,
        "made": made,
        "tmp": tmp_path,
    }


def install_into(world, **overrides) -> int:
    options = installer.Options(
        payload=world["payload"], target=world["target"], app_dir=world["data"],
        shortcuts=True, desktop_icon=True, launch=False,
        portable=overrides.pop("portable", False),
    )
    for key, value in overrides.items():
        setattr(options, key, value)
    # The steps call `create_shortcuts` without a creator, so patch the default.
    original = installer.powershell_shortcut
    installer.powershell_shortcut = world["create"]
    try:
        return installer.install(options, say=lambda _line: None)
    finally:
        installer.powershell_shortcut = original


# where things go


def test_default_target_is_per_user_and_needs_no_admin(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert installer.default_target() == tmp_path / "Local" / "Programs" / "Captioner"
    assert installer.data_dir() == tmp_path / "Local" / "Captioner"
    # The data folder is next to the programs folder, never inside it: an
    # uninstall replaces the program folder wholesale.
    assert installer.default_target() not in installer.data_dir().parents


def test_a_relative_target_is_made_absolute(tmp_path, monkeypatch):
    """--target Captioner from a USB stick must not write a relative data path."""
    monkeypatch.chdir(tmp_path)
    args = installer.build_parser().parse_args(["--target", "Captioner"])
    assert installer.resolve(args).target == (tmp_path / "Captioner").resolve()


def test_data_dir_survives_a_missing_localappdata(monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    assert installer.data_dir().name == "Captioner"


def test_shortcuts_name_the_exe_the_data_folder_and_the_uninstaller(world):
    world["data"].mkdir(parents=True)
    (world["data"] / "uninstall.cmd").write_text("@echo off", encoding="utf-8")
    links = installer.shortcuts_for(world["target"], world["data"], world["data"])

    start = next(link for link in links if link.link.parent.name == "Captioner")
    assert start.target == world["target"] / installer.EXE_NAME
    assert start.link.name == "Captioner.lnk"

    uninstall = next(link for link in links
                     if link.link.name.startswith("Uninstall"))
    assert uninstall.target == world["data"] / "uninstall.cmd"
    # Removing the program folder must not take the uninstaller with it.
    assert uninstall.target != start.target

    folder = next(link for link in links if "data folder" in link.link.name)
    assert folder.target == world["data"]

    desktop = [link for link in links if link.link.parent ==
               world["tmp"] / "home" / "Desktop"]
    assert [link.link.name for link in desktop] == ["Captioner.lnk"]
    assert len(links) == 4


def test_shortcuts_can_skip_the_desktop_and_the_start_menu(world):
    assert installer.shortcuts_for(world["target"], world["data"], world["data"],
                                   desktop=False, start_menu=False) == []
    no_menu = installer.shortcuts_for(world["target"], world["data"], world["data"],
                                      start_menu=False)
    assert [link.link.name for link in no_menu] == ["Captioner.lnk"]


# creating and removing links


def test_every_shortcut_is_verified_after_it_is_claimed(world):
    """A creator that says yes and writes nothing must not be believed."""
    wanted = installer.shortcuts_for(world["target"], world["data"], world["data"])
    made = installer.create_shortcuts(wanted, create=lambda _s: True)
    assert made == []
    assert all(not link.link.exists() for link in wanted)


def test_a_failing_shortcut_does_not_stop_the_others(world):
    wanted = installer.shortcuts_for(world["target"], world["data"], world["data"])

    def flaky(shortcut) -> bool:
        if "data folder" in shortcut.link.name:
            return False
        shortcut.link.write_text("link", encoding="utf-8")
        return True

    made = installer.create_shortcuts(wanted, create=flaky)
    assert len(made) == len(wanted) - 1
    assert all(link.exists() for link in made)


def test_removing_shortcuts_cleans_up_its_own_start_menu_folder(world):
    wanted = installer.shortcuts_for(world["target"], world["data"], world["data"])
    installer.create_shortcuts(wanted, create=world["create"])
    menu = wanted[0].link.parent
    assert menu.is_dir()
    assert len(installer.remove_shortcuts(wanted)) == len(wanted)
    assert not menu.exists()          # the empty Start Menu folder goes too
    assert installer.remove_shortcuts(wanted) == []   # idempotent


# the install itself


def test_install_copies_the_app_and_writes_what_it_promised(world):
    assert install_into(world) == 0
    target, data = world["target"], world["data"]
    assert (target / installer.EXE_NAME).read_bytes() == b"MZ fake executable"
    assert (target / "_internal" / "python313.dll").exists()
    assert (target / "version.txt").read_text(encoding="utf-8") == "9.9.9\n"
    assert (data / "data" / "outputs").is_dir()
    assert (data / "data" / "logs").is_dir()
    assert "# HF_TOKEN=hf_your_token_here" in (data / ".env").read_text(encoding="utf-8")
    assert len(world["made"]) == 4


def test_install_never_overwrites_a_settings_file(world):
    (world["data"]).mkdir(parents=True)
    (world["data"] / ".env").write_text("HF_TOKEN=mine\n", encoding="utf-8")
    assert install_into(world) == 0
    assert (world["data"] / ".env").read_text(encoding="utf-8") == "HF_TOKEN=mine\n"


def test_install_replaces_an_older_version_wholesale(world):
    stale = world["target"] / "_internal"
    stale.mkdir(parents=True)
    (stale / "gone.dll").write_bytes(b"old")
    (world["target"] / installer.EXE_NAME).write_bytes(b"old")
    assert install_into(world) == 0
    assert not (stale / "gone.dll").exists()
    assert (world["target"] / installer.EXE_NAME).read_bytes() == b"MZ fake executable"


def test_install_refuses_an_empty_setup(world):
    (world["payload"] / installer.EXE_NAME).unlink()
    assert install_into(world) == 1
    assert not world["target"].exists()


def test_install_refuses_to_replace_a_folder_that_is_not_ours(world):
    """Installing deletes the target first; that must never take other files."""
    world["target"].mkdir(parents=True)
    thesis = world["target"] / "thesis.docx"
    thesis.write_bytes(b"do not delete")

    assert install_into(world) == 1
    assert thesis.read_bytes() == b"do not delete"


def test_install_refuses_drive_roots_and_the_home_folder(world, monkeypatch):
    """A mistyped --target is the one way this could erase a machine."""
    home = world["tmp"] / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))

    with pytest.raises(installer.InstallError):
        installer.check_target(home)
    with pytest.raises(installer.InstallError):
        installer.check_target(Path(home.anchor))
    # A file where the folder should be is refused too, not crashed on.
    afile = world["tmp"] / "notafolder"
    afile.write_bytes(b"x")
    with pytest.raises(installer.InstallError):
        installer.check_target(afile)
    # An empty folder and one that already holds the app are both fine.
    empty = world["tmp"] / "empty"
    empty.mkdir()
    assert installer.check_target(empty) == empty.resolve()
    assert installer.check_target(world["target"]) == world["target"].resolve()


def test_uninstall_refuses_a_folder_that_is_not_ours(world):
    world["target"].mkdir(parents=True)
    photos = world["target"] / "photos"
    photos.mkdir()
    (photos / "a.jpg").write_bytes(b"x")

    assert installer.uninstall(world["target"], world["data"],
                               say=lambda _line: None) == 1
    assert (photos / "a.jpg").exists()


def test_portable_install_keeps_everything_in_the_install_folder(world):
    assert install_into(world, portable=True) == 0
    env = (world["target"] / ".env").read_text(encoding="utf-8")
    assert f"CAPTIONER_DATA_DIR={world['target'] / 'data'}" in env
    assert (world["target"] / "data" / "outputs").is_dir()
    # No machine-wide settings template, no uninstaller: nothing outside the
    # install folder belongs to a portable copy.
    assert not (world["data"] / ".env").exists()
    assert not (world["data"] / "uninstall.cmd").exists()


def test_shortcuts_can_be_skipped(world):
    assert install_into(world, shortcuts=False) == 0
    assert world["made"] == []


def test_a_portable_install_really_moves_the_data_folder(world, tmp_path):
    """The promise of `--portable`, checked end to end in a fresh interpreter.

    `captioner.config` fixes `DATA_DIR` when it is imported, so the only honest
    way to prove that the `.env` this installer writes actually moves the data
    folder is to start Python again and ask. (CAPTIONER_ENV_FILE stands in for
    "the .env beside the executable", which needs a real frozen build.)
    """
    root = Path(installer.__file__).resolve().parents[1]
    stick = tmp_path / "stick"
    stick.mkdir()
    (stick / ".env").write_text(installer.portable_env(stick), encoding="utf-8")

    env = {key: value for key, value in os.environ.items()
           if key != "CAPTIONER_DATA_DIR"}
    env["CAPTIONER_ENV_FILE"] = str(stick / ".env")
    done = subprocess.run(
        [sys.executable, "-c", "import captioner.config as c; print(c.DATA_DIR)"],
        cwd=str(root), env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(stick / "data")


def test_the_uninstaller_knows_the_links_that_were_made(world):
    install_into(world)
    script = (world["data"] / "uninstall.cmd").read_text(encoding="utf-8")
    assert str(world["target"]) in script
    for shortcut in world["made"]:
        assert f'"{shortcut.link}"' in script


def test_the_uninstaller_escapes_a_percent_in_the_install_path(world):
    """`%` is legal in a folder name and batch would expand it."""
    target = world["tmp"] / "100% done"
    script = installer.write_uninstaller(world["data"], target, [])
    text = script.read_text(encoding="utf-8")
    assert '100%% done"' in text
    # Doubled everywhere it is used: batch sees one literal % either way.
    assert "100% done" not in text


def test_the_uninstaller_script_is_shaped_correctly(world):
    """Its content is the contract: what it deletes, what it keeps, and how it
    ends. The self-delete in particular: asking the running copy of cmd to
    remove the file it is reading leaves a stray "the batch file cannot be
    found" on screen, so it is done from a second copy instead."""
    script = installer.write_uninstaller(
        world["data"], world["target"], [world["tmp"] / "a.lnk"],
    ).read_text(encoding="utf-8")
    assert f'rmdir /s /q "{world["target"]}"' in script      # the app goes
    assert "for %%L in (" in script                          # so do the links
    assert f'"{world["tmp"] / "a.lnk"}"' in script
    assert f'rmdir /q "{installer._start_menu() / "Captioner"}"' in script
    assert str(world["data"]) in script                       # the data stays, named
    assert 'start "" /b' in script and "%~f0" in script
    assert not script.strip().endswith(">nul 2>&1")      # no bare self-delete


def test_the_uninstaller_always_deletes_a_list_even_when_empty(world):
    """`for %%L in ()` is a syntax error, so an empty list has to be quoted."""
    script = installer.write_uninstaller(world["data"], world["target"], []).read_text(
        encoding="utf-8")
    assert "for %%L in (\"\")" in script


def test_an_unwritable_target_is_reported_not_raised(world):
    blocker = world["target"].parent
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("in the way", encoding="utf-8")   # a file where a folder goes
    assert install_into(world) == 1


# uninstalling


def test_uninstall_keeps_the_users_captions(world):
    install_into(world)
    (world["data"] / "data" / "outputs" / "job_1").mkdir()
    (world["data"] / "data" / "outputs" / "job_1" / "captions.srt").write_text("hi")
    assert installer.uninstall(world["target"], world["data"],
                               say=lambda _line: None) == 0
    assert not world["target"].exists()
    assert (world["data"] / "data" / "outputs" / "job_1" / "captions.srt").exists()
    assert (world["data"] / "uninstall.cmd").exists() is False
    assert world["made"] and not any(shortcut.link.exists() for shortcut in world["made"])


def test_purge_data_is_the_only_way_to_lose_the_data(world):
    install_into(world)
    (world["data"] / "data" / "captioner.db").write_text("jobs")
    assert installer.uninstall(world["target"], world["data"], purge_data=True,
                               say=lambda _line: None) == 0
    assert not world["data"].exists()


def test_uninstalling_something_that_is_not_there_is_not_an_error(world):
    lines: list[str] = []
    assert installer.uninstall(world["target"], world["data"],
                               say=lines.append) == 0
    assert any("Nothing installed" in line for line in lines)


# helpers


def test_write_file_once_leaves_an_existing_file_alone(world):
    first = world["tmp"] / "notes.txt"
    assert installer.write_file_once(first, "one") is True
    assert installer.write_file_once(first, "two") is False
    assert first.read_text(encoding="utf-8") == "one"


def test_human_bytes_reads_like_a_file_size():
    assert installer.human_bytes(512) == "512 B"
    assert installer.human_bytes(1536) == "1.5 KiB"
    assert installer.human_bytes(3 * 1024 ** 3 + 512 * 1024 ** 2) == "3.5 GiB"


@pytest.mark.parametrize("stamp, version", [
    ("9.9.9\n", "9.9.9"),                       # a build from before this stamp
    ("9.9.9\nslim\ncuda\n", "9.9.9"),
    ("\n", "0.0.0"),                            # an empty stamp is not a version
])
def test_the_version_is_the_stamps_first_line(world, stamp, version):
    (world["payload"] / "version.txt").write_text(stamp, encoding="utf-8")
    assert installer.payload_version(world["payload"]) == version


def test_the_setup_says_what_it_is_about_to_install(world):
    """The set-up window is the last chance to say "this is the 1.4 GB one"."""
    (world["payload"] / "version.txt").write_text("9.9.9\nslim\ncuda\n",
                                                  encoding="utf-8")
    slim = installer.payload_note(world["payload"])
    assert "Slim build" in slim and "CUDA" in slim
    assert "live captions" in slim                      # what it does do
    assert "Local translation" in slim and "not" in slim  # and what it does not
    assert "4.8 GiB" in slim                            # the alternative, priced

    (world["payload"] / "version.txt").write_text("9.9.9\nfull\ncuda\n",
                                                  encoding="utf-8")
    full = installer.payload_note(world["payload"])
    assert "Full build" in full and "Everything is included" in full
    assert "fine-tuning" in full


# a payload too big for PyInstaller's 32-bit one-file archive


def _zip_payload(world, stamp: str = "9.9.9\nfull\ncuda\n") -> Path:
    """The full build's shape: one compressed file instead of the folder."""
    archive = world["tmp"] / "payload.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in world["payload"].rglob("*"):
            if path.is_file() and path.name != "version.txt":
                bundle.write(path, path.relative_to(world["payload"]))
        bundle.writestr("version.txt", stamp)
    return archive


def test_a_compressed_payload_installs_like_a_folder(world):
    archive = _zip_payload(world)
    # The folder is not there to copy from any more - only the archive is.
    (world["payload"] / installer.EXE_NAME).unlink()
    assert install_into(world, archive=archive) == 0
    assert (world["target"] / installer.EXE_NAME).read_bytes() == b"MZ fake executable"
    assert (world["target"] / "_internal" / "python313.dll").exists()
    assert (world["data"] / ".env").exists()


def test_a_portable_zip_installs_its_top_folder_away(world):
    """The portable archive is the payload; its ``Captioner/`` level goes."""
    archive = world["tmp"] / "portable.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in world["payload"].rglob("*"):
            if path.is_file():
                bundle.write(path, "Captioner/" + path.relative_to(
                    world["payload"]).as_posix())
    installer.copy_app(world["payload"], world["target"], archive=archive)
    assert (world["target"] / installer.EXE_NAME).exists()
    assert not (world["target"] / "Captioner").exists()


def test_a_compressed_payload_replaces_an_older_install(world):
    world["target"].mkdir(parents=True)
    # An older install: it has the exe (which is how a target is recognised as
    # ours) and files this build no longer ships.
    (world["target"] / installer.EXE_NAME).write_bytes(b"old version")
    (world["target"] / "stale.dll").write_bytes(b"old")
    assert install_into(world, archive=_zip_payload(world)) == 0
    assert not (world["target"] / "stale.dll").exists()
    assert (world["target"] / installer.EXE_NAME).read_bytes() == b"MZ fake executable"


def test_the_setup_reads_its_description_out_of_the_archive(world):
    # The real payload is the portable zip, which keeps its Captioner/ folder.
    archive = world["tmp"] / "portable.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Captioner/version.txt", "9.9.9\nfull\ncuda\n")
        for path in world["payload"].rglob("*"):
            if path.is_file() and path.name != "version.txt":
                bundle.write(path, "Captioner/" + path.relative_to(
                    world["payload"]).as_posix())
    (world["payload"] / "version.txt").unlink()
    assert installer.payload_version(world["payload"], archive) == "9.9.9"
    assert "Full build" in installer.payload_note(world["payload"], archive)
    expected = sum(path.stat().st_size for path in world["payload"].rglob("*")
                   if path.is_file() and path.name != "version.txt")
    assert installer.payload_bytes(world["payload"], archive) == expected + len(
        "9.9.9\nfull\ncuda\n"
    )


def test_an_archive_without_the_app_is_refused(world, tmp_path):
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("readme.txt", "nothing here")
    with pytest.raises(installer.InstallError, match="no application"):
        installer.copy_app(world["payload"], world["target"], archive=archive)
    assert not world["target"].exists()


def test_the_command_line_takes_a_compressed_payload(world, capsys):
    archive = _zip_payload(world)
    missing = world["tmp"] / "no-folder"
    assert installer.main(["--version", "--payload", str(missing),
                           "--payload-zip", str(archive)]) == 0
    out = capsys.readouterr().out
    assert "9.9.9" in out and "Full build" in out


def test_a_payload_with_no_stamp_is_described_honestly(world, capsys):
    (world["payload"] / "version.txt").unlink()
    note = installer.payload_note(world["payload"])
    assert "does not say" in note
    assert "CPU only" in note              # no cuda line either
    assert installer.main(["--version", "--payload", str(world["payload"])]) == 0
    out = capsys.readouterr().out
    assert "0.0.0" in out and "does not say" in out


def test_payload_version_and_size_are_read_from_the_payload(world):
    assert installer.payload_version(world["payload"]) == "9.9.9"
    assert installer.payload_version(world["tmp"] / "nope") == "0.0.0"
    expected = sum(path.stat().st_size for path in world["payload"].rglob("*")
                   if path.is_file())
    assert installer.payload_bytes(world["payload"]) == expected
    assert installer.payload_bytes(world["payload"]) > 2048  # every file counts


# the command line


def test_cli_version_reports_the_payload(world, capsys):
    code = installer.main(["--version", "--payload", str(world["payload"])])
    assert code == 0
    assert "9.9.9" in capsys.readouterr().out


def test_cli_version_without_a_payload_says_so(world, capsys):
    assert installer.main(["--version", "--payload",
                           str(world["tmp"] / "nothing")]) == 1
    assert "no application payload" in capsys.readouterr().out


def test_cli_silent_install_and_uninstall(world, capsys, monkeypatch):
    install = ["--silent", "--payload", str(world["payload"]), "--target",
               str(world["target"]), "--no-shortcuts", "--no-launch"]
    assert installer.main(install) == 0
    out = capsys.readouterr().out
    assert str(world["target"]) in out
    assert (world["target"] / installer.EXE_NAME).exists()
    # A silent install also leaves an audit trail: a windowed build has no
    # console, and nobody is watching an unattended install.
    log = world["data"] / "logs" / "setup.log"
    assert "Copying the application" in log.read_text(encoding="utf-8")

    assert installer.main(["--silent", "--uninstall", "--target",
                           str(world["target"])]) == 0
    assert not world["target"].exists()
    assert world["data"].exists()


def test_cli_silent_install_of_an_empty_setup_fails_before_writing(world, capsys):
    (world["payload"] / installer.EXE_NAME).unlink()
    assert installer.main(["--silent", "--payload", str(world["payload"]),
                           "--target", str(world["target"])]) == 1
    captured = capsys.readouterr()
    assert "carries no application" in captured.err + captured.out
    assert not world["target"].exists()


def test_the_wizard_window_can_be_built_and_closed(monkeypatch, tmp_path):
    """The wizard's wiring has no other test, so build every widget once.

    The event loop is stubbed out (the window destroys itself instead of
    running), which is what makes this cheap: it catches a typo in the widget
    tree without a display server - which is exactly how the four checkboxes
    passing a BooleanVar where their parent frame belongs were found. Skipped
    where tkinter cannot open a window at all (a headless CI box).
    """
    tkinter = pytest.importorskip("tkinter")
    try:
        tkinter.Tk().destroy()
    except tkinter.TclError:
        pytest.skip("no display for tkinter")
    monkeypatch.setattr(tkinter.Tk, "mainloop", lambda self: self.destroy())

    options = installer.Options(
        payload=tmp_path / "payload", target=tmp_path / "app",
        app_dir=tmp_path / "data", shortcuts=True, desktop_icon=True,
        launch=True, portable=False,
    )
    # 1 is the wizard's "nothing was installed" default, which is what a window
    # that opens and closes without a click must report.
    assert installer._run_wizard(options, "9.9.9") == 1


def test_the_default_target_is_shown_in_the_help(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    # A wide terminal, or argparse wraps the path across two lines and the
    # assertion below would fail for a reason that has nothing to do with it.
    monkeypatch.setenv("COLUMNS", "400")
    with pytest.raises(SystemExit):
        installer.build_parser().parse_args(["--help"])
    assert str(tmp_path / "Programs" / "Captioner") in capsys.readouterr().out
