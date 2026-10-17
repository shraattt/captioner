"""Tests for scripts/package.py: the recipes a build runs, not the build.

Freezing the app takes minutes and gigabytes, so the build itself is not part of
the test suite (scripts/package.py --smoke does that on demand). What *is* worth
testing is everything that decides what gets packed: the two artifacts' recipes,
which wheels a slim build leaves out, and the version stamps agreeing with each
other - all of which are cheap to check and expensive to get wrong, because a
mistake here only shows up on someone else's laptop.
"""
import ast
import zipfile

import pytest

from captioner import __version__
from scripts import package


def test_the_two_artifacts_are_written_by_two_recipes(tmp_path):
    """The app is one folder, the setup is one file, and the folder is inside it."""
    setup = package.setup_spec(payload=tmp_path)
    app = package.app_spec(full=False, cuda=True)
    setup_text = setup.read_text(encoding="utf-8")
    app_text = app.read_text(encoding="utf-8")

    assert f"name='{package.SETUP_NAME}-{package.version()}'" in setup_text
    assert "name='Captioner'" in app_text
    # The app folder travels inside the setup as a data directory called
    # "payload" (the name scripts/installer.py looks for in _MEIPASS).
    assert "'payload'" in setup_text
    assert repr(str(tmp_path)) in setup_text
    assert "'payload'" not in app_text


def test_the_app_is_one_folder_and_the_setup_is_one_file(tmp_path):
    app = package.app_spec(full=False, cuda=False).read_text(encoding="utf-8")
    setup = package.setup_spec(payload=tmp_path).read_text(encoding="utf-8")

    # One folder: binaries live beside the executable, collected by COLLECT.
    assert "exclude_binaries=True" in app
    assert "coll = COLLECT(exe, a.binaries, a.datas" in app
    # One file: everything goes *into* the executable, and stays there.
    assert "exclude_binaries=True" not in setup
    assert "a.binaries, a.datas" in setup
    assert "COLLECT" not in setup


def test_both_recipes_are_valid_python(tmp_path):
    """A spec with a syntax error fails a 10-minute build at the last second."""
    for spec in (package.app_spec(full=False, cuda=True),
                 package.setup_spec(payload=tmp_path)):
        ast.parse(spec.read_text(encoding="utf-8"))


def test_both_recipes_are_windowed_and_carry_the_icon():
    for text in (package.app_spec(full=False, cuda=False).read_text(encoding="utf-8"),
                 package.setup_spec().read_text(encoding="utf-8")):
        assert "console=False" in text
        assert "icon=" in text and "captioner.ico" in text


def test_a_payload_too_big_for_one_file_travels_compressed(tmp_path):
    """PyInstaller's one-file archive is 32-bit; a full build needs a zip."""
    assert package.needs_archive(package.RAW_PAYLOAD_LIMIT + 1)
    assert not package.needs_archive(package.RAW_PAYLOAD_LIMIT)

    archive = tmp_path / "payload.zip"
    archive.write_bytes(b"PK")
    text = package.setup_spec(payload_zip=archive).read_text(encoding="utf-8")
    # Packed at the top of _MEIPASS under its own name - which is why the
    # caller hard-links it to payload.zip (the name the installer looks for).
    assert f"({str(archive)!r}, '.',)" in text or f"({str(archive)!r}, '.')" in text
    # The folder is not packed as well: it would not fit, and it would double
    # the size of the setup.
    assert "'payload'" not in text


def test_the_payload_archive_is_the_portable_zip_and_is_reused(tmp_path,
                                                             monkeypatch):
    """One archive, two uses - and not built twice for one setup."""
    folder = tmp_path / "Captioner"
    (folder / "_internal").mkdir(parents=True)
    (folder / "Captioner.exe").write_bytes(b"MZ")
    (folder / "_internal" / "python313.dll").write_bytes(b"dll")
    monkeypatch.setattr(package, "DIST", tmp_path / "dist")

    archive = package.payload_archive(folder)
    assert archive.name == f"Captioner-{package.version()}-portable.zip"
    with zipfile.ZipFile(archive) as bundle:
        assert "Captioner/Captioner.exe" in bundle.namelist()
    # Already newer than the folder: the second build must not recompress it.
    stamp = archive.stat().st_mtime_ns
    assert package.payload_archive(folder) == archive
    assert archive.stat().st_mtime_ns == stamp


def test_the_setup_names_tkinter_explicitly():
    """The wizard is the one thing that must open on a machine without Python."""
    setup = package.setup_spec().read_text(encoding="utf-8")
    for module in ("tkinter", "tkinter.ttk", "tkinter.messagebox"):
        assert f"'{module}'" in setup


def test_the_app_resources_are_data_not_imports():
    datas = package.app_datas(cuda=False)
    sources = {source.replace("\\", "/") for source, _ in datas}
    assert any(source.endswith("captioner.ico") for source in sources)
    assert any(source.endswith("ui/web") for source in sources)
    assert any(source.endswith("LICENSE") for source in sources)
    # The icon has to sit at the top of the bundle: ui/dashboard looks for it
    # beside its own package root, which is _MEIPASS in a frozen build, and
    # the browser UI keeps its own folder.
    assert any(dest == "." and source.endswith("captioner.ico")
               for source, dest in datas)
    assert any(dest == "ui/web" for _, dest in datas)


def test_cuda_wheels_are_packed_when_asked_for():
    with_cuda = package.app_datas(cuda=True)
    targets = {dest for _, dest in with_cuda}
    installed = (package.site_packages() / "nvidia" / "cublas").is_dir()
    if installed:
        assert "nvidia/cublas/bin" in targets
    assert not any("nvidia" in dest for _, dest in package.app_datas(cuda=False))


def test_a_full_build_does_not_pack_the_cuda_runtime_twice():
    """torch/lib already has cublas/cudnn/nvrtc; the nvidia wheels are 0.8 GiB."""
    torch_lib = package.site_packages() / "torch" / "lib"
    full = package.app_datas(cuda=True, torch_bundled=True)
    if (torch_lib / "cublas64_12.dll").is_file():
        assert not any(dest.startswith("nvidia") for _, dest in full)
    # The app still gets what it reads at runtime, and a slim build (no torch)
    # still travels with the wheels it needs.
    assert any(dest == "ui/web" for _, dest in full)
    slim = package.app_datas(cuda=True, torch_bundled=False)
    assert any(dest.startswith("nvidia/") for _, dest in slim) == (
        (package.site_packages() / "nvidia").is_dir()
    )


def test_a_slim_build_leaves_the_transformers_stack_out():
    assert "torch" in package.SLIM_EXCLUDES
    assert "transformers" in package.SLIM_EXCLUDES
    assert "pyannote" in package.SLIM_EXCLUDES
    # ...but a full build keeps them, and both leave the dev tools out.
    assert "torch" not in package.DEV_EXCLUDES
    assert "pytest" in package.DEV_EXCLUDES
    assert set(package.DEV_EXCLUDES) <= set(package.SLIM_EXCLUDES)


def test_the_version_is_read_from_pyproject_and_matches_the_app():
    """Three stamps, one number: a mismatch ships the wrong version to a user."""
    assert package.version() == __version__
    from ui.dashboard.constants import APP_VERSION

    assert APP_VERSION == __version__


def test_human_sizes_read_like_file_sizes():
    assert package.human(999) == "999 B"
    assert package.human(2 * 1024 ** 3) == "2.0 GiB"


def test_the_app_folder_and_the_setup_land_in_dist():
    assert package.APP_FOLDER == package.DIST / "Captioner"
    assert package.BUILD.parts[-2:] == ("build", "package")


@pytest.mark.parametrize("argv", [["--help"], ["--specs-only"]])
def test_the_command_line_offers_a_dry_run(argv, capsys):
    """--specs-only is the one build path a test can run end to end."""
    if argv == ["--help"]:
        with pytest.raises(SystemExit):
            package.main(argv)
        assert "--installer" in capsys.readouterr().out
        return
    assert package.main(["--specs-only", "--no-cuda"]) == 0
    out = capsys.readouterr().out
    assert "app.spec" in out and "setup.spec" in out
    for spec in (package.BUILD / "app.spec", package.BUILD / "setup.spec"):
        ast.parse(spec.read_text(encoding="utf-8"))
