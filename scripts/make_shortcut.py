"""Create a "Captioner" shortcut on the user's Desktop (Windows).

Uses the Explorer COM API through PowerShell, so no extra packages are
needed. Re-running is safe: the shortcut is simply overwritten.

    python scripts/make_shortcut.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The .vbs launches pythonw (no console window); the .bat stays as the
# visible fallback for first-run setup and troubleshooting.
TARGET = ROOT / "Captioner.vbs"
ICON = ROOT / "captioner.ico"
SHORTCUT_NAME = "Captioner"


def main() -> int:
    if sys.platform != "win32":
        print("This helper is Windows-only (it creates a .lnk).")
        return 1
    if not TARGET.exists():
        print(f"missing {TARGET} — run this from the captioner checkout")
        return 1

    desktop = Path.home() / "Desktop"
    if not desktop.exists():  # OneDrive-redirected desktops
        desktop = Path.home() / "OneDrive" / "Desktop"
    lnk = desktop / f"{SHORTCUT_NAME}.lnk"

    icon = str(ICON) if ICON.exists() else str(ROOT / ".venv" / "Scripts" / "pythonw.exe")
    ps = f"""
$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}')
$s.TargetPath = '{TARGET}'
$s.WorkingDirectory = '{ROOT}'
$s.Description = 'Captioner - local AI captions (batch + live)'
$s.IconLocation = '{icon}'
$s.Save()
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0 or not lnk.exists():
        print("could not create the shortcut:")
        print(result.stderr.strip() or result.stdout.strip())
        return 1
    print(f"Desktop shortcut created: {lnk}")
    print("Double-click it any time - it launches silently (no console window).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
