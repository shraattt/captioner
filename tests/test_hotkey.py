"""Tests for the Windows global hotkey (Ctrl+Alt+L).

The real combination is registered against the OS and fired with a
synthesized keystroke; on a machine where Ctrl+Alt+L is already taken those
tests skip honestly rather than fail. Dashboard wiring runs offscreen with
fakes, no audio device involved.
"""
import ctypes
import ctypes.wintypes
import sys
import time

import pytest

pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.hotkey import MOD_ALT, MOD_CONTROL, VK_L, GlobalHotkey  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


def _pump_until(qapp, predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _input_desktop_available() -> bool:
    """False when the workstation is locked (input desktop is Winlogon).

    RegisterHotKey still succeeds while locked, but synthesized keystrokes
    are injected into the (absent) interactive desktop and never deliver
    WM_HOTKEY — an environmental failure, not a code one.
    """
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    user32.OpenInputDesktop.argtypes = [
        ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD
    ]
    user32.OpenInputDesktop.restype = ctypes.wintypes.HANDLE
    DESKTOP_GENERIC_READ = 0x0001
    hwnd = user32.OpenInputDesktop(0, False, DESKTOP_GENERIC_READ)
    if hwnd:
        user32.CloseDesktop(hwnd)
        return True
    return False


def _keystrokes_can_be_injected(qapp) -> bool:
    """True if the session delivers synthesized keystrokes at all.

    Some session policies (and some remote-desktop / locked-session states)
    silently drop injected input; RegisterHotKey itself still works and real
    user keystrokes still fire. Probe with a throwaway hotkey on an unused
    combination so the real test's registration is unaffected.
    """
    if sys.platform != "win32":
        return False
    probe = GlobalHotkey(MOD_CONTROL | MOD_ALT, 0x79)  # Ctrl+Alt+F9
    fired = []
    probe.activated.connect(lambda: fired.append(1))
    probe.start()
    try:
        if probe.error:
            return False
        time.sleep(0.3)
        user32 = ctypes.windll.user32
        KEYEVENTF_KEYUP = 0x0002
        user32.keybd_event(0xA2, 0, 0, 0)   # Left Ctrl
        user32.keybd_event(0xA4, 0, 0, 0)   # Left Alt
        user32.keybd_event(0x79, 0, 0, 0)   # F9
        user32.keybd_event(0x79, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(0xA4, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(0xA2, 0, KEYEVENTF_KEYUP, 0)
        return bool(_pump_until(qapp, lambda: bool(fired), timeout=2.0))
    finally:
        probe.stop()


def test_hotkey_registers_and_fires(qapp):
    if not _input_desktop_available():
        pytest.skip("workstation is locked — synthesized keystrokes cannot deliver")
    if not _keystrokes_can_be_injected(qapp):
        pytest.skip(
            "this session drops synthesized keystrokes (policy/remote session); "
            "real keystrokes still work — verify Ctrl+Alt+L by hand"
        )
    hk = GlobalHotkey(MOD_CONTROL | MOD_ALT, VK_L)
    fires = []
    hk.activated.connect(lambda: fires.append(1))
    hk.start()
    try:
        if hk.error:  # the combination is owned by another program here
            pytest.skip(f"Ctrl+Alt+L not registrable on this machine: {hk.error}")
        user32 = ctypes.windll.user32
        KEYEVENTF_KEYUP = 0x0002
        # Synthesize the real keystroke; the OS delivers WM_HOTKEY to our window.
        user32.keybd_event(0xA2, 0, 0, 0)              # Left Ctrl down
        user32.keybd_event(0xA4, 0, 0, 0)              # Left Alt down
        user32.keybd_event(0x4C, 0, 0, 0)              # L down
        user32.keybd_event(0x4C, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(0xA4, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(0xA2, 0, KEYEVENTF_KEYUP, 0)
        # the signal crosses threads; pump the Qt loop to receive it.
        # 5s: the hotkey thread loads win32 DLLs on first use, which is slow
        # on a cold start right after a full-suite compile.
        assert _pump_until(qapp, lambda: bool(fires), timeout=5.0), \
            "registered hotkey did not fire on the synthesized keystroke"
    finally:
        hk.stop()
    assert hk._thread is None


def test_hotkey_stop_is_idempotent(qapp):
    hk = GlobalHotkey(MOD_CONTROL | MOD_ALT, VK_L)
    hk.start()
    if hk.error:
        pytest.skip("combination taken on this machine")
    hk.stop()
    hk.stop()  # must not raise
    assert hk._thread is None


def test_hotkey_stop_without_start_is_safe(qapp):
    hk = GlobalHotkey()
    hk.stop()  # never started: must not raise
    assert hk._thread is None


# dashboard wiring (fakes; no real device or model)

class FakeWorker:
    def __init__(self, device_id):
        self.device_id = device_id
        self._running = False
        self.stopped = False

    def isRunning(self):
        return self._running

    def start(self):
        self._running = True

    def stop(self):
        self.stopped = True
        self._running = False

    def wait(self, ms):
        return True


@pytest.fixture()
def dashboard(qapp):
    import ui.dashboard as dashboard_mod

    win = dashboard_mod.Dashboard(persist_settings=False)
    yield win
    if win._hotkey is not None:
        win._hotkey.stop()
    win._drafts.clear()
    win.deleteLater()


def test_toggle_live_stops_a_running_session(dashboard, monkeypatch):
    launches = []
    monkeypatch.setattr(dashboard, "_launch_live", lambda d: launches.append(d["id"]))
    dashboard._live_device = {"id": "dev-1", "name": "Fake Mic"}
    fake = FakeWorker("dev-1")
    dashboard._live_worker = fake
    fake.start()

    dashboard.toggle_live()  # running -> stops
    assert launches == []
    assert fake.stopped
    assert dashboard._live_worker is None  # stop_live clears the slot

    dashboard.toggle_live()  # second press: relaunches the remembered device
    assert launches == ["dev-1"]


def test_toggle_live_asks_once_then_remembers(dashboard, monkeypatch):
    picked = {"id": "dev-9", "name": "Picked Mic"}
    monkeypatch.setattr(dashboard, "_pick_live_device", lambda: picked)

    def fake_launch(device):
        dashboard._live_device = device  # like the real _launch_live

    monkeypatch.setattr(dashboard, "_launch_live", fake_launch)

    dashboard._live_device = None
    dashboard.toggle_live()  # first press: picker runs, choice is remembered
    assert dashboard._live_device["id"] == "dev-9"

    monkeypatch.setattr(dashboard, "_pick_live_device", lambda: None)
    dashboard.toggle_live()  # a canceled picker does nothing
    assert dashboard._live_device["id"] == "dev-9"
