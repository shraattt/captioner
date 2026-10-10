"""System-wide hotkey (Windows): Ctrl+Alt+L toggles live captions.

Qt shortcuts only fire while our own window has focus. ``RegisterHotKey``
makes the combination work in *any* app — that is what this module wraps.

Two consequences worth knowing:
  - the combination is **consumed system-wide** while the app runs: other
    programs stop seeing Ctrl+Alt+L;
  - only one process can own a given combination. If another app already
    registered it, registration fails — the failure is reported through a
    signal instead of being swallowed, so the UI can say so honestly.

Implementation: an invisible native window is created on a worker thread;
the Win32 message loop delivers ``WM_HOTKEY`` there and it is forwarded as
a Qt signal. ``stop()`` unregisters, posts WM_QUIT, and joins the thread.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import threading

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
HOTKEY_ID = 0xC4B  # arbitrary but unique inside our message window

VK_L = 0x4C  # virtual-key code for 'L'


def describe(modifiers: int, vk: int) -> str:
    """Human-readable name for a modifier/vk pair, e.g. 'Ctrl+Alt+L'."""
    parts = []
    if modifiers & MOD_CONTROL:
        parts.append("Ctrl")
    if modifiers & MOD_ALT:
        parts.append("Alt")
    if modifiers & MOD_SHIFT:
        parts.append("Shift")
    name = {VK_L: "L"}.get(vk, f"vk{vk:#x}")
    return "+".join(parts + [name])


class GlobalHotkey(QObject):
    """Owns the registration + message loop; emits ``activated`` on press.

    A QObject so signals emitted from the hotkey thread queue into the GUI
    thread automatically (auto connections turn queued across threads).

    >>> hk = GlobalHotkey()
    >>> hk.activated.connect(toggle_live)
    >>> hk.start()          # registration happens on the hotkey thread
    >>> hk.error            # None, or a human-readable reason it failed
    >>> hk.stop()           # app shutdown: release the combination
    """

    activated = Signal()
    registration_failed = Signal(str)

    def __init__(self, modifiers: int = MOD_CONTROL | MOD_ALT, vk: int = VK_L):
        super().__init__()  # QObject init: signals need a valid C++ object
        self._modifiers = modifiers
        self._vk = vk
        self._thread: threading.Thread | None = None
        self._thread_id = None
        self._hwnd = None
        self._registered = threading.Event()
        self._failed = threading.Event()
        self._failure: str | None = None
        self.error: str | None = None
        self._stop_called = threading.Event()

    # -- public API
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_called.clear()
        self._failed.clear()
        self._failure = None
        self.error = None
        self._thread = threading.Thread(
            target=self._run, name="global-hotkey", daemon=True
        )
        self._thread.start()
        # registration success/failure is known once the loop is running
        if not self._registered.wait(5.0):
            self._failed.wait(5.0)  # returns as soon as a failure is recorded
        if self._failure:
            self.error = self._failure

    def stop(self) -> None:
        self._stop_called.set()
        hwnd, thread_id = self._hwnd, self._thread_id
        if hwnd is not None and thread_id is not None:
            user32 = ctypes.windll.user32
            user32.UnregisterHotKey(hwnd, HOTKEY_ID)
            user32.PostThreadMessageW(thread_id, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._hwnd = None
        self._thread_id = None

    # -- thread body
    def _run(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        # Explicit prototypes: without them ctypes passes ints as 32-bit
        # values, and HWND_MESSAGE (-3) zero-extends into an invalid 64-bit
        # handle - the window creation then fails on every call.
        user32.CreateWindowExW.restype = ctypes.wintypes.HWND
        user32.CreateWindowExW.argtypes = [
            ctypes.wintypes.DWORD,   # dwExStyle
            ctypes.wintypes.LPCWSTR,  # lpClassName
            ctypes.wintypes.LPCWSTR,  # lpWindowName
            ctypes.wintypes.DWORD,   # dwStyle
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.wintypes.HWND,     # hWndParent (HWND_MESSAGE lives here)
            ctypes.wintypes.HMENU,
            ctypes.wintypes.HINSTANCE,
            ctypes.wintypes.LPVOID,
        ]
        user32.RegisterHotKey.argtypes = [
            ctypes.wintypes.HWND, ctypes.c_int, ctypes.wintypes.UINT, ctypes.wintypes.UINT
        ]
        user32.RegisterHotKey.restype = ctypes.wintypes.BOOL
        user32.UnregisterHotKey.argtypes = [ctypes.wintypes.HWND, ctypes.c_int]
        user32.PostThreadMessageW.argtypes = [
            ctypes.wintypes.DWORD, ctypes.wintypes.UINT,
            ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM,
        ]

        # A message-only window: never visible, but it receives WM_HOTKEY.
        hwnd = user32.CreateWindowExW(
            0, "STATIC", "", 0, 0, 0, 0, 0,
            ctypes.wintypes.HWND(-3),  # HWND_MESSAGE
            None, None, None,
        )
        if not hwnd:
            err = kernel32.GetLastError()
            self._failure = (
                f"could not create the hotkey message window (WinError {err})"
            )
            self._failed.set()
            return
        self._hwnd = hwnd
        self._thread_id = kernel32.GetCurrentThreadId()

        if not user32.RegisterHotKey(hwnd, HOTKEY_ID, self._modifiers, self._vk):
            self._failure = (
                f"{describe(self._modifiers, self._vk)} is already taken by "
                "another program — start live captions from the Live menu instead"
            )
            self._failed.set()
            self.registration_failed.emit(self._failure)
            # keep pumping until stop() so the app can shut down cleanly
            msg = ctypes.wintypes.MSG()
            while not self._stop_called.is_set():
                if not user32.GetMessageW(ctypes.byref(msg), None, 0, 0):
                    break
            return

        self._registered.set()
        logger.info("global hotkey Ctrl+Alt+L registered")

        msg = ctypes.wintypes.MSG()
        while not self._stop_called.is_set():
            # GetMessageW blocks here; WM_QUIT (posted by stop()) ends it.
            r = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if r <= 0:  # 0 == WM_QUIT, -1 == error
                break
            if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                self.activated.emit()
        logger.info("global hotkey thread exiting")
