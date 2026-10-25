"""Keyboard input to the Strive window and its geometry (Windows only).

Keys are sent with SendInput as hardware scan codes, which the game reads
whatever the keyboard layout. Input only goes out while the Strive window is
the foreground window: :func:`press` brings it forward and refuses to type
into anything else.

Importing this module is safe on any OS; calling into it off Windows raises.
"""
from __future__ import annotations

import ctypes
import sys
import time

GAME_EXE = "GGST-Win64-Shipping"

SCAN = {
    "esc": 0x01, "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06, "6": 0x07,
    "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13, "t": 0x14, "y": 0x15, "u": 0x16,
    "i": 0x17, "o": 0x18, "p": 0x19, "enter": 0x1C, "a": 0x1E, "s": 0x1F, "d": 0x20,
    "f": 0x21, "g": 0x22, "h": 0x23, "j": 0x24, "k": 0x25, "l": 0x26, ";": 0x27, "'": 0x28, "z": 0x2C,
    "x": 0x2D, "c": 0x2E, "v": 0x2F, "b": 0x30, "n": 0x31, "m": 0x32, "space": 0x39,
    "bksp": 0x0E, "tab": 0x0F, "lshift": 0x2A,
    "f1": 0x3B, "f2": 0x3C, "f3": 0x3D, "f4": 0x3E, "f5": 0x3F, "f6": 0x40, "f7": 0x41, "f8": 0x42, "f9": 0x43, "f10": 0x44,
    # extended keys (E0 prefix)
    "up": 0xE048, "down": 0xE050, "left": 0xE04B, "right": 0xE04D,
}

INPUT_KEYBOARD, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, KEYEVENTF_EXTENDEDKEY = 1, 2, 8, 1
ALT = 0x38


class NotWindows(RuntimeError):
    pass


class FocusError(RuntimeError):
    pass


def require_windows() -> None:
    if sys.platform != "win32":
        raise NotWindows("Exporting Strive replays works on Windows only "
                         "(it drives the game window with Windows input and capture APIs).")


_user32 = None
_structs = None


def _u32():
    global _user32, _structs
    require_windows()
    if _user32 is None:
        import ctypes.wintypes as wt
        _user32 = ctypes.WinDLL("user32", use_last_error=True)

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                        ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

        class INPUT(ctypes.Structure):
            class _U(ctypes.Union):
                _fields_ = [("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]
            _anonymous_ = ("u",)
            _fields_ = [("type", wt.DWORD), ("u", _U)]

        _structs = (KEYBDINPUT, INPUT)
    return _user32


def send_scan(scan: int, up: bool) -> None:
    u = _u32()
    KEYBDINPUT, INPUT = _structs
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    if scan > 0xFF:
        flags |= KEYEVENTF_EXTENDEDKEY
        scan &= 0xFF
    inp = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(0, scan, flags, 0, 0))
    u.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def find_window() -> int:
    """The Strive window handle (visible, title contains GUILTY GEAR), or 0."""
    import ctypes.wintypes as wt
    u = _u32()
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        buf = ctypes.create_unicode_buffer(256)
        u.GetWindowTextW(hwnd, buf, 256)
        if u.IsWindowVisible(hwnd) and "GUILTY GEAR" in buf.value.upper():
            found.append(hwnd)
        return True

    u.EnumWindows(cb, 0)
    return found[0] if found else 0


def focus(verify: bool = False, tries: int = 3) -> int:
    """Bring Strive to the foreground. With ``verify``, raise FocusError when
    it does not stay there (keys would land in another window)."""
    u = _u32()
    hwnd = find_window()
    if not hwnd:
        raise FocusError("Strive window not found: is the game running?")
    for _ in range(tries):
        if u.GetForegroundWindow() == hwnd:
            return hwnd
        # An Alt tap lets SetForegroundWindow succeed from a background process.
        send_scan(ALT, False)
        send_scan(ALT, True)
        if u.IsIconic(hwnd):
            u.ShowWindow(hwnd, 9)  # SW_RESTORE
        u.SetForegroundWindow(hwnd)
        time.sleep(0.3)
    if verify and u.GetForegroundWindow() != hwnd:
        raise FocusError("could not bring the Strive window to the foreground")
    return hwnd


def press(seq: str, gap_ms: int = 150, verify_focus: bool = False) -> None:
    """Press keys in order: "u", "s s s", "a+s:100" (hold a and s for 100 ms)."""
    focus(verify=verify_focus)
    for tok in seq.split():
        name, _, ms = tok.partition(":")
        codes = [SCAN[k] for k in name.split("+")]
        if verify_focus and _u32().GetForegroundWindow() != find_window():
            raise FocusError("the Strive window lost focus; not sending keys")
        for c in codes:
            send_scan(c, False)
        time.sleep(int(ms or 60) / 1000)
        for c in reversed(codes):
            send_scan(c, True)
        time.sleep(gap_ms / 1000)


def client_rect(hwnd: int | None = None) -> tuple[int, int, int, int]:
    """(x, y, width, height) of the window's client area in screen pixels."""
    import ctypes.wintypes as wt
    u = _u32()
    hwnd = hwnd or find_window()
    if not hwnd:
        raise FocusError("Strive window not found: is the game running?")
    try:  # physical pixels, not DPI-scaled ones
        ctypes.WinDLL("user32").SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        pass
    rc = wt.RECT()
    u.GetClientRect(hwnd, ctypes.byref(rc))
    pt = wt.POINT(0, 0)
    u.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, rc.right - rc.left, rc.bottom - rc.top


def monitors() -> list[tuple[int, int, int, int]]:
    """Monitor rectangles (left, top, right, bottom) in enumeration order."""
    import ctypes.wintypes as wt
    u = _u32()
    out = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HMONITOR, wt.HDC, ctypes.POINTER(wt.RECT), wt.LPARAM)
    def cb(_m, _dc, rect, _):
        r = rect.contents
        out.append((r.left, r.top, r.right, r.bottom))
        return True

    u.EnumDisplayMonitors(0, 0, cb, 0)
    return out


def window_on_monitor(rect: tuple[int, int, int, int],
                      mons: list[tuple[int, int, int, int]]) -> tuple[int, int, int]:
    """(monitor index, x, y relative to it) for the monitor holding the
    window's centre: what ddagrab's output_idx/offset_x/offset_y want. The
    monitor order is EnumDisplayMonitors', which matches DXGI's output order on
    single-GPU systems."""
    x, y, w, h = rect
    cx, cy = x + w // 2, y + h // 2
    for i, (l, t, r, b) in enumerate(mons):
        if l <= cx < r and t <= cy < b:
            return i, x - l, y - t
    return 0, x, y
