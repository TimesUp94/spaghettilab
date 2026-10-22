"""Drive Guilty Gear Strive from a script: screenshots of its window and keyboard input.

Usage: python ggst.py shot <out.png>
       python ggst.py keys "<key>[:ms] <key>[:ms] ..." [gap_ms]
Keys are scan-code names (see SCAN); input goes to the Strive window, which is
brought to the foreground first. Screenshots use FFmpeg's gfxcapture, so they
work while the window is covered.
"""
import ctypes
import ctypes.wintypes as wt
import glob
import os
import subprocess
import sys
import time

# FFmpeg 9+ (needs the gfxcapture source); override with GGST_FFMPEG.
FFMPEG = os.environ.get("GGST_FFMPEG") or glob.glob(os.path.join(
    os.environ.get("GGST_DATA", r"G:\Projects\replanal-data"), "ggst_mod", "dl", "ffmpeg-*", "bin", "ffmpeg.exe"))[0]

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

user32 = ctypes.WinDLL("user32", use_last_error=True)
INPUT_KEYBOARD, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, KEYEVENTF_EXTENDEDKEY = 1, 2, 8, 1


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _U)]


def _send(scan: int, up: bool) -> None:
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    if scan > 0xFF:
        flags |= KEYEVENTF_EXTENDEDKEY
        scan &= 0xFF
    inp = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(0, scan, flags, 0, 0))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def find_window() -> int:
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        if user32.IsWindowVisible(hwnd) and "GUILTY GEAR" in buf.value.upper():
            found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else 0


def focus() -> None:
    hwnd = find_window()
    if not hwnd:
        raise SystemExit("Strive window not found")
    if user32.GetForegroundWindow() != hwnd:
        # Alt tap lets SetForegroundWindow succeed from a background process.
        _send(0x38, False); _send(0x38, True)
        user32.ShowWindow(hwnd, 9)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.3)


def keys(seq: str, gap_ms: int = 150) -> None:
    """Press keys in order. 'a+s:100' holds a and s together for 100 ms."""
    focus()
    for tok in seq.split():
        name, _, ms = tok.partition(":")
        codes = [SCAN[k] for k in name.split("+")]
        for c in codes:
            _send(c, False)
        time.sleep(int(ms or 60) / 1000)
        for c in reversed(codes):
            _send(c, True)
        time.sleep(gap_ms / 1000)


def shot(out: str) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-filter_complex",
                    "gfxcapture=window_exe=GGST-Win64-Shipping:capture_cursor=0,hwdownload,format=bgra",
                    "-frames:v", "1", out], check=True)


if __name__ == "__main__":
    if sys.argv[1] == "shot":
        shot(sys.argv[2])
    elif sys.argv[1] == "keys":
        keys(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 150)
