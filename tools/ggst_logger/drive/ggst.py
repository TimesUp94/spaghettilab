"""Drive Guilty Gear Strive from a script: screenshots of its window and keyboard input.

Usage: python ggst.py shot <out.png>
       python ggst.py place               (borderless 1920x1080 at GGST_WINDOW_POS)
       python ggst.py keys "<key>[:ms] <key>[:ms] ..." [gap_ms]
Keys are scan-code names (see SCAN). By default they are posted to the Strive window
as WM_KEYDOWN/WM_KEYUP, which the game takes while in the background, so the PC stays
usable; GGST_FOREGROUND=1 brings it to the front and uses SendInput instead (waiting
while someone else is using the PC). Screenshots use FFmpeg's gfxcapture, so they
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
    global _sent_tick
    _sent_tick = kernel32.GetTickCount()
    _sent.append(_sent_tick)
    del _sent[:-200]


kernel32 = ctypes.WinDLL("kernel32")
kernel32.GetTickCount.restype = wt.DWORD
_sent_tick = 0
_sent = []             # GetTickCount of our recent SendInput calls
USER_QUIET_S = 60       # look this far back for someone else's keyboard/mouse input
USER_EVENTS = 3         # this many separate inputs in that time = someone is at the PC
# A lone input now and then is not a person: some program on this PC sends one exactly
# every 50 s (2026-10-26), which a plain "no input for 60 s" check never gets past.


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wt.UINT), ("dwTime", wt.DWORD)]


_inputs = []            # GetTickCount of each input change seen that was not ours
_watch_t0 = None


def _near_ours(tick: int) -> bool:
    """Input stamped within a second of one of our SendInput calls (either side: the
    event is stamped before _send records the time, and the watcher may see it late)."""
    for t in reversed(_sent):
        d = (tick - t) & 0xFFFFFFFF
        if d <= 1000 or d >= 0xFFFFFFFF - 1000:
            return True
    return False


def _watch() -> None:
    last = None
    while True:
        li = _LASTINPUTINFO(ctypes.sizeof(_LASTINPUTINFO), 0)
        user32.GetLastInputInfo(ctypes.byref(li))
        # SendInput also counts as input: skip changes within a second of our own keys
        if li.dwTime != last and last is not None and not _near_ours(li.dwTime):
            _inputs.append(li.dwTime)
            del _inputs[:-50]
        last = li.dwTime
        time.sleep(0.1)


def wait_user_idle() -> None:
    """Wait while someone else is using the PC: USER_EVENTS separate inputs (not ours)
    within USER_QUIET_S. Watches input in a thread; the first call watches 10 s first."""
    global _watch_t0
    if _watch_t0 is None:
        import threading
        threading.Thread(target=_watch, daemon=True).start()
        _watch_t0 = time.time()
    time.sleep(max(0.0, _watch_t0 + 10 - time.time()))
    said = False
    while True:
        now = kernel32.GetTickCount()
        recent = [t for t in _inputs if (now - t) & 0xFFFFFFFF < USER_QUIET_S * 1000]
        if len(recent) < USER_EVENTS:
            return
        if not said:
            print("  someone is using the PC; waiting for them to stop", flush=True)
            said = True
        time.sleep(5)


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
    for _ in range(5):
        if user32.GetForegroundWindow() == hwnd:
            return
        # Alt tap lets SetForegroundWindow succeed from a background process.
        _send(0x38, False); _send(0x38, True)
        user32.ShowWindow(hwnd, 9)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.3)
    # SendInput types into whatever window is in front: never into someone else's app
    raise SystemExit("could not bring the Strive window to the front; no keys sent")


FOREGROUND = os.environ.get("GGST_FOREGROUND") == "1"
WM_KEYDOWN, WM_KEYUP, MAPVK_VSC_TO_VK_EX = 0x0100, 0x0101, 3


def _post(hwnd: int, scan: int, up: bool) -> None:
    """Key message to the game window only: never reaches the app someone is using."""
    vk = user32.MapVirtualKeyW(scan, MAPVK_VSC_TO_VK_EX)
    lp = 1 | (scan & 0xFF) << 16 | (1 << 24 if scan > 0xFF else 0) | (3 << 30 if up else 0)
    if not user32.PostMessageW(hwnd, WM_KEYUP if up else WM_KEYDOWN, vk, lp):
        raise SystemExit("could not post a key to the Strive window")


def keys(seq: str, gap_ms: int = 150) -> None:
    """Press keys in order. 'a+s:100' holds a and s together for 100 ms."""
    if not FOREGROUND:
        hwnd = find_window()
        if not hwnd:
            raise SystemExit("Strive window not found")
        for tok in seq.split():
            name, _, ms = tok.partition(":")
            codes = [SCAN[k] for k in name.split("+")]
            for c in codes:
                _post(hwnd, c, False)
            time.sleep(int(ms or 60) / 1000)
            for c in reversed(codes):
                _post(hwnd, c, True)
            time.sleep(gap_ms / 1000)
        return
    wait_user_idle()
    focus()
    hwnd = find_window()
    for tok in seq.split():
        if user32.GetForegroundWindow() != hwnd:
            raise SystemExit("the Strive window lost focus; stopped sending keys")
        name, _, ms = tok.partition(":")
        codes = [SCAN[k] for k in name.split("+")]
        for c in codes:
            _send(c, False)
        time.sleep(int(ms or 60) / 1000)
        for c in reversed(codes):
            _send(c, True)
        time.sleep(gap_ms / 1000)


# Where the game window goes (left, top of a borderless 1920x1080 window): this PC's
# monitor 3, bottom right. The screen checks need a 1920x1080 client area.
WINDOW_POS = tuple(int(v) for v in os.environ.get("GGST_WINDOW_POS", "5760,1080").split(","))


def place_window() -> None:
    """Make the game window borderless 1920x1080 at WINDOW_POS, without activating it."""
    hwnd = find_window()
    if not hwnd:
        raise SystemExit("Strive window not found")
    user32.SetProcessDPIAware()
    style = user32.GetWindowLongW(hwnd, -16)
    user32.SetWindowLongW(hwnd, -16, style & ~(0x00C00000 | 0x00040000))  # no caption / sizing frame
    # NOZORDER | NOACTIVATE | FRAMECHANGED
    user32.SetWindowPos(hwnd, 0, WINDOW_POS[0], WINDOW_POS[1], 1920, 1080, 0x0004 | 0x0010 | 0x0020)
    rc = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rc))
    if (rc.right, rc.bottom) != (1920, 1080):
        raise SystemExit(f"game area is {rc.right}x{rc.bottom}, not 1920x1080")


def shot(out: str) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-filter_complex",
                    "gfxcapture=window_exe=GGST-Win64-Shipping:capture_cursor=0,hwdownload,format=bgra",
                    "-frames:v", "1", out], check=True)


if __name__ == "__main__":
    if sys.argv[1] == "shot":
        shot(sys.argv[2])
    elif sys.argv[1] == "place":
        place_window()
    elif sys.argv[1] == "keys":
        keys(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 150)
