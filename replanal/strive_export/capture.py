"""FFmpeg discovery, window capture and recording of the Strive window.

Two capture sources:

* ``gfxcapture`` (FFmpeg 8+, Windows Graphics Capture): grabs the game
  window itself, even while it is covered by other windows. Preferred.
* ``ddagrab`` (FFmpeg 6+, Desktop Duplication): grabs the screen area of the
  game's client rectangle, so the window must stay visible and on top.
  Used only when no FFmpeg with gfxcapture is found.

The encoder is picked by test-encoding a few frames: h264_nvenc, then
h264_mf (Media Foundation, any GPU vendor), then libx264.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import winio

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _run(args, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("timeout", 60)
    if CREATE_NO_WINDOW:
        kw.setdefault("creationflags", CREATE_NO_WINDOW)
    return subprocess.run(args, **kw)


@dataclass
class FfmpegInfo:
    path: str
    version: str = ""
    filters: set = field(default_factory=set)
    encoders: set = field(default_factory=set)

    @property
    def gfxcapture(self) -> bool:
        return "gfxcapture" in self.filters

    @property
    def ddagrab(self) -> bool:
        return "ddagrab" in self.filters


def ffmpeg_candidates(explicit: str | None = None) -> list[str]:
    """FFmpeg executables to consider, best guess first: --ffmpeg,
    GGST_FFMPEG, SPAGHETTI_FFMPEG (what the app resolved), PATH, an FFmpeg
    unpacked under $GGST_DATA/ggst_mod/dl or %LOCALAPPDATA%/SpaghettiLab/ffmpeg,
    then imageio-ffmpeg's (FFmpeg 7, no gfxcapture)."""
    out: list[str] = []

    def add(p):
        if p and os.path.isfile(p) and os.path.abspath(p) not in [os.path.abspath(o) for o in out]:
            out.append(p)

    add(explicit)
    add(os.environ.get("GGST_FFMPEG"))
    add(os.environ.get("SPAGHETTI_FFMPEG"))
    add(shutil.which("ffmpeg"))
    roots = [os.path.join(os.environ.get("GGST_DATA", r"G:\Projects\replanal-data"), "ggst_mod", "dl")]
    if os.environ.get("LOCALAPPDATA"):
        roots.append(os.path.join(os.environ["LOCALAPPDATA"], "SpaghettiLab", "ffmpeg"))
    for root in roots:
        for pat in ("ffmpeg*/bin/ffmpeg.exe", "ffmpeg*/ffmpeg.exe", "bin/ffmpeg.exe", "ffmpeg.exe"):
            for p in sorted(glob.glob(os.path.join(root, pat)), reverse=True):
                add(p)
    try:
        import imageio_ffmpeg
        add(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    return out


def probe(path: str, run=_run) -> FfmpegInfo:
    info = FfmpegInfo(path)
    try:
        v = run([path, "-hide_banner", "-version"])
        first = (v.stdout or b"").decode(errors="replace").splitlines()
        info.version = first[0] if first else ""
        f = run([path, "-hide_banner", "-filters"])
        for line in (f.stdout or b"").decode(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                info.filters.add(parts[1])
        e = run([path, "-hide_banner", "-encoders"])
        for line in (e.stdout or b"").decode(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 6:
                info.encoders.add(parts[1])
    except (OSError, subprocess.SubprocessError):
        pass
    return info


@dataclass
class CaptureChoice:
    ffmpeg: FfmpegInfo
    method: str            # "gfxcapture" | "ddagrab"
    warning: str | None = None


def choose_ffmpeg(candidates: list[str], probe_fn=probe) -> CaptureChoice:
    infos = [probe_fn(p) for p in candidates]
    for info in infos:
        if info.gfxcapture:
            return CaptureChoice(info, "gfxcapture")
    for info in infos:
        if info.ddagrab:
            return CaptureChoice(info, "ddagrab", warning=(
                "No FFmpeg with window capture (gfxcapture, FFmpeg 8 or newer) was found; "
                f"falling back to screen capture with {info.path}. Keep the Strive window "
                "visible and on top while exporting: anything covering it is recorded too. "
                "For window capture, install FFmpeg 8+ and set SPAGHETTI_FFMPEG or GGST_FFMPEG "
                "to its ffmpeg.exe."))
    if not infos:
        raise RuntimeError("ffmpeg not found. Install FFmpeg 8 or newer (it has window capture) "
                           "and set GGST_FFMPEG to its ffmpeg.exe, or pass --ffmpeg.")
    raise RuntimeError("None of the FFmpeg builds found can capture the screen (gfxcapture or "
                       "ddagrab): " + ", ".join(i.path for i in infos) +
                       ". Install FFmpeg 8 or newer and pass --ffmpeg.")


# Encoder candidates, best first: (name, output args).
ENCODERS = [
    ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "21", "-b:v", "20M"]),
    ("h264_mf", ["-c:v", "h264_mf", "-hw_encoding", "1", "-b:v", "25M"]),
    ("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]),
]


def pick_encoder(info: FfmpegInfo, run=_run, prefer: str | None = None) -> tuple[str, list[str]]:
    """The first encoder that encodes a few 1080p frames without error."""
    order = ENCODERS
    if prefer:
        order = [e for e in ENCODERS if e[0] == prefer] + [e for e in ENCODERS if e[0] != prefer]
    tried = []
    for name, args in order:
        if info.encoders and name not in info.encoders:
            continue
        cmd = [info.path, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
               "-i", "testsrc2=size=1920x1080:rate=60", "-frames:v", "10",
               "-pix_fmt", "nv12", *args, "-f", "null", "-"]
        try:
            r = run(cmd, timeout=60)
            ok = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        if ok:
            return name, args
        tried.append(name)
    raise RuntimeError(f"No H.264 encoder works with {info.path} (tried {', '.join(tried) or 'none'}).")


def source_filter(method: str, fps: int, rect=None, monitor=None, still: bool = False) -> str:
    """The FFmpeg filter graph that produces 1920x1080 NV12 frames of the game
    (``still``: BGRA frames at the window's own size, for a screenshot)."""
    if method == "gfxcapture":
        src = f"gfxcapture=window_exe={winio.GAME_EXE}:capture_cursor=0:max_framerate={fps}"
    else:
        idx, ox, oy = monitor
        _, _, w, h = rect
        src = (f"ddagrab=output_idx={idx}:framerate={fps}:draw_mouse=0:"
               f"offset_x={ox}:offset_y={oy}:video_size={w}x{h}")
    if still:
        return f"{src},hwdownload,format=bgra"
    # crop: the centred 16:9 picture of a window of another shape (letterboxed)
    return (f"{src},hwdownload,format=bgra,crop=w='min(iw,ih*16/9)':h='min(ih,iw*9/16)',"
            f"scale=1920:1080,fps={fps},format=nv12")


class Capture:
    """Screenshots and recordings of the Strive window through FFmpeg."""

    def __init__(self, choice: CaptureChoice, encoder: tuple[str, list[str]] | None = None, fps: int = 60):
        self.choice = choice
        self.ffmpeg = choice.ffmpeg.path
        self.method = choice.method
        self.encoder = encoder
        self.fps = fps

    def _dda_geometry(self):
        hwnd = winio.find_window()
        rect = winio.client_rect(hwnd)
        return rect, winio.window_on_monitor(rect, winio.monitors())

    def _graph(self, fps: int, still: bool = False) -> str:
        rect = mon = None
        if self.method == "ddagrab":
            rect, mon = self._dda_geometry()
        return source_filter(self.method, fps, rect, mon, still=still)

    def grab(self) -> np.ndarray | None:
        """One frame of the game window as BGR, or None (captures sometimes
        come back empty; the caller retries)."""
        try:
            graph = self._graph(10, still=True)
            r = _run([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-filter_complex", graph,
                      "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], timeout=20)
        except (OSError, subprocess.SubprocessError, RuntimeError):
            return None
        if r.returncode != 0 or not r.stdout:
            return None
        img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 100:
            return None
        return img

    def start_recording(self, path: str, max_seconds: int = 900) -> "Recording":
        if self.encoder is None:
            raise RuntimeError("no encoder selected")
        graph = self._graph(self.fps)
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-filter_complex", graph,
               "-t", str(max_seconds), *self.encoder[1], "-movflags", "+faststart", path]
        return Recording(cmd, path)


class Recording:
    """A running FFmpeg recording; :meth:`stop` finalises the mp4 by sending
    FFmpeg its 'q' command (a killed FFmpeg leaves an unplayable file)."""

    def __init__(self, cmd: list[str], path: str):
        self.cmd = cmd
        self.path = path
        self.started = time.time()
        self._result = None
        kw = {"creationflags": CREATE_NO_WINDOW} if CREATE_NO_WINDOW else {}
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, **kw)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout: float = 60) -> tuple[int, str]:
        """(exit code, FFmpeg's error output). Safe to call more than once."""
        if self._result is not None:
            return self._result
        if self.alive():
            try:
                self.proc.stdin.write(b"q")
                self.proc.stdin.flush()
            except OSError:
                pass
        try:
            _, err = self.proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            _, err = self.proc.communicate()
        self._result = (self.proc.returncode, (err or b"").decode(errors="replace").strip())
        return self._result

    @property
    def duration(self) -> float:
        return time.time() - self.started


_VERSION_RE = re.compile(r"ffmpeg version (?:n)?(\d+)\.(\d+)")


def major_version(info: FfmpegInfo) -> int | None:
    m = _VERSION_RE.search(info.version)
    return int(m.group(1)) if m else None
