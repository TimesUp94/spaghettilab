"""The chain's frame readers must return the frame the pipeline read, at any source rate and through dropped frames. Short synthetic
clips whose frames carry their own number, with rates far enough from 60 that a few seconds show the drift a long 59.94 video would."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ggst_logger" / "model"))
vc = pytest.importorskip("video_crops")
br = pytest.importorskip("banner_reader")
vs = pytest.importorskip("video_scene")
W, H = 64, 36


def _marked(n: int) -> np.ndarray:
    """A gray frame carrying n: one bright or dark block per bit, two rows of five."""
    im = np.zeros((H, W), np.uint8)
    for b in range(10):
        if n >> b & 1:
            y0, y1 = (0, H // 2) if b < 5 else (H // 2, H)
            im[y0:y1, (b % 5) * W // 5:((b % 5) + 1) * W // 5] = 235
    return im


def _read(img: np.ndarray) -> int:
    g = img[..., 0]
    h, w = g.shape
    n = 0
    for b in range(10):
        y = h // 4 if b < 5 else 3 * h // 4
        if g[y, (b % 5) * w // 5 + w // 10] > 118:
            n |= 1 << b
    return n


def _clip(path: Path, frames: int, rate: str, hole=None) -> None:
    """frames frames at rate; hole (frame, frames lost): the timestamps jump there, as in a stream hiccup."""
    vf = "format=yuv420p"
    if hole:
        at, lost = hole
        vf = f"setpts='(N+gt(N\\,{at})*{lost})/({rate})/TB'," + vf
    cmd = [vc.FFMPEG, "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", rate, "-i", "-", "-vf", vf,
           "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "10", "-g", "30", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(frames):
        p.stdin.write(_marked(n).tobytes())
    p.stdin.close()
    assert p.wait() == 0


CASES = [("60", None), ("59.94", None), ("59.6", None), ("30", None), ("29.97", None), ("29.5", None), ("60", (300, 150))]


@pytest.mark.parametrize("rate,hole", CASES)
def test_every_reader_returns_the_pipelines_frame(tmp_path, rate, hole):
    if not (Path(vc.FFMPEG).exists() or shutil.which(vc.FFMPEG)):
        pytest.skip("no ffmpeg")
    video = tmp_path / "clip.mp4"
    frames = 1000
    _clip(video, frames, rate, hole)
    fps = vc.video_format(video)[0]
    # what the chain does: pipeline frame n -> 60 fps index f, the readers asked for f
    ns = [5, 40, 299, 301, 450, frames - 40]
    f60 = [int(round(n * 60.0 / fps)) for n in ns]
    for name, got in (
        ("crops", dict(vc.read_frames(video, f60))),
        ("banner", dict(br.frames_at(video, f60))),
        ("scene", vs.frames_at(video, f60)),
    ):
        assert sorted(got) == f60, name
        assert [_read(got[f]) for f in f60] == ns, name


def test_repeated_source_frames_keep_the_count(tmp_path):
    if not (Path(vc.FFMPEG).exists() or shutil.which(vc.FFMPEG)):
        pytest.skip("no ffmpeg")
    video = tmp_path / "clip.mp4"
    _clip(video, 100, "30")
    wanted = list(range(20, 41))                 # the crop window: 21 frames at 60 fps
    for reader in (vc.read_frames, br.frames_at):
        got = list(reader(video, wanted))
        assert [f for f, _ in got] == wanted
        assert [_read(im) for _, im in got] == [f // 2 for f in wanted]


@pytest.mark.parametrize("fps", [60.0, 59.94, 59.824, 30.0, 29.97])
def test_source_frame_inverts_the_chain(fps):
    ns = range(0, 200000, 137)
    off = [abs(vc.source_frame(int(round(n * 60.0 / fps)), fps) - n) for n in ns]
    assert max(off) <= 1 and sum(off) <= len(off) * 0.001         # only an exact .5 lands on the frame before
