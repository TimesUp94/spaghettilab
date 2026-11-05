"""The frame selection expression: ffmpeg refuses one nested over 100 deep, so a long set's spans must stay a shallow tree."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ggst_logger" / "model"))
vc = pytest.importorskip("video_crops")


def _depth(e: str) -> int:
    d = m = 0
    for c in e:
        d += (c == "(") - (c == ")")
        m = max(m, d)
    return m


def test_shape():
    assert vc.select_expr([(1, 2)]) == "between(n\\,1\\,2)"
    assert vc.select_expr([(1, 2), (5, 6)]) == "(between(n\\,1\\,2)+between(n\\,5\\,6))"
    e = vc.select_expr([(i, i) for i in range(1000)])
    assert e.count("between") == 1000
    assert _depth(e) <= 12


@pytest.mark.parametrize("n", [3, 101, 300])
def test_ffmpeg_takes_it(n):
    ff = vc.FFMPEG if Path(vc.FFMPEG).exists() else shutil.which(vc.FFMPEG)
    if not ff:
        pytest.skip("no ffmpeg")
    e = vc.select_expr([(i * 30, i * 30 + 20) for i in range(n)])
    r = subprocess.run([ff, "-hide_banner", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s=16x16:r=60:d={n // 2 + 1}", "-vf", f"select='{e}'",
                        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    assert r.returncode == 0, r.stderr[-300:]
    assert len(r.stdout) == 16 * 16 * 21 * n
