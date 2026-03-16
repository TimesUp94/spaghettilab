"""Quick diagnostic: test health bar extraction on a video."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replanal.extractors.health import HealthBarExtractor
from replanal.extractors.scene import SceneDetector
from replanal.models import FrameContext, FrameData

VIDEO = str(ROOT / "data" / "sets" / "04_UF_GuessingGame_vs_Ricbionicle.mp4")


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fps:.0f}fps, {total} frames")

    scene = SceneDetector()
    ext = HealthBarExtractor()

    frame_idx = 0
    prev_p1 = prev_p2 = None
    p1_jumps = p2_jumps = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts_ms = frame_idx * 1000.0 / fps
        ctx = FrameContext(
            video_path=VIDEO,
            frame_number=frame_idx,
            timestamp_ms=ts_ms,
            frame_bgr=frame,
            rois={},
        )
        data = FrameData(frame_number=frame_idx, timestamp_ms=ts_ms)

        scene.extract(ctx, data)
        if not data.is_gameplay:
            frame_idx += 1
            continue

        ext.extract(ctx, data)

        p1 = data.p1_health.health_pct if data.p1_health else None
        p2 = data.p2_health.health_pct if data.p2_health else None

        # Track big jumps (>30% between consecutive gameplay frames)
        if p1 is not None and prev_p1 is not None:
            if abs(p1 - prev_p1) > 0.30:
                p1_jumps += 1
        if p2 is not None and prev_p2 is not None:
            if abs(p2 - prev_p2) > 0.30:
                p2_jumps += 1
        if p1 is not None:
            prev_p1 = p1
        if p2 is not None:
            prev_p2 = p2

        # Print HP at key timestamps
        if (44 <= ts_ms/1000 <= 170 and frame_idx % 150 == 0) or frame_idx % 450 == 0:
            p1w = data.p1_health.bar_pixels_filled if data.p1_health else 0
            p2w = data.p2_health.bar_pixels_filled if data.p2_health else 0
            if p1 is not None:
                print(f"  t={ts_ms/1000:6.1f}s f={frame_idx:4d}: "
                      f"P1={p1:.3f} ({p1w:3d}px)  P2={p2:.3f} ({p2w:3d}px)  "
                      f"band={ext._active}")
            else:
                print(f"  t={ts_ms/1000:6.1f}s f={frame_idx:4d}: no HP")

        frame_idx += 1

    cap.release()
    print(f"\nMax widths: low={ext._max_w['low']}, high={ext._max_w['high']}")
    print(f"Big jumps (>30%): P1={p1_jumps}, P2={p2_jumps}")


if __name__ == "__main__":
    main()
