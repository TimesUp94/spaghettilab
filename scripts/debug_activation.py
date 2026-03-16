"""Debug band activation around t=90-110s where band switches from low to high."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replanal.extractors.health import HealthBarExtractor
from replanal.extractors.scene import SceneDetector
from replanal.models import FrameContext, FrameData

VIDEO = str(ROOT / "data" / "sets" / "04_UF_GuessingGame_vs_Ricbionicle.mp4")


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)

    scene = SceneDetector()
    ext = HealthBarExtractor()

    # Process all frames up to t=110s, only printing detail around t=88-108s
    target_end = int(110 * fps)
    detail_start = int(88 * fps)

    frame_idx = 0
    while frame_idx <= target_end:
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

        if frame_idx >= detail_start:
            # Print scene detection status
            if not data.is_gameplay:
                print(f"  f={frame_idx:4d} t={ts_ms/1000:6.1f}s: NON-GAMEPLAY")
                frame_idx += 1
                continue

            # Manually measure both bands BEFORE calling extract
            low_y1, low_y2 = ext.BAND_LOW
            high_y1, high_y2 = ext.BAND_HIGH

            for band_name, y1, y2 in [("low", low_y1, low_y2), ("high", high_y1, high_y2)]:
                strip = frame[y1:y2]
                hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
                p1_warm = ext._measure_warm_fill(hsv, ext.P1_X1, ext.P1_X2)
                p2_warm = ext._measure_warm_fill(hsv, ext.P2_X1, ext.P2_X2)
                p2_cool = ext._measure_cool_fill(hsv, ext.P2_X1, ext.P2_X2)
                p2_best = max(p2_warm, p2_cool)
                print(f"  f={frame_idx:4d} t={ts_ms/1000:6.1f}s: "
                      f"GAMEPLAY  band={band_name}  "
                      f"P1w={p1_warm:3d}  P2w={p2_warm:3d}  P2c={p2_cool:3d}  P2best={p2_best:3d}")

            # Now call extract
            prev_active = ext._active
            ext.extract(ctx, data)
            new_active = ext._active

            p1_pct = data.p1_health.health_pct if data.p1_health else None
            p2_pct = data.p2_health.health_pct if data.p2_health else None
            status = ""
            if prev_active != new_active:
                status = f"  *** BAND CHANGED: {prev_active} -> {new_active} ***"
            if data.p1_health:
                print(f"           => active={ext._active}  "
                      f"P1={p1_pct:.3f}  P2={p2_pct:.3f}  "
                      f"cand={ext._activate_candidate}/{ext._activate_count}{status}")
            else:
                print(f"           => active={ext._active}  no HP  "
                      f"cand={ext._activate_candidate}/{ext._activate_count}{status}")
        else:
            if not data.is_gameplay:
                frame_idx += 1
                continue
            ext.extract(ctx, data)

        frame_idx += 1

    cap.release()
    print(f"\nFinal max_w: low={ext._max_w['low']}, high={ext._max_w['high']}")


if __name__ == "__main__":
    main()
