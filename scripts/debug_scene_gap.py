"""Check scene detector gameplay status around the set transition (t=30-50s)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replanal.extractors.scene import SceneDetector
from replanal.models import FrameContext, FrameData

VIDEO = str(ROOT / "data" / "sets" / "04_UF_GuessingGame_vs_Ricbionicle.mp4")


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)

    scene = SceneDetector()
    prev_gp = None
    run_start = 0

    for frame_idx in range(int(55 * fps)):  # Check first 55 seconds
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

        if data.is_gameplay != prev_gp:
            if prev_gp is not None:
                dur = frame_idx - run_start
                print(f"  frames {run_start:4d}-{frame_idx-1:4d} "
                      f"(t={run_start/fps:.1f}-{(frame_idx-1)/fps:.1f}s): "
                      f"{'GAMEPLAY' if prev_gp else 'non-gameplay'} "
                      f"({dur} frames, {dur/fps:.1f}s)")
            prev_gp = data.is_gameplay
            run_start = frame_idx

    # Print last run
    dur = frame_idx - run_start + 1
    print(f"  frames {run_start:4d}-{frame_idx:4d} "
          f"(t={run_start/fps:.1f}-{frame_idx/fps:.1f}s): "
          f"{'GAMEPLAY' if prev_gp else 'non-gameplay'} "
          f"({dur} frames, {dur/fps:.1f}s)")

    cap.release()


if __name__ == "__main__":
    main()
