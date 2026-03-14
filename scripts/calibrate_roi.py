"""Interactive ROI calibration tool.

Opens a sample frame and draws all configured ROIs. Shows each cropped ROI
in separate windows and displays the health bar binary mask so you can tune
coordinates and the background_threshold interactively.

Usage:
    python scripts/calibrate_roi.py [frame_path] [--config config/default.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replanal.config import PipelineConfig


def draw_rois(frame: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Return a copy of *frame* with ROI rectangles drawn."""
    overlay = frame.copy()
    colors = {
        "p1_health_bar": (0, 255, 0),
        "p2_health_bar": (0, 255, 0),
        "timer": (255, 255, 0),
        "p1_tension_bar": (255, 0, 0),
        "p2_tension_bar": (255, 0, 0),
    }
    for name, roi in config.rois.items():
        color = colors.get(name, (0, 200, 200))
        cv2.rectangle(overlay, (roi.x1, roi.y1), (roi.x2, roi.y2), color, 2)
        cv2.putText(overlay, name, (roi.x1, roi.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return overlay


def show_health_mask(bar_roi: np.ndarray, threshold: int, label: str) -> np.ndarray:
    """Show brightness mask and column fill ratios for a health bar ROI."""
    if bar_roi.size == 0:
        return np.zeros((50, 200, 3), dtype=np.uint8)

    hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    mask = (value > threshold).astype(np.uint8) * 255

    # Scale up for visibility
    h, w = mask.shape
    scale = max(1, 400 // w)
    mask_vis = cv2.resize(mask, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    # Compute fill ratio
    col_fill = np.mean(mask // 255, axis=0)
    filled = int(np.sum(col_fill > 0.5))
    pct = filled / w if w > 0 else 0
    cv2.putText(mask_bgr, f"{label}: {pct:.1%} ({filled}/{w} cols)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return mask_bgr


def main():
    parser = argparse.ArgumentParser(description="ROI calibration tool")
    parser.add_argument("frame", nargs="?", help="Path to a sample frame image")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(Path(args.config))

    # Find a sample frame
    frame_path = args.frame
    if frame_path is None:
        id_frames = sorted((ROOT / "data" / "id_frames").glob("*.png"))
        if not id_frames:
            print("No sample frames found. Pass a frame path as argument.")
            return
        frame_path = str(id_frames[0])

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Cannot read image: {frame_path}")
        return

    print(f"Frame: {frame_path} ({frame.shape[1]}x{frame.shape[0]})")
    print("Press 'q' to quit, '+'/'-' to adjust threshold")

    threshold = config.health_bar.background_threshold

    while True:
        overlay = draw_rois(frame, config)

        # Show health bar masks
        p1_roi = config.rois.get("p1_health_bar")
        p2_roi = config.rois.get("p2_health_bar")
        masks = []
        if p1_roi:
            masks.append(show_health_mask(p1_roi.crop(frame), threshold, "P1"))
        if p2_roi:
            masks.append(show_health_mask(p2_roi.crop(frame), threshold, "P2"))

        # Show ROI crops
        for name, roi in config.rois.items():
            cropped = roi.crop(frame)
            if cropped.size > 0:
                h, w = cropped.shape[:2]
                scale = max(1, 200 // max(w, 1))
                display = cv2.resize(cropped, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(f"ROI: {name}", display)

        cv2.putText(overlay, f"threshold={threshold}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Scale down main view for display
        h, w = overlay.shape[:2]
        display_w = min(w, 1280)
        scale = display_w / w
        overlay_small = cv2.resize(overlay, (display_w, int(h * scale)))
        cv2.imshow("ROI Calibration", overlay_small)

        if masks:
            combined = np.vstack(masks) if all(m.shape[1] == masks[0].shape[1] for m in masks) else masks[0]
            cv2.imshow("Health Masks", combined)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            threshold = min(255, threshold + 5)
            print(f"threshold = {threshold}")
        elif key == ord("-"):
            threshold = max(0, threshold - 5)
            print(f"threshold = {threshold}")

    cv2.destroyAllWindows()

    # Offer to save threshold
    if threshold != config.health_bar.background_threshold:
        print(f"\nThreshold changed: {config.health_bar.background_threshold} -> {threshold}")
        resp = input("Save to config? [y/N] ").strip().lower()
        if resp == "y":
            config_path = Path(args.config)
            with open(config_path) as f:
                raw = yaml.safe_load(f)
            raw.setdefault("health_bar", {})["background_threshold"] = threshold
            with open(config_path, "w") as f:
                yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)
            print(f"Saved to {config_path}")


if __name__ == "__main__":
    main()
