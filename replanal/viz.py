"""Debug overlays and health timeline charts."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from replanal.config import PipelineConfig
from replanal.models import DamageEvent, FrameData, Side


def draw_roi_overlay(frame: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Return a copy of *frame* with ROI rectangles drawn for debugging."""
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
        cv2.putText(
            overlay, name, (roi.x1, roi.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )
    return overlay


def save_health_timeline(
    frames: list[FrameData],
    events: list[DamageEvent],
    replay_id: str,
    output_path: Path,
) -> Path:
    """Generate and save a health-over-time chart. Returns the saved path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps = [f.timestamp_ms / 1000 for f in frames]
    p1_health = [f.p1_health.health_pct if f.p1_health else float('nan') for f in frames]
    p2_health = [f.p2_health.health_pct if f.p2_health else float('nan') for f in frames]

    # Rolling median smooth (31-frame window, matching Rust round detection)
    def _rolling_median(data: list[float], window: int) -> list[float]:
        n = len(data)
        out = [float('nan')] * n
        half = window // 2
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            vals = [v for v in data[start:end] if not np.isnan(v)]
            if vals:
                out[i] = float(np.median(vals))
        return out

    p1_smooth = _rolling_median(p1_health, 31)
    p2_smooth = _rolling_median(p2_health, 31)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(timestamps, p1_smooth, label="P1 Health", color="blue", linewidth=0.8)
    ax.plot(timestamps, p2_smooth, label="P2 Health", color="red", linewidth=0.8)

    for e in events:
        color = "blue" if e.target_side == Side.P1 else "red"
        ax.axvline(e.timestamp_ms / 1000, color=color, alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Health %")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Health Timeline — {replay_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    chart_path = output_path / f"{replay_id}_health.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path
