from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from replanal.roi import ROIRect


@dataclass
class HealthBarConfig:
    background_threshold: int = 50
    min_delta_pct: float = 0.005
    smoothing_window: int = 3


@dataclass
class PipelineConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    sample_every_n_frames: int = 1
    rois: dict[str, ROIRect] = field(default_factory=dict)
    health_bar: HealthBarConfig = field(default_factory=HealthBarConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)

        resolution = raw.get("resolution", {})
        rois = {}
        for name, coords in raw.get("rois", {}).items():
            rois[name] = ROIRect(
                x1=coords["x1"],
                y1=coords["y1"],
                x2=coords["x2"],
                y2=coords["y2"],
            )

        hb_raw = raw.get("health_bar", {})
        health_bar = HealthBarConfig(
            background_threshold=hb_raw.get("background_threshold", 50),
            min_delta_pct=hb_raw.get("min_delta_pct", 0.005),
            smoothing_window=hb_raw.get("smoothing_window", 3),
        )

        return cls(
            width=resolution.get("width", 1920),
            height=resolution.get("height", 1080),
            fps=raw.get("fps", 30),
            sample_every_n_frames=raw.get("sample_every_n_frames", 1),
            rois=rois,
            health_bar=health_bar,
        )
