from __future__ import annotations

import cv2
import numpy as np

from replanal.config import HealthBarConfig
from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData, HealthReading, Side


class HealthBarExtractor(BaseExtractor):
    """Extracts health percentage from P1 and P2 health bars.

    Uses brightness-based column scanning: the empty portion of a GGS health
    bar is consistently dark regardless of character, so we threshold on the
    V channel in HSV and count filled columns.
    """

    def __init__(self, config: HealthBarConfig | None = None):
        self.config = config or HealthBarConfig()

    @property
    def required_rois(self) -> list[str]:
        return ["p1_health_bar", "p2_health_bar"]

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        data.p1_health = self._read_bar(ctx.rois["p1_health_bar"], Side.P1)
        data.p2_health = self._read_bar(ctx.rois["p2_health_bar"], Side.P2)

    def _read_bar(self, bar_roi: np.ndarray, side: Side) -> HealthReading:
        if bar_roi.size == 0:
            return HealthReading(side=side, health_pct=0.0, bar_pixels_filled=0, bar_pixels_total=0)

        hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]

        # A column is "filled" if >50% of its vertical pixels are brighter
        # than the background threshold.
        filled_mask = value_channel > self.config.background_threshold
        col_fill_ratio = np.mean(filled_mask, axis=0)
        col_filled = col_fill_ratio > 0.5

        total_cols = len(col_filled)
        filled_cols = int(np.sum(col_filled))
        health_pct = filled_cols / total_cols if total_cols > 0 else 0.0

        return HealthReading(
            side=side,
            health_pct=health_pct,
            bar_pixels_filled=filled_cols,
            bar_pixels_total=total_cols,
        )
