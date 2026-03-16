"""Tension gauge extractor.

Reads P1 and P2 tension percentage from the GGS HUD. The tension bars
are green-hued (H 35-85) and fill from the center of the screen outward:
  - P1: fills right-to-left (x=750 toward x=260)
  - P2: fills left-to-right (x=1170 toward x=1660)

Tension starts at 0% each round and can freely increase or decrease.
A rolling median filter smooths screen effects.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData


class TensionExtractor(BaseExtractor):
    """Extracts tension gauge percentage for P1 and P2."""

    # Tension bar regions (confirmed from pixel inspection)
    P1_X1, P1_X2 = 260, 750
    P2_X1, P2_X2 = 1170, 1660
    BAND_Y1, BAND_Y2 = 1000, 1040

    # Green hue detection for tension bars
    HUE_LO, HUE_HI = 35, 85
    SAT_MIN = 40
    VAL_MIN = 60

    # Column fill threshold
    _COL_FILL_THRESHOLD = 0.2

    # Rolling median window
    _MEDIAN_WINDOW = 15

    # Non-gameplay gap that triggers reset
    _RESET_GAP = 15

    def __init__(self) -> None:
        self._last_frame: int = -9999
        self._p1_max_w: int = 0
        self._p2_max_w: int = 0
        self._p1_buf: deque[int] = deque(maxlen=self._MEDIAN_WINDOW)
        self._p2_buf: deque[int] = deque(maxlen=self._MEDIAN_WINDOW)

    @property
    def required_rois(self) -> list[str]:
        return []

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        frame = ctx.frame_bgr

        # Reset after non-gameplay gap
        if ctx.frame_number - self._last_frame > self._RESET_GAP:
            self._p1_max_w = 0
            self._p2_max_w = 0
            self._p1_buf.clear()
            self._p2_buf.clear()

        strip = frame[self.BAND_Y1:self.BAND_Y2]
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

        p1_w = self._measure_fill(hsv, self.P1_X1, self.P1_X2)
        p2_w = self._measure_fill(hsv, self.P2_X1, self.P2_X2)

        # Need at least some signal to consider this a valid frame
        if p1_w == 0 and p2_w == 0:
            return

        self._last_frame = ctx.frame_number

        # Update max widths
        if p1_w > self._p1_max_w:
            self._p1_max_w = p1_w
        if p2_w > self._p2_max_w:
            self._p2_max_w = p2_w

        # Rolling median
        self._p1_buf.append(p1_w)
        self._p2_buf.append(p2_w)
        p1_smooth = int(np.median(list(self._p1_buf)))
        p2_smooth = int(np.median(list(self._p2_buf)))

        # Compute percentages
        data.p1_tension_pct = p1_smooth / self._p1_max_w if self._p1_max_w > 0 else 0.0
        data.p2_tension_pct = p2_smooth / self._p2_max_w if self._p2_max_w > 0 else 0.0

    def _measure_fill(self, hsv: np.ndarray, x1: int, x2: int) -> int:
        """Measure tension bar fill width using green pixel detection."""
        h_ch = hsv[:, x1:x2, 0]
        s_ch = hsv[:, x1:x2, 1]
        v_ch = hsv[:, x1:x2, 2]

        green = (
            (h_ch >= self.HUE_LO) & (h_ch <= self.HUE_HI)
            & (s_ch > self.SAT_MIN) & (v_ch > self.VAL_MIN)
        )
        col_ratio = np.mean(green, axis=0)
        is_filled = col_ratio > self._COL_FILL_THRESHOLD

        return self._longest_run(is_filled)

    @staticmethod
    def _longest_run(is_filled: np.ndarray) -> int:
        """Find longest continuous run of True values."""
        best = 0
        current = 0
        for val in is_filled:
            if val:
                current += 1
                if current > best:
                    best = current
            else:
                current = 0
        return best

    def reset_calibration(self) -> None:
        """Reset all state (call between replays)."""
        self._last_frame = -9999
        self._p1_max_w = 0
        self._p2_max_w = 0
        self._p1_buf.clear()
        self._p2_buf.clear()
