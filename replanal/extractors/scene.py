"""Scene / gameplay detector.

Detects whether a frame is active gameplay vs. non-gameplay (lobby, loading,
transitions, stream overlay scenes). Sets ``data.is_gameplay`` accordingly.

Detection signals:
1. **Health bar check**: During gameplay, the health bar region contains
   warm-hued (red/pink) pixels from the health bar fill. Checks a wide
   y-range (50-140) to handle different tournament overlay layouts.
2. **Timer region check**: The timer circle area has distinctive bright
   elements during gameplay.
3. **Small game detection**: Tournament recordings sometimes start with the
   game rendered in a smaller window. We detect this by checking if the
   expected HUD regions are missing.
"""
from __future__ import annotations

import cv2
import numpy as np

from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData


class SceneDetector(BaseExtractor):
    """Marks frames as gameplay or non-gameplay."""

    # Wide y-range to catch health bars across different overlay layouts:
    #   - Some overlays place bars at y≈65-100
    #   - Others at y≈118-138
    BAR_Y1, BAR_Y2 = 50, 140
    # Health bar x-ranges for checking
    P1_HEALTH_X = (200, 800)
    P2_HEALTH_X = (1120, 1720)
    # Timer region (the ornate circle with digits)
    TIMER_Y1, TIMER_Y2 = 100, 180
    TIMER_X1, TIMER_X2 = 880, 1040

    @property
    def required_rois(self) -> list[str]:
        return []

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        frame = ctx.frame_bgr
        h, w = frame.shape[:2]
        if h < 1080 or w < 1920:
            data.is_gameplay = False
            return

        data.is_gameplay = self._is_gameplay(frame)

    def _is_gameplay(self, frame: np.ndarray) -> bool:
        # Signal 1: Health bar warm pixel check (P1 side — always warm/red)
        bar_strip = frame[self.BAR_Y1 : self.BAR_Y2]
        hsv = cv2.cvtColor(bar_strip, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        warm = ((h_ch < 30) | (h_ch > 150)) & (s_ch > 30) & (v_ch > 80)

        p1_warm = float(
            np.mean(warm[:, self.P1_HEALTH_X[0] : self.P1_HEALTH_X[1]])
        )

        # P1 bar is always warm — sufficient to detect gameplay
        health_ok = p1_warm > 0.10

        # Signal 2: Timer region brightness
        timer = frame[self.TIMER_Y1 : self.TIMER_Y2, self.TIMER_X1 : self.TIMER_X2]
        timer_gray = cv2.cvtColor(timer, cv2.COLOR_BGR2GRAY)
        timer_bright = float(np.mean(timer_gray > 100))

        # Timer area should have some bright pixels (the timer digits/frame)
        timer_ok = timer_bright > 0.05

        # Require health bars visible; timer is a secondary signal
        if health_ok:
            return True
        if timer_ok and p1_warm > 0.03:
            return True

        return False
