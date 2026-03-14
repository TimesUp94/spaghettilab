"""Scene / gameplay detector.

Detects whether a frame is active gameplay vs. non-gameplay (lobby, loading,
transitions, stream overlay scenes). Sets ``data.is_gameplay`` accordingly.

Detection signals:
1. **Tension bar check**: During gameplay, at least one tension bar shows
   colored (saturated) pixels.
2. **Health bar check**: During gameplay, the health bar regions contain
   bright pixels (the bars themselves or the bar background).
3. **Timer circle check**: The ornate timer circle has distinctive bright
   elements during gameplay.
"""
from __future__ import annotations

import cv2
import numpy as np

from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData


class SceneDetector(BaseExtractor):
    """Marks frames as gameplay or non-gameplay."""

    # Tension bar ROIs
    P1_TENSION = (1040, 1058, 50, 440)   # y1, y2, x1, x2
    P2_TENSION = (1040, 1058, 1480, 1870)
    # Health bar ROIs (just check a small region for brightness)
    P1_HEALTH = (55, 72, 200, 800)
    P2_HEALTH = (55, 72, 1120, 1720)
    # Timer ornate frame (the gold ring, not the digits)
    TIMER_RING = (90, 100, 930, 990)  # bottom of the ornate ring

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
        # Signal 1: Tension bar saturation (high during gameplay)
        t1 = frame[self.P1_TENSION[0]:self.P1_TENSION[1],
                    self.P1_TENSION[2]:self.P1_TENSION[3]]
        t2 = frame[self.P2_TENSION[0]:self.P2_TENSION[1],
                    self.P2_TENSION[2]:self.P2_TENSION[3]]
        hsv1 = cv2.cvtColor(t1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(t2, cv2.COLOR_BGR2HSV)
        t1_score = float(np.mean((hsv1[:, :, 1] > 40) & (hsv1[:, :, 2] > 30)))
        t2_score = float(np.mean((hsv2[:, :, 1] > 40) & (hsv2[:, :, 2] > 30)))
        tension_ok = t1_score > 0.15 or t2_score > 0.15

        # Signal 2: Health bar region has bright pixels
        h1 = frame[self.P1_HEALTH[0]:self.P1_HEALTH[1],
                    self.P1_HEALTH[2]:self.P1_HEALTH[3]]
        h2 = frame[self.P2_HEALTH[0]:self.P2_HEALTH[1],
                    self.P2_HEALTH[2]:self.P2_HEALTH[3]]
        h1_bright = float(np.mean(cv2.cvtColor(h1, cv2.COLOR_BGR2GRAY) > 60))
        h2_bright = float(np.mean(cv2.cvtColor(h2, cv2.COLOR_BGR2GRAY) > 60))
        health_ok = h1_bright > 0.10 or h2_bright > 0.10

        # Gameplay requires tension bars visible OR health bars visible
        # (tension alone is sufficient; health alone needs higher confidence)
        if tension_ok:
            return True
        if health_ok and (h1_bright > 0.30 or h2_bright > 0.30):
            return True

        return False
