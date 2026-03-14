"""Round counter (heart) detector.

Counts rounds won by each player by examining the heart indicators above
each player's health bar. In GGS Strive:

- **Red heart** = active (round not yet lost by this player)
- **White/gray heart** = lost (opponent won that round)

So ``rounds_won_by_player = number_of_white_hearts_on_opponent_side``.

Detection: checks saturation at known heart center positions.
High saturation → red (active). Low saturation + bright → white (lost).
"""
from __future__ import annotations

import cv2
import numpy as np

from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData

# Heart center positions in full 1920x1080 frame coordinates.
# Each player has two hearts (best-of-3 rounds format).
# "inner" = closer to center, "outer" = closer to edge.
P1_HEART_POSITIONS = [(843, 83), (799, 88)]  # inner, outer
P2_HEART_POSITIONS = [(1077, 83), (1121, 88)]

# Size of the square ROI to sample around each heart center
SAMPLE_HALF = 10  # 20x20 pixel ROI


class RoundCounterExtractor(BaseExtractor):
    """Counts rounds won by each player from heart indicators."""

    @property
    def required_rois(self) -> list[str]:
        # We read directly from the full frame at known positions
        return []

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        frame = ctx.frame_bgr
        h, w = frame.shape[:2]
        if h < 1080 or w < 1920:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, s_ch, v_ch = cv2.split(hsv)

        p1_lost = self._count_lost_hearts(s_ch, v_ch, P1_HEART_POSITIONS)
        p2_lost = self._count_lost_hearts(s_ch, v_ch, P2_HEART_POSITIONS)

        if p1_lost is not None and p2_lost is not None:
            # Rounds won by a player = rounds lost by their opponent
            data.p1_rounds_won = p2_lost
            data.p2_rounds_won = p1_lost

    def _count_lost_hearts(
        self, s_ch: np.ndarray, v_ch: np.ndarray, positions: list[tuple[int, int]]
    ) -> int | None:
        """Count how many hearts at the given positions are white (lost).

        Returns None if the hearts aren't visible (non-gameplay frame).
        """
        lost = 0
        visible = 0

        for cx, cy in positions:
            y1 = max(0, cy - SAMPLE_HALF)
            y2 = cy + SAMPLE_HALF
            x1 = max(0, cx - SAMPLE_HALF)
            x2 = cx + SAMPLE_HALF

            roi_s = s_ch[y1:y2, x1:x2]
            roi_v = v_ch[y1:y2, x1:x2]
            mean_s = float(roi_s.mean())
            mean_v = float(roi_v.mean())

            if mean_v < 40:
                # Too dark — heart not visible (transition/loading screen)
                continue

            visible += 1

            if mean_s < 60 and mean_v > 100:
                # White/gray heart = round lost
                lost += 1
            # else: red heart (active) or uncertain — don't count as lost

        if visible == 0:
            return None
        return lost
