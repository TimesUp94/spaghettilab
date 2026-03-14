"""Timer digit reader.

Reads the two-digit round timer from the GGS HUD using Otsu thresholding
and template matching against pre-extracted digit templates.

The timer digits are gold/cream colored on a dark circular background.
Otsu's method cleanly separates the digits from the background.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData

# Standard size for digit templates
TEMPLATE_H = 50
TEMPLATE_W = 30

# Default path to digit templates
_DEFAULT_TEMPLATES = Path(__file__).resolve().parents[2] / "data" / "templates" / "timer_digits.npz"


class TimerExtractor(BaseExtractor):
    """Extracts the round timer value (0-99) from the GGS HUD."""

    def __init__(self, templates_path: Path | None = None):
        path = templates_path or _DEFAULT_TEMPLATES
        self._templates: dict[int, np.ndarray] = {}
        if path.exists():
            data = np.load(str(path))
            for key in data.files:
                self._templates[int(key)] = data[key]

    @property
    def required_rois(self) -> list[str]:
        return ["timer_digits"]

    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        roi = ctx.rois.get("timer_digits")
        if roi is None or roi.size == 0 or not self._templates:
            return

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Check if the region looks like a valid timer (dark background present)
        bg_dark_ratio = float(np.mean(gray < 50))
        if bg_dark_ratio < 0.25:
            # Too bright — likely a transition screen, super, or effect
            return

        # Otsu threshold to segment digits
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fill = float(mask.mean()) / 255.0
        if fill < 0.15 or fill > 0.45:
            # Fill outside expected range for two digits
            return

        h, w = mask.shape

        # Split into left (tens) and right (ones) halves
        left_mask = mask[:, : w // 2]
        right_mask = mask[:, w // 2 :]

        tens = self._recognize_digit(left_mask)
        ones = self._recognize_digit(right_mask)

        if tens is not None and ones is not None:
            data.timer_value = tens * 10 + ones

    def _recognize_digit(self, digit_mask: np.ndarray) -> int | None:
        """Match a digit mask against templates, return best digit or None."""
        # Find bounding box of the digit
        cols = np.where(digit_mask.any(axis=0))[0]
        rows = np.where(digit_mask.any(axis=1))[0]
        if len(cols) < 3 or len(rows) < 3:
            return None

        cropped = digit_mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if cropped.shape[0] < 5 or cropped.shape[1] < 5:
            return None

        resized = cv2.resize(cropped, (TEMPLATE_W, TEMPLATE_H), interpolation=cv2.INTER_AREA)
        _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        best_digit = None
        best_score = -1.0

        for digit, template in self._templates.items():
            # Normalized cross-correlation
            score = float(np.sum(resized == template)) / (TEMPLATE_H * TEMPLATE_W)
            if score > best_score:
                best_score = score
                best_digit = digit

        # Require reasonable confidence
        if best_score < 0.70:
            return None

        return best_digit
