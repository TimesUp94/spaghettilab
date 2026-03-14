from __future__ import annotations

from abc import ABC, abstractmethod

from replanal.models import FrameContext, FrameData


class BaseExtractor(ABC):
    """All extractors implement this interface."""

    @abstractmethod
    def extract(self, ctx: FrameContext, data: FrameData) -> None:
        """Mutate *data* in-place with extracted information."""
        ...

    @property
    @abstractmethod
    def required_rois(self) -> list[str]:
        """ROI names this extractor needs pre-cropped in ctx.rois."""
        ...
