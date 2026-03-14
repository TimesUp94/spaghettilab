from __future__ import annotations

from replanal.config import PipelineConfig
from replanal.extractors.base import BaseExtractor
from replanal.models import FrameContext, FrameData
from replanal.video import iter_frames


class ReplayPipeline:
    """Orchestrates frame iteration, ROI cropping, and extractor execution."""

    def __init__(self, config: PipelineConfig, extractors: list[BaseExtractor]):
        self.config = config
        self.extractors = extractors
        self.required_rois: set[str] = set()
        for ext in extractors:
            self.required_rois.update(ext.required_rois)

    def process_video(self, video_path: str) -> list[FrameData]:
        results: list[FrameData] = []
        for frame_number, timestamp_ms, frame_bgr in iter_frames(
            video_path,
            fps=self.config.fps,
            sample_every=self.config.sample_every_n_frames,
        ):
            # Pre-crop all needed ROIs once per frame
            rois: dict[str, object] = {}
            for roi_name in self.required_rois:
                roi_def = self.config.rois.get(roi_name)
                if roi_def is not None:
                    rois[roi_name] = roi_def.crop(frame_bgr)

            ctx = FrameContext(
                video_path=video_path,
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                frame_bgr=frame_bgr,
                rois=rois,
            )
            data = FrameData(frame_number=frame_number, timestamp_ms=timestamp_ms)

            for extractor in self.extractors:
                extractor.extract(ctx, data)
                # Skip remaining extractors if scene detector marks non-gameplay
                if not data.is_gameplay:
                    break

            results.append(data)

        return results
