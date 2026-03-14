from replanal.extractors.health import HealthBarExtractor
from replanal.models import FrameContext, FrameData


class TestHealthBarExtractor:
    def test_reads_health_from_ggs_frame(self, sample_ggs_frame, default_config):
        """Goldlewis vs Potemkin at 10s — both should have significant health."""
        extractor = HealthBarExtractor(default_config.health_bar)
        rois = {}
        for name in extractor.required_rois:
            roi_def = default_config.rois[name]
            rois[name] = roi_def.crop(sample_ggs_frame)

        ctx = FrameContext(
            video_path="test",
            frame_number=300,
            timestamp_ms=10000,
            frame_bgr=sample_ggs_frame,
            rois=rois,
        )
        data = FrameData(frame_number=300, timestamp_ms=10000)
        extractor.extract(ctx, data)

        assert data.p1_health is not None
        assert data.p2_health is not None
        # Both should be between 0 and 1
        assert 0.0 <= data.p1_health.health_pct <= 1.0
        assert 0.0 <= data.p2_health.health_pct <= 1.0
        # At 10s into a match, both players should still have health
        assert data.p1_health.health_pct > 0.1
        assert data.p2_health.health_pct > 0.1

    def test_reads_health_different_characters(self, sample_ggs_frame_2, default_config):
        """Anji vs Baiken — different character colors should still work."""
        extractor = HealthBarExtractor(default_config.health_bar)
        rois = {}
        for name in extractor.required_rois:
            roi_def = default_config.rois[name]
            rois[name] = roi_def.crop(sample_ggs_frame_2)

        ctx = FrameContext(
            video_path="test",
            frame_number=300,
            timestamp_ms=10000,
            frame_bgr=sample_ggs_frame_2,
            rois=rois,
        )
        data = FrameData(frame_number=300, timestamp_ms=10000)
        extractor.extract(ctx, data)

        assert data.p1_health is not None
        assert data.p2_health is not None
        assert 0.0 <= data.p1_health.health_pct <= 1.0
        assert 0.0 <= data.p2_health.health_pct <= 1.0

    def test_roi_crop_dimensions(self, sample_ggs_frame, default_config):
        """Verify ROI crops produce expected dimensions."""
        for name in ("p1_health_bar", "p2_health_bar"):
            roi = default_config.rois[name]
            cropped = roi.crop(sample_ggs_frame)
            assert cropped.shape[1] == roi.width
            assert cropped.shape[0] == roi.height
            assert cropped.shape[2] == 3  # BGR
