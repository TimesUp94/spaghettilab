from replanal.roi import ROIRect


class TestROIRect:
    def test_width_height(self):
        roi = ROIRect(x1=100, y1=50, x2=300, y2=80)
        assert roi.width == 200
        assert roi.height == 30

    def test_crop_dimensions(self, sample_ggs_frame, default_config):
        """All configured ROIs produce correctly sized crops."""
        for name, roi in default_config.rois.items():
            cropped = roi.crop(sample_ggs_frame)
            assert cropped.shape[1] == roi.width, f"{name} width mismatch"
            assert cropped.shape[0] == roi.height, f"{name} height mismatch"
            assert cropped.shape[2] == 3, f"{name} not BGR"

    def test_crop_within_bounds(self, sample_ggs_frame, default_config):
        """ROI coordinates stay within frame dimensions."""
        h, w = sample_ggs_frame.shape[:2]
        for name, roi in default_config.rois.items():
            assert roi.x1 >= 0, f"{name} x1 < 0"
            assert roi.y1 >= 0, f"{name} y1 < 0"
            assert roi.x2 <= w, f"{name} x2 > frame width"
            assert roi.y2 <= h, f"{name} y2 > frame height"
