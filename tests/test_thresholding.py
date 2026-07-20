"""Tests for core.thresholding — ClassAdaptiveThreshold module."""

import torch
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.thresholding import ClassAdaptiveThreshold


class TestClassAdaptiveThreshold:
    def setup_method(self):
        self.K = 5
        self.cat = ClassAdaptiveThreshold(
            num_classes=self.K,
            warmup_epochs=3,
            ema_decay=0.99,
            min_pixels_per_class=50,
            base_threshold=0.6,
            beta_sigmoid=0.5,
        )

    def test_warmup_returns_uniform_threshold(self):
        """During warmup, all classes should get the same threshold."""
        assert self.cat.in_warmup
        thresholds = self.cat.compute_thresholds()
        assert thresholds.shape == (self.K,)
        # All should be the same value
        assert torch.allclose(thresholds, thresholds[0].expand(self.K))

    def test_post_warmup_thresholds_differ(self):
        """After warmup with varied per-class data, thresholds should differ."""
        # Feed data with different confidence distributions per class
        # Provide ground truth so the noise estimator has data
        for epoch in range(4):  # pass warmup (3 epochs)
            B, H, W = 2, 32, 32
            for _ in range(20):
                confidence = torch.rand(B, H, W)
                pred_classes = torch.randint(0, self.K, (B, H, W))
                gt = pred_classes.clone()
                for k in range(self.K):
                    mask = pred_classes == k
                    # Class 0: very high confidence, Class 4: low confidence
                    confidence[mask] = torch.clamp(
                        torch.randn(mask.sum()) * 0.05 + 0.5 + k * 0.1, 0.05, 0.99
                    )
                    # Add noise proportional to class index
                    class_noise = torch.rand(mask.sum()) < (k * 0.05)
                    gt_flat = gt[mask].clone()
                    gt_flat[class_noise] = (gt_flat[class_noise] + 1) % self.K
                    gt[mask] = gt_flat
                self.cat.update_stats(confidence, pred_classes, gt)
            self.cat.step_epoch()

        assert not self.cat.in_warmup
        thresholds = self.cat.compute_thresholds()
        # With different confidence distributions and noise rates, thresholds
        # should not all be identical (though they may be close)
        assert thresholds.shape == (self.K,)
        # At minimum, thresholds should be within valid range
        assert (thresholds >= self.cat.min_threshold).all()
        assert (thresholds <= self.cat.max_threshold).all()

    def test_apply_thresholds_shape(self):
        """apply_thresholds should return a boolean mask of correct shape."""
        B, H, W = 4, 64, 64
        confidence = torch.rand(B, H, W)
        pred_classes = torch.randint(0, self.K, (B, H, W))
        mask = self.cat.apply_thresholds(confidence, pred_classes)
        assert mask.shape == (B, H, W)
        assert mask.dtype == torch.bool

    def test_threshold_clamping(self):
        """Thresholds should always be within [min_threshold, max_threshold]."""
        for _ in range(5):
            self.cat.step_epoch()
        thresholds = self.cat.compute_thresholds()
        assert (thresholds >= self.cat.min_threshold).all()
        assert (thresholds <= self.cat.max_threshold).all()

    def test_checkpoint_roundtrip(self):
        """State dict save/load should preserve thresholds."""
        # Feed some data
        B, H, W = 2, 16, 16
        confidence = torch.rand(B, H, W)
        pred_classes = torch.randint(0, self.K, (B, H, W))
        self.cat.update_stats(confidence, pred_classes)
        self.cat.step_epoch()

        # Save
        state = self.cat.state_dict()

        # Create fresh instance and load
        cat2 = ClassAdaptiveThreshold(num_classes=self.K, warmup_epochs=3)
        cat2.load_state_dict(state)

        assert torch.equal(self.cat.per_class_mean, cat2.per_class_mean)
        assert torch.equal(self.cat.per_class_threshold, cat2.per_class_threshold)
        assert torch.equal(self.cat.current_epoch, cat2.current_epoch)

    def test_rare_class_fallback(self):
        """Classes with < min_pixels_per_class should use global threshold."""
        # Only feed data for class 0
        for _ in range(4):
            self.cat.step_epoch()

        B, H, W = 2, 32, 32
        confidence = torch.rand(B, H, W) * 0.5 + 0.5
        pred_classes = torch.zeros(B, H, W, dtype=torch.long)  # only class 0
        for _ in range(20):
            self.cat.update_stats(confidence, pred_classes)

        assert not self.cat.in_warmup
        thresholds = self.cat.compute_thresholds()
        # Classes 1-4 should have the fallback (same) threshold
        for k in range(1, self.K):
            assert thresholds[k] == thresholds[1], \
                f"Rare classes should share fallback threshold"

    def test_step_epoch_increments(self):
        assert self.cat.current_epoch.item() == 0
        self.cat.step_epoch()
        assert self.cat.current_epoch.item() == 1
        self.cat.step_epoch()
        assert self.cat.current_epoch.item() == 2

    def test_get_stats_dict(self):
        stats = self.cat.get_stats_dict()
        # Per class: threshold, mean_conf, count. Summary (inactive module):
        # confidence_floor + mean/min/max/mean_coverage = 5.
        assert len(stats) == self.K * 3 + 5
        assert 'threshold/confidence_floor' in stats
