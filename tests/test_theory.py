"""Tests for core.theory — Hoeffding bounds, Beta fitting, risk minimization."""

import math
import numpy as np
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.theory import PerClassNoiseEstimator, BetaDistributionFitter, TheoreticalRiskMinimizer


class TestPerClassNoiseEstimator:
    def test_perfect_predictions_have_zero_noise(self):
        """If predictions always match GT, empirical noise should be ~0."""
        est = PerClassNoiseEstimator(num_classes=5, delta=0.05)
        N = 5000
        confidences = torch.rand(N) * 0.5 + 0.5  # [0.5, 1.0]
        classes = torch.randint(0, 5, (N,))
        gt = classes.clone()  # perfect predictions
        est.update_from_labeled(confidences, classes, gt)

        for k in range(5):
            eps_hat, n_k = est.compute_empirical_noise_rate(k, 0.5)
            assert eps_hat == 0.0, f"Class {k}: expected 0 noise, got {eps_hat}"

    def test_hoeffding_bound_holds(self):
        """Hoeffding upper bound should be >= empirical noise rate."""
        est = PerClassNoiseEstimator(num_classes=3, delta=0.05)
        N = 10000
        confidences = torch.rand(N)
        classes = torch.randint(0, 3, (N,))
        # Introduce 20% noise
        gt = classes.clone()
        noise_mask = torch.rand(N) < 0.2
        gt[noise_mask] = (gt[noise_mask] + 1) % 3

        est.update_from_labeled(confidences, classes, gt)

        for k in range(3):
            eps_hat, n_k = est.compute_empirical_noise_rate(k, 0.3)
            eps_upper = est.compute_hoeffding_bound(k, 0.3)
            assert eps_upper >= eps_hat, (
                f"Class {k}: bound {eps_upper:.4f} < empirical {eps_hat:.4f}")

    def test_empty_class_returns_max_noise(self):
        """A class with no samples should return noise rate 1.0."""
        est = PerClassNoiseEstimator(num_classes=5)
        eps_hat, n_k = est.compute_empirical_noise_rate(4, 0.5)
        assert eps_hat == 1.0
        assert n_k == 0


class TestBetaDistributionFitter:
    def test_fits_high_confidence_class(self):
        """A class with high confidence should get threshold < 0.95."""
        fitter = BetaDistributionFitter(num_classes=3, ema_decay=0.0)  # no smoothing
        # Class 0: high confidence ~ Beta(10, 2) ≈ mean 0.83
        confs = torch.from_numpy(
            np.random.beta(10, 2, size=5000).astype(np.float32))
        classes = torch.zeros(5000, dtype=torch.long)
        fitter.update(confs, classes)

        tau = fitter.compute_threshold(0, epsilon_target=0.1)
        assert tau is not None
        assert 0.5 < tau < 0.99, f"Expected threshold in (0.5, 0.99), got {tau}"

    def test_fits_low_confidence_class(self):
        """A class with low confidence should get a lower threshold."""
        fitter = BetaDistributionFitter(num_classes=3, ema_decay=0.0)
        # Class 1: low confidence ~ Beta(2, 5) ≈ mean 0.29
        confs = torch.from_numpy(
            np.random.beta(2, 5, size=5000).astype(np.float32))
        classes = torch.ones(5000, dtype=torch.long)
        fitter.update(confs, classes)

        tau = fitter.compute_threshold(1, epsilon_target=0.1)
        assert tau is not None
        # Threshold for low-confidence class should be lower
        assert tau < 0.7, f"Expected threshold < 0.7, got {tau}"

    def test_insufficient_samples_returns_none(self):
        fitter = BetaDistributionFitter(num_classes=3, min_samples=100)
        confs = torch.rand(10)
        classes = torch.zeros(10, dtype=torch.long)
        fitter.update(confs, classes)
        assert fitter.compute_threshold(0, 0.1) is None

    def test_get_all_thresholds(self):
        fitter = BetaDistributionFitter(num_classes=3, ema_decay=0.0, min_samples=10)
        for k in range(3):
            confs = torch.rand(200) * 0.5 + k * 0.15
            classes = torch.full((200,), k, dtype=torch.long)
            fitter.update(confs, classes)
        thresholds = fitter.get_all_thresholds(0.1)
        assert thresholds.shape == (3,)
        assert not np.any(np.isnan(thresholds))


class TestTheoreticalRiskMinimizer:
    def test_optimal_threshold_in_valid_range(self):
        est = PerClassNoiseEstimator(num_classes=3)
        fitter = BetaDistributionFitter(num_classes=3, ema_decay=0.0, min_samples=10)

        # Simulate data
        N = 5000
        confs = torch.rand(N)
        classes = torch.randint(0, 3, (N,))
        gt = classes.clone()
        noise_mask = torch.rand(N) < 0.15
        gt[noise_mask] = (gt[noise_mask] + 1) % 3

        est.update_from_labeled(confs, classes, gt)
        fitter.update(confs, classes)

        minimizer = TheoreticalRiskMinimizer(est, fitter, lambda_coverage=0.1)
        for k in range(3):
            tau = minimizer.compute_optimal_threshold(k)
            assert 0.0 < tau < 1.0, f"Class {k}: threshold {tau} out of range"

    def test_compute_all_optimal_thresholds(self):
        est = PerClassNoiseEstimator(num_classes=5)
        fitter = BetaDistributionFitter(num_classes=5, ema_decay=0.0, min_samples=10)
        N = 2000
        confs = torch.rand(N)
        classes = torch.randint(0, 5, (N,))
        gt = classes.clone()
        est.update_from_labeled(confs, classes, gt)
        fitter.update(confs, classes)

        minimizer = TheoreticalRiskMinimizer(est, fitter)
        thresholds = minimizer.compute_all_optimal_thresholds()
        assert thresholds.shape == (5,)
        assert np.all(thresholds > 0) and np.all(thresholds < 1)

    def test_bound_tightness_dict(self):
        est = PerClassNoiseEstimator(num_classes=2)
        fitter = BetaDistributionFitter(num_classes=2, ema_decay=0.0, min_samples=10)
        confs = torch.rand(1000)
        classes = torch.randint(0, 2, (1000,))
        gt = classes.clone()
        est.update_from_labeled(confs, classes, gt)
        fitter.update(confs, classes)

        minimizer = TheoreticalRiskMinimizer(est, fitter)
        info = minimizer.compute_bound_tightness(0)
        assert 'empirical' in info
        assert 'bound' in info
        assert 'gap' in info
        assert info['gap'] >= 0
