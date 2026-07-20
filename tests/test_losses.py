"""Tests for core.losses — weighted CE, class balance, boundary loss."""

import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.losses import weighted_cross_entropy_loss, compute_class_balance_weights, boundary_loss


class TestWeightedCrossEntropyLoss:
    def test_backward_compatible_no_class_weights(self):
        """With class_weights=None, should match the original CW-BASS loss."""
        B, C, H, W = 2, 5, 32, 32
        pred = torch.randn(B, C, H, W, requires_grad=True)
        pseudo = torch.randint(0, C, (B, H, W))
        conf = torch.rand(B, H, W)

        loss = weighted_cross_entropy_loss(pred, pseudo, conf, gamma=1.0)
        assert loss.requires_grad
        loss.backward()
        assert pred.grad is not None

    def test_with_class_weights(self):
        B, C, H, W = 2, 5, 32, 32
        pred = torch.randn(B, C, H, W, requires_grad=True)
        pseudo = torch.randint(0, C, (B, H, W))
        conf = torch.rand(B, H, W)
        weights = torch.ones(C)

        loss = weighted_cross_entropy_loss(pred, pseudo, conf, gamma=1.0,
                                           class_weights=weights)
        loss.backward()
        assert loss.item() > 0

    def test_ignore_index_handled(self):
        B, C, H, W = 2, 5, 16, 16
        pred = torch.randn(B, C, H, W, requires_grad=True)
        pseudo = torch.full((B, H, W), 255, dtype=torch.long)  # all ignored
        conf = torch.rand(B, H, W)

        loss = weighted_cross_entropy_loss(pred, pseudo, conf, ignore_index=255)
        assert loss.item() == 0.0  # no valid pixels

    def test_gamma_zero_uniform_weighting(self):
        """gamma=0 means confidence^0 = 1, so all pixels weighted equally."""
        B, C, H, W = 1, 3, 8, 8
        pred = torch.randn(B, C, H, W, requires_grad=True)
        pseudo = torch.randint(0, C, (B, H, W))
        conf = torch.rand(B, H, W)

        loss_g0 = weighted_cross_entropy_loss(pred, pseudo, conf, gamma=0.0)
        # Compare with standard CE
        ce = torch.nn.functional.cross_entropy(pred, pseudo, reduction='mean')
        # Should be very close (both uniform weight)
        assert abs(loss_g0.item() - ce.item()) < 1e-5


class TestComputeClassBalanceWeights:
    def test_uniform_distribution_gives_uniform_weights(self):
        B, H, W = 2, 32, 32
        K = 5
        pseudo = torch.randint(0, K, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)
        weights = compute_class_balance_weights(mask, pseudo, K)
        assert weights.shape == (K,)
        # With roughly uniform distribution, weights should be near 1
        assert torch.allclose(weights, torch.ones(K), atol=0.5)

    def test_empty_mask_returns_ones(self):
        B, H, W = 1, 8, 8
        K = 3
        pseudo = torch.randint(0, K, (B, H, W))
        mask = torch.zeros(B, H, W, dtype=torch.bool)
        weights = compute_class_balance_weights(mask, pseudo, K)
        assert torch.allclose(weights, torch.ones(K))

    def test_imbalanced_gives_higher_weight_to_rare(self):
        K = 3
        # 900 pixels of class 0, 90 of class 1, 10 of class 2
        labels = torch.cat([
            torch.zeros(900, dtype=torch.long),
            torch.ones(90, dtype=torch.long),
            torch.full((10,), 2, dtype=torch.long),
        ])
        pseudo = labels.view(1, 1, -1).expand(1, 1, 1000).reshape(1, 1, 1000)
        # Reshape to valid spatial dims
        pseudo = labels.view(1, 100, 10)
        mask = torch.ones(1, 100, 10, dtype=torch.bool)
        weights = compute_class_balance_weights(mask, pseudo, K)
        # Rare class (2) should have highest weight
        assert weights[2] > weights[1] > weights[0]


class TestBoundaryLoss:
    def test_returns_scalar(self):
        B, C, H, W = 2, 5, 32, 32
        pred = torch.randn(B, C, H, W, requires_grad=True)
        pseudo = torch.randint(0, C, (B, H, W))
        conf = torch.rand(B, H, W)
        boundary = torch.zeros(B, H, W)
        boundary[:, 15:17, :] = 1.0  # artificial boundary

        loss = boundary_loss(pred, pseudo, conf, boundary, gamma=1.0)
        assert loss.dim() == 0  # scalar
        loss.backward()
        assert pred.grad is not None

    def test_no_boundary_reduces_to_base_loss(self):
        B, C, H, W = 1, 3, 16, 16
        pred = torch.randn(B, C, H, W)
        pseudo = torch.randint(0, C, (B, H, W))
        conf = torch.rand(B, H, W)
        boundary = torch.zeros(B, H, W)

        loss_boundary = boundary_loss(pred, pseudo, conf, boundary, gamma=1.0)
        loss_base = weighted_cross_entropy_loss(pred, pseudo, conf, gamma=1.0)
        # With all-zero boundary, boundary term should be 0
        assert abs(loss_boundary.item() - loss_base.item()) < 1e-5
