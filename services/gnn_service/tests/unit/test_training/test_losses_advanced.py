"""Unit tests for advanced loss functions.

Tests:
- AsymmetricL1Loss
- QuantileRULLoss
- PhysicsAwareFocalLoss
- DomainAdversarialLoss
- ConfidenceWeightedLoss
"""

import pytest
import torch
from torch import nn

from src.training.losses_advanced import (
    AsymmetricL1Loss,
    QuantileRULLoss,
    PhysicsAwareFocalLoss,
    DomainAdversarialLoss,
    ConfidenceWeightedLoss,
)


class TestAsymmetricL1Loss:
    """Tests for AsymmetricL1Loss."""

    def test_underestimation_penalty(self):
        """Test underestimation is penalized more."""
        loss_fn = AsymmetricL1Loss(tau=0.7)

        # Underestimation: predict 100, true 50 (predict too late)
        pred_under = torch.tensor([[100.0]])
        true_under = torch.tensor([[50.0]])
        loss_under = loss_fn(pred_under, true_under)

        # Overestimation: predict 50, true 100 (predict too early)
        pred_over = torch.tensor([[50.0]])
        true_over = torch.tensor([[100.0]])
        loss_over = loss_fn(pred_over, true_over)

        # Underestimation should have higher loss
        assert loss_under > loss_over
        assert loss_under / loss_over > 2.0  # ~2.33x with tau=0.7

    def test_tau_effect(self):
        """Test tau parameter effect."""
        pred = torch.tensor([[100.0]])
        true = torch.tensor([[50.0]])

        loss_tau_05 = AsymmetricL1Loss(tau=0.5)(pred, true)
        loss_tau_07 = AsymmetricL1Loss(tau=0.7)(pred, true)
        loss_tau_09 = AsymmetricL1Loss(tau=0.9)(pred, true)

        # Higher tau = more penalty for underestimation
        assert loss_tau_09 > loss_tau_07 > loss_tau_05

    def test_invalid_tau(self):
        """Test invalid tau raises error."""
        with pytest.raises(ValueError):
            AsymmetricL1Loss(tau=0.0)
        with pytest.raises(ValueError):
            AsymmetricL1Loss(tau=1.0)
        with pytest.raises(ValueError):
            AsymmetricL1Loss(tau=1.5)


class TestQuantileRULLoss:
    """Tests for QuantileRULLoss."""

    def test_pinball_loss(self):
        """Test pinball loss computation."""
        loss_fn = QuantileRULLoss(quantiles=[0.1, 0.5, 0.9])

        pred = torch.tensor([[40.0, 50.0, 60.0]])  # 3 quantiles
        true = torch.tensor([[50.0]])  # True RUL

        loss = loss_fn(pred, true)
        assert loss.item() >= 0

    def test_asymmetric_penalty(self):
        """Test asymmetric penalty for different quantiles."""
        loss_fn_low = QuantileRULLoss(quantiles=[0.1])
        loss_fn_high = QuantileRULLoss(quantiles=[0.9])

        pred = torch.tensor([[40.0]])  # Underestimate
        true = torch.tensor([[50.0]])

        loss_low = loss_fn_low(pred, true)
        loss_high = loss_fn_high(pred, true)

        # q=0.9 should penalize underestimation more
        assert loss_high > loss_low

    def test_invalid_quantiles(self):
        """Test invalid quantiles raise error."""
        with pytest.raises(ValueError):
            QuantileRULLoss(quantiles=[0.0, 0.5])
        with pytest.raises(ValueError):
            QuantileRULLoss(quantiles=[0.5, 1.0])
        with pytest.raises(ValueError):
            QuantileRULLoss(quantiles=[-0.1, 0.5])


class TestPhysicsAwareFocalLoss:
    """Tests for PhysicsAwareFocalLoss."""

    def test_component_weighting(self):
        """Test component importance weighting."""
        component_weights = torch.tensor([1.0, 0.5, 0.1])

        loss_fn = PhysicsAwareFocalLoss(
            alpha=0.25,
            gamma=2.0,
            component_weights=component_weights,
        )

        logits = torch.randn(3, 9)  # 3 components, 9 anomalies
        targets = torch.randint(0, 2, (3, 9)).float()

        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_severity_weighting(self):
        """Test anomaly severity weighting."""
        severity_weights = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

        loss_fn = PhysicsAwareFocalLoss(
            alpha=0.25,
            gamma=2.0,
            severity_weights=severity_weights,
        )

        logits = torch.randn(5, 9)
        targets = torch.randint(0, 2, (5, 9)).float()

        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_combined_weighting(self):
        """Test combined component + severity weighting."""
        component_weights = torch.tensor([1.0, 0.8])
        severity_weights = torch.tensor([1.0] * 9)

        loss_fn = PhysicsAwareFocalLoss(
            component_weights=component_weights,
            severity_weights=severity_weights,
        )

        logits = torch.randn(2, 9)
        targets = torch.ones(2, 9)

        loss = loss_fn(logits, targets)
        assert loss.item() >= 0


class TestDomainAdversarialLoss:
    """Tests for DomainAdversarialLoss."""

    def test_domain_classification(self):
        """Test domain adversarial loss."""
        loss_fn = DomainAdversarialLoss(lambda_domain=0.5, num_domains=2)

        domain_pred = torch.randn(32, 2)  # 32 samples, 2 domains
        domain_labels = torch.randint(0, 2, (32,))

        loss = loss_fn(domain_pred, domain_labels)
        assert loss.item() >= 0

    def test_lambda_domain_scaling(self):
        """Test lambda_domain scales loss."""
        domain_pred = torch.randn(16, 2)
        domain_labels = torch.zeros(16, dtype=torch.long)

        loss_lambda_03 = DomainAdversarialLoss(lambda_domain=0.3)(domain_pred, domain_labels)
        loss_lambda_07 = DomainAdversarialLoss(lambda_domain=0.7)(domain_pred, domain_labels)

        # Higher lambda = higher loss
        assert loss_lambda_07 > loss_lambda_03


class TestConfidenceWeightedLoss:
    """Tests for ConfidenceWeightedLoss."""

    def test_confidence_weighting(self):
        """Test loss is weighted by confidence."""
        base_loss = nn.MSELoss(reduction="none")
        conf_loss = ConfidenceWeightedLoss(base_loss, min_confidence=0.1)

        pred = torch.randn(10, 1)
        target = torch.randn(10, 1)
        confidence = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

        loss = conf_loss(pred, target, confidence.unsqueeze(-1))
        assert loss.item() >= 0

    def test_low_confidence_reduction(self):
        """Test low confidence reduces loss contribution."""
        base_loss = nn.MSELoss(reduction="none")
        conf_loss = ConfidenceWeightedLoss(base_loss)

        pred = torch.tensor([[1.0], [1.0]])
        target = torch.tensor([[0.0], [0.0]])  # Same error

        # High confidence
        conf_high = torch.tensor([[1.0], [1.0]])
        loss_high = conf_loss(pred, target, conf_high)

        # Low confidence
        conf_low = torch.tensor([[0.2], [0.2]])
        loss_low = conf_loss(pred, target, conf_low)

        # Low confidence should give lower loss
        assert loss_low < loss_high

    def test_min_confidence_clamp(self):
        """Test minimum confidence clamping."""
        base_loss = nn.MSELoss(reduction="none")
        conf_loss = ConfidenceWeightedLoss(base_loss, min_confidence=0.3)

        pred = torch.tensor([[1.0]])
        target = torch.tensor([[0.0]])

        # Confidence below min
        conf_very_low = torch.tensor([[0.01]])
        loss = conf_loss(pred, target, conf_very_low)

        # Should be clamped to min_confidence=0.3
        assert loss.item() > 0
