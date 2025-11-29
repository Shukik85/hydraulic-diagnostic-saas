"""Unit tests for training loss functions.

Tests for:
- FocalLoss
- WingLoss
- QuantileRULLoss
- UncertaintyWeighting
- MultiTaskLoss
"""

import pytest
import torch
import torch.nn as nn

from src.training.losses import (
    FocalLoss,
    WingLoss,
    QuantileRULLoss,
    UncertaintyWeighting,
    MultiTaskLoss,
)


class TestFocalLoss:
    """Tests for FocalLoss."""
    
    def test_focal_loss_basic(self):
        """Test basic focal loss computation."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        logits = torch.randn(32, 9)
        targets = torch.randint(0, 2, (32, 9)).float()
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_focal_loss_gamma_effect(self):
        """Test gamma parameter effect."""
        logits = torch.randn(16, 9)
        targets = torch.randint(0, 2, (16, 9)).float()
        
        # Higher gamma should give different loss
        loss_gamma_0 = FocalLoss(gamma=0.0)(logits, targets)
        loss_gamma_2 = FocalLoss(gamma=2.0)(logits, targets)
        
        # Losses should be different
        assert not torch.allclose(loss_gamma_0, loss_gamma_2)
    
    def test_focal_loss_alpha_weighting(self):
        """Test alpha weighting."""
        logits = torch.randn(16, 9)
        targets = torch.ones(16, 9)  # All positive
        
        loss_alpha_025 = FocalLoss(alpha=0.25)(logits, targets)
        loss_alpha_075 = FocalLoss(alpha=0.75)(logits, targets)
        
        # Different alpha should give different loss
        assert not torch.allclose(loss_alpha_025, loss_alpha_075)
    
    def test_focal_loss_reduction(self):
        """Test reduction modes."""
        logits = torch.randn(8, 9)
        targets = torch.randint(0, 2, (8, 9)).float()
        
        loss_mean = FocalLoss(reduction="mean")(logits, targets)
        loss_sum = FocalLoss(reduction="sum")(logits, targets)
        loss_none = FocalLoss(reduction="none")(logits, targets)
        
        assert loss_mean.shape == torch.Size([])
        assert loss_sum.shape == torch.Size([])
        assert loss_none.shape == torch.Size([8, 9])
        
        # Sum should be larger than mean
        assert loss_sum > loss_mean


class TestWingLoss:
    """Tests for WingLoss."""
    
    def test_wing_loss_basic(self):
        """Test basic wing loss computation."""
        loss_fn = WingLoss(omega=10.0, epsilon=2.0)
        
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_wing_loss_omega_threshold(self):
        """Test omega threshold behavior."""
        loss_fn = WingLoss(omega=1.0, epsilon=0.5)
        
        # Small error (below omega)
        pred_small = torch.tensor([[0.5]])
        target_small = torch.tensor([[0.6]])
        loss_small = loss_fn(pred_small, target_small)
        
        # Large error (above omega)
        pred_large = torch.tensor([[0.0]])
        target_large = torch.tensor([[5.0]])
        loss_large = loss_fn(pred_large, target_large)
        
        # Large error should have larger loss
        assert loss_large > loss_small
    
    def test_wing_loss_vs_mse(self):
        """Test Wing loss is more robust than MSE."""
        pred = torch.tensor([[1.0], [2.0], [100.0]])  # Outlier
        target = torch.tensor([[1.1], [2.1], [2.1]])  # Outlier target
        
        wing_loss = WingLoss()(pred, target)
        mse_loss = nn.MSELoss()(pred, target)
        
        # Wing should be less affected by outlier
        assert wing_loss < mse_loss
    
    def test_wing_loss_reduction(self):
        """Test reduction modes."""
        pred = torch.randn(8, 1)
        target = torch.randn(8, 1)
        
        loss_mean = WingLoss(reduction="mean")(pred, target)
        loss_sum = WingLoss(reduction="sum")(pred, target)
        loss_none = WingLoss(reduction="none")(pred, target)
        
        assert loss_mean.shape == torch.Size([])
        assert loss_sum.shape == torch.Size([])
        assert loss_none.shape == torch.Size([8, 1])


class TestQuantileRULLoss:
    """Tests for QuantileRULLoss."""
    
    def test_quantile_rul_loss_basic(self):
        """Test basic quantile RUL loss computation."""
        loss_fn = QuantileRULLoss(quantiles=[0.1, 0.5, 0.9])
        
        pred = torch.randn(32, 1).abs() * 100  # Positive RUL
        target = torch.randn(32, 1).abs() * 100
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_quantile_rul_asymmetric_penalty(self):
        """Test asymmetric penalties (underestimation > overestimation)."""
        loss_fn = QuantileRULLoss(quantiles=[0.9])  # Heavy penalty for underestimation
        
        # Underestimation: predict 50, true 100 (predict too early)
        pred_under = torch.tensor([[50.0]])
        target_under = torch.tensor([[100.0]])
        loss_under = loss_fn(pred_under, target_under)
        
        # Overestimation: predict 100, true 50 (predict too late)
        pred_over = torch.tensor([[100.0]])
        target_over = torch.tensor([[50.0]])
        loss_over = loss_fn(pred_over, target_over)
        
        # Underestimation should be penalized more
        # With q=0.9: under penalty = 0.9 * 50 = 45
        #              over penalty = 0.1 * 50 = 5
        assert loss_under > loss_over
        assert loss_under / loss_over > 5.0  # ~9x penalty
    
    def test_quantile_rul_log_normalization(self):
        """Test with log-normalized targets."""
        loss_fn = QuantileRULLoss()
        
        pred = torch.randn(16, 1).abs() * 100
        target = torch.randn(16, 1).abs() * 100
        
        # Log-normalize
        pred_log = torch.log1p(pred)
        target_log = torch.log1p(target)
        
        loss = loss_fn(pred_log, target_log)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_quantile_rul_multiple_quantiles(self):
        """Test with multiple quantiles."""
        loss_fn = QuantileRULLoss(quantiles=[0.1, 0.5, 0.9])
        
        pred = torch.randn(8, 1).abs() * 50
        target = torch.randn(8, 1).abs() * 50
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_quantile_rul_invalid_quantile(self):
        """Test that invalid quantiles raise error."""
        with pytest.raises(ValueError):
            QuantileRULLoss(quantiles=[0.0, 0.5, 1.0])  # 0 and 1 invalid
        
        with pytest.raises(ValueError):
            QuantileRULLoss(quantiles=[-0.1, 0.5])  # Negative invalid


class TestUncertaintyWeighting:
    """Tests for UncertaintyWeighting."""
    
    def test_uncertainty_weighting_init(self):
        """Test initialization."""
        weighter = UncertaintyWeighting(num_tasks=3)
        
        assert weighter.log_vars.shape == torch.Size([3])
        assert torch.allclose(weighter.log_vars, torch.zeros(3))
    
    def test_uncertainty_weighting_forward(self):
        """Test forward pass."""
        weighter = UncertaintyWeighting(num_tasks=3)
        
        losses = {
            "health": torch.tensor(0.5),
            "degradation": torch.tensor(0.3),
            "anomaly": torch.tensor(0.8)
        }
        
        total_loss = weighter(losses)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.shape == torch.Size([])
        assert total_loss.item() >= 0
    
    def test_uncertainty_weighting_learning(self):
        """Test that weights are learnable."""
        weighter = UncertaintyWeighting(num_tasks=3)
        
        # Check parameters are registered
        params = list(weighter.parameters())
        assert len(params) == 1
        assert params[0].shape == torch.Size([3])
        assert params[0].requires_grad


class TestMultiTaskLoss:
    """Tests for MultiTaskLoss."""
    
    def test_multitask_loss_fixed_weighting(self):
        """Test fixed weighting strategy."""
        multi_loss = MultiTaskLoss(
            weighting="fixed",
            loss_weights={"health": 1.0, "degradation": 0.5, "anomaly": 2.0}
        )
        
        health_pred = torch.randn(16, 1)
        degradation_pred = torch.randn(16, 1)
        anomaly_logits = torch.randn(16, 9)
        
        health_true = torch.randn(16, 1)
        degradation_true = torch.randn(16, 1)
        anomaly_true = torch.randint(0, 2, (16, 9)).float()
        
        total_loss, loss_dict = multi_loss(
            health_pred, degradation_pred, anomaly_logits,
            health_true, degradation_true, anomaly_true
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert "health" in loss_dict
        assert "degradation" in loss_dict
        assert "anomaly" in loss_dict
        assert "total" in loss_dict
    
    def test_multitask_loss_uncertainty_weighting(self):
        """Test uncertainty weighting strategy."""
        multi_loss = MultiTaskLoss(weighting="uncertainty")
        
        health_pred = torch.randn(16, 1)
        degradation_pred = torch.randn(16, 1)
        anomaly_logits = torch.randn(16, 9)
        
        health_true = torch.randn(16, 1)
        degradation_true = torch.randn(16, 1)
        anomaly_true = torch.randint(0, 2, (16, 9)).float()
        
        total_loss, loss_dict = multi_loss(
            health_pred, degradation_pred, anomaly_logits,
            health_true, degradation_true, anomaly_true
        )
        
        assert isinstance(total_loss, torch.Tensor)
        
        # Check uncertainty weighter exists
        assert hasattr(multi_loss, "uncertainty_weighter")
        
        # Check learnable parameters
        params = list(multi_loss.uncertainty_weighter.parameters())
        assert len(params) == 1
    
    def test_multitask_loss_custom_losses(self):
        """Test with custom loss functions."""
        multi_loss = MultiTaskLoss(
            health_loss=WingLoss(),
            degradation_loss=WingLoss(),
            anomaly_loss=FocalLoss(),
            weighting="fixed"
        )
        
        health_pred = torch.randn(8, 1)
        degradation_pred = torch.randn(8, 1)
        anomaly_logits = torch.randn(8, 9)
        
        health_true = torch.randn(8, 1)
        degradation_true = torch.randn(8, 1)
        anomaly_true = torch.randint(0, 2, (8, 9)).float()
        
        total_loss, loss_dict = multi_loss(
            health_pred, degradation_pred, anomaly_logits,
            health_true, degradation_true, anomaly_true
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0
