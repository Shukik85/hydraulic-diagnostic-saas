"""Unit tests for multi-level metrics system.

Comprehensive test coverage for:
    - RegressionMetrics (MAE, RMSE, R², MAPE)
    - ClassificationMetrics (Precision, Recall, F1, AUC)
    - RULMetrics (Horizon accuracy, asymmetric loss)
    - MultiLevelMetrics (Integration)

Python 3.14 Features:
    - Deferred annotations
    - Improved unittest framework
"""

from __future__ import annotations

import warnings
import pytest
import torch
from torch_geometric.data import Data, Batch

from src.training.metrics import (
    RegressionMetrics,
    ClassificationMetrics,
    RULMetrics,
    MultiLevelMetrics,
    MetricConfig,
    create_metrics,
)


# ===== Fixtures =====

@pytest.fixture
def regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic regression data.
    
    Returns:
        preds: Predictions [10, 1]
        targets: Ground truth [10, 1]
    """
    torch.manual_seed(42)
    
    targets = torch.rand(10, 1) * 100  # 0-100 range
    preds = targets + torch.randn(10, 1) * 5  # Add noise
    
    return preds, targets


@pytest.fixture
def classification_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic classification data.
    
    Returns:
        logits: Predictions [20, 9]
        targets: Binary labels [20, 9]
    """
    torch.manual_seed(42)
    
    # Generate targets (sparse multi-label)
    targets = torch.zeros(20, 9)
    for i in range(20):
        num_labels = torch.randint(1, 4, (1,)).item()
        indices = torch.randperm(9)[:num_labels]
        targets[i, indices] = 1
    
    # Generate logits (correlated with targets)
    logits = torch.randn(20, 9)
    logits = torch.where(targets == 1, logits + 2, logits - 2)
    
    return logits, targets


@pytest.fixture
def rul_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic RUL data.
    
    Returns:
        preds: RUL predictions [16, 1]
        targets: True RUL [16, 1]
    """
    torch.manual_seed(42)
    
    targets = torch.rand(16, 1) * 500  # 0-500 hours
    preds = targets + torch.randn(16, 1) * 30  # Add noise
    
    return preds, targets


@pytest.fixture
def batch_data() -> Data:
    """Synthetic graph batch with all targets.
    
    Returns:
        Batch object with component and graph targets
    """
    torch.manual_seed(42)
    
    # Graph structure (2 graphs, 10 nodes each)
    x = torch.randn(20, 34)
    edge_index = torch.tensor([
        [0, 1, 2, 10, 11, 12],
        [1, 2, 0, 11, 12, 10]
    ])
    edge_attr = torch.randn(6, 8)
    batch_tensor = torch.tensor([0]*10 + [1]*10)
    
    # Component-level targets
    y_component_health = torch.rand(20, 1)
    y_component_anomaly = torch.randint(0, 2, (20, 9)).float()
    
    # Graph-level targets
    y_graph_health = torch.rand(2, 1)
    y_graph_degradation = torch.rand(2, 1)
    y_graph_anomaly = torch.randint(0, 2, (2, 9)).float()
    y_graph_rul = torch.rand(2, 1) * 500
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch_tensor,
        y_component_health=y_component_health,
        y_component_anomaly=y_component_anomaly,
        y_graph_health=y_graph_health,
        y_graph_degradation=y_graph_degradation,
        y_graph_anomaly=y_graph_anomaly,
        y_graph_rul=y_graph_rul,
    )
    
    return data


# ===== RegressionMetrics Tests =====

def test_regression_metrics_mae(regression_data):
    """Test MAE computation."""
    preds, targets = regression_data
    
    metrics = RegressionMetrics(prefix="test_")
    metrics.update(preds, targets)
    result = metrics.compute()
    
    # Check keys
    assert "test_mae" in result
    assert "test_rmse" in result
    assert "test_r2" in result
    assert "test_mape" in result
    
    # MAE should be positive and reasonable
    mae = result["test_mae"].item()
    assert mae > 0
    assert mae < 100  # Given noise level


def test_regression_metrics_rmse(regression_data):
    """Test RMSE computation."""
    preds, targets = regression_data
    
    metrics = RegressionMetrics(prefix="test_")
    metrics.update(preds, targets)
    result = metrics.compute()
    
    rmse = result["test_rmse"].item()
    mae = result["test_mae"].item()
    
    # RMSE >= MAE (always true)
    assert rmse >= mae


def test_regression_metrics_r2(regression_data):
    """Test R² computation."""
    preds, targets = regression_data
    
    metrics = RegressionMetrics(prefix="test_")
    metrics.update(preds, targets)
    result = metrics.compute()
    
    r2 = result["test_r2"].item()
    
    # R² should be high for correlated data
    assert r2 > 0.5
    assert r2 <= 1.0


def test_regression_metrics_reset(regression_data):
    """Test metric reset."""
    preds, targets = regression_data
    
    metrics = RegressionMetrics(prefix="test_")
    
    # First update
    metrics.update(preds, targets)
    result1 = metrics.compute()
    
    # Reset
    metrics.reset()
    
    # Second update with different data
    preds2 = torch.rand_like(preds)
    targets2 = torch.rand_like(targets)
    metrics.update(preds2, targets2)
    result2 = metrics.compute()
    
    # Results should differ after reset
    assert result1["test_mae"] != result2["test_mae"]


# ===== ClassificationMetrics Tests =====

def test_classification_metrics_precision(classification_data):
    """Test Precision computation."""
    logits, targets = classification_data
    
    metrics = ClassificationMetrics(
        num_classes=9,
        average="macro",
        prefix="test_"
    )
    
    metrics.update(logits, targets)
    result = metrics.compute()
    
    # Check keys
    assert "test_precision" in result
    assert "test_recall" in result
    assert "test_f1" in result
    assert "test_auroc" in result
    
    # Precision should be in [0, 1]
    precision = result["test_precision"].item()
    assert 0 <= precision <= 1


def test_classification_metrics_f1(classification_data):
    """Test F1 score computation."""
    logits, targets = classification_data
    
    metrics = ClassificationMetrics(
        num_classes=9,
        average="macro",
        prefix="test_"
    )
    
    metrics.update(logits, targets)
    result = metrics.compute()
    
    f1 = result["test_f1"].item()
    precision = result["test_precision"].item()
    recall = result["test_recall"].item()
    
    # F1 is harmonic mean of precision and recall
    assert 0 <= f1 <= 1
    assert f1 <= max(precision, recall)


def test_classification_metrics_auroc(classification_data):
    """Test AUC-ROC computation."""
    logits, targets = classification_data
    
    metrics = ClassificationMetrics(
        num_classes=9,
        average="macro",
        prefix="test_"
    )
    
    metrics.update(logits, targets)
    result = metrics.compute()
    
    auroc = result["test_auroc"].item()
    
    # AUROC should be reasonable for correlated data
    assert auroc > 0.5  # Better than random
    assert auroc <= 1.0


def test_classification_metrics_threshold(classification_data):
    """Test different thresholds."""
    logits, targets = classification_data
    
    # Threshold 0.3 (more sensitive)
    metrics_low = ClassificationMetrics(
        num_classes=9,
        threshold=0.3,
        prefix="low_"
    )
    
    # Threshold 0.7 (more specific)
    metrics_high = ClassificationMetrics(
        num_classes=9,
        threshold=0.7,
        prefix="high_"
    )
    
    metrics_low.update(logits, targets)
    metrics_high.update(logits, targets)
    
    result_low = metrics_low.compute()
    result_high = metrics_high.compute()
    
    # Lower threshold -> higher recall
    assert result_low["low_recall"] >= result_high["high_recall"]


# ===== RULMetrics Tests =====

def test_rul_metrics_horizon_accuracy(rul_data):
    """Test horizon accuracy computation."""
    preds, targets = rul_data
    
    metrics = RULMetrics(
        horizons=[24, 72, 168],
        prefix="test_"
    )
    
    metrics.update(preds, targets)
    result = metrics.compute()
    
    # Check horizon accuracy keys
    assert "test_acc_h24" in result
    assert "test_acc_h72" in result
    assert "test_acc_h168" in result
    
    # Accuracy should increase with horizon
    acc_24 = result["test_acc_h24"].item()
    acc_72 = result["test_acc_h72"].item()
    acc_168 = result["test_acc_h168"].item()
    
    assert acc_24 <= acc_72 <= acc_168
    assert 0 <= acc_168 <= 1


def test_rul_metrics_asymmetric_loss(rul_data):
    """Test asymmetric loss (penalizes late predictions)."""
    preds, targets = rul_data
    
    metrics = RULMetrics(prefix="test_")
    metrics.update(preds, targets)
    result = metrics.compute()
    
    # Check asymmetric loss
    assert "test_asymmetric_loss" in result
    asymmetric = result["test_asymmetric_loss"].item()
    
    # Should be positive
    assert asymmetric > 0


def test_rul_metrics_late_penalty():
    """Test that late predictions are penalized more."""
    # Early predictions (pred > target)
    preds_early = torch.tensor([[150.0], [250.0]])
    targets = torch.tensor([[100.0], [200.0]])
    
    # Late predictions (pred < target)
    preds_late = torch.tensor([[50.0], [150.0]])
    
    # Compute asymmetric loss for early
    metrics_early = RULMetrics(prefix="early_")
    metrics_early.update(preds_early, targets)
    result_early = metrics_early.compute()
    
    # Compute asymmetric loss for late
    metrics_late = RULMetrics(prefix="late_")
    metrics_late.update(preds_late, targets)
    result_late = metrics_late.compute()
    
    # Late should have higher loss (2x penalty)
    assert result_late["late_asymmetric_loss"] > result_early["early_asymmetric_loss"]


def test_rul_metrics_custom_horizons():
    """Test custom horizon configuration."""
    preds = torch.tensor([[100.0], [200.0], [300.0]])
    targets = torch.tensor([[110.0], [190.0], [320.0]])
    
    metrics = RULMetrics(
        horizons=[12, 48],  # Custom horizons
        prefix="custom_"
    )
    
    metrics.update(preds, targets)
    result = metrics.compute()
    
    # Check custom horizon keys
    assert "custom_acc_h12" in result
    assert "custom_acc_h48" in result
    assert "custom_acc_h24" not in result  # Not in custom list


# ===== MultiLevelMetrics Tests =====

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multilevel_metrics_integration(batch_data):
    """Test complete multi-level metrics integration."""
    config = MetricConfig(num_anomaly_classes=9)
    metrics = MultiLevelMetrics(config, stage="val")
    
    # Create synthetic outputs
    outputs = {
        'component': {
            'health': torch.rand(20, 1),
            'anomaly': torch.randn(20, 9),
        },
        'graph': {
            'health': torch.rand(2, 1),
            'degradation': torch.rand(2, 1),
            'anomaly': torch.randn(2, 9),
            'rul': torch.rand(2, 1) * 500,
        }
    }
    
    # Update metrics
    metrics.update(outputs, batch_data)
    
    # Compute all metrics
    result = metrics.compute()
    
    # Check that all metric types are present
    # Component-level
    assert any("component_health" in k for k in result.keys())
    assert any("component_anomaly" in k for k in result.keys())
    
    # Graph-level
    assert any("graph_health" in k for k in result.keys())
    assert any("graph_degradation" in k for k in result.keys())
    assert any("graph_anomaly" in k for k in result.keys())
    assert any("graph_rul" in k for k in result.keys())
    
    # Check stage prefix
    assert all(k.startswith("val/") for k in result.keys())


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multilevel_metrics_reset(batch_data):
    """Test multi-level metrics reset."""
    metrics = MultiLevelMetrics(stage="train")
    
    # First update
    outputs1 = {
        'component': {
            'health': torch.rand(20, 1),
            'anomaly': torch.randn(20, 9),
        },
        'graph': {
            'health': torch.rand(2, 1),
            'degradation': torch.rand(2, 1),
            'anomaly': torch.randn(2, 9),
            'rul': torch.rand(2, 1) * 500,
        }
    }
    
    metrics.update(outputs1, batch_data)
    result1 = metrics.compute()
    
    # Reset
    metrics.reset()
    
    # Second update with different data
    outputs2 = {
        'component': {
            'health': torch.rand(20, 1) * 2,  # Different scale
            'anomaly': torch.randn(20, 9),
        },
        'graph': {
            'health': torch.rand(2, 1) * 2,
            'degradation': torch.rand(2, 1) * 2,
            'anomaly': torch.randn(2, 9),
            'rul': torch.rand(2, 1) * 1000,  # Different scale
        }
    }
    
    metrics.update(outputs2, batch_data)
    result2 = metrics.compute()
    
    # Results should differ after reset
    component_mae_key = "train/component_health_mae"
    assert result1[component_mae_key] != result2[component_mae_key]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multilevel_metrics_log_dict(batch_data):
    """Test conversion to float dict for Lightning logging."""
    metrics = MultiLevelMetrics(stage="val")
    
    outputs = {
        'component': {
            'health': torch.rand(20, 1),
            'anomaly': torch.randn(20, 9),
        },
        'graph': {
            'health': torch.rand(2, 1),
            'degradation': torch.rand(2, 1),
            'anomaly': torch.randn(2, 9),
            'rul': torch.rand(2, 1) * 500,
        }
    }
    
    metrics.update(outputs, batch_data)
    
    # Get log dict
    log_dict = metrics.log_dict()
    
    # All values should be floats
    assert all(isinstance(v, float) for v in log_dict.values())
    
    # Should have multiple metrics
    assert len(log_dict) > 10


# ===== Factory Function Tests =====

def test_create_metrics_factory():
    """Test metrics factory function."""
    metrics = create_metrics(
        stage="test",
        num_anomaly_classes=9,
        rul_horizons=[24, 72]
    )
    
    assert isinstance(metrics, MultiLevelMetrics)
    assert metrics.stage == "test"
    assert metrics.config.num_anomaly_classes == 9
    assert metrics.config.rul_horizons == [24, 72]


def test_metrics_config_defaults():
    """Test MetricConfig default values."""
    config = MetricConfig()
    
    assert config.num_anomaly_classes == 9
    assert config.rul_horizons == [24, 72, 168]
    assert config.component_average == "macro"
    assert config.anomaly_average == "macro"
    assert config.anomaly_threshold == 0.5
