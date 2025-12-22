"""Unit tests for MultiTaskLoss (Phase 2).

Test coverage:
- 6-task loss computation
- Nested dict input/output
- Individual loss functions
- Weighted combination
- Pos weight updates

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import pytest
import torch

from models import MultiTaskLoss, MultiTaskLossConfig, compute_pos_weights


class TestMultiTaskLoss:
    """Test MultiTaskLoss for Phase 2 (6 tasks)."""

    @pytest.fixture
    def config(self) -> MultiTaskLossConfig:
        """Default loss configuration."""
        return MultiTaskLossConfig(
            component_health_weight=1.0,
            component_anomaly_weight=0.5,
            graph_health_weight=1.0,
            graph_degradation_weight=1.0,
            graph_anomaly_weight=0.5,
            graph_rul_weight=2.0,
            rul_huber_delta=1.0,
        )

    @pytest.fixture
    def loss_fn(self, config: MultiTaskLossConfig) -> MultiTaskLoss:
        """Initialized loss function."""
        return MultiTaskLoss(config)

    @pytest.fixture
    def sample_outputs(self) -> dict[str, dict[str, torch.Tensor]]:
        """Sample model outputs (Phase 2)."""
        return {
            'component': {
                'health': torch.randn(10, 1),      # [N, 1]
                'anomaly': torch.randn(10, 9)      # [N, 9] logits
            },
            'graph': {
                'health': torch.randn(4, 1),       # [B, 1]
                'degradation': torch.randn(4, 1),  # [B, 1]
                'anomaly': torch.randn(4, 9),      # [B, 9] logits
                'rul': torch.randn(4, 1)           # [B, 1]
            }
        }

    @pytest.fixture
    def sample_targets(self) -> dict[str, dict[str, torch.Tensor]]:
        """Sample ground truth targets."""
        return {
            'component': {
                'health': torch.rand(10, 1),                          # [0, 1]
                'anomaly': torch.randint(0, 2, (10, 9)).float()      # Binary labels
            },
            'graph': {
                'health': torch.rand(4, 1),                          # [0, 1]
                'degradation': torch.rand(4, 1),                     # [0, 1]
                'anomaly': torch.randint(0, 2, (4, 9)).float(),      # Binary labels
                'rul': torch.rand(4, 1) * 1000                       # Hours
            }
        }

    def test_loss_initialization(self, config: MultiTaskLossConfig):
        """Loss function should initialize without errors."""
        loss_fn = MultiTaskLoss(config)
        assert loss_fn.config == config
        assert loss_fn.mse_loss is not None
        assert loss_fn.huber_loss is not None

    def test_forward_all_tasks(self, loss_fn: MultiTaskLoss, sample_outputs, sample_targets):
        """Test loss computation with all 6 tasks."""
        losses = loss_fn(sample_outputs, sample_targets)

        # Check all expected keys
        assert 'total' in losses
        assert 'component_health' in losses
        assert 'component_anomaly' in losses
        assert 'graph_health' in losses
        assert 'graph_degradation' in losses
        assert 'graph_anomaly' in losses
        assert 'graph_rul' in losses

        # All losses should be scalars
        for key, loss in losses.items():
            assert isinstance(loss, torch.Tensor)
            assert loss.ndim == 0  # Scalar
            assert not torch.isnan(loss)
            assert loss.item() >= 0

    def test_weighted_combination(self, sample_outputs, sample_targets):
        """Test weighted combination of losses."""
        config = MultiTaskLossConfig(
            component_health_weight=2.0,
            component_anomaly_weight=1.0,
            graph_health_weight=0.5,
            graph_degradation_weight=0.5,
            graph_anomaly_weight=1.0,
            graph_rul_weight=3.0,
        )
        loss_fn = MultiTaskLoss(config)

        losses = loss_fn(sample_outputs, sample_targets)

        # Manual weighted sum
        expected_total = (
            2.0 * losses['component_health'] +
            1.0 * losses['component_anomaly'] +
            0.5 * losses['graph_health'] +
            0.5 * losses['graph_degradation'] +
            1.0 * losses['graph_anomaly'] +
            3.0 * losses['graph_rul']
        )

        assert torch.allclose(losses['total'], expected_total, atol=1e-6)

    def test_missing_component_tasks(self, loss_fn: MultiTaskLoss, sample_targets):
        """Loss should handle missing component-level outputs."""
        # Only graph-level outputs
        outputs = {
            'graph': {
                'health': torch.randn(4, 1),
                'degradation': torch.randn(4, 1),
                'anomaly': torch.randn(4, 9),
                'rul': torch.randn(4, 1)
            }
        }

        losses = loss_fn(outputs, sample_targets)

        # Should only have graph losses
        assert 'component_health' not in losses
        assert 'component_anomaly' not in losses
        assert 'graph_health' in losses
        assert 'total' in losses

    def test_missing_graph_tasks(self, loss_fn: MultiTaskLoss, sample_targets):
        """Loss should handle missing graph-level outputs."""
        # Only component-level outputs
        outputs = {
            'component': {
                'health': torch.randn(10, 1),
                'anomaly': torch.randn(10, 9)
            }
        }

        losses = loss_fn(outputs, sample_targets)

        # Should only have component losses
        assert 'component_health' in losses
        assert 'component_anomaly' in losses
        assert 'graph_health' not in losses
        assert 'total' in losses

    def test_pos_weight_update(self, loss_fn: MultiTaskLoss, sample_outputs, sample_targets):
        """Test updating positive class weights."""
        # Initial loss
        losses_before = loss_fn(sample_outputs, sample_targets)

        # Update weights
        new_weights = torch.tensor([2.0] * 9)
        loss_fn.update_anomaly_weights(
            component_anomaly_weights=new_weights,
            graph_anomaly_weights=new_weights
        )

        # Loss should change
        losses_after = loss_fn(sample_outputs, sample_targets)

        # Anomaly losses should be different
        assert not torch.allclose(
            losses_before['component_anomaly'],
            losses_after['component_anomaly']
        )

    def test_backward_pass(self, loss_fn: MultiTaskLoss, sample_outputs, sample_targets):
        """Test gradient flow through all losses."""
        # Make outputs require grad
        for level in sample_outputs.values():
            for tensor in level.values():
                tensor.requires_grad = True

        losses = loss_fn(sample_outputs, sample_targets)
        losses['total'].backward()

        # Check gradients exist
        for level in sample_outputs.values():
            for tensor in level.values():
                assert tensor.grad is not None
                assert not torch.isnan(tensor.grad).any()

    def test_device_compatibility(self, config: MultiTaskLossConfig):
        """Test loss works on CPU."""
        loss_fn = MultiTaskLoss(config).to('cpu')

        outputs = {
            'component': {
                'health': torch.randn(5, 1, device='cpu'),
                'anomaly': torch.randn(5, 9, device='cpu')
            },
            'graph': {
                'health': torch.randn(2, 1, device='cpu'),
                'degradation': torch.randn(2, 1, device='cpu'),
                'anomaly': torch.randn(2, 9, device='cpu'),
                'rul': torch.randn(2, 1, device='cpu')
            }
        }
        targets = {
            'component': {
                'health': torch.rand(5, 1, device='cpu'),
                'anomaly': torch.randint(0, 2, (5, 9), device='cpu').float()
            },
            'graph': {
                'health': torch.rand(2, 1, device='cpu'),
                'degradation': torch.rand(2, 1, device='cpu'),
                'anomaly': torch.randint(0, 2, (2, 9), device='cpu').float(),
                'rul': torch.rand(2, 1, device='cpu') * 1000
            }
        }

        losses = loss_fn(outputs, targets)
        assert losses['total'].device.type == 'cpu'


class TestComputePosWeights:
    """Test compute_pos_weights utility."""

    def test_balanced_classes(self):
        """Test with balanced classes."""
        labels = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        weights = compute_pos_weights(labels, num_classes=3)

        # Class 0: 2 pos, 2 neg -> weight = 1.0
        # Class 1: 2 pos, 2 neg -> weight = 1.0
        # Class 2: 2 pos, 2 neg -> weight = 1.0
        assert torch.allclose(weights, torch.tensor([1.0, 1.0, 1.0]))

    def test_imbalanced_classes(self):
        """Test with imbalanced classes."""
        labels = torch.tensor([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 1]
        ])

        weights = compute_pos_weights(labels, num_classes=3)

        # Class 0: 3 pos, 1 neg -> weight = 1/3
        # Class 1: 1 pos, 3 neg -> weight = 3
        # Class 2: 1 pos, 3 neg -> weight = 3
        expected = torch.tensor([1.0/3.0, 3.0, 3.0])
        assert torch.allclose(weights, expected)

    def test_all_positive(self):
        """Test with all positive labels."""
        labels = torch.ones(5, 3)

        weights = compute_pos_weights(labels, num_classes=3)

        # All positive -> weight = 0
        assert torch.allclose(weights, torch.tensor([0.0, 0.0, 0.0]))
