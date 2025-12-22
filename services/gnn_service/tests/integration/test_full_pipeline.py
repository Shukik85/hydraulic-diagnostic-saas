"""Integration tests for full training pipeline with UniversalTemporalGNNv2.

Tests:
- Model forward pass (single + temporal)
- Training/validation/test steps
- Gradient flow
- Inference determinism
- Checkpoint compatibility (future)

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from models import ModelConfig, MultiTaskLoss, UniversalTemporalGNNv2


@pytest.mark.integration
class TestFullTrainingPipeline:
    """Test complete training pipeline with v2 model."""

    @pytest.fixture
    def config(self) -> ModelConfig:
        """Model configuration for testing."""
        return ModelConfig(
            node_features=34,
            edge_features=14,
            gat_hidden_dim=64,  # Smaller for tests
            gat_num_layers=2,
            lstm_hidden_dim=64,
            lstm_num_layers=1,
            component_health_num_classes=5,
            anomaly_type_num_classes=4,
        )

    @pytest.fixture
    def model(self, config: ModelConfig) -> UniversalTemporalGNNv2:
        """Create v2 model for testing."""
        return UniversalTemporalGNNv2(config)

    @pytest.fixture
    def loss_fn(self) -> MultiTaskLoss:
        """Multi-task loss function."""
        return MultiTaskLoss()

    @pytest.fixture
    def sample_graph(self) -> Data:
        """Sample single graph for testing."""
        return Data(
            x=torch.randn(10, 34),  # 10 nodes, 34 features
            edge_index=torch.randint(0, 10, (2, 20)),  # 20 edges
            edge_attr=torch.randn(20, 14),  # 14D edge features
            y_node=torch.randint(0, 5, (10,)),  # Component health (5 classes)
            y_graph=torch.randint(0, 4, (1,)),  # Anomaly type (4 classes)
        )

    @pytest.fixture
    def sample_batch(self, sample_graph: Data) -> Batch:
        """Sample batch (4 graphs)."""
        return Batch.from_data_list([sample_graph for _ in range(4)])

    @pytest.fixture
    def sample_temporal_sequence(self, sample_graph: Data) -> list[Data]:
        """Sample temporal sequence (5 timesteps)."""
        return [sample_graph for _ in range(5)]

    def test_training_step_single(self, model: UniversalTemporalGNNv2, loss_fn: MultiTaskLoss, sample_graph: Data):
        """Test single graph training step."""
        model.train()

        # Forward pass
        outputs = model(sample_graph, temporal=False)

        # Compute loss
        losses = loss_fn(
            outputs['node_logits'], sample_graph.y_node,
            outputs['graph_logits'], sample_graph.y_graph
        )
        total_loss = losses['total']

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0
        assert not torch.isnan(total_loss)
        assert total_loss.requires_grad  # Gradient should be enabled

    def test_training_step_temporal(self, model: UniversalTemporalGNNv2, loss_fn: MultiTaskLoss, sample_temporal_sequence: list[Data]):
        """Test temporal sequence training step."""
        model.train()

        # Forward pass (temporal)
        outputs = model(sample_temporal_sequence, temporal=True)

        # Use last timestep targets
        last_graph = sample_temporal_sequence[-1]
        losses = loss_fn(
            outputs['node_logits'], last_graph.y_node,
            outputs['graph_logits'], last_graph.y_graph
        )
        total_loss = losses['total']

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0
        assert not torch.isnan(total_loss)

    def test_validation_step(self, model: UniversalTemporalGNNv2, loss_fn: MultiTaskLoss, sample_graph: Data):
        """Test validation step (inference mode)."""
        model.eval()

        with torch.no_grad():
            outputs = model(sample_graph, temporal=False)
            losses = loss_fn(
                outputs['node_logits'], sample_graph.y_node,
                outputs['graph_logits'], sample_graph.y_graph
            )
            val_loss = losses['total']

        assert isinstance(val_loss, torch.Tensor)
        assert val_loss.item() >= 0
        assert not torch.isnan(val_loss)

    def test_test_step(self, model: UniversalTemporalGNNv2, loss_fn: MultiTaskLoss, sample_batch: Batch):
        """Test step on batched data."""
        model.eval()

        with torch.no_grad():
            outputs = model(sample_batch, temporal=False)
            losses = loss_fn(
                outputs['node_logits'], sample_batch.y_node,
                outputs['graph_logits'], sample_batch.y_graph
            )
            test_loss = losses['total']

        assert isinstance(test_loss, torch.Tensor)
        assert test_loss.item() >= 0

    def test_backward_pass(self, model: UniversalTemporalGNNv2, loss_fn: MultiTaskLoss, sample_graph: Data):
        """Test backward pass and gradient flow."""
        model.train()

        outputs = model(sample_graph, temporal=False)
        losses = loss_fn(
            outputs['node_logits'], sample_graph.y_node,
            outputs['graph_logits'], sample_graph.y_graph
        )
        total_loss = losses['total']
        total_loss.backward()

        # Check gradients exist for all trainable parameters (except LSTM in single mode)
        for name, param in model.named_parameters():
            if param.requires_grad and not name.startswith('lstm.'):
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_inference_deterministic(self, model: UniversalTemporalGNNv2, sample_graph: Data):
        """Test inference is deterministic with fixed seed."""
        model.eval()

        # Run 1
        torch.manual_seed(42)
        with torch.no_grad():
            outputs1 = model(sample_graph, temporal=False)

        # Run 2 (same seed)
        torch.manual_seed(42)
        with torch.no_grad():
            outputs2 = model(sample_graph, temporal=False)

        # Outputs should be identical
        assert torch.allclose(outputs1['node_logits'], outputs2['node_logits'], atol=1e-6)
        assert torch.allclose(outputs1['graph_logits'], outputs2['graph_logits'], atol=1e-6)

    def test_batch_processing(self, model: UniversalTemporalGNNv2, sample_batch: Batch):
        """Test model handles batched graphs correctly."""
        model.eval()

        with torch.no_grad():
            outputs = model(sample_batch, temporal=False)

        # Check output shapes
        assert outputs['node_logits'].shape[0] == sample_batch.x.shape[0]  # All nodes
        assert outputs['node_logits'].shape[1] == 5  # 5 health classes
        assert outputs['graph_logits'].shape[0] == sample_batch.num_graphs  # 4 graphs
        assert outputs['graph_logits'].shape[1] == 4  # 4 anomaly types

    def test_temporal_all_timesteps(self, model: UniversalTemporalGNNv2, sample_temporal_sequence: list[Data]):
        """Test temporal mode returns all timesteps when requested."""
        model.eval()

        with torch.no_grad():
            outputs = model(
                sample_temporal_sequence,
                temporal=True,
                return_all_timesteps=True
            )

        # Should have node predictions for all timesteps
        assert 'node_logits_seq' in outputs
        assert len(outputs['node_logits_seq']) == len(sample_temporal_sequence)

        # Graph prediction is still final timestep
        assert outputs['graph_logits'].shape == (1, 4)

    def test_attention_weights_extraction(self, model: UniversalTemporalGNNv2, sample_graph: Data):
        """Test attention weights can be extracted for interpretability."""
        model.eval()

        with torch.no_grad():
            outputs = model(sample_graph, temporal=False, return_attention=True)

        # Should have attention weights
        assert 'attention_weights' in outputs
        assert isinstance(outputs['attention_weights'], dict)
        assert len(outputs['attention_weights']) > 0  # At least one GAT layer

    def test_model_device_compatibility(self, config: ModelConfig, sample_graph: Data):
        """Test model works on CPU (GPU test would require CUDA)."""
        model = UniversalTemporalGNNv2(config)
        model.eval()

        # Ensure everything on CPU
        model = model.to('cpu')
        sample_graph = sample_graph.to('cpu')

        with torch.no_grad():
            outputs = model(sample_graph, temporal=False)

        # Check outputs are on CPU
        assert outputs['node_logits'].device.type == 'cpu'
        assert outputs['graph_logits'].device.type == 'cpu'
