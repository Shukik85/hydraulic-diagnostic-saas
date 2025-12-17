"""Unit tests for HydraulicGNNModule.

Tests:
- Loss computation
- Forward pass
- Optimization
- Checkpoint saving/loading
"""

import pytest
import torch
from pytorch_lightning import Trainer

from src.training.lightning_module import HydraulicGNNModule


class TestHydraulicGNNModule:
    """Tests for HydraulicGNNModule."""

    @pytest.fixture
    def module(self) -> HydraulicGNNModule:
        """Create a module instance."""
        return HydraulicGNNModule(
            in_channels=34,
            hidden_channels=64,
            num_heads=4,
            num_gat_layers=2,
            lstm_hidden=128,
            lstm_layers=1,
            learning_rate=0.001,
            loss_weighting="fixed",
        )

    def test_module_initialization(self, module):
        """Test module initializes correctly."""
        assert module.model is not None
        assert hasattr(module, "graph_health_loss")
        assert hasattr(module, "graph_anomaly_loss")
        assert hasattr(module, "uncertainty_weighter") or module.loss_weighting == "fixed"

    def test_forward_pass(self, module, sample_batch):
        """Test forward pass returns correct outputs."""
        outputs = module(
            x=sample_batch.x,
            edge_index=sample_batch.edge_index,
            edge_attr=sample_batch.edge_attr,
            batch=sample_batch.batch,
        )

        assert isinstance(outputs, dict)
        assert "graph" in outputs
        assert "component" in outputs
        assert "health" in outputs["graph"]
        assert "anomaly" in outputs["graph"]

    def test_loss_computation(self, module, sample_batch):
        """Test loss computation."""
        outputs = module(
            x=sample_batch.x,
            edge_index=sample_batch.edge_index,
            edge_attr=sample_batch.edge_attr,
            batch=sample_batch.batch,
        )

        total_loss, loss_dict = module.compute_loss(outputs, sample_batch)

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert total_loss.item() >= 0

    def test_configure_optimizers(self, module):
        """Test optimizer configuration."""
        config = module.configure_optimizers()

        assert isinstance(config, dict)
        assert "optimizer" in config
        assert config["optimizer"] is not None

    def test_loss_weighting_validation(self):
        """Test invalid loss_weighting raises error."""
        with pytest.raises(ValueError):
            HydraulicGNNModule(
                in_channels=34,
                loss_weighting="invalid",
            )

    def test_batch_field_validation(self, module, sample_batch):
        """Test batch field validation in compute_loss."""
        # Create incomplete batch
        incomplete_batch = sample_batch.clone()
        del incomplete_batch.y_graph_health

        outputs = module(
            x=incomplete_batch.x,
            edge_index=incomplete_batch.edge_index,
            edge_attr=incomplete_batch.edge_attr,
            batch=incomplete_batch.batch,
        )

        # Should raise AttributeError
        with pytest.raises(AttributeError):
            module.compute_loss(outputs, incomplete_batch)
