"""Integration tests for full training pipeline.

Tests:
- Data loading
- Model training
- Inference
- Checkpoint save/load
"""

import pytest
import torch
from pytorch_lightning import Trainer

from src.training.lightning_module import HydraulicGNNModule


@pytest.mark.integration
class TestFullTrainingPipeline:
    """Test complete training pipeline."""

    @pytest.fixture
    def trainer(self, tmp_log_dir, tmp_checkpoint_dir) -> Trainer:
        """Create a trainer for testing."""
        return Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_log_dir),
            logger=False,
            enable_checkpointing=False,
        )

    @pytest.fixture
    def module(self) -> HydraulicGNNModule:
        """Create module for testing."""
        return HydraulicGNNModule(
            in_channels=34,
            hidden_channels=32,
            num_heads=2,
            num_gat_layers=1,
            lstm_hidden=64,
            lstm_layers=1,
        )

    def test_training_step(self, module, sample_batch):
        """Test single training step."""
        module.train()

        loss = module.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_validation_step(self, module, sample_batch):
        """Test single validation step."""
        module.eval()

        with torch.no_grad():
            loss = module.validation_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_test_step(self, module, sample_batch):
        """Test single test step."""
        module.eval()

        with torch.no_grad():
            loss = module.test_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_backward_pass(self, module, sample_batch):
        """Test backward pass works."""
        module.train()
        loss = module.training_step(sample_batch, batch_idx=0)
        loss.backward()

        # Check gradients exist
        for param in module.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_inference_deterministic(self, module, sample_graph):
        """Test inference is deterministic."""
        module.eval()
        torch.manual_seed(42)

        with torch.no_grad():
            out1 = module(
                x=sample_graph.x,
                edge_index=sample_graph.edge_index,
                edge_attr=sample_graph.edge_attr,
                batch=torch.zeros(sample_graph.x.shape[0], dtype=torch.long),
            )

        torch.manual_seed(42)
        with torch.no_grad():
            out2 = module(
                x=sample_graph.x,
                edge_index=sample_graph.edge_index,
                edge_attr=sample_graph.edge_attr,
                batch=torch.zeros(sample_graph.x.shape[0], dtype=torch.long),
            )

        # Outputs should be identical
        for key in out1["graph"]:
            assert torch.allclose(out1["graph"][key], out2["graph"][key])
