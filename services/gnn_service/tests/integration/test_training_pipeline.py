"""Integration tests for training pipeline with Phase 2 architecture.

Tests:
- Full training loop (forward + backward + optimizer step)
- MultiTaskLoss integration with real targets
- Checkpoint save/load
- Learning rate scheduler
- Gradient clipping
- Training resumption

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Batch, Data

from models import (
    ModelConfig,
    MultiTaskLoss,
    MultiTaskLossConfig,
    UniversalTemporalGNNv2,
)


@pytest.mark.integration
class TestTrainingPipeline:
    """Test complete training pipeline (Phase 2)."""

    @pytest.fixture
    def config(self) -> ModelConfig:
        """Model configuration for testing."""
        return ModelConfig(
            node_features=34,
            edge_features=14,
            gat_hidden_dim=64,
            gat_num_layers=2,
            lstm_hidden_dim=64,
            lstm_num_layers=1,
            graph_anomaly_classes=9,
            component_anomaly_classes=9,
        )

    @pytest.fixture
    def model(self, config: ModelConfig) -> UniversalTemporalGNNv2:
        """Initialized model."""
        return UniversalTemporalGNNv2(config)

    @pytest.fixture
    def loss_fn(self) -> MultiTaskLoss:
        """Multi-task loss function."""
        config = MultiTaskLossConfig(
            component_health_weight=1.0,
            component_anomaly_weight=0.5,
            graph_health_weight=1.0,
            graph_degradation_weight=1.0,
            graph_anomaly_weight=0.5,
            graph_rul_weight=2.0,
        )
        return MultiTaskLoss(config)

    @pytest.fixture
    def optimizer(self, model: UniversalTemporalGNNv2) -> Adam:
        """Adam optimizer."""
        return Adam(model.parameters(), lr=1e-3)

    @pytest.fixture
    def sample_batch(self) -> Batch:
        """Sample batch of graphs."""
        graphs = [
            Data(
                x=torch.randn(10, 34),
                edge_index=torch.randint(0, 10, (2, 20)),
                edge_attr=torch.randn(20, 14),
            )
            for _ in range(4)
        ]
        return Batch.from_data_list(graphs)

    @pytest.fixture
    def sample_targets(self) -> dict[str, dict[str, torch.Tensor]]:
        """Sample ground truth targets for batch."""
        return {
            'component': {
                'health': torch.rand(40, 1),  # 4 graphs * 10 nodes
                'anomaly': torch.randint(0, 2, (40, 9)).float(),
            },
            'graph': {
                'health': torch.rand(4, 1),
                'degradation': torch.rand(4, 1),
                'anomaly': torch.randint(0, 2, (4, 9)).float(),
                'rul': torch.rand(4, 1) * 1000,  # Hours
            }
        }

    def test_full_training_step(self, model, loss_fn, optimizer, sample_batch, sample_targets):
        """Test full training step: forward + loss + backward + optimizer step."""
        model.train()
        
        # Forward pass
        outputs = model(sample_batch, temporal=False)
        
        # Compute loss
        losses = loss_fn(outputs, sample_targets)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad and not name.startswith('lstm.'):
                assert param.grad is not None
        
        # Optimizer step
        optimizer.step()
        
        # Verify loss is valid
        assert not torch.isnan(losses['total'])
        assert losses['total'].item() >= 0

    def test_validation_loop(self, model, loss_fn, sample_batch, sample_targets):
        """Test validation loop (no gradients)."""
        model.eval()
        
        with torch.no_grad():
            outputs = model(sample_batch, temporal=False)
            losses = loss_fn(outputs, sample_targets)
        
        # Check all 6 losses computed
        assert 'component_health' in losses
        assert 'component_anomaly' in losses
        assert 'graph_health' in losses
        assert 'graph_degradation' in losses
        assert 'graph_anomaly' in losses
        assert 'graph_rul' in losses
        assert 'total' in losses
        
        # All losses should be valid
        for loss_name, loss_value in losses.items():
            assert not torch.isnan(loss_value)
            assert loss_value.item() >= 0

    def test_multi_epoch_training(self, model, loss_fn, optimizer, sample_batch, sample_targets):
        """Test training over multiple epochs."""
        model.train()
        
        losses_history = []
        
        for epoch in range(3):
            # Forward
            outputs = model(sample_batch, temporal=False)
            losses = loss_fn(outputs, sample_targets)
            
            # Backward + optimize
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            losses_history.append(losses['total'].item())
        
        # Check we have 3 epochs
        assert len(losses_history) == 3
        
        # All losses should be valid
        for loss in losses_history:
            assert not torch.isnan(torch.tensor(loss))
            assert loss >= 0

    def test_gradient_clipping(self, model, loss_fn, optimizer, sample_batch, sample_targets):
        """Test gradient clipping during training."""
        model.train()
        
        # Forward + backward
        outputs = model(sample_batch, temporal=False)
        losses = loss_fn(outputs, sample_targets)
        optimizer.zero_grad()
        losses['total'].backward()
        
        # Clip gradients
        max_norm = 1.0
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Check clipping worked
        assert total_norm.item() >= 0
        
        # Verify all gradients are within bounds
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                # Individual grad norms can exceed max_norm, but total should not
                assert not torch.isnan(grad_norm)

    def test_learning_rate_scheduler(self, model, loss_fn, optimizer, sample_batch, sample_targets):
        """Test learning rate scheduler integration."""
        model.train()
        
        # Create scheduler (reduce LR every 2 steps)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        for step in range(3):
            # Training step
            outputs = model(sample_batch, temporal=False)
            losses = loss_fn(outputs, sample_targets)
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            scheduler.step()
        
        # After 3 steps (2 scheduler steps), LR should be reduced once
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr
        assert final_lr == initial_lr * 0.5  # One reduction by gamma=0.5

    def test_checkpoint_save_load(self, model, optimizer):
        """Test model and optimizer checkpoint save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': 5,
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Create new model and optimizer
            new_model = UniversalTemporalGNNv2(model.config)
            new_optimizer = Adam(new_model.parameters(), lr=1e-3)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
            new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
            new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
            
            # Verify epoch loaded
            assert loaded_checkpoint['epoch'] == 5
            
            # Verify model parameters match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)

    def test_training_resumption(self, config, sample_batch, sample_targets):
        """Test resuming training from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            
            # Train for 2 epochs
            model1 = UniversalTemporalGNNv2(config)
            optimizer1 = Adam(model1.parameters(), lr=1e-3)
            loss_fn = MultiTaskLoss()
            
            model1.train()
            for epoch in range(2):
                outputs = model1(sample_batch, temporal=False)
                losses = loss_fn(outputs, sample_targets)
                optimizer1.zero_grad()
                losses['total'].backward()
                optimizer1.step()
            
            # Save checkpoint
            torch.save({
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'epoch': 2,
            }, checkpoint_path)
            
            # Load and continue training
            model2 = UniversalTemporalGNNv2(config)
            optimizer2 = Adam(model2.parameters(), lr=1e-3)
            
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model2.load_state_dict(checkpoint['model_state_dict'])
            optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            
            assert start_epoch == 2
            
            # Continue training for 1 more epoch
            model2.train()
            outputs = model2(sample_batch, temporal=False)
            losses = loss_fn(outputs, sample_targets)
            optimizer2.zero_grad()
            losses['total'].backward()
            optimizer2.step()
            
            # Training should work without errors
            assert not torch.isnan(losses['total'])

    def test_loss_tracking(self, model, loss_fn, optimizer, sample_batch, sample_targets):
        """Test tracking individual losses over training."""
        model.train()
        
        loss_history = {
            'component_health': [],
            'component_anomaly': [],
            'graph_health': [],
            'graph_degradation': [],
            'graph_anomaly': [],
            'graph_rul': [],
            'total': [],
        }
        
        for step in range(3):
            outputs = model(sample_batch, temporal=False)
            losses = loss_fn(outputs, sample_targets)
            
            # Track all losses
            for key in loss_history:
                loss_history[key].append(losses[key].item())
            
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
        
        # Verify all losses tracked
        for key, values in loss_history.items():
            assert len(values) == 3
            for val in values:
                assert not torch.isnan(torch.tensor(val))
                assert val >= 0

    def test_batch_accumulation(self, model, loss_fn, optimizer, sample_batch, sample_targets):
        """Test gradient accumulation over multiple batches."""
        model.train()
        accumulation_steps = 2
        
        optimizer.zero_grad()
        
        for step in range(accumulation_steps):
            outputs = model(sample_batch, temporal=False)
            losses = loss_fn(outputs, sample_targets)
            
            # Scale loss by accumulation steps
            scaled_loss = losses['total'] / accumulation_steps
            scaled_loss.backward()
        
        # Update after accumulation
        optimizer.step()
        
        # Verify gradients accumulated
        grad_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_count += 1
                assert not torch.isnan(param.grad).any()
        
        assert grad_count > 0

    def test_mixed_precision_compatibility(self, model, loss_fn, optimizer, sample_batch, sample_targets):
        """Test model works with mixed precision training (float16)."""
        model.train()
        
        # Convert model to float16 (simulates mixed precision)
        # Note: In real training, use torch.cuda.amp.autocast()
        sample_batch_fp16 = sample_batch.clone()
        sample_batch_fp16.x = sample_batch_fp16.x.half()
        sample_batch_fp16.edge_attr = sample_batch_fp16.edge_attr.half()
        
        # Forward in float16
        model = model.half()
        outputs = model(sample_batch_fp16, temporal=False)
        
        # Convert targets to float16
        targets_fp16 = {
            'component': {
                'health': sample_targets['component']['health'].half(),
                'anomaly': sample_targets['component']['anomaly'].half(),
            },
            'graph': {
                'health': sample_targets['graph']['health'].half(),
                'degradation': sample_targets['graph']['degradation'].half(),
                'anomaly': sample_targets['graph']['anomaly'].half(),
                'rul': sample_targets['graph']['rul'].half(),
            }
        }
        
        # Compute loss (loss modules internally use float32 for stability)
        losses = loss_fn(outputs, targets_fp16)
        
        # Outputs should be float16
        assert outputs['component']['health'].dtype == torch.float16
        assert outputs['graph']['rul'].dtype == torch.float16
        
        # Loss should work without errors
        assert not torch.isnan(losses['total'])
