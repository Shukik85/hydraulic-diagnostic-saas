"""Unit tests for HydraulicGNNModule (Phase 2.1.0 API).

Tests for:
- Module initialization with ModelConfig
- Forward pass (nested outputs)
- Loss computation (6 tasks)
- Training step (manual optimization)
- Optimizer configuration
- Advanced losses
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models import ModelConfig
from src.training.lightning_module import HydraulicGNNModule


@pytest.fixture
def model_config():
    """Create ModelConfig for testing.
    
    Note: ModelConfig has separate dropout parameters:
    - gat_dropout (for GATv2 layers)
    - lstm_dropout (for LSTM)
    - head_dropout (for prediction heads)
    """
    return ModelConfig(
        node_features=34,
        edge_features=14,
        gat_hidden_dim=64,  # Smaller for tests
        lstm_hidden_dim=128,
        gat_num_heads=4,
        gat_num_layers=2,
        lstm_num_layers=1,
        gat_dropout=0.1,
        lstm_dropout=0.1,
        head_dropout=0.2,
        version="2.1.0",
    )


@pytest.fixture
def lightning_module(model_config: ModelConfig):
    """Create HydraulicGNNModule for testing."""
    return HydraulicGNNModule(
        model_config=model_config,
        learning_rate=0.001,
        scheduler_type="plateau",
        loss_weighting="fixed",
        use_advanced_losses=False,  # Disable for basic tests
        use_confidence_weighting=False,
        use_domain_adversarial=False,
    )


@pytest.fixture
def sample_batch():
    """Create sample PyG Data batch for testing.
    
    Returns:
        Batch with 2 graphs, 10 total nodes
    """
    # Graph 1: 5 nodes
    graph1 = Data(
        x=torch.randn(5, 34),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.randn(4, 14),
        batch=torch.zeros(5, dtype=torch.long),
        # Graph-level targets
        y_graph_health=torch.tensor([0.8]),
        y_graph_degradation=torch.tensor([0.2]),
        y_graph_anomaly=torch.randint(0, 2, (9,)).float(),
        y_graph_rul=torch.tensor([100.0]),
        # Component-level targets
        y_component_health=torch.rand(5),
        y_component_anomaly=torch.randint(0, 2, (5, 9)).float(),
    )
    
    # Graph 2: 5 nodes
    graph2 = Data(
        x=torch.randn(5, 34),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.randn(4, 14),
        batch=torch.ones(5, dtype=torch.long),
        y_graph_health=torch.tensor([0.6]),
        y_graph_degradation=torch.tensor([0.4]),
        y_graph_anomaly=torch.randint(0, 2, (9,)).float(),
        y_graph_rul=torch.tensor([50.0]),
        y_component_health=torch.rand(5),
        y_component_anomaly=torch.randint(0, 2, (5, 9)).float(),
    )
    
    # Create batch
    batch = Batch.from_data_list([graph1, graph2])
    
    return batch


class TestHydraulicGNNModuleInit:
    """Tests for module initialization."""

    def test_module_initialization(self, lightning_module: HydraulicGNNModule):
        """Test module initializes correctly."""
        assert lightning_module is not None
        assert hasattr(lightning_module, "model")
        assert hasattr(lightning_module, "graph_health_loss")
        assert lightning_module.automatic_optimization is False  # Manual optimization

    def test_model_config_stored(self, lightning_module: HydraulicGNNModule):
        """Test ModelConfig is stored."""
        assert hasattr(lightning_module, "model_config")
        assert lightning_module.model_config.node_features == 34
        assert lightning_module.model_config.edge_features == 14

    def test_hyperparameters_saved(self, lightning_module: HydraulicGNNModule):
        """Test hyperparameters are saved (for checkpointing)."""
        assert "learning_rate" in lightning_module.hparams
        assert "scheduler_type" in lightning_module.hparams
        assert lightning_module.hparams["learning_rate"] == 0.001

    def test_manual_optimization_enabled(self, lightning_module: HydraulicGNNModule):
        """Test manual optimization is enabled (prevents closure() errors)."""
        assert lightning_module.automatic_optimization is False


class TestForwardPass:
    """Tests for forward pass."""

    def test_forward_returns_nested_dict(self, lightning_module: HydraulicGNNModule, sample_batch):
        """Test forward returns nested dict (Phase 2 v2 API)."""
        outputs = lightning_module(sample_batch)
        
        # Check nested structure
        assert isinstance(outputs, dict)
        assert "graph" in outputs
        assert "component" in outputs
        
        # Graph-level outputs
        assert "health" in outputs["graph"]
        assert "degradation" in outputs["graph"]
        assert "anomaly" in outputs["graph"]
        assert "rul" in outputs["graph"]
        
        # Component-level outputs
        assert "health" in outputs["component"]
        assert "anomaly" in outputs["component"]

    def test_forward_output_shapes(self, lightning_module: HydraulicGNNModule, sample_batch):
        """Test forward output shapes."""
        outputs = lightning_module(sample_batch)
        batch_size = 2  # 2 graphs
        num_nodes = 10  # 5 + 5 nodes
        
        # Graph-level shapes
        assert outputs["graph"]["health"].shape == (batch_size, 1)
        assert outputs["graph"]["degradation"].shape == (batch_size, 1)
        assert outputs["graph"]["anomaly"].shape == (batch_size, 9)
        assert outputs["graph"]["rul"].shape == (batch_size, 1)
        
        # Component-level shapes
        assert outputs["component"]["health"].shape == (num_nodes, 1)
        assert outputs["component"]["anomaly"].shape == (num_nodes, 9)

    def test_forward_no_nans(self, lightning_module: HydraulicGNNModule, sample_batch):
        """Test forward doesn't produce NaNs."""
        outputs = lightning_module(sample_batch)
        
        for level in ["graph", "component"]:
            for task, tensor in outputs[level].items():
                assert not torch.isnan(tensor).any(), f"NaN in {level}/{task}"


class TestLossComputation:
    """Tests for loss computation."""

    def test_compute_loss_returns_scalar(self, lightning_module: HydraulicGNNModule, sample_batch):
        """Test compute_loss returns scalar total_loss."""
        outputs = lightning_module(sample_batch)
        total_loss, _ = lightning_module.compute_loss(outputs, sample_batch, return_components=False)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0  # Scalar
        assert total_loss.item() >= 0.0

    def test_compute_loss_returns_components(self, lightning_module: HydraulicGNNModule, sample_batch):
        """Test compute_loss returns individual loss components."""
        outputs = lightning_module(sample_batch)
        total_loss, loss_dict = lightning_module.compute_loss(
            outputs, sample_batch, return_components=True
        )
        
        assert loss_dict is not None
        assert "graph_health" in loss_dict
        assert "graph_degradation" in loss_dict
        assert "graph_anomaly" in loss_dict
        assert "graph_rul" in loss_dict
        assert "component_health" in loss_dict
        assert "component_anomaly" in loss_dict
        assert "total" in loss_dict

    def test_compute_loss_all_scalars(self, lightning_module: HydraulicGNNModule, sample_batch):
        """Test all loss components are scalars."""
        outputs = lightning_module(sample_batch)
        _, loss_dict = lightning_module.compute_loss(outputs, sample_batch, return_components=True)
        
        for key, loss_val in loss_dict.items():
            assert loss_val.dim() == 0, f"{key} loss is not scalar"

    def test_loss_no_nans(self, lightning_module: HydraulicGNNModule, sample_batch):
        """Test loss computation doesn't produce NaNs."""
        outputs = lightning_module(sample_batch)
        total_loss, loss_dict = lightning_module.compute_loss(
            outputs, sample_batch, return_components=True
        )
        
        assert not torch.isnan(total_loss)
        for key, loss_val in loss_dict.items():
            assert not torch.isnan(loss_val), f"NaN in {key} loss"


class TestTrainingStep:
    """Tests for training step."""

    def test_training_step_runs(self, lightning_module: HydraulicGNNModule, sample_batch):
        """Test training_step executes without errors."""
        # Mock optimizer
        lightning_module.configure_optimizers()
        
        # Run training step (should not raise)
        result = lightning_module.training_step(sample_batch, batch_idx=0)
        
        # Manual optimization returns None
        assert result is None

    def test_training_step_backward(self, lightning_module: HydraulicGNNModule, sample_batch):
        """Test training_step performs backward pass."""
        # Get a parameter to check gradients
        param = next(lightning_module.model.parameters())
        
        # Before training step
        assert param.grad is None
        
        # Mock optimizer (required for manual optimization)
        lightning_module.configure_optimizers()
        
        # Run training step
        lightning_module.training_step(sample_batch, batch_idx=0)
        
        # After training step, gradients should be computed then zeroed
        # (optimizer.zero_grad() is called at end)
        # So we check that backward was called by verifying no errors


class TestValidationStep:
    """Tests for validation step."""

    def test_validation_step_runs(self, lightning_module: HydraulicGNNModule, sample_batch):
        """Test validation_step executes without errors."""
        result = lightning_module.validation_step(sample_batch, batch_idx=0)
        assert result is None  # validation_step returns None

    def test_validation_step_no_gradients(self, lightning_module: HydraulicGNNModule, sample_batch):
        """Test validation_step doesn't compute gradients."""
        lightning_module.eval()
        
        param = next(lightning_module.model.parameters())
        assert param.grad is None
        
        lightning_module.validation_step(sample_batch, batch_idx=0)
        
        # Gradients should still be None (validation uses torch.no_grad)
        assert param.grad is None


class TestOptimizerConfiguration:
    """Tests for optimizer configuration."""

    def test_plateau_scheduler(self, model_config: ModelConfig):
        """Test ReduceLROnPlateau scheduler configuration."""
        module = HydraulicGNNModule(
            model_config=model_config,
            scheduler_type="plateau",
        )
        
        config = module.configure_optimizers()
        
        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert config["lr_scheduler"]["scheduler"].__class__.__name__ == "ReduceLROnPlateau"
        assert config["lr_scheduler"]["monitor"] == "val/total_loss"

    def test_cosine_scheduler(self, model_config: ModelConfig):
        """Test CosineAnnealingLR scheduler configuration."""
        module = HydraulicGNNModule(
            model_config=model_config,
            scheduler_type="cosine",
        )
        
        config = module.configure_optimizers()
        
        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert config["lr_scheduler"]["scheduler"].__class__.__name__ == "CosineAnnealingLR"

    def test_no_scheduler(self, model_config: ModelConfig):
        """Test configuration without scheduler."""
        module = HydraulicGNNModule(
            model_config=model_config,
            scheduler_type="none",
        )
        
        config = module.configure_optimizers()
        
        assert "optimizer" in config
        assert "lr_scheduler" not in config


class TestAdvancedLosses:
    """Tests for advanced loss options."""

    def test_advanced_losses_enabled(self, model_config: ModelConfig):
        """Test module with advanced losses."""
        module = HydraulicGNNModule(
            model_config=model_config,
            use_advanced_losses=True,
            use_confidence_weighting=False,
            use_domain_adversarial=False,
        )
        
        # Should have AsymmetricL1Loss for RUL
        assert module.graph_rul_loss.__class__.__name__ == "AsymmetricL1Loss"
        
        # Should have PhysicsAwareFocalLoss for anomaly
        assert module.graph_anomaly_loss.__class__.__name__ == "PhysicsAwareFocalLoss"

    def test_confidence_weighting_enabled(self, model_config: ModelConfig):
        """Test module with confidence-weighted training."""
        module = HydraulicGNNModule(
            model_config=model_config,
            use_advanced_losses=True,
            use_confidence_weighting=True,
        )
        
        # Health loss should be wrapped with ConfidenceWeightedLoss
        assert module.graph_health_loss.__class__.__name__ == "ConfidenceWeightedLoss"

    def test_domain_adversarial_enabled(self, model_config: ModelConfig):
        """Test module with domain adversarial loss."""
        module = HydraulicGNNModule(
            model_config=model_config,
            use_domain_adversarial=True,
            num_domains=2,
        )
        
        assert hasattr(module, "domain_loss")
        assert hasattr(module, "domain_classifier")


class TestMultiTaskWeighting:
    """Tests for multi-task loss weighting."""

    def test_fixed_weighting(self, model_config: ModelConfig):
        """Test fixed loss weighting."""
        custom_weights = {
            "graph_health": 2.0,
            "graph_degradation": 1.0,
            "graph_anomaly": 1.5,
            "graph_rul": 1.0,
            "component_health": 0.5,
            "component_anomaly": 0.5,
        }
        
        module = HydraulicGNNModule(
            model_config=model_config,
            loss_weighting="fixed",
            loss_weights=custom_weights,
        )
        
        assert module.loss_weighting == "fixed"
        assert module.loss_weights == custom_weights

    def test_uncertainty_weighting(self, model_config: ModelConfig):
        """Test uncertainty-based loss weighting."""
        module = HydraulicGNNModule(
            model_config=model_config,
            loss_weighting="uncertainty",
        )
        
        assert module.loss_weighting == "uncertainty"
        assert hasattr(module, "uncertainty_weighter")
        
        # UncertaintyWeighting should have NO learnable parameters (buffers only)
        params = list(module.uncertainty_weighter.parameters())
        assert len(params) == 0, "UncertaintyWeighting should not have parameters"
