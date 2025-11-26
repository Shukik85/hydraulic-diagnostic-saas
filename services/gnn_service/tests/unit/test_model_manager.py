"""Unit tests for ModelManager.

Tests:
- Model loading from checkpoint
- Caching behavior
- Device management
- Error handling
- Warmup
- Thread safety
"""

import pytest
import torch
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.inference import ModelManager, ModelConfig
from src.models import UniversalTemporalGNN


# ==================== FIXTURES ====================

@pytest.fixture
def mock_checkpoint(tmp_path):
    """Create mock checkpoint file."""
    checkpoint_path = tmp_path / "model.ckpt"
    
    # Create minimal checkpoint
    checkpoint = {
        "model_config": {
            "in_channels": 34,
            "hidden_channels": 128,
            "num_heads": 8,
            "num_gat_layers": 3,
            "lstm_hidden": 256,
            "lstm_layers": 2
        },
        "model_state_dict": {
            # Minimal state dict (will be mocked)
        },
        "epoch": 100,
        "best_val_loss": 0.0234
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def mock_model():
    """Create mock model."""
    model = Mock(spec=UniversalTemporalGNN)
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)
    model.parameters = Mock(return_value=[torch.randn(10, 10)])  # Fake parameters
    model.training = False
    model.forward = Mock(return_value=(torch.randn(1, 1), torch.randn(1, 1), torch.randn(1, 9)))
    return model


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset ModelManager singleton between tests."""
    ModelManager._instance = None
    yield
    ModelManager._instance = None


# ==================== TESTS ====================

class TestModelManagerBasics:
    """Basic ModelManager functionality."""
    
    def test_singleton_pattern(self):
        """Test ModelManager is singleton."""
        manager1 = ModelManager()
        manager2 = ModelManager()
        
        assert manager1 is manager2
    
    def test_initialization(self):
        """Test manager initializes correctly."""
        manager = ModelManager()
        
        assert hasattr(manager, "_models")
        assert hasattr(manager, "_configs")
        assert isinstance(manager._models, dict)
        assert isinstance(manager._configs, dict)
        assert len(manager._models) == 0


class TestModelLoading:
    """Test model loading functionality."""
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_load_model_success(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        mock_checkpoint,
        mock_model
    ):
        """Test successful model loading."""
        # Setup mocks
        mock_load_checkpoint.return_value = {
            "model_config": {
                "in_channels": 34,
                "hidden_channels": 128,
                "num_heads": 8,
                "num_gat_layers": 3,
                "lstm_hidden": 256,
                "lstm_layers": 2
            },
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        manager = ModelManager()
        model = manager.load_model(
            model_path=mock_checkpoint,
            device="cpu",
            use_compile=False
        )
        
        # Assertions
        assert model is not None
        assert model is mock_model
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called()
    
    def test_load_model_invalid_path(self):
        """Test loading with invalid path."""
        manager = ModelManager()
        
        with pytest.raises(FileNotFoundError):
            manager.load_model(
                model_path="/invalid/path/model.ckpt",
                device="cpu"
            )
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_load_model_device_auto_cpu(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        mock_checkpoint,
        mock_model
    ):
        """Test device='auto' falls back to CPU."""
        mock_load_checkpoint.return_value = {
            "model_config": {},
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        with patch('torch.cuda.is_available', return_value=False):
            manager = ModelManager()
            model = manager.load_model(
                model_path=mock_checkpoint,
                device="auto",
                use_compile=False
            )
            
            # Should use CPU
            mock_model.to.assert_called_with("cpu")
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_load_model_device_auto_cuda(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        mock_checkpoint,
        mock_model
    ):
        """Test device='auto' uses CUDA when available."""
        mock_load_checkpoint.return_value = {
            "model_config": {},
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        with patch('torch.cuda.is_available', return_value=True):
            manager = ModelManager()
            model = manager.load_model(
                model_path=mock_checkpoint,
                device="auto",
                use_compile=False
            )
            
            # Should use CUDA
            mock_model.to.assert_called_with("cuda")


class TestModelCaching:
    """Test model caching behavior."""
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_model_cached_between_calls(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        mock_checkpoint,
        mock_model
    ):
        """Test model is cached and reused."""
        mock_load_checkpoint.return_value = {
            "model_config": {},
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        manager = ModelManager()
        
        # First load
        model1 = manager.load_model(
            model_path=mock_checkpoint,
            device="cpu",
            use_compile=False
        )
        
        # Second load (should use cache)
        model2 = manager.load_model(
            model_path=mock_checkpoint,
            device="cpu",
            use_compile=False
        )
        
        # Same instance
        assert model1 is model2
        
        # load_checkpoint called only once
        assert mock_load_checkpoint.call_count == 1
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_force_reload_bypasses_cache(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        mock_checkpoint,
        mock_model
    ):
        """Test force_reload bypasses cache."""
        mock_load_checkpoint.return_value = {
            "model_config": {},
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        manager = ModelManager()
        
        # First load
        manager.load_model(
            model_path=mock_checkpoint,
            device="cpu",
            use_compile=False
        )
        
        # Force reload
        manager.load_model(
            model_path=mock_checkpoint,
            device="cpu",
            use_compile=False,
            force_reload=True
        )
        
        # load_checkpoint called twice
        assert mock_load_checkpoint.call_count == 2
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_get_model_returns_cached(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        mock_checkpoint,
        mock_model
    ):
        """Test get_model returns cached model."""
        mock_load_checkpoint.return_value = {
            "model_config": {},
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        manager = ModelManager()
        
        # Load model
        loaded_model = manager.load_model(
            model_path=mock_checkpoint,
            device="cpu",
            use_compile=False
        )
        
        # Get from cache
        cached_model = manager.get_model(mock_checkpoint)
        
        assert cached_model is loaded_model
    
    def test_get_model_returns_none_if_not_loaded(self, mock_checkpoint):
        """Test get_model returns None if not loaded."""
        manager = ModelManager()
        
        model = manager.get_model(mock_checkpoint)
        
        assert model is None
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_clear_cache_specific_model(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        mock_checkpoint,
        mock_model
    ):
        """Test clearing specific model from cache."""
        mock_load_checkpoint.return_value = {
            "model_config": {},
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        manager = ModelManager()
        
        # Load model
        manager.load_model(
            model_path=mock_checkpoint,
            device="cpu",
            use_compile=False
        )
        
        # Clear
        manager.clear_cache(mock_checkpoint)
        
        # Should be None now
        assert manager.get_model(mock_checkpoint) is None
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_clear_cache_all_models(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        tmp_path,
        mock_model
    ):
        """Test clearing all models from cache."""
        mock_load_checkpoint.return_value = {
            "model_config": {},
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        # Create two checkpoints
        ckpt1 = tmp_path / "model1.ckpt"
        ckpt2 = tmp_path / "model2.ckpt"
        torch.save({}, ckpt1)
        torch.save({}, ckpt2)
        
        manager = ModelManager()
        
        # Load both
        manager.load_model(model_path=ckpt1, device="cpu", use_compile=False)
        manager.load_model(model_path=ckpt2, device="cpu", use_compile=False)
        
        # Clear all
        manager.clear_cache()
        
        # Both should be None
        assert manager.get_model(ckpt1) is None
        assert manager.get_model(ckpt2) is None


class TestModelInfo:
    """Test model info and statistics."""
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_get_model_info(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        mock_checkpoint,
        mock_model
    ):
        """Test get_model_info returns correct info."""
        mock_load_checkpoint.return_value = {
            "model_config": {},
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        manager = ModelManager()
        manager.load_model(
            model_path=mock_checkpoint,
            device="cpu",
            use_compile=False
        )
        
        info = manager.get_model_info(mock_checkpoint)
        
        assert info is not None
        assert "path" in info
        assert "device" in info
        assert "num_parameters" in info
        assert "compiled" in info
    
    def test_get_model_info_not_loaded(self, mock_checkpoint):
        """Test get_model_info returns None if not loaded."""
        manager = ModelManager()
        
        info = manager.get_model_info(mock_checkpoint)
        
        assert info is None
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_list_cached_models(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        tmp_path,
        mock_model
    ):
        """Test listing cached models."""
        mock_load_checkpoint.return_value = {
            "model_config": {},
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        ckpt1 = tmp_path / "model1.ckpt"
        ckpt2 = tmp_path / "model2.ckpt"
        torch.save({}, ckpt1)
        torch.save({}, ckpt2)
        
        manager = ModelManager()
        manager.load_model(model_path=ckpt1, device="cpu", use_compile=False)
        manager.load_model(model_path=ckpt2, device="cpu", use_compile=False)
        
        cached = manager.list_cached_models()
        
        assert len(cached) == 2
        assert str(ckpt1.resolve()) in cached
        assert str(ckpt2.resolve()) in cached


class TestThreadSafety:
    """Test thread safety of ModelManager."""
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_concurrent_load_same_model(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        mock_checkpoint,
        mock_model
    ):
        """Test concurrent loading of same model is safe."""
        mock_load_checkpoint.return_value = {
            "model_config": {},
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        manager = ModelManager()
        results = []
        errors = []
        
        def load_model_thread():
            try:
                model = manager.load_model(
                    model_path=mock_checkpoint,
                    device="cpu",
                    use_compile=False
                )
                results.append(model)
            except Exception as e:
                errors.append(e)
        
        # Create 10 threads
        threads = [threading.Thread(target=load_model_thread) for _ in range(10)]
        
        # Start all
        for t in threads:
            t.start()
        
        # Wait all
        for t in threads:
            t.join()
        
        # No errors
        assert len(errors) == 0
        
        # All threads got model
        assert len(results) == 10
        
        # All same instance
        assert all(model is results[0] for model in results)


class TestWarmup:
    """Test model warmup functionality."""
    
    @patch('src.inference.model_manager.UniversalTemporalGNN')
    @patch('src.inference.model_manager.load_checkpoint')
    def test_warmup_runs_forward_passes(
        self,
        mock_load_checkpoint,
        mock_gnn_class,
        mock_checkpoint,
        mock_model
    ):
        """Test warmup runs forward passes."""
        mock_load_checkpoint.return_value = {
            "model_config": {},
            "model_state_dict": {}
        }
        mock_gnn_class.return_value = mock_model
        
        manager = ModelManager()
        manager.load_model(
            model_path=mock_checkpoint,
            device="cpu",
            use_compile=False
        )
        
        # Warmup
        manager.warmup(mock_checkpoint, batch_size=2)
        
        # Model forward called (3 warmup iterations)
        assert mock_model.forward.call_count >= 3
    
    def test_warmup_model_not_loaded_raises(self, mock_checkpoint):
        """Test warmup raises if model not loaded."""
        manager = ModelManager()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            manager.warmup(mock_checkpoint)
