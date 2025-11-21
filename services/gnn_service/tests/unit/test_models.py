"""Unit tests для GNN models.

Полное покрытие:
- UniversalTemporalGNN forward pass
- Custom layers (EdgeGATv2, ARMA-LSTM, Spectral)
- Attention mechanisms
- Model utilities
- Checkpoint compatibility

Pytest fixtures в conftest.py.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from pathlib import Path
import tempfile

from src.models import (
    UniversalTemporalGNN,
    EdgeConditionedGATv2Layer,
    ARMAAttentionLSTM,
    SpectralTemporalLayer,
    MultiHeadAttention,
    CrossTaskAttention,
    EdgeAwareAttention,
)
from src.models.utils import (
    initialize_model,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    model_summary,
)


# ==================== FIXTURES ====================

@pytest.fixture
def device():
    """Устройство для тестов."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_graph_data():
    """Sample PyG Data object."""
    # 5 nodes, 6 edges
    x = torch.randn(5, 12)  # [N=5, F=12]
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],  # Source nodes
        [1, 0, 2, 1, 3, 2]   # Target nodes
    ], dtype=torch.long)  # [2, E=6]
    edge_attr = torch.randn(6, 8)  # [E=6, F_edge=8]
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def sample_batch_data(sample_graph_data):
    """Sample batch of graphs."""
    # 3 graphs in batch
    graph1 = sample_graph_data
    graph2 = Data(
        x=torch.randn(4, 12),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_attr=torch.randn(3, 8)
    )
    graph3 = Data(
        x=torch.randn(6, 12),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
        edge_attr=torch.randn(5, 8)
    )
    
    batch = Batch.from_data_list([graph1, graph2, graph3])
    return batch


@pytest.fixture
def small_model():
    """Small model для быстрых тестов."""
    return UniversalTemporalGNN(
        in_channels=12,
        hidden_channels=32,  # Small for speed
        num_heads=4,
        num_gat_layers=2,
        lstm_hidden=64,
        lstm_layers=1,
        ar_order=2,
        ma_order=1,
        use_compile=False  # Disable для тестов
    )


# ==================== MODEL TESTS ====================

class TestUniversalTemporalGNN:
    """Tests для UniversalTemporalGNN."""
    
    def test_model_creation(self, small_model):
        """Создание модели."""
        assert isinstance(small_model, nn.Module)
        assert small_model.in_channels == 12
        assert small_model.hidden_channels == 32
        assert small_model.num_gat_layers == 2
    
    def test_forward_pass_single_graph(self, small_model, sample_graph_data):
        """Тест forward pass на одном графе."""
        small_model.eval()
        
        with torch.no_grad():
            health, degradation, anomaly = small_model(
                x=sample_graph_data.x,
                edge_index=sample_graph_data.edge_index,
                edge_attr=sample_graph_data.edge_attr,
                batch=None
            )
        
        # Check shapes
        assert health.shape == (1, 1)  # Single graph
        assert degradation.shape == (1, 1)
        assert anomaly.shape == (1, 9)  # 9 anomaly types
        
        # Check ranges
        assert 0 <= health.item() <= 1
        assert 0 <= degradation.item() <= 1
    
    def test_forward_pass_batch(self, small_model, sample_batch_data):
        """Тест forward pass на batch."""
        small_model.eval()
        
        with torch.no_grad():
            health, degradation, anomaly = small_model(
                x=sample_batch_data.x,
                edge_index=sample_batch_data.edge_index,
                edge_attr=sample_batch_data.edge_attr,
                batch=sample_batch_data.batch
            )
        
        batch_size = sample_batch_data.num_graphs  # 3 graphs
        
        assert health.shape == (batch_size, 1)
        assert degradation.shape == (batch_size, 1)
        assert anomaly.shape == (batch_size, 9)
    
    def test_return_attention_weights(self, small_model, sample_graph_data):
        """Тест return attention weights."""
        small_model.eval()
        
        with torch.no_grad():
            health, degradation, anomaly, attn_weights = small_model(
                x=sample_graph_data.x,
                edge_index=sample_graph_data.edge_index,
                edge_attr=sample_graph_data.edge_attr,
                batch=None,
                return_attention=True
            )
        
        # Should have attention для каждого GAT layer
        assert len(attn_weights) == small_model.num_gat_layers
    
    def test_predict_method(self, small_model, sample_graph_data):
        """Тест inference mode predict()."""
        result = small_model.predict(
            x=sample_graph_data.x,
            edge_index=sample_graph_data.edge_index,
            edge_attr=sample_graph_data.edge_attr
        )
        
        assert "health" in result
        assert "degradation" in result
        assert "anomaly_logits" in result
        assert "anomaly_probs" in result
        
        # Probabilities sum to 1
        probs_sum = result["anomaly_probs"].sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)
    
    def test_get_model_config(self, small_model):
        """Тест get_model_config()."""
        config = small_model.get_model_config()
        
        assert config["in_channels"] == 12
        assert config["hidden_channels"] == 32
        assert config["num_heads"] == 4
        assert "use_compile" in config


# ==================== LAYER TESTS ====================

class TestEdgeConditionedGATv2Layer:
    """Tests для EdgeConditionedGATv2Layer."""
    
    def test_layer_creation(self):
        """Создание layer."""
        layer = EdgeConditionedGATv2Layer(
            in_channels=32,
            out_channels=8,
            heads=4,
            edge_dim=8
        )
        
        assert layer.in_channels == 32
        assert layer.out_channels == 8
        assert layer.heads == 4
    
    def test_forward_without_edge_features(self):
        """Тест без edge features."""
        layer = EdgeConditionedGATv2Layer(
            in_channels=32,
            out_channels=8,
            heads=4,
            edge_dim=None  # No edge features
        )
        
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        out = layer(x, edge_index, edge_attr=None)
        
        # Concat: out_channels * heads
        assert out.shape == (5, 8 * 4)
    
    def test_forward_with_edge_features(self):
        """Тест с edge features."""
        layer = EdgeConditionedGATv2Layer(
            in_channels=32,
            out_channels=8,
            heads=4,
            edge_dim=8
        )
        
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, 8)
        
        out = layer(x, edge_index, edge_attr=edge_attr)
        
        assert out.shape == (5, 8 * 4)
    
    def test_return_attention_weights(self):
        """Тест return attention."""
        layer = EdgeConditionedGATv2Layer(
            in_channels=32,
            out_channels=8,
            heads=4,
            edge_dim=8
        )
        
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, 8)
        
        out, (edge_idx, alpha) = layer(
            x, edge_index, edge_attr=edge_attr,
            return_attention_weights=True
        )
        
        assert out.shape == (5, 8 * 4)
        assert alpha.shape[0] == 3  # 3 edges
        assert alpha.shape[1] == 4  # 4 heads


class TestARMAAttentionLSTM:
    """Tests для ARMAAttentionLSTM."""
    
    def test_lstm_creation(self):
        """Создание LSTM."""
        lstm = ARMAAttentionLSTM(
            input_dim=32,
            hidden_dim=64,
            num_layers=2,
            ar_order=3,
            ma_order=2
        )
        
        assert lstm.input_dim == 32
        assert lstm.hidden_dim == 64
        assert lstm.ar_order == 3
        assert lstm.ma_order == 2
    
    def test_forward_pass(self):
        """Тест forward pass."""
        lstm = ARMAAttentionLSTM(
            input_dim=32,
            hidden_dim=64,
            num_layers=1,
            ar_order=2,
            ma_order=1
        )
        
        x = torch.randn(2, 10, 32)  # [B=2, T=10, F=32]
        
        out, (h_n, c_n) = lstm(x)
        
        assert out.shape == (2, 10, 64)  # [B, T, hidden]
        assert h_n.shape == (1, 2, 64)  # [num_layers, B, hidden]
        assert c_n.shape == (1, 2, 64)
    
    def test_arma_modulation_computation(self):
        """Тест ARMA modulation matrix."""
        lstm = ARMAAttentionLSTM(
            input_dim=32,
            hidden_dim=64,
            ar_order=3,
            ma_order=2
        )
        
        modulation = lstm._compute_arma_modulation(seq_len=10)
        
        assert modulation.shape == (10, 10)
        assert torch.all(modulation > 0)  # exp() always positive


class TestSpectralTemporalLayer:
    """Tests для SpectralTemporalLayer."""
    
    def test_layer_creation(self):
        """Создание layer."""
        layer = SpectralTemporalLayer(hidden_dim=64, num_frequencies=16)
        
        assert layer.hidden_dim == 64
        assert layer.num_frequencies == 16
    
    def test_forward_pass(self):
        """Тест FFT processing."""
        layer = SpectralTemporalLayer(hidden_dim=64, num_frequencies=16)
        
        x = torch.randn(2, 20, 64)  # [B=2, T=20, H=64]
        out = layer(x)
        
        assert out.shape == x.shape  # Same shape
        assert not torch.allclose(out, x)  # Modified


# ==================== ATTENTION TESTS ====================

class TestMultiHeadAttention:
    """Tests для MultiHeadAttention."""
    
    def test_attention_creation(self):
        """Создание attention."""
        attn = MultiHeadAttention(embed_dim=128, num_heads=8)
        
        assert attn.embed_dim == 128
        assert attn.num_heads == 8
        assert attn.head_dim == 16  # 128 / 8
    
    def test_forward_pass(self):
        """Тест attention forward."""
        attn = MultiHeadAttention(embed_dim=128, num_heads=8)
        
        query = torch.randn(2, 10, 128)  # [B=2, T_q=10, E=128]
        key = torch.randn(2, 10, 128)
        value = torch.randn(2, 10, 128)
        
        out = attn(query, key, value)
        
        assert out.shape == query.shape
    
    def test_return_attention_weights(self):
        """Тест return attention weights."""
        attn = MultiHeadAttention(embed_dim=128, num_heads=8)
        
        query = torch.randn(2, 10, 128)
        key = torch.randn(2, 10, 128)
        value = torch.randn(2, 10, 128)
        
        out, attn_weights = attn(query, key, value, return_attention=True)
        
        assert out.shape == (2, 10, 128)
        assert attn_weights.shape == (2, 8, 10, 10)  # [B, heads, T_q, T_k]


class TestCrossTaskAttention:
    """Tests для CrossTaskAttention."""
    
    def test_attention_creation(self):
        """Создание cross-task attention."""
        attn = CrossTaskAttention(hidden_dim=64, num_tasks=3, num_heads=4)
        
        assert attn.hidden_dim == 64
        assert attn.num_tasks == 3
    
    def test_forward_pass(self):
        """Тест task representation."""
        attn = CrossTaskAttention(hidden_dim=64, num_tasks=3, num_heads=4)
        
        x = torch.randn(8, 64)  # [B=8, H=64]
        task_repr = attn(x)
        
        # Should return [num_tasks, B, H]
        assert task_repr.shape == (3, 8, 64)


# ==================== UTILS TESTS ====================

class TestModelUtils:
    """Tests для model utilities."""
    
    def test_count_parameters(self, small_model):
        """Тест parameter counting."""
        total = count_parameters(small_model, trainable_only=False)
        trainable = count_parameters(small_model, trainable_only=True)
        
        assert total > 0
        assert trainable > 0
        assert trainable <= total
    
    def test_model_summary(self, small_model):
        """Тест model summary."""
        summary = model_summary(small_model)
        
        assert "total_params" in summary
        assert "trainable_params" in summary
        assert "memory_mb" in summary
        assert "model_type" in summary
        assert summary["model_type"] == "UniversalTemporalGNN"
    
    def test_checkpoint_save_load(self, small_model):
        """Тест checkpoint save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.ckpt"
            
            # Save
            optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
            save_checkpoint(
                model=small_model,
                optimizer=optimizer,
                epoch=10,
                loss=0.123,
                metrics={"health_mae": 0.045},
                save_path=checkpoint_path,
                model_config=small_model.get_model_config()
            )
            
            assert checkpoint_path.exists()
            
            # Load
            new_model = UniversalTemporalGNN(
                in_channels=12,
                hidden_channels=32,
                num_heads=4,
                num_gat_layers=2,
                lstm_hidden=64,
                lstm_layers=1,
                use_compile=False
            )
            
            checkpoint = load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=new_model
            )
            
            assert checkpoint["epoch"] == 10
            assert checkpoint["loss"] == 0.123
            assert "health_mae" in checkpoint["metrics"]
    
    def test_initialize_model(self, small_model):
        """Тест weight initialization."""
        # Re-initialize
        initialized = initialize_model(small_model, method="xavier_uniform")
        
        # Check that parameters exist
        params = list(initialized.parameters())
        assert len(params) > 0
        
        # Check that at least one param не zero
        has_nonzero = any(torch.any(p != 0) for p in params)
        assert has_nonzero


# ==================== INTEGRATION TESTS ====================

class TestModelIntegration:
    """Integration tests с PyG Data."""
    
    def test_model_with_pyg_batch(self, small_model):
        """Тест с PyG Batch."""
        # Create multiple graphs
        graphs = [
            Data(
                x=torch.randn(4, 12),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                edge_attr=torch.randn(3, 8)
            )
            for _ in range(5)
        ]
        
        batch = Batch.from_data_list(graphs)
        
        small_model.eval()
        with torch.no_grad():
            health, degradation, anomaly = small_model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch
            )
        
        assert health.shape == (5, 1)  # 5 graphs
        assert degradation.shape == (5, 1)
        assert anomaly.shape == (5, 9)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_model_on_gpu(self, small_model, sample_graph_data):
        """Тест на GPU."""
        device = torch.device("cuda")
        
        model = small_model.to(device)
        data = sample_graph_data.to(device)
        
        model.eval()
        with torch.no_grad():
            health, degradation, anomaly = model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr
            )
        
        assert health.device.type == "cuda"
        assert degradation.device.type == "cuda"
        assert anomaly.device.type == "cuda"


# ==================== GRADIENT TESTS ====================

class TestModelGradients:
    """Tests для gradient flow."""
    
    def test_backward_pass(self, small_model, sample_graph_data):
        """Тест backward pass."""
        small_model.train()
        
        health, degradation, anomaly = small_model(
            x=sample_graph_data.x,
            edge_index=sample_graph_data.edge_index,
            edge_attr=sample_graph_data.edge_attr
        )
        
        # Dummy loss
        loss = health.sum() + degradation.sum() + anomaly.sum()
        loss.backward()
        
        # Check gradients exist
        for param in small_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_no_gradient_in_eval_mode(self, small_model, sample_graph_data):
        """Тест no gradients в eval mode."""
        small_model.eval()
        
        with torch.no_grad():
            health, degradation, anomaly = small_model(
                x=sample_graph_data.x,
                edge_index=sample_graph_data.edge_index,
                edge_attr=sample_graph_data.edge_attr
            )
        
        # Should not require gradients
        assert not health.requires_grad
        assert not degradation.requires_grad
        assert not anomaly.requires_grad
