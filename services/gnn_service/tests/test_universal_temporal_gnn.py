"""Unit tests for UniversalTemporalGNNv2.

Test coverage:
- Configuration validation
- Forward pass correctness
- Output shapes (Phase 2: 6 tasks)
- Edge cases
- Batch processing

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from models import ModelConfig, UniversalTemporalGNNv2


class TestModelConfig:
    """Test ModelConfig validation."""
    
    def test_valid_config(self):
        """Valid configuration should not raise."""
        config = ModelConfig(
            node_features=34,
            edge_features=14,
            gat_hidden_dim=128,
            gat_num_layers=3
        )
        assert config.node_features == 34
        assert config.gat_num_layers == 3
        assert config.graph_anomaly_classes == 9
        assert config.component_anomaly_classes == 9
    
    def test_invalid_dropout(self):
        """Invalid dropout should raise ValueError."""
        with pytest.raises(ValueError, match="gat_dropout must be in"):
            ModelConfig(gat_dropout=1.5)
        
        with pytest.raises(ValueError, match="gat_dropout must be in"):
            ModelConfig(gat_dropout=-0.1)
    
    def test_invalid_dimensions(self):
        """Invalid dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="node_features must be > 0"):
            ModelConfig(node_features=0)
        
        with pytest.raises(ValueError, match="gat_hidden_dim must be > 0"):
            ModelConfig(gat_hidden_dim=-10)
    
    def test_invalid_layer_counts(self):
        """Invalid layer counts should raise ValueError."""
        with pytest.raises(ValueError, match="gat_num_layers must be >= 1"):
            ModelConfig(gat_num_layers=0)
        
        with pytest.raises(ValueError, match="lstm_num_layers must be >= 1"):
            ModelConfig(lstm_num_layers=0)
    
    def test_invalid_num_classes(self):
        """Invalid class counts should raise ValueError (Phase 2)."""
        with pytest.raises(ValueError, match="graph_anomaly_classes must be >= 2"):
            ModelConfig(graph_anomaly_classes=1)
        
        with pytest.raises(ValueError, match="component_anomaly_classes must be >= 2"):
            ModelConfig(component_anomaly_classes=0)
    
    def test_virtual_node_validation(self):
        """Virtual node dim must be positive if enabled."""
        with pytest.raises(ValueError, match="virtual_node_dim must be > 0"):
            ModelConfig(use_virtual_nodes=True, virtual_node_dim=0)


class TestUniversalTemporalGNNv2:
    """Test UniversalTemporalGNNv2 model (Phase 2)."""
    
    @pytest.fixture
    def config(self) -> ModelConfig:
        """Default test configuration."""
        return ModelConfig(
            node_features=34,
            edge_features=14,
            gat_hidden_dim=64,  # Smaller for tests
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
    def sample_graph_3_nodes(self) -> Data:
        """Sample graph with 3 nodes."""
        return Data(
            x=torch.randn(3, 34),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
            edge_attr=torch.randn(4, 14),
        )
    
    @pytest.fixture
    def sample_graph_10_nodes(self) -> Data:
        """Sample graph with 10 nodes."""
        num_nodes = 10
        num_edges = 20
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        return Data(
            x=torch.randn(num_nodes, 34),
            edge_index=edge_index,
            edge_attr=torch.randn(num_edges, 14),
        )
    
    def test_model_initialization(self, config: ModelConfig):
        """Model should initialize without errors."""
        model = UniversalTemporalGNNv2(config)
        assert model.config == config
        assert model.gat_output_dim > 0
    
    def test_forward_single_graph(self, model: UniversalTemporalGNNv2, sample_graph_3_nodes: Data):
        """Forward pass on single graph should work (Phase 2)."""
        model.eval()
        with torch.no_grad():
            outputs = model(sample_graph_3_nodes, temporal=False)
        
        # Check nested structure
        assert 'component' in outputs
        assert 'graph' in outputs
        
        # Component-level
        assert 'health' in outputs['component']
        assert 'anomaly' in outputs['component']
        assert outputs['component']['health'].shape == (3, 1)  # [N, 1]
        assert outputs['component']['anomaly'].shape == (3, 9)  # [N, 9]
        
        # Graph-level
        assert 'health' in outputs['graph']
        assert 'degradation' in outputs['graph']
        assert 'anomaly' in outputs['graph']
        assert 'rul' in outputs['graph']
        assert outputs['graph']['health'].shape == (1, 1)  # [B, 1]
        assert outputs['graph']['degradation'].shape == (1, 1)  # [B, 1]
        assert outputs['graph']['anomaly'].shape == (1, 9)  # [B, 9]
        assert outputs['graph']['rul'].shape == (1, 1)  # [B, 1]
    
    def test_forward_with_attention(self, model: UniversalTemporalGNNv2, sample_graph_3_nodes: Data):
        """Forward pass should return attention weights if requested."""
        model.eval()
        with torch.no_grad():
            outputs = model(sample_graph_3_nodes, temporal=False, return_attention=True)
        
        assert 'attention_weights' in outputs
        assert isinstance(outputs['attention_weights'], dict)
        assert len(outputs['attention_weights']) == 2  # 2 GAT layers
    
    def test_forward_temporal_last_timestep(self, model: UniversalTemporalGNNv2, sample_graph_3_nodes: Data):
        """Temporal forward should return last timestep predictions."""
        # Create sequence of 5 timesteps
        sequence = [sample_graph_3_nodes for _ in range(5)]
        
        model.eval()
        with torch.no_grad():
            outputs = model(sequence, temporal=True)
        
        # Check nested structure
        assert 'component' in outputs
        assert 'graph' in outputs
        
        # Component-level (last timestep)
        assert outputs['component']['health'].shape == (3, 1)
        assert outputs['component']['anomaly'].shape == (3, 9)
        
        # Graph-level (from LSTM)
        assert outputs['graph']['health'].shape == (1, 1)
        assert outputs['graph']['degradation'].shape == (1, 1)
        assert outputs['graph']['anomaly'].shape == (1, 9)
        assert outputs['graph']['rul'].shape == (1, 1)
    
    def test_forward_temporal_all_timesteps(self, model: UniversalTemporalGNNv2, sample_graph_3_nodes: Data):
        """Temporal forward should return all timesteps if requested."""
        sequence = [sample_graph_3_nodes for _ in range(5)]
        
        model.eval()
        with torch.no_grad():
            outputs = model(sequence, temporal=True, return_all_timesteps=True)
        
        assert 'component_seq' in outputs
        assert len(outputs['component_seq']) == 5  # 5 timesteps
        assert outputs['component_seq'][0]['health'].shape == (3, 1)
        assert outputs['component_seq'][0]['anomaly'].shape == (3, 9)
    
    def test_forward_temporal_all_attention(self, model: UniversalTemporalGNNv2, sample_graph_3_nodes: Data):
        """Should return attention for all timesteps if requested."""
        sequence = [sample_graph_3_nodes for _ in range(3)]
        
        model.eval()
        with torch.no_grad():
            outputs = model(
                sequence, 
                temporal=True, 
                return_all_timesteps=True,
                return_attention=True
            )
        
        assert 'attention_seq' in outputs
        assert len(outputs['attention_seq']) == 3  # 3 timesteps
    
    def test_different_graph_sizes(self, model: UniversalTemporalGNNv2):
        """Model should handle different graph sizes."""
        sizes = [3, 7, 10]
        
        model.eval()
        for size in sizes:
            graph = Data(
                x=torch.randn(size, 34),
                edge_index=torch.randint(0, size, (2, size * 2)),
                edge_attr=torch.randn(size * 2, 14)
            )
            
            with torch.no_grad():
                outputs = model(graph, temporal=False)
            
            assert outputs['component']['health'].shape[0] == size
            assert outputs['graph']['health'].shape == (1, 1)
    
    def test_batch_processing(self, model: UniversalTemporalGNNv2):
        """Model should handle batched graphs."""
        graphs = [
            Data(
                x=torch.randn(3, 34),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_attr=torch.randn(2, 14)
            )
            for _ in range(4)
        ]
        
        batch = Batch.from_data_list(graphs)
        
        model.eval()
        with torch.no_grad():
            outputs = model(batch, temporal=False)
        
        # Component-level: 4 graphs * 3 nodes = 12 total nodes
        assert outputs['component']['health'].shape[0] == 12
        assert outputs['component']['anomaly'].shape[0] == 12
        
        # Graph-level: 4 graphs
        assert outputs['graph']['health'].shape == (4, 1)
        assert outputs['graph']['degradation'].shape == (4, 1)
        assert outputs['graph']['anomaly'].shape == (4, 9)
        assert outputs['graph']['rul'].shape == (4, 1)
    
    def test_size_embedding_small_graph(self, model: UniversalTemporalGNNv2):
        """Model should handle small graphs."""
        graph = Data(
            x=torch.randn(3, 34),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_attr=torch.randn(2, 14)
        )
        
        model.eval()
        with torch.no_grad():
            outputs = model(graph, temporal=False)
        
        assert outputs['graph']['health'].shape == (1, 1)
        assert outputs['graph']['rul'].shape == (1, 1)
    
    def test_size_embedding_large_graph(self, model: UniversalTemporalGNNv2):
        """Model should handle large graphs (1000+ nodes)."""
        num_nodes = 1000
        graph = Data(
            x=torch.randn(num_nodes, 34),
            edge_index=torch.randint(0, num_nodes, (2, num_nodes * 2)),
            edge_attr=torch.randn(num_nodes * 2, 14)
        )
        
        model.eval()
        with torch.no_grad():
            outputs = model(graph, temporal=False)
        
        assert outputs['component']['health'].shape[0] == num_nodes
        assert outputs['graph']['health'].shape == (1, 1)
    
    def test_size_embedding_extra_large_graph(self, model: UniversalTemporalGNNv2):
        """Model should handle extra large graphs (10000+ nodes)."""
        num_nodes = 15000
        graph = Data(
            x=torch.randn(num_nodes, 34),
            edge_index=torch.randint(0, num_nodes, (2, num_nodes)),
            edge_attr=torch.randn(num_nodes, 14)
        )
        
        model.eval()
        with torch.no_grad():
            outputs = model(graph, temporal=False)
        
        assert outputs['component']['health'].shape[0] == num_nodes
        assert outputs['graph']['health'].shape == (1, 1)
    
    def test_no_attention_returns_empty_dict(self, model: UniversalTemporalGNNv2, sample_graph_3_nodes: Data):
        """Should not return attention_weights key when not requested."""
        model.eval()
        with torch.no_grad():
            outputs = model(sample_graph_3_nodes, temporal=False, return_attention=False)
        
        # Should not have attention_weights key
        assert 'attention_weights' not in outputs
    
    def test_bidirectional_lstm(self):
        """Model should work with bidirectional LSTM."""
        config = ModelConfig(
            node_features=34,
            edge_features=14,
            lstm_bidirectional=True
        )
        model = UniversalTemporalGNNv2(config)
        
        sequence = [
            Data(
                x=torch.randn(3, 34),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_attr=torch.randn(2, 14)
            )
            for _ in range(5)
        ]
        
        model.eval()
        with torch.no_grad():
            outputs = model(sequence, temporal=True)
        
        assert outputs['graph']['health'].shape == (1, 1)
        assert outputs['graph']['rul'].shape == (1, 1)
    
    def test_training_mode(self, model: UniversalTemporalGNNv2, sample_graph_3_nodes: Data):
        """Forward pass should work in training mode."""
        model.train()
        
        outputs = model(sample_graph_3_nodes, temporal=False)
        
        # Should enable dropout
        assert model.training
        assert outputs['component']['health'].shape == (3, 1)
        assert outputs['graph']['health'].shape == (1, 1)
    
    def test_gradient_flow(self, model: UniversalTemporalGNNv2, sample_graph_3_nodes: Data):
        """Gradients should flow correctly (single mode)."""
        model.train()
        
        outputs = model(sample_graph_3_nodes, temporal=False)
        
        # Sum all outputs for backward
        loss = (
            outputs['component']['health'].sum() + 
            outputs['component']['anomaly'].sum() +
            outputs['graph']['health'].sum() + 
            outputs['graph']['degradation'].sum() +
            outputs['graph']['anomaly'].sum() +
            outputs['graph']['rul'].sum()
        )
        loss.backward()
        
        # Check that gradients exist (excluding LSTM which is not used in single mode)
        for name, param in model.named_parameters():
            if param.requires_grad and not name.startswith('lstm.'):
                assert param.grad is not None, f"No gradient for {name}"
