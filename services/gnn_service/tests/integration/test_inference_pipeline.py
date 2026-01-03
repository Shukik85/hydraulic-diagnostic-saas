"""Integration tests for inference pipeline with Phase 2 architecture.

Tests:
- Single graph inference
- Batch inference
- Temporal sequence inference
- Response formatting
- Error handling
- Performance benchmarks

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import time

import pytest
import torch
from torch_geometric.data import Batch, Data

from models import ModelConfig, UniversalTemporalGNNv2


@pytest.mark.integration
class TestInferencePipeline:
    """Test inference pipeline (Phase 2)."""

    @pytest.fixture
    def config(self) -> ModelConfig:
        """Model configuration for inference."""
        return ModelConfig(
            node_features=34,
            edge_features=14,
            gat_hidden_dim=128,
            gat_num_layers=3,
            lstm_hidden_dim=256,
            lstm_num_layers=2,
            graph_anomaly_classes=9,
            component_anomaly_classes=9,
        )

    @pytest.fixture
    def model(self, config: ModelConfig) -> UniversalTemporalGNNv2:
        """Model in eval mode."""
        model = UniversalTemporalGNNv2(config)
        model.eval()
        return model

    @pytest.fixture
    def single_graph(self) -> Data:
        """Single graph for inference."""
        return Data(
            x=torch.randn(10, 34),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.randn(20, 14),
        )

    @pytest.fixture
    def batch_graphs(self) -> Batch:
        """Batch of graphs for inference."""
        graphs = [
            Data(
                x=torch.randn(10, 34),
                edge_index=torch.randint(0, 10, (2, 20)),
                edge_attr=torch.randn(20, 14),
            )
            for _ in range(8)
        ]
        return Batch.from_data_list(graphs)

    @pytest.fixture
    def temporal_sequence(self, single_graph: Data) -> list[Data]:
        """Temporal sequence for inference."""
        return [single_graph for _ in range(5)]

    def test_single_graph_inference(self, model, single_graph):
        """Test inference on single graph."""
        with torch.no_grad():
            outputs = model(single_graph, temporal=False)
        
        # Validate Phase 2 structure
        assert 'component' in outputs
        assert 'graph' in outputs
        
        # Component-level outputs
        assert outputs['component']['health'].shape == (10, 1)
        assert outputs['component']['anomaly'].shape == (10, 9)
        
        # Graph-level outputs
        assert outputs['graph']['health'].shape == (1, 1)
        assert outputs['graph']['degradation'].shape == (1, 1)
        assert outputs['graph']['anomaly'].shape == (1, 9)
        assert outputs['graph']['rul'].shape == (1, 1)
        
        # All outputs should be valid
        for level in outputs.values():
            for tensor in level.values():
                assert not torch.isnan(tensor).any()

    def test_batch_inference(self, model, batch_graphs):
        """Test inference on batch of graphs."""
        with torch.no_grad():
            outputs = model(batch_graphs, temporal=False)
        
        # Component-level: 8 graphs * 10 nodes = 80
        assert outputs['component']['health'].shape[0] == 80
        assert outputs['component']['anomaly'].shape[0] == 80
        
        # Graph-level: 8 graphs
        assert outputs['graph']['health'].shape == (8, 1)
        assert outputs['graph']['degradation'].shape == (8, 1)
        assert outputs['graph']['anomaly'].shape == (8, 9)
        assert outputs['graph']['rul'].shape == (8, 1)
        
        # All outputs valid
        for level in outputs.values():
            for tensor in level.values():
                assert not torch.isnan(tensor).any()

    def test_temporal_inference(self, model, temporal_sequence):
        """Test temporal sequence inference."""
        with torch.no_grad():
            outputs = model(temporal_sequence, temporal=True)
        
        # Should return last timestep predictions
        assert 'component' in outputs
        assert 'graph' in outputs
        
        # Component from last timestep
        assert outputs['component']['health'].shape == (10, 1)
        
        # Graph from LSTM
        assert outputs['graph']['rul'].shape == (1, 1)
        
        # All valid
        for level in outputs.values():
            for tensor in level.values():
                assert not torch.isnan(tensor).any()

    def test_response_formatting(self, model, single_graph):
        """Test converting model output to response format."""
        with torch.no_grad():
            outputs = model(single_graph, temporal=False)
        
        # Simulate API response formatting
        response = {
            'component_predictions': [],
            'system_predictions': {},
        }
        
        # Format component-level
        num_components = outputs['component']['health'].shape[0]
        for i in range(num_components):
            component_pred = {
                'component_id': f'comp_{i}',
                'health': outputs['component']['health'][i, 0].item(),
                'anomalies': outputs['component']['anomaly'][i].tolist(),
            }
            response['component_predictions'].append(component_pred)
        
        # Format graph-level
        response['system_predictions'] = {
            'health': outputs['graph']['health'][0, 0].item(),
            'degradation_rate': outputs['graph']['degradation'][0, 0].item(),
            'anomalies': outputs['graph']['anomaly'][0].tolist(),
            'rul_hours': outputs['graph']['rul'][0, 0].item(),
        }
        
        # Validate response structure
        assert len(response['component_predictions']) == 10
        assert isinstance(response['system_predictions']['health'], float)
        assert len(response['system_predictions']['anomalies']) == 9

    def test_inference_speed(self, model, single_graph):
        """Benchmark inference speed."""
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(single_graph, temporal=False)
        
        # Measure
        num_iterations = 50
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(single_graph, temporal=False)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        # Should be reasonably fast (< 100ms on CPU)
        assert avg_time < 0.1  # 100ms
        print(f"\nAverage inference time: {avg_time*1000:.2f}ms")

    def test_batch_vs_single_performance(self, model, single_graph, batch_graphs):
        """Compare batch vs sequential single inference."""
        # Warmup
        with torch.no_grad():
            _ = model(single_graph, temporal=False)
            _ = model(batch_graphs, temporal=False)
        
        # Batch inference
        start = time.time()
        with torch.no_grad():
            _ = model(batch_graphs, temporal=False)
        batch_time = time.time() - start
        
        # Sequential single inference (8 times)
        start = time.time()
        with torch.no_grad():
            for _ in range(8):
                _ = model(single_graph, temporal=False)
        sequential_time = time.time() - start
        
        # Batch should be faster than sequential
        print(f"\nBatch time: {batch_time*1000:.2f}ms")
        print(f"Sequential time: {sequential_time*1000:.2f}ms")
        print(f"Speedup: {sequential_time/batch_time:.2f}x")
        
        # Batch should be at least as fast as sequential
        assert batch_time <= sequential_time

    def test_error_handling_invalid_shape(self, model):
        """Test error handling for invalid input shapes."""
        # Wrong node feature dimension
        invalid_graph = Data(
            x=torch.randn(10, 20),  # Should be 34
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.randn(20, 14),
        )
        
        with pytest.raises(RuntimeError):
            with torch.no_grad():
                _ = model(invalid_graph, temporal=False)

    def test_error_handling_empty_graph(self, model):
        """Test handling of empty graph (no nodes)."""
        empty_graph = Data(
            x=torch.randn(0, 34),  # No nodes
            edge_index=torch.randint(0, 1, (2, 0)),  # No edges
            edge_attr=torch.randn(0, 14),
        )
        
        # Should handle gracefully or raise meaningful error
        try:
            with torch.no_grad():
                outputs = model(empty_graph, temporal=False)
            # If it doesn't raise, check outputs are valid
            assert outputs['component']['health'].shape[0] == 0
        except (RuntimeError, ValueError) as e:
            # Expected - model doesn't support empty graphs
            assert "empty" in str(e).lower() or "0" in str(e)


@pytest.mark.integration
class TestInferenceDeterminism:
    """Test inference determinism and reproducibility."""

    @pytest.fixture
    def config(self) -> ModelConfig:
        """Model configuration."""
        return ModelConfig(
            node_features=34,
            edge_features=14,
            graph_anomaly_classes=9,
            component_anomaly_classes=9,
        )

    @pytest.fixture
    def sample_graph(self) -> Data:
        """Sample graph."""
        return Data(
            x=torch.randn(10, 34),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.randn(20, 14),
        )

    def test_deterministic_inference_same_seed(self, config, sample_graph):
        """Test inference is deterministic with same seed."""
        # Run 1
        torch.manual_seed(42)
        model1 = UniversalTemporalGNNv2(config)
        model1.eval()
        
        torch.manual_seed(42)
        with torch.no_grad():
            outputs1 = model1(sample_graph, temporal=False)
        
        # Run 2 (same seed)
        torch.manual_seed(42)
        model2 = UniversalTemporalGNNv2(config)
        model2.eval()
        
        torch.manual_seed(42)
        with torch.no_grad():
            outputs2 = model2(sample_graph, temporal=False)
        
        # Should be identical
        assert torch.allclose(
            outputs1['component']['health'],
            outputs2['component']['health'],
            atol=1e-6
        )
        assert torch.allclose(
            outputs1['graph']['rul'],
            outputs2['graph']['rul'],
            atol=1e-6
        )

    def test_different_runs_same_model(self, config, sample_graph):
        """Test multiple runs with same model give same results."""
        model = UniversalTemporalGNNv2(config)
        model.eval()
        
        outputs_list = []
        
        with torch.no_grad():
            for _ in range(3):
                outputs = model(sample_graph, temporal=False)
                outputs_list.append(outputs)
        
        # All runs should give identical results (eval mode)
        for i in range(1, 3):
            assert torch.allclose(
                outputs_list[0]['graph']['health'],
                outputs_list[i]['graph']['health']
            )
