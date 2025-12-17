"""Unit tests for TemporalHydraulicDataLoader.

Tests:
- Temporal snapshot creation
- Dynamic edge construction
- Missing data handling
- Batch creation
"""

import pytest
import torch
from torch_geometric.data import Data


class TestTemporalSnapshotCreation:
    """Test temporal snapshot creation."""

    def test_snapshot_shape(self, sample_graph):
        """Test snapshot has correct shape."""
        assert sample_graph.x.shape[0] > 0
        assert sample_graph.x.shape[1] == 34  # 34 features
        assert sample_graph.edge_index.shape[0] == 2
        assert sample_graph.edge_index.max() < sample_graph.x.shape[0]

    def test_snapshot_targets(self, sample_graph):
        """Test snapshot has all required targets."""
        required_attrs = [
            "y_graph_health",
            "y_graph_degradation",
            "y_graph_anomaly",
            "y_graph_rul",
            "y_component_health",
            "y_component_anomaly",
        ]
        for attr in required_attrs:
            assert hasattr(sample_graph, attr)
            assert isinstance(getattr(sample_graph, attr), torch.Tensor)

    def test_edge_index_validity(self, sample_graph):
        """Test edge_index contains valid node indices."""
        n_nodes = sample_graph.x.shape[0]
        assert sample_graph.edge_index.min() >= 0
        assert sample_graph.edge_index.max() < n_nodes


class TestDynamicEdgeConstruction:
    """Test dynamic edge detection."""

    def test_edge_attributes(self, sample_graph):
        """Test edge attributes have correct shape."""
        if sample_graph.edge_attr is not None:
            assert sample_graph.edge_attr.shape[0] == sample_graph.edge_index.shape[1]
            assert sample_graph.edge_attr.shape[1] == 14  # 14 edge features

    def test_no_self_loops(self, sample_graph):
        """Test no self-loops in edge_index."""
        edge_index = sample_graph.edge_index
        self_loops = (edge_index[0] == edge_index[1]).sum().item()
        assert self_loops == 0


class TestMissingDataHandling:
    """Test missing data in snapshots."""

    def test_feature_validity(self, sample_graph):
        """Test features don't contain NaN or Inf."""
        assert not torch.isnan(sample_graph.x).any()
        assert not torch.isinf(sample_graph.x).any()

    def test_target_validity(self, sample_graph):
        """Test targets are in valid ranges."""
        # Health should be [0, 1]
        assert 0 <= sample_graph.y_graph_health <= 1

        # RUL should be positive
        assert sample_graph.y_graph_rul >= 0

        # Component health should be [0, 1]
        assert (sample_graph.y_component_health >= 0).all()
        assert (sample_graph.y_component_health <= 1).all()


class TestBatchCreation:
    """Test batch creation from snapshots."""

    def test_batch_size(self, sample_batch):
        """Test batch has correct total nodes."""
        expected_nodes = 5 * 10  # 5 graphs * 10 nodes each
        assert sample_batch.x.shape[0] == expected_nodes

    def test_batch_attributes(self, sample_batch):
        """Test batch has required attributes."""
        required_attrs = [
            "x",
            "edge_index",
            "batch",
            "y_graph_health",
            "y_component_health",
        ]
        for attr in required_attrs:
            assert hasattr(sample_batch, attr)

    def test_batch_vector_length(self, sample_batch):
        """Test batch vector marks all nodes correctly."""
        assert sample_batch.batch.shape[0] == sample_batch.x.shape[0]
        assert sample_batch.batch.max().item() == 4  # 0-4 for 5 graphs
