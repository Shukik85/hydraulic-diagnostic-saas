"""Unit tests for ImputationEngine.

Tests imputation functionality:
- Missing data handling
- Neighbor propagation
- Asymmetric noise modeling
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.training.imputation_engine import ImputationEngine


class TestImputationEngine:
    """Test ImputationEngine class."""

    @pytest.fixture
    def engine(self) -> ImputationEngine:
        """Create engine fixture."""
        return ImputationEngine(confidence_threshold=0.5, propagation_hops=2)

    @pytest.fixture
    def simple_graph_data(self):
        """Create simple graph data for testing."""
        n_nodes = 5
        n_features = 4
        n_edges = 4

        x = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 0.0, 0.0],  # Missing
        ])

        mask_nodes = np.array([True, True, True, True, False])

        edge_index = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ])

        edge_weights = np.array([1.0, 1.0, 1.0, 1.0])

        return x, mask_nodes, edge_index, edge_weights

    def test_init(self):
        """Test initialization."""
        engine = ImputationEngine(confidence_threshold=0.3, propagation_hops=3)
        assert engine.confidence_threshold == 0.3
        assert engine.propagation_hops == 3

    def test_invalid_threshold(self):
        """Test invalid threshold raises error."""
        with pytest.raises(ValueError):
            ImputationEngine(confidence_threshold=1.5)

    def test_impute_missing_features_shape(self, engine, simple_graph_data):
        """Test output shape is preserved."""
        x, mask_nodes, edge_index, edge_weights = simple_graph_data
        x_imputed, conf = engine.impute_missing_features(
            x, mask_nodes, edge_index, edge_weights
        )

        assert x_imputed.shape == x.shape
        assert conf.shape == mask_nodes.shape

    def test_impute_missing_features_fills_gaps(self, engine, simple_graph_data):
        """Test that missing values are filled."""
        x, mask_nodes, edge_index, edge_weights = simple_graph_data
        x_imputed, _ = engine.impute_missing_features(
            x, mask_nodes, edge_index, edge_weights
        )

        # Last node should be filled (not all zeros)
        assert not (x_imputed[4] == 0).all()

    def test_confidence_scores_valid_range(self, engine, simple_graph_data):
        """Test confidence scores are in [0, 1]."""
        x, mask_nodes, edge_index, edge_weights = simple_graph_data
        _, conf = engine.impute_missing_features(
            x, mask_nodes, edge_index, edge_weights
        )

        assert (conf >= 0).all()
        assert (conf <= 1).all()

    def test_confidence_high_for_present_nodes(self, engine, simple_graph_data):
        """Test high confidence for present nodes."""
        x, mask_nodes, edge_index, edge_weights = simple_graph_data
        _, conf = engine.impute_missing_features(
            x, mask_nodes, edge_index, edge_weights
        )

        # Present nodes should have high confidence
        assert conf[0] > 0.9
        assert conf[4] < 0.5  # Missing node has low confidence

    def test_no_missing_data(self, engine):
        """Test when no data is missing."""
        x = np.random.randn(5, 4)
        mask_nodes = np.ones(5, dtype=bool)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        edge_weights = np.ones(3)

        x_imputed, conf = engine.impute_missing_features(
            x, mask_nodes, edge_index, edge_weights
        )

        # Should be nearly identical
        assert np.allclose(x_imputed, x, atol=1e-5)
        assert np.allclose(conf, 1.0)

    def test_asymmetric_noise_application(self, engine):
        """Test asymmetric noise application."""
        x = np.random.randn(5, 4)
        conf = np.array([1.0, 0.8, 0.5, 0.2, 0.1])

        x_noisy = engine.apply_asymmetric_noise_model(x, conf, noise_scale=0.1)

        # Low confidence nodes should have more noise
        noise = np.abs(x_noisy - x)
        assert noise[4].mean() > noise[0].mean()

    def test_invalid_noise_scale(self, engine):
        """Test invalid noise scale raises error."""
        x = np.random.randn(5, 4)
        conf = np.ones(5)

        with pytest.raises(AssertionError):
            engine.apply_asymmetric_noise_model(x, conf, noise_scale=1.5)

    def test_bfs_neighbors(self, engine, simple_graph_data):
        """Test BFS neighbor finding."""
        _, mask_nodes, edge_index, edge_weights = simple_graph_data
        adjacency = engine._build_adjacency_dict(edge_index, edge_weights, n_nodes=5)

        # Find neighbors of node 0
        neighbors = engine._find_neighbors_bfs(
            node_idx=0,
            adjacency=adjacency,
            max_hops=2,
            mask_nodes=mask_nodes,
        )

        # Should find nodes 1, 2, 3
        neighbor_indices = [n[0] for n in neighbors]
        assert 1 in neighbor_indices or 2 in neighbor_indices or 3 in neighbor_indices
