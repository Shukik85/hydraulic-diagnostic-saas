"""Unit tests for ImputationEngine.

Tests:
- Missing data imputation
- Confidence weighting
- Asymmetric noise
"""

import numpy as np
import pytest

from src.training.imputation_engine import ImputationEngine


class TestImputationEngine:
    """Tests for ImputationEngine."""

    @pytest.fixture
    def engine(self) -> ImputationEngine:
        """Create imputation engine."""
        return ImputationEngine(
            confidence_threshold=0.5,
            propagation_hops=2,
        )

    @pytest.fixture
    def missing_data(self):
        """Create sample data with missing values.
        
        Returns:
            (x, mask_nodes, edge_index, edge_weights)
        """
        n_nodes = 10
        n_features = 34
        n_edges = 15

        # Features with some missing
        x = np.random.randn(n_nodes, n_features)
        x[5:, :] = 0  # Nodes 5-9 are missing

        # Mask (True = present, False = missing)
        mask_nodes = np.array([True] * 5 + [False] * 5)

        # Edges
        edge_index = np.array(
            [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 6, 7, 8, 9, 5]]
        )

        edge_weights = np.ones(n_edges)

        return x, mask_nodes, edge_index, edge_weights

    def test_imputation_shape(self, engine, missing_data):
        """Test imputation preserves shapes."""
        x, mask_nodes, edge_index, edge_weights = missing_data

        x_imputed, conf = engine.impute_missing_features(
            x, mask_nodes, edge_index, edge_weights
        )

        assert x_imputed.shape == x.shape
        assert conf.shape == mask_nodes.shape

    def test_imputation_fills_missing(self, engine, missing_data):
        """Test imputation fills missing values."""
        x, mask_nodes, edge_index, edge_weights = missing_data

        x_imputed, _ = engine.impute_missing_features(
            x, mask_nodes, edge_index, edge_weights
        )

        # Check missing nodes were filled
        assert not (x_imputed[5:, :] == 0).all()

    def test_confidence_scores(self, engine, missing_data):
        """Test confidence scores in valid range."""
        x, mask_nodes, edge_index, edge_weights = missing_data

        _, conf = engine.impute_missing_features(
            x, mask_nodes, edge_index, edge_weights
        )

        # Confidence should be in [0, 1]
        assert (conf >= 0).all()
        assert (conf <= 1).all()

        # Present nodes should have high confidence
        assert (conf[:5] >= 0.9).all()

    def test_no_missing_data(self, engine):
        """Test with no missing data."""
        n_nodes = 10
        n_features = 34

        x = np.random.randn(n_nodes, n_features)
        mask_nodes = np.ones(n_nodes, dtype=bool)
        edge_index = np.random.randint(0, n_nodes, (2, 15))
        edge_weights = np.ones(15)

        x_imputed, conf = engine.impute_missing_features(
            x, mask_nodes, edge_index, edge_weights
        )

        # Should return same data
        assert np.allclose(x_imputed, x)
        assert np.allclose(conf, 1.0)

    def test_asymmetric_noise(self, engine, missing_data):
        """Test asymmetric noise application."""
        _, _, _, _ = missing_data

        x = np.random.randn(10, 34)
        conf = np.array([1.0, 0.9, 0.8, 0.5, 0.3, 0.1, 0.05, 0.01, 0.0, 0.0])

        x_noisy = engine.apply_asymmetric_noise_model(x, conf, noise_scale=0.1)

        # Low confidence nodes should have more noise
        noise = np.abs(x_noisy - x)
        assert noise[5:].mean() > noise[:3].mean()

    def test_invalid_threshold(self):
        """Test invalid confidence_threshold raises error."""
        with pytest.raises(ValueError):
            ImputationEngine(confidence_threshold=1.5)  # > 1

        with pytest.raises(ValueError):
            ImputationEngine(confidence_threshold=-0.1)  # < 0
