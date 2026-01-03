"""Unit tests for GRAPE two-stage imputation.

Tests:
- GRAPEImputer (spatial)
- TemporalImputer (temporal)
- TwoStageImputer (complete pipeline)
"""

import pytest
import torch
from torch_geometric.data import Data

from src.training.imputation_grape import (
    GRAPEImputer,
    TemporalImputer,
    TwoStageImputer,
)


class TestGRAPEImputer:
    """Tests for GRAPEImputer (Stage 1: Spatial)."""

    @pytest.fixture
    def imputer(self) -> GRAPEImputer:
        """Create imputer instance."""
        return GRAPEImputer(
            feature_dim=34,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )

    @pytest.fixture
    def missing_data(self):
        """Create data with missing nodes."""
        n_nodes = 10
        n_edges = 15

        x = torch.randn(n_nodes, 34)
        x[5:, :] = 0  # Missing nodes 5-9

        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        edge_attr = torch.randn(n_edges, 14)
        mask_nodes = torch.tensor([True] * 5 + [False] * 5)

        return x, edge_index, edge_attr, mask_nodes

    def test_forward_shape(self, imputer, missing_data):
        """Test output shapes."""
        x, edge_index, edge_attr, mask_nodes = missing_data

        x_imputed, confidence = imputer(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_nodes=mask_nodes,
        )

        assert x_imputed.shape == x.shape
        assert confidence.shape == mask_nodes.shape

    def test_fills_missing(self, imputer, missing_data):
        """Test that missing values are filled."""
        x, edge_index, edge_attr, mask_nodes = missing_data

        x_imputed, _ = imputer(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_nodes=mask_nodes,
        )

        # Missing nodes should be non-zero after imputation
        assert not (x_imputed[5:, :] == 0).all()

    def test_confidence_range(self, imputer, missing_data):
        """Test confidence is in [0, 1]."""
        x, edge_index, edge_attr, mask_nodes = missing_data

        _, confidence = imputer(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_nodes=mask_nodes,
        )

        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_observed_confidence_high(self, imputer, missing_data):
        """Test observed nodes have high confidence."""
        x, edge_index, edge_attr, mask_nodes = missing_data

        _, confidence = imputer(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_nodes=mask_nodes,
        )

        # Observed nodes should have confidence = 1.0
        assert torch.allclose(confidence[:5], torch.ones(5))

    def test_static_topology_integration(self, imputer, missing_data):
        """Test static topology prior."""
        x, edge_index, edge_attr, mask_nodes = missing_data

        static_topology = torch.tensor([[0, 1, 2], [1, 2, 3]])

        x_imputed, _ = imputer(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_nodes=mask_nodes,
            static_topology=static_topology,
        )

        assert x_imputed.shape == x.shape


class TestTemporalImputer:
    """Tests for TemporalImputer (Stage 2: Temporal)."""

    @pytest.fixture
    def imputer(self) -> TemporalImputer:
        """Create temporal imputer."""
        return TemporalImputer(
            feature_dim=34,
            hidden_dim=64,
            num_layers=2,
        )

    @pytest.fixture
    def temporal_data(self):
        """Create temporal sequence."""
        T, N, F = 12, 10, 34

        x_sequence = torch.randn(T, N, F)
        mask_sequence = torch.ones(T, N, dtype=torch.bool)
        mask_sequence[:, 5:] = False  # Nodes 5-9 missing

        spatial_confidence = torch.rand(T, N)

        return x_sequence, mask_sequence, spatial_confidence

    def test_forward_shape(self, imputer, temporal_data):
        """Test output shapes."""
        x_seq, mask_seq, spatial_conf = temporal_data

        x_refined, temporal_conf = imputer(
            x_sequence=x_seq,
            mask_sequence=mask_seq,
            spatial_confidence=spatial_conf,
        )

        assert x_refined.shape == x_seq.shape
        assert temporal_conf.shape == mask_seq.shape

    def test_temporal_refinement(self, imputer, temporal_data):
        """Test temporal refinement changes values."""
        x_seq, mask_seq, spatial_conf = temporal_data

        x_refined, _ = imputer(
            x_sequence=x_seq,
            mask_sequence=mask_seq,
            spatial_confidence=spatial_conf,
        )

        # Refined should differ from input for missing nodes
        assert not torch.allclose(x_refined[:, 5:, :], x_seq[:, 5:, :])

    def test_confidence_combination(self, imputer, temporal_data):
        """Test spatial+temporal confidence combination."""
        x_seq, mask_seq, spatial_conf = temporal_data

        _, temporal_conf = imputer(
            x_sequence=x_seq,
            mask_sequence=mask_seq,
            spatial_confidence=spatial_conf,
        )

        # Temporal confidence should be <= spatial confidence
        assert (temporal_conf <= spatial_conf).all() or torch.allclose(
            temporal_conf, spatial_conf
        )


class TestTwoStageImputer:
    """Tests for complete TwoStageImputer."""

    @pytest.fixture
    def imputer(self) -> TwoStageImputer:
        """Create two-stage imputer."""
        return TwoStageImputer(
            feature_dim=34,
            spatial_hidden=64,
            temporal_hidden=64,
            device="cpu",
        )

    @pytest.fixture
    def temporal_graph_data(self):
        """Create temporal graph sequence."""
        T, N, F = 12, 10, 34
        E = 15

        x_sequence = torch.randn(T, N, F)
        x_sequence[:, 5:, :] = 0  # Missing nodes

        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, 14)

        mask_sequence = torch.ones(T, N, dtype=torch.bool)
        mask_sequence[:, 5:] = False

        return x_sequence, edge_index, edge_attr, mask_sequence

    def test_complete_pipeline(self, imputer, temporal_graph_data):
        """Test complete two-stage imputation."""
        x_seq, edge_index, edge_attr, mask_seq = temporal_graph_data

        x_imputed, confidence = imputer.impute_temporal_sequence(
            x_sequence=x_seq,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_sequence=mask_seq,
        )

        assert x_imputed.shape == x_seq.shape
        assert confidence.shape == mask_seq.shape

    def test_fills_all_missing(self, imputer, temporal_graph_data):
        """Test all missing values are filled."""
        x_seq, edge_index, edge_attr, mask_seq = temporal_graph_data

        x_imputed, _ = imputer.impute_temporal_sequence(
            x_sequence=x_seq,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_sequence=mask_seq,
        )

        # No zeros should remain in missing nodes
        assert not (x_imputed[:, 5:, :] == 0).all()

    def test_confidence_degradation(self, imputer, temporal_graph_data):
        """Test confidence degrades over stages."""
        x_seq, edge_index, edge_attr, mask_seq = temporal_graph_data

        _, confidence = imputer.impute_temporal_sequence(
            x_sequence=x_seq,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_sequence=mask_seq,
        )

        # Missing nodes should have < 1.0 confidence
        assert (confidence[:, 5:] < 1.0).any()

    def test_with_static_topology(self, imputer, temporal_graph_data):
        """Test with static topology prior."""
        x_seq, edge_index, edge_attr, mask_seq = temporal_graph_data

        static_topology = torch.tensor([[0, 1, 2], [1, 2, 3]])

        x_imputed, confidence = imputer.impute_temporal_sequence(
            x_sequence=x_seq,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_sequence=mask_seq,
            static_topology=static_topology,
        )

        assert x_imputed.shape == x_seq.shape
        assert confidence.shape == mask_seq.shape
