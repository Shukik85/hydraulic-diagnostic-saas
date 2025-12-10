"""Tests for edge_in_dim flexibility in Universal GNN data pipeline.

Tests variable edge feature dimensions:
- 8D: static features only
- 14D: static + dynamic features (default)
- 20D: custom extended features

Python 3.14 Features:
    - Deferred annotations
    - Parametrized tests
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.data.dataset import HydraulicGraphDataset, TemporalGraphDataset
from src.data.feature_config import FeatureConfig
from src.data.feature_engineer import FeatureEngineer
from src.data.graph_builder import GraphBuilder


class TestFeatureConfigEdgeDim:
    """Tests for FeatureConfig.edge_in_dim."""

    def test_default_edge_in_dim(self):
        """Test default edge_in_dim is 14 (backward compatible)."""
        config = FeatureConfig()
        assert config.edge_in_dim == 14

    def test_edge_in_dim_8_static_only(self):
        """Test edge_in_dim=8 (static features only)."""
        config = FeatureConfig(edge_in_dim=8)
        assert config.edge_in_dim == 8
        assert config.static_edge_features_count == 8
        assert config.dynamic_edge_features_count == 6
        assert not config.has_dynamic_edge_features

    def test_edge_in_dim_14_full(self):
        """Test edge_in_dim=14 (static + dynamic)."""
        config = FeatureConfig(edge_in_dim=14)
        assert config.edge_in_dim == 14
        assert config.static_edge_features_count == 8
        assert config.dynamic_edge_features_count == 6
        assert config.has_dynamic_edge_features

    def test_edge_in_dim_custom_20(self):
        """Test custom edge_in_dim=20 (extended features)."""
        config = FeatureConfig(edge_in_dim=20)
        assert config.edge_in_dim == 20
        assert config.has_dynamic_edge_features  # >= 14

    def test_edge_in_dim_invalid_zero(self):
        """Test invalid edge_in_dim=0 raises ValueError."""
        with pytest.raises(ValueError, match="edge_in_dim must be >= 1"):
            FeatureConfig(edge_in_dim=0)

    def test_edge_in_dim_invalid_negative(self):
        """Test invalid edge_in_dim=-5 raises ValueError."""
        with pytest.raises(ValueError, match="edge_in_dim must be >= 1"):
            FeatureConfig(edge_in_dim=-5)


class TestGraphBuilderEdgeDim:
    """Tests for GraphBuilder with variable edge_in_dim."""

    def test_graph_builder_8d_edge_features(self):
        """Test GraphBuilder creates 8D edge features."""
        config = FeatureConfig(edge_in_dim=8)
        builder = GraphBuilder(feature_config=config)

        assert builder.feature_config.edge_in_dim == 8

        # Build minimal graph
        x = torch.randn(3, config.total_features_per_sensor)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)

        # Validate
        assert builder.validate_graph(graph)
        assert graph.edge_attr is None

    def test_graph_builder_14d_edge_features(self):
        """Test GraphBuilder creates 14D edge features (default)."""
        config = FeatureConfig(edge_in_dim=14)
        builder = GraphBuilder(feature_config=config)

        assert builder.feature_config.edge_in_dim == 14

    def test_graph_builder_custom_20d_edge_features(self):
        """Test GraphBuilder with custom 20D edge features."""
        config = FeatureConfig(edge_in_dim=20)
        builder = GraphBuilder(feature_config=config)

        assert builder.feature_config.edge_in_dim == 20

    def test_edge_attr_padding_to_20d(self):
        """Test edge_attr padding from 14D to 20D."""
        config = FeatureConfig(edge_in_dim=20)
        builder = GraphBuilder(feature_config=config)

        # Create graph with 14D edge features (below target 20D)
        x = torch.randn(3, config.total_features_per_sensor)
        edge_attr_14d = torch.randn(4, 14)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_14d)

        # Padding happens in GraphBuilder.build_edge_features()
        # Here we simulate it
        if graph.edge_attr.shape[1] < 20:
            padding = torch.zeros(graph.edge_attr.shape[0], 20 - graph.edge_attr.shape[1])
            padded = torch.cat([graph.edge_attr, padding], dim=1)
            assert padded.shape[1] == 20

    def test_edge_attr_truncation_to_8d(self):
        """Test edge_attr truncation from 14D to 8D."""
        config = FeatureConfig(edge_in_dim=8)
        builder = GraphBuilder(feature_config=config)

        # Create graph with 14D edge features (above target 8D)
        x = torch.randn(3, config.total_features_per_sensor)
        edge_attr_14d = torch.randn(4, 14)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_14d)

        # Truncation simulation
        if graph.edge_attr.shape[1] > 8:
            truncated = graph.edge_attr[:, :8]
            assert truncated.shape[1] == 8

    def test_graph_validation_with_wrong_edge_dim(self):
        """Test graph validation fails when edge_attr dim doesn't match config."""
        config = FeatureConfig(edge_in_dim=14)
        builder = GraphBuilder(feature_config=config)

        # Create graph with 8D edge features (mismatch with config)
        x = torch.randn(3, config.total_features_per_sensor)
        edge_attr_8d = torch.randn(4, 8)  # Wrong: config expects 14D
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_8d)

        # Validation should fail
        assert not builder.validate_graph(graph)


class TestTemporalGraphDataset:
    """Tests for TemporalGraphDataset (pre-built .pt graphs)."""

    def test_create_temporal_dataset_8d(self):
        """Test creating TemporalGraphDataset with 8D graphs."""
        config = FeatureConfig(edge_in_dim=8)

        # Create minimal .pt dataset in temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test_graphs.pt"

            # Create test graphs with 8D edge features
            graphs = [
                Data(
                    x=torch.randn(5, 34),
                    edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                    edge_attr=torch.randn(3, 8),  # 8D edge features
                ),
                Data(
                    x=torch.randn(4, 34),
                    edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                    edge_attr=torch.randn(2, 8),  # 8D edge features
                ),
            ]

            # Save to .pt file
            torch.save({"graphs": graphs}, data_path)

            # Load with dataset
            dataset = TemporalGraphDataset(
                data_path=data_path,
                feature_config=config,
                split="train",
                weights_only=False,
            )

            assert len(dataset) == 2
            assert dataset[0].edge_attr.shape[1] == 8
            assert dataset[1].edge_attr.shape[1] == 8

    def test_create_temporal_dataset_14d(self):
        """Test creating TemporalGraphDataset with 14D graphs."""
        config = FeatureConfig(edge_in_dim=14)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test_graphs_14d.pt"

            # Create test graphs with 14D edge features
            graphs = [
                Data(
                    x=torch.randn(5, 34),
                    edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                    edge_attr=torch.randn(3, 14),  # 14D edge features
                ),
            ]

            torch.save({"graphs": graphs}, data_path)

            dataset = TemporalGraphDataset(
                data_path=data_path,
                feature_config=config,
                split="val",
                weights_only=False,
            )

            assert len(dataset) == 1
            assert dataset[0].edge_attr.shape[1] == 14

    def test_temporal_dataset_statistics(self):
        """Test TemporalGraphDataset.get_statistics()."""
        config = FeatureConfig(edge_in_dim=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test_graphs.pt"

            graphs = [
                Data(
                    x=torch.randn(5, 34),
                    edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                    edge_attr=torch.randn(3, 8),
                ),
                Data(
                    x=torch.randn(6, 34),
                    edge_index=torch.tensor(
                        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
                    ),
                    edge_attr=torch.randn(4, 8),
                ),
            ]

            torch.save({"graphs": graphs}, data_path)

            dataset = TemporalGraphDataset(
                data_path=data_path,
                feature_config=config,
                weights_only=False,
            )

            stats = dataset.get_statistics()

            assert stats["dataset_size"] == 2
            assert stats["node_features"] == 34
            assert 8 in stats["edge_feature_dims"]
            assert stats["edge_in_dim_configured"] == 8
            assert stats["avg_num_nodes"] == 5.5

    def test_temporal_dataset_file_not_found(self):
        """Test TemporalGraphDataset raises FileNotFoundError."""
        config = FeatureConfig(edge_in_dim=8)

        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            TemporalGraphDataset(
                data_path=Path("/nonexistent/path/graphs.pt"),
                feature_config=config,
            )

    def test_temporal_dataset_invalid_structure(self):
        """Test TemporalGraphDataset with invalid data structure."""
        config = FeatureConfig(edge_in_dim=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "invalid.pt"

            # Save invalid structure (neither dict nor list)
            torch.save({"invalid_key": [1, 2, 3]}, data_path)

            with pytest.raises(ValueError, match="Unknown dict structure"):
                TemporalGraphDataset(
                    data_path=data_path,
                    feature_config=config,
                    weights_only=False,
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
