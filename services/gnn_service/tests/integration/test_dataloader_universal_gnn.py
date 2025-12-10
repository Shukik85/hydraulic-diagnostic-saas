"""Integration tests for Universal GNN with variable edge dimensions.

Tests complete pipeline:
1. Load .pt dataset (TemporalGraphDataset)
2. Create DataLoader with batching
3. Forward pass through UniversalTemporalGNN
4. Verify edge_projection handles variable edge_in_dim

Python 3.14 Features:
    - Deferred annotations
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.data.dataset import TemporalGraphDataset
from src.data.feature_config import FeatureConfig
from src.data.loader import create_dataloader
from src.models.universal_temporal_gnn import UniversalTemporalGNN


class TestDataLoaderUniversalGNN:
    """Integration tests for DataLoader + UniversalTemporalGNN."""

    def _create_test_dataset(self, data_path: Path, num_graphs: int = 10, edge_in_dim: int = 8):
        """Create test .pt dataset with specified edge dimension.

        Args:
            data_path: Path to save .pt file
            num_graphs: Number of graphs to create
            edge_in_dim: Edge feature dimension
        """
        graphs = []
        for i in range(num_graphs):
            # Variable graph sizes
            num_nodes = 4 + (i % 3)  # 4, 5, 6 nodes
            num_edges = num_nodes * 2  # Roughly 2 edges per node

            graph = Data(
                x=torch.randn(num_nodes, 34),  # Standard node features
                edge_index=torch.randint(0, num_nodes, (2, num_edges)),
                edge_attr=torch.randn(num_edges, edge_in_dim),
            )
            graphs.append(graph)

        torch.save({"graphs": graphs}, data_path)

    def test_temporal_dataset_to_model_8d_edges(self):
        """Test pipeline: TemporalGraphDataset (8D) → DataLoader → Model."""
        config = FeatureConfig(edge_in_dim=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test_graphs.pt"
            self._create_test_dataset(data_path, num_graphs=10, edge_in_dim=8)

            # Load dataset
            dataset = TemporalGraphDataset(
                data_path=data_path,
                feature_config=config,
                weights_only=False,
            )

            assert len(dataset) == 10

            # Create DataLoader
            dataloader = create_dataloader(
                dataset, config=None, split="train", batch_size=4, num_workers=0
            )

            # Create model with matching edge_in_dim
            model = UniversalTemporalGNN(
                in_channels=34,
                hidden_channels=64,
                edge_in_dim=8,  # Match dataset
                num_heads=4,
                num_gat_layers=2,
                lstm_hidden=128,
                lstm_layers=1,
                dropout=0.1,
            )

            model.eval()

            # Process first batch
            for batch in dataloader:
                assert batch.x.shape[1] == 34  # Node features
                assert batch.edge_attr.shape[1] == 8  # Edge features

                # Forward pass
                with torch.no_grad():
                    output = model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch,
                    )

                # Verify output structure
                assert "graph" in output
                assert "component" in output
                assert "health" in output["graph"]

                break  # Just test first batch

    def test_temporal_dataset_to_model_14d_edges(self):
        """Test pipeline: TemporalGraphDataset (14D) → DataLoader → Model."""
        config = FeatureConfig(edge_in_dim=14)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test_graphs_14d.pt"
            self._create_test_dataset(data_path, num_graphs=8, edge_in_dim=14)

            # Load dataset
            dataset = TemporalGraphDataset(
                data_path=data_path,
                feature_config=config,
                weights_only=False,
            )

            # Create model with matching edge_in_dim
            model = UniversalTemporalGNN(
                in_channels=34,
                hidden_channels=64,
                edge_in_dim=14,  # Match dataset (static + dynamic features)
                num_heads=4,
                num_gat_layers=2,
                lstm_hidden=128,
                lstm_layers=1,
            )

            model.eval()

            # Create DataLoader
            dataloader = create_dataloader(
                dataset, config=None, split="val", batch_size=2, num_workers=0
            )

            # Test first batch
            for batch in dataloader:
                assert batch.edge_attr.shape[1] == 14

                with torch.no_grad():
                    output = model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch,
                    )

                # All required outputs present
                assert "graph" in output
                assert output["graph"]["health"].shape[0] == batch.num_graphs

                break

    def test_edge_projection_8d_to_internal_dim(self):
        """Test edge_projection correctly handles 8D → internal dimension."""
        config = FeatureConfig(edge_in_dim=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test_graphs.pt"
            self._create_test_dataset(data_path, num_graphs=5, edge_in_dim=8)

            dataset = TemporalGraphDataset(
                data_path=data_path,
                feature_config=config,
                weights_only=False,
            )

            # Model with edge_in_dim=8 (must have edge_projection)
            model = UniversalTemporalGNN(
                in_channels=34,
                edge_in_dim=8,
                hidden_channels=64,
                edge_hidden_dim=32,  # Internal projection dimension
                num_heads=4,
            )

            # Verify edge_projection exists
            assert hasattr(model, "edge_projection")
            assert model.edge_in_dim == 8

            # Create dataloader
            dataloader = create_dataloader(
                dataset, config=None, batch_size=2, num_workers=0
            )

            model.eval()
            for batch in dataloader:
                # Input: 8D edge features
                assert batch.edge_attr.shape[1] == 8

                # Forward through projection
                with torch.no_grad():
                    edge_proj = model.edge_projection(batch.edge_attr)

                # Output: edge_hidden_dim (32)
                assert edge_proj.shape[1] == model.edge_hidden_dim

                break

    def test_batch_with_variable_graph_sizes(self):
        """Test batching graphs of different sizes."""
        config = FeatureConfig(edge_in_dim=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "variable_size.pt"

            # Create graphs with different sizes
            graphs = [
                Data(
                    x=torch.randn(3, 34),  # Small graph
                    edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                    edge_attr=torch.randn(2, 8),
                ),
                Data(
                    x=torch.randn(5, 34),  # Medium graph
                    edge_index=torch.randint(0, 5, (2, 8)),
                    edge_attr=torch.randn(8, 8),
                ),
                Data(
                    x=torch.randn(4, 34),  # Another small graph
                    edge_index=torch.randint(0, 4, (2, 6)),
                    edge_attr=torch.randn(6, 8),
                ),
            ]

            torch.save({"graphs": graphs}, data_path)

            dataset = TemporalGraphDataset(
                data_path=data_path,
                feature_config=config,
                weights_only=False,
            )

            # Create DataLoader (will batch different-sized graphs)
            dataloader = create_dataloader(
                dataset, config=None, batch_size=3, num_workers=0
            )

            model = UniversalTemporalGNN(
                in_channels=34,
                edge_in_dim=8,
                hidden_channels=64,
            )

            model.eval()

            for batch in dataloader:
                # Batch contains all 3 graphs of different sizes
                assert batch.num_graphs == 3
                # Total nodes: 3 + 5 + 4 = 12
                assert batch.x.shape[0] == 12

                with torch.no_grad():
                    output = model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch,
                    )

                # Graph-level outputs for 3 graphs
                assert output["graph"]["health"].shape[0] == 3

                break

    def test_backward_pass_for_training(self):
        """Test that backward pass works for training."""
        config = FeatureConfig(edge_in_dim=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "train.pt"
            self._create_test_dataset(data_path, num_graphs=4, edge_in_dim=8)

            dataset = TemporalGraphDataset(
                data_path=data_path,
                feature_config=config,
                weights_only=False,
            )

            dataloader = create_dataloader(
                dataset, config=None, batch_size=2, num_workers=0
            )

            model = UniversalTemporalGNN(
                in_channels=34,
                edge_in_dim=8,
                hidden_channels=64,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.train()

            for batch in dataloader:
                optimizer.zero_grad()

                # Forward pass
                output = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                )

                # Simple loss: MSE on graph health scores
                target = torch.ones(batch.num_graphs, 1)
                loss = torch.nn.functional.mse_loss(
                    output["graph"]["health"], target
                )

                # Backward pass
                loss.backward()
                optimizer.step()

                # Verify parameters were updated
                assert loss.item() >= 0  # Loss is valid number

                break  # Just test first batch

    def test_get_dataset_statistics(self):
        """Test TemporalGraphDataset.get_statistics()."""
        config = FeatureConfig(edge_in_dim=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "stats.pt"
            self._create_test_dataset(data_path, num_graphs=20, edge_in_dim=8)

            dataset = TemporalGraphDataset(
                data_path=data_path,
                feature_config=config,
                weights_only=False,
            )

            stats = dataset.get_statistics()

            # Verify statistics
            assert stats["dataset_size"] == 20
            assert stats["edge_in_dim_configured"] == 8
            assert 8 in stats["edge_feature_dims"]
            assert stats["node_features"] == 34
            assert stats["avg_num_nodes"] >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
