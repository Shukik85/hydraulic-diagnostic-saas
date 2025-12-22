"""Unit tests for TemporalGraphDataset (Phase 2 v2 API).

Tests for:
- Dataset loading from .pt files
- Graph structure validation
- Edge feature dimensions
- Multi-task targets
- Statistics and transforms
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.data.dataset import TemporalGraphDataset
from src.data.feature_config import FeatureConfig


@pytest.fixture
def sample_pt_file(tmp_path: Path):
    """Create a temporary .pt file with sample PyG Data graphs.
    
    Returns:
        Path to .pt file with 10 graphs (5 nodes, 6 edges each)
    """
    # Create 10 sample graphs
    graphs = []
    for i in range(10):
        # Node features: [N, 34] (5 nodes, 34 features each)
        x = torch.randn(5, 34)
        
        # Edge index: [2, E] (6 edges)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], 
            dtype=torch.long
        )
        
        # Edge attributes: [E, 8] (8D features - static only)
        edge_attr = torch.randn(6, 8)
        
        # Multi-task targets (Phase 2 v2)
        # Graph-level
        y_graph_health = torch.tensor([0.8 - i * 0.05])  # Scalar per graph
        y_graph_degradation = torch.tensor([0.2 + i * 0.03])
        y_graph_anomaly = torch.randint(0, 2, (9,)).float()  # 9 anomaly classes
        y_graph_rul = torch.tensor([100.0 - i * 5])  # RUL in hours
        
        # Component-level
        y_component_health = torch.rand(5)  # Per node
        y_component_anomaly = torch.randint(0, 2, (5, 9)).float()  # Per node
        
        # Batch index (for DataLoader compatibility)
        batch = torch.zeros(5, dtype=torch.long)  # All nodes belong to graph 0
        
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_graph_health=y_graph_health,
            y_graph_degradation=y_graph_degradation,
            y_graph_anomaly=y_graph_anomaly,
            y_graph_rul=y_graph_rul,
            y_component_health=y_component_health,
            y_component_anomaly=y_component_anomaly,
            batch=batch,
        )
        graphs.append(graph)
    
    # Save to .pt file
    pt_file = tmp_path / "test_graphs.pt"
    torch.save({"graphs": graphs}, pt_file)
    
    return pt_file


@pytest.fixture
def feature_config():
    """Create FeatureConfig for testing."""
    return FeatureConfig(
        edge_in_dim=14,  # Model expects 14D (8D static + 6D dynamic)
        total_features_per_sensor=34,
    )


@pytest.fixture
def temporal_dataset(sample_pt_file: Path, feature_config: FeatureConfig):
    """Create TemporalGraphDataset for testing."""
    return TemporalGraphDataset(
        data_path=sample_pt_file,
        feature_config=feature_config,
        split="train",
        weights_only=False,
    )


class TestTemporalGraphDatasetInit:
    """Tests for TemporalGraphDataset initialization."""

    def test_dataset_loads_from_pt_file(self, temporal_dataset: TemporalGraphDataset):
        """Test that dataset loads graphs from .pt file."""
        assert len(temporal_dataset) == 10
        assert temporal_dataset.split == "train"
        assert temporal_dataset.feature_config.edge_in_dim == 14

    def test_dataset_missing_file_raises_error(self, feature_config: FeatureConfig):
        """Test that missing .pt file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TemporalGraphDataset(
                data_path="/nonexistent/path.pt",
                feature_config=feature_config,
            )

    def test_dataset_getitem_returns_data(self, temporal_dataset: TemporalGraphDataset):
        """Test that __getitem__ returns PyG Data object."""
        graph = temporal_dataset[0]
        assert isinstance(graph, Data)
        assert hasattr(graph, "x")
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "edge_attr")


class TestGraphStructureValidation:
    """Tests for graph structure validation."""

    def test_node_features_shape(self, temporal_dataset: TemporalGraphDataset):
        """Test node feature shape [N, 34]."""
        graph = temporal_dataset[0]
        assert graph.x.shape == (5, 34)
        assert graph.x.dtype == torch.float32

    def test_edge_index_shape(self, temporal_dataset: TemporalGraphDataset):
        """Test edge_index shape [2, E]."""
        graph = temporal_dataset[0]
        assert graph.edge_index.shape == (2, 6)
        assert graph.edge_index.dtype == torch.long

    def test_edge_index_validity(self, temporal_dataset: TemporalGraphDataset):
        """Test edge_index contains valid node indices."""
        graph = temporal_dataset[0]
        num_nodes = graph.x.shape[0]
        
        assert graph.edge_index.min() >= 0
        assert graph.edge_index.max() < num_nodes

    def test_no_self_loops(self, temporal_dataset: TemporalGraphDataset):
        """Test that graph has no self-loops."""
        graph = temporal_dataset[0]
        edge_index = graph.edge_index
        
        # Check src != dst for all edges
        src, dst = edge_index[0], edge_index[1]
        assert not torch.any(src == dst)


class TestEdgeFeatureDimensions:
    """Tests for edge feature dimensions."""

    def test_edge_attr_loaded_dimension(self, temporal_dataset: TemporalGraphDataset):
        """Test that loaded graphs have 8D edge features."""
        graph = temporal_dataset[0]
        assert graph.edge_attr.shape == (6, 8)  # 8D static features

    def test_edge_attr_not_14d_yet(self, temporal_dataset: TemporalGraphDataset):
        """Test that loaded edge_attr is 8D (projection happens in model)."""
        graph = temporal_dataset[0]
        
        # Dataset loads 8D, model will project to 14D
        assert graph.edge_attr.shape[1] == 8
        assert graph.edge_attr.shape[1] != temporal_dataset.feature_config.edge_in_dim

    def test_edge_projection_handled_by_model(self, temporal_dataset: TemporalGraphDataset):
        """Verify that dataset doesn't project edges (model's job)."""
        # TemporalGraphDataset only loads data as-is
        # edge_projection layer in UniversalTemporalGNNv2 handles 8D → 14D
        graph = temporal_dataset[0]
        
        assert graph.edge_attr.shape[1] == 8, "Dataset should not modify edge dimensions"


class TestMultiTaskTargets:
    """Tests for multi-task target validation."""

    def test_graph_health_target(self, temporal_dataset: TemporalGraphDataset):
        """Test graph-level health target."""
        graph = temporal_dataset[0]
        assert hasattr(graph, "y_graph_health")
        assert graph.y_graph_health.shape == (1,)  # Scalar per graph
        assert 0.0 <= graph.y_graph_health.item() <= 1.0

    def test_graph_degradation_target(self, temporal_dataset: TemporalGraphDataset):
        """Test graph-level degradation target."""
        graph = temporal_dataset[0]
        assert hasattr(graph, "y_graph_degradation")
        assert graph.y_graph_degradation.shape == (1,)
        assert 0.0 <= graph.y_graph_degradation.item() <= 1.0

    def test_graph_anomaly_target(self, temporal_dataset: TemporalGraphDataset):
        """Test graph-level anomaly target (9 classes)."""
        graph = temporal_dataset[0]
        assert hasattr(graph, "y_graph_anomaly")
        assert graph.y_graph_anomaly.shape == (9,)
        assert graph.y_graph_anomaly.dtype == torch.float32

    def test_graph_rul_target(self, temporal_dataset: TemporalGraphDataset):
        """Test graph-level RUL target."""
        graph = temporal_dataset[0]
        assert hasattr(graph, "y_graph_rul")
        assert graph.y_graph_rul.shape == (1,)
        assert graph.y_graph_rul.item() >= 0.0

    def test_component_health_target(self, temporal_dataset: TemporalGraphDataset):
        """Test component-level health target."""
        graph = temporal_dataset[0]
        assert hasattr(graph, "y_component_health")
        assert graph.y_component_health.shape == (5,)  # Per node
        assert torch.all((graph.y_component_health >= 0.0) & (graph.y_component_health <= 1.0))

    def test_component_anomaly_target(self, temporal_dataset: TemporalGraphDataset):
        """Test component-level anomaly target."""
        graph = temporal_dataset[0]
        assert hasattr(graph, "y_component_anomaly")
        assert graph.y_component_anomaly.shape == (5, 9)  # Per node, 9 classes
        assert graph.y_component_anomaly.dtype == torch.float32


class TestDatasetStatistics:
    """Tests for dataset statistics."""

    def test_get_statistics(self, temporal_dataset: TemporalGraphDataset):
        """Test get_statistics returns valid stats."""
        stats = temporal_dataset.get_statistics()
        
        assert stats["dataset_size"] == 10
        assert stats["split"] == "train"
        assert stats["avg_num_nodes"] == 5.0
        assert stats["avg_num_edges"] == 6.0
        assert stats["node_features"] == 34
        assert stats["edge_feature_dims"] == [8]  # Only 8D in dataset
        assert stats["edge_in_dim_configured"] == 14  # Model expects 14D

    def test_statistics_sample_size(self, temporal_dataset: TemporalGraphDataset):
        """Test statistics uses sample_size correctly."""
        stats = temporal_dataset.get_statistics()
        
        # Should sample all 10 graphs (dataset size < 10)
        assert stats["sample_size"] == 10
        assert stats["min_num_nodes"] == 5
        assert stats["max_num_nodes"] == 5


class TestDatasetTransforms:
    """Tests for dataset transforms."""

    def test_transform_applied(self, sample_pt_file: Path, feature_config: FeatureConfig):
        """Test that transform is applied to graphs."""
        def add_noise_transform(data: Data) -> Data:
            """Add noise to node features."""
            data.x = data.x + torch.randn_like(data.x) * 0.01
            return data
        
        dataset = TemporalGraphDataset(
            data_path=sample_pt_file,
            feature_config=feature_config,
            transform=add_noise_transform,
        )
        
        graph = dataset[0]
        # Transform should be applied (hard to verify noise, but check shape)
        assert graph.x.shape == (5, 34)

    def test_no_transform(self, temporal_dataset: TemporalGraphDataset):
        """Test dataset works without transforms."""
        graph = temporal_dataset[0]
        assert graph.x.shape == (5, 34)


class TestDatasetIteration:
    """Tests for dataset iteration."""

    def test_dataset_iteration(self, temporal_dataset: TemporalGraphDataset):
        """Test iterating through dataset."""
        graphs = [temporal_dataset[i] for i in range(len(temporal_dataset))]
        assert len(graphs) == 10
        
        # All should be Data objects
        for graph in graphs:
            assert isinstance(graph, Data)

    def test_dataset_indexing(self, temporal_dataset: TemporalGraphDataset):
        """Test indexing specific graphs."""
        first = temporal_dataset[0]
        last = temporal_dataset[-1]
        
        assert isinstance(first, Data)
        assert isinstance(last, Data)
        
        # Targets should be different
        assert not torch.allclose(first.y_graph_health, last.y_graph_health)
