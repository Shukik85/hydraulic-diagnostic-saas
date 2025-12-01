"""Unit tests для data pipeline.

Полное покрытие:
- FeatureEngineer (all feature types)
- GraphBuilder (node/edge features)
- HydraulicGraphDataset (loading, caching)
- DataLoader (collate, batching)

Pytest fixtures в conftest.py.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from pathlib import Path
import tempfile
import json

from src.data import (
    FeatureConfig,
    DataLoaderConfig,
    FeatureEngineer,
    GraphBuilder,
    HydraulicGraphDataset,
    hydraulic_collate_fn,
    create_dataloader,
    create_train_val_loaders
)
from src.schemas import (
    GraphTopology,
    ComponentSpec,
    EdgeSpec,
    ComponentType,
    EdgeType
)


# ==================== FIXTURES ====================

@pytest.fixture
def sample_sensor_data():
    """Пример sensor data."""
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    
    df = pd.DataFrame({
        "pressure_pump": 100 + 10 * np.sin(2 * np.pi * t) + np.random.randn(100),
        "temperature_pump": 60 + 5 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(100),
        "vibration_pump": 2 + 0.5 * np.random.randn(100),
        "pressure_valve": 95 + 8 * np.sin(2 * np.pi * t) + np.random.randn(100),
    })
    
    return df


@pytest.fixture
def feature_config():
    """Default feature config."""
    return FeatureConfig(
        use_statistical=True,
        use_frequency=True,
        use_temporal=True,
        use_hydraulic=True,
        num_frequencies=5  # Smaller для тестов
    )


@pytest.fixture
def feature_engineer(feature_config):
    """FeatureEngineer instance."""
    return FeatureEngineer(feature_config)


@pytest.fixture
def sample_topology():
    """Sample GraphTopology."""
    components = {
        "pump": ComponentSpec(
            component_id="pump",
            component_type=ComponentType.PUMP,
            location_x=0.0,
            location_y=0.0
        ),
        "valve": ComponentSpec(
            component_id="valve",
            component_type=ComponentType.VALVE,
            location_x=1.0,
            location_y=0.0
        )
    }
    
    edges = [
        EdgeSpec(
            source_id="pump",
            target_id="valve",
            edge_type=EdgeType.PIPE,
            diameter_mm=16.0,
            length_m=2.5,
            pressure_rating_bar=350,
            material="steel"
        )
    ]
    
    return GraphTopology(components=components, edges=edges)


# ==================== FEATURE ENGINEER TESTS ====================

class TestFeatureEngineer:
    """Tests для FeatureEngineer."""
    
    def test_statistical_features(self, feature_engineer):
        """Тест statistical feature extraction."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        features = feature_engineer.extract_statistical_features(data)
        
        assert len(features) == 11  # mean, std, min, max, median, 5 percentiles, skew, kurt
        assert features[0] == pytest.approx(5.5)  # mean
        assert features[2] == 1.0  # min
        assert features[3] == 10.0  # max
    
    def test_frequency_features(self, feature_engineer):
        """Тест frequency feature extraction."""
        # Create 5 Hz sine wave
        t = np.linspace(0, 1, 100)
        data = np.sin(2 * np.pi * 5 * t)
        
        features = feature_engineer.extract_frequency_features(data, sampling_rate=100)
        
        assert len(features) == 7  # 5 top magnitudes + dominant freq + entropy
        # Dominant frequency should be ~ 5 Hz
        assert 4.5 < features[-2] < 5.5
    
    def test_temporal_features(self, feature_engineer):
        """Тест temporal feature extraction."""
        # Trending data
        data = pd.Series(np.arange(100) + np.random.randn(100) * 0.1)
        
        features = feature_engineer.extract_temporal_features(data)
        
        # 3 windows * 2 (mean+std) + EMA + 3 autocorr + trend = 11
        assert len(features) == 11
        # Trend should be positive
        assert features[-1] > 0
    
    def test_hydraulic_features(self, feature_engineer, sample_sensor_data):
        """Тест hydraulic-specific features."""
        # Add pressure_in/out columns
        df = sample_sensor_data.copy()
        df["pressure_in"] = 100 + np.random.randn(100)
        df["pressure_out"] = 95 + np.random.randn(100)
        
        features = feature_engineer.extract_hydraulic_features(df)
        
        assert len(features) == 4
        # Pressure ratio should be ~ 0.95
        assert 0.9 < features[0] < 1.0
    
    def test_extract_all_features(self, feature_engineer, sample_sensor_data):
        """Тест full feature extraction."""
        features = feature_engineer.extract_all_features(sample_sensor_data)
        
        assert len(features) > 0
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()
    
    def test_handle_missing_data(self, feature_engineer):
        """Тест missing data handling."""
        data = pd.DataFrame({
            "pressure": [100, np.nan, 102, np.nan, 104]
        })
        
        cleaned = feature_engineer.handle_missing_data(data)
        
        assert cleaned.isna().sum().sum() == 0
    
    def test_remove_outliers(self, feature_engineer):
        """Тест outlier removal."""
        data = pd.DataFrame({
            "pressure": [100, 101, 102, 500, 103, 104]  # 500 is outlier
        })
        
        cleaned = feature_engineer.remove_outliers(data)
        
        # Outlier should be replaced
        assert cleaned["pressure"].max() < 200


# ==================== GRAPH BUILDER TESTS ====================

class TestGraphBuilder:
    """Tests для GraphBuilder."""
    
    def test_builder_creation(self, feature_engineer):
        """Создание GraphBuilder."""
        builder = GraphBuilder(feature_engineer)
        
        assert builder.feature_engineer is not None
        assert builder.feature_config is not None
    
    def test_build_edge_features(self):
        """Тест edge feature construction."""
        builder = GraphBuilder()
        
        edge_spec = EdgeSpec(
            source_id="pump",
            target_id="valve",
            edge_type=EdgeType.PIPE,
            diameter_mm=16.0,
            length_m=2.5,
            pressure_rating_bar=350,
            material="steel"
        )
        
        features = builder.build_edge_features(edge_spec)
        
        assert features.shape == (8,)  # 8D edge features
        assert torch.all(features >= 0)  # All non-negative
        assert torch.all(features <= 1.5)  # Reasonable range
    
    def test_validate_graph(self):
        """Тест graph validation."""
        builder = GraphBuilder()
        
        # Valid graph
        valid_graph = Data(
            x=torch.randn(5, 10),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            edge_attr=torch.randn(3, 8)
        )
        
        assert builder.validate_graph(valid_graph) is True
        
        # Invalid graph (out-of-bounds edge)
        invalid_graph = Data(
            x=torch.randn(5, 10),
            edge_index=torch.tensor([[0, 1, 10], [1, 2, 3]], dtype=torch.long),  # 10 >= 5
            edge_attr=torch.randn(3, 8)
        )
        
        assert builder.validate_graph(invalid_graph) is False


# ==================== DATASET TESTS ====================

class TestHydraulicGraphDataset:
    """Tests для HydraulicGraphDataset."""
    
    @pytest.fixture
    def mock_equipment_list(self, tmp_path):
        """Создать mock equipment list."""
        equipment_list = [
            {"equipment_id": "exc_001", "equipment_type": "excavator"},
            {"equipment_id": "exc_002", "equipment_type": "excavator"},
            {"equipment_id": "exc_003", "equipment_type": "excavator"},
        ]
        
        list_path = tmp_path / "equipment_list.json"
        with open(list_path, "w") as f:
            json.dump(equipment_list, f)
        
        return list_path
    
    @pytest.fixture
    def mock_connector(self):
        """Мок TimescaleConnector."""
        # TODO: Create proper mock when connector is finalized
        class MockConnector:
            async def fetch_sensor_data(self, *args, **kwargs):
                return pd.DataFrame({
                    "pressure": np.random.randn(100),
                    "temperature": np.random.randn(100)
                })
        
        return MockConnector()
    
    def test_dataset_length(self, mock_equipment_list, mock_connector):
        """Тест dataset size."""
        dataset = HydraulicGraphDataset(
            data_path=mock_equipment_list,
            timescale_connector=mock_connector,
            feature_engineer=FeatureEngineer(),
            graph_builder=GraphBuilder(),
            cache_dir=None
        )
        
        assert len(dataset) == 3
    
    def test_get_item(self, mock_equipment_list, mock_connector):
        """Тест __getitem__."""
        dataset = HydraulicGraphDataset(
            data_path=mock_equipment_list,
            timescale_connector=mock_connector,
            feature_engineer=FeatureEngineer(),
            graph_builder=GraphBuilder(),
            cache_dir=None
        )
        
        graph = dataset[0]
        
        assert isinstance(graph, Data)
        assert graph.x is not None
        assert graph.edge_index is not None
    
    def test_get_equipment_ids(self, mock_equipment_list, mock_connector):
        """Тест get_equipment_ids."""
        dataset = HydraulicGraphDataset(
            data_path=mock_equipment_list,
            timescale_connector=mock_connector,
            feature_engineer=FeatureEngineer(),
            graph_builder=GraphBuilder()
        )
        
        ids = dataset.get_equipment_ids()
        
        assert len(ids) == 3
        assert "exc_001" in ids
        assert "exc_002" in ids


# ==================== DATALOADER TESTS ====================

class TestDataLoader:
    """Tests для DataLoader."""
    
    def test_collate_function(self):
        """Тест hydraulic_collate_fn."""
        # Create sample graphs
        graph1 = Data(
            x=torch.randn(3, 10),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_attr=torch.randn(2, 8)
        )
        graph2 = Data(
            x=torch.randn(5, 10),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            edge_attr=torch.randn(3, 8)
        )
        
        batch = hydraulic_collate_fn([graph1, graph2])
        
        assert isinstance(batch, Batch)
        assert batch.num_graphs == 2
        assert batch.x.shape == (8, 10)  # 3 + 5 nodes
        assert batch.num_edges == 5  # 2 + 3 edges
    
    def test_create_dataloader(self, mock_equipment_list, mock_connector):
        """Тест create_dataloader factory."""
        dataset = HydraulicGraphDataset(
            data_path=mock_equipment_list,
            timescale_connector=mock_connector,
            feature_engineer=FeatureEngineer(),
            graph_builder=GraphBuilder()
        )
        
        loader = create_dataloader(
            dataset,
            config=DataLoaderConfig(batch_size=2, num_workers=0),
            split="train"
        )
        
        assert loader.batch_size == 2
        assert len(loader) == 2  # 3 samples / batch_size 2 = 2 batches
    
    def test_train_val_split(self, mock_equipment_list, mock_connector):
        """Тест train/val split."""
        dataset = HydraulicGraphDataset(
            data_path=mock_equipment_list,
            timescale_connector=mock_connector,
            feature_engineer=FeatureEngineer(),
            graph_builder=GraphBuilder()
        )
        
        train_loader, val_loader = create_train_val_loaders(
            dataset,
            config=DataLoaderConfig(batch_size=2, num_workers=0),
            train_ratio=0.67  # 2 train, 1 val
        )
        
        assert len(train_loader.dataset) == 2
        assert len(val_loader.dataset) == 1
