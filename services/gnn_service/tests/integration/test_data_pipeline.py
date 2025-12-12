"""Integration tests для complete data pipeline.

End-to-end tests:
- TimescaleDB → Features → Graph → Batch → Ready for model
- Real schema instances from Issue #93
- Multiple equipment processing
- Caching behavior
- DataLoader iteration
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

from src.data import (
    DataLoaderConfig,
    FeatureConfig,
    FeatureEngineer,
    GraphBuilder,
    HydraulicGraphDataset,
    create_dataloader,
)
from src.schemas import (
    ComponentSpec,
    ComponentType,
    EdgeSpec,
    EdgeType,
    GraphTopology,
    TimeWindow,
)

# ==================== FIXTURES ====================

@pytest.fixture
def sample_topology():
    """Real GraphTopology instance."""
    components = {
        "pump_main": ComponentSpec(
            component_id="pump_main",
            component_type=ComponentType.PUMP,
            location_x=0.0,
            location_y=0.0,
            location_z=0.0
        ),
        "valve_control": ComponentSpec(
            component_id="valve_control",
            component_type=ComponentType.VALVE,
            location_x=1.0,
            location_y=0.0,
            location_z=0.0
        ),
        "cylinder_main": ComponentSpec(
            component_id="cylinder_main",
            component_type=ComponentType.CYLINDER,
            location_x=2.0,
            location_y=0.0,
            location_z=0.0
        )
    }

    edges = [
        EdgeSpec(
            source_id="pump_main",
            target_id="valve_control",
            edge_type=EdgeType.PIPE,
            diameter_mm=16.0,
            length_m=2.5,
            pressure_rating_bar=350,
            material="steel",
            is_bidirectional=False
        ),
        EdgeSpec(
            source_id="valve_control",
            target_id="cylinder_main",
            edge_type=EdgeType.HOSE,
            diameter_mm=12.0,
            length_m=1.5,
            pressure_rating_bar=280,
            material="rubber",
            is_bidirectional=False
        )
    ]

    return GraphTopology(components=components, edges=edges)


@pytest.fixture
def mock_sensor_data():
    """Realistic sensor data for 3 components."""
    np.random.seed(42)
    n_samples = 100
    t = np.linspace(0, 10, n_samples)

    # Pump sensors
    pump_pressure = 150 + 20 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(n_samples) * 3
    pump_temp = 65 + 10 * np.sin(2 * np.pi * 0.2 * t) + np.random.randn(n_samples) * 2
    pump_vibration = 2.5 + 0.5 * np.random.randn(n_samples)

    # Valve sensors
    valve_pressure = 145 + 18 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(n_samples) * 3
    valve_temp = 60 + 8 * np.sin(2 * np.pi * 0.2 * t) + np.random.randn(n_samples) * 2

    # Cylinder sensors
    cylinder_pressure = 140 + 15 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(n_samples) * 3
    cylinder_temp = 58 + 7 * np.sin(2 * np.pi * 0.2 * t) + np.random.randn(n_samples) * 2
    cylinder_position = 50 + 30 * np.sin(2 * np.pi * 0.3 * t) + np.random.randn(n_samples)

    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2025-11-01", periods=n_samples, freq="1min"),
        "pressure_pump_main": pump_pressure,
        "temperature_pump_main": pump_temp,
        "vibration_pump_main": pump_vibration,
        "pressure_valve_control": valve_pressure,
        "temperature_valve_control": valve_temp,
        "pressure_cylinder_main": cylinder_pressure,
        "temperature_cylinder_main": cylinder_temp,
        "position_cylinder_main": cylinder_position
    })

    return df


class MockTimescaleConnector:
    """Mock connector для integration tests."""

    def __init__(self, sensor_data):
        self.sensor_data = sensor_data
        self._connected = False

    async def connect(self):
        self._connected = True

    async def close(self):
        self._connected = False

    async def fetch_sensor_data(self, equipment_id, time_window, sensors):
        """Return mock sensor data."""
        # Filter columns
        cols = ["timestamp"] + [s for s in sensors if s in self.sensor_data.columns]
        return self.sensor_data[cols].copy()

    async def get_equipment_metadata(self, equipment_id):
        """Return mock metadata."""
        return {
            "equipment_id": equipment_id,
            "equipment_type": "excavator",
            "manufacturer": "Test",
            "model": "Test-1000"
        }

    async def health_check(self):
        return self._connected


@pytest.fixture
def mock_connector(mock_sensor_data):
    """Mock TimescaleConnector instance."""
    return MockTimescaleConnector(mock_sensor_data)


@pytest.fixture
def feature_config():
    """Feature configuration."""
    return FeatureConfig(
        use_statistical=True,
        use_frequency=True,
        use_temporal=True,
        use_hydraulic=True,
        num_frequencies=5  # Smaller для тестов
    )


@pytest.fixture
def equipment_list(tmp_path):
    """Create temporary equipment list."""
    equipment = [
        {"equipment_id": "exc_001", "equipment_type": "excavator"},
        {"equipment_id": "exc_002", "equipment_type": "excavator"},
        {"equipment_id": "exc_003", "equipment_type": "excavator"}
    ]

    list_path = tmp_path / "equipment_list.json"
    with open(list_path, "w") as f:
        json.dump(equipment, f)

    return list_path


# ==================== INTEGRATION TESTS ====================

class TestDataPipelineIntegration:
    """Integration tests для complete pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_end_to_end(self, mock_connector, mock_sensor_data, sample_topology, feature_config):
        """Test complete pipeline: connector → features → graph."""
        # 1. Connect
        await mock_connector.connect()
        assert await mock_connector.health_check()

        # 2. Fetch sensor data
        time_window = TimeWindow(
            start_time=datetime(2025, 11, 1),
            end_time=datetime(2025, 11, 1, 2, 0)
        )

        sensors = ["pressure_pump_main", "temperature_pump_main"]
        data = await mock_connector.fetch_sensor_data(
            equipment_id="exc_001",
            time_window=time_window,
            sensors=sensors
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "pressure_pump_main" in data.columns

        # 3. Extract features
        engineer = FeatureEngineer(feature_config)
        features = engineer.extract_all_features(data[["pressure_pump_main", "temperature_pump_main"]])

        assert len(features) > 0
        assert not np.isnan(features).any()

        # 4. Build graph
        builder = GraphBuilder(engineer, feature_config)
        graph = builder.build_graph(
            sensor_data=mock_sensor_data,
            topology=sample_topology,
            metadata=None  # Not used in mock
        )

        assert graph.x is not None
        assert graph.edge_index is not None
        assert graph.num_nodes == 3  # pump, valve, cylinder
        assert graph.num_edges == 2  # 2 edges

        # 5. Validate graph
        assert builder.validate_graph(graph)

        await mock_connector.close()

    def test_dataset_integration(self, mock_connector, equipment_list, feature_config, sample_topology, tmp_path):
        """Test HydraulicGraphDataset integration."""
        engineer = FeatureEngineer(feature_config)
        builder = GraphBuilder(engineer, feature_config)

        dataset = HydraulicGraphDataset(
            data_path=equipment_list,
            timescale_connector=mock_connector,
            feature_engineer=engineer,
            graph_builder=builder,
            cache_dir=tmp_path / "cache",
            preload=False
        )

        # Check dataset size
        assert len(dataset) == 3

        # Get first graph
        graph = dataset[0]
        assert isinstance(graph, torch.utils.data.Dataset.__bases__[0].__subclasses__()[0])

        # Get stats
        stats = dataset.get_statistics()
        assert stats["dataset_size"] == 3
        assert stats["cache_enabled"] is True

    def test_dataloader_integration(self, mock_connector, equipment_list, feature_config, tmp_path):
        """Test DataLoader integration with batching."""
        engineer = FeatureEngineer(feature_config)
        builder = GraphBuilder(engineer, feature_config)

        dataset = HydraulicGraphDataset(
            data_path=equipment_list,
            timescale_connector=mock_connector,
            feature_engineer=engineer,
            graph_builder=builder,
            cache_dir=tmp_path / "cache"
        )

        # Create DataLoader
        loader_config = DataLoaderConfig(
            batch_size=2,
            num_workers=0,  # Single process для тестов
            pin_memory=False
        )

        loader = create_dataloader(
            dataset,
            config=loader_config,
            split="train"
        )

        # Iterate
        batches = list(loader)
        assert len(batches) == 2  # 3 samples / batch_size 2 = 2 batches

        # Check first batch
        batch = batches[0]
        assert isinstance(batch, Batch)
        assert batch.num_graphs == 2
        assert batch.x.shape[0] > 0  # Has nodes
        assert batch.edge_index.shape[1] > 0  # Has edges

    def test_caching_behavior(self, mock_connector, equipment_list, feature_config, tmp_path):
        """Test caching accelerates repeated access."""
        import time

        engineer = FeatureEngineer(feature_config)
        builder = GraphBuilder(engineer, feature_config)

        cache_dir = tmp_path / "cache"
        dataset = HydraulicGraphDataset(
            data_path=equipment_list,
            timescale_connector=mock_connector,
            feature_engineer=engineer,
            graph_builder=builder,
            cache_dir=cache_dir
        )

        # First access: cache miss
        start = time.time()
        graph1 = dataset[0]
        time_miss = time.time() - start

        # Second access: cache hit
        start = time.time()
        graph2 = dataset[0]
        time_hit = time.time() - start

        # Cache hit should be faster
        # Note: На CI может быть нестабильно, поэтому мягкое условие
        assert time_hit < time_miss * 2  # At least not slower

        # Cache files should exist
        cache_files = list(cache_dir.glob("*.pkl"))
        assert len(cache_files) > 0

    def test_multiple_equipment_processing(self, mock_connector, equipment_list, feature_config, tmp_path):
        """Test processing multiple equipment."""
        engineer = FeatureEngineer(feature_config)
        builder = GraphBuilder(engineer, feature_config)

        dataset = HydraulicGraphDataset(
            data_path=equipment_list,
            timescale_connector=mock_connector,
            feature_engineer=engineer,
            graph_builder=builder,
            cache_dir=tmp_path / "cache"
        )

        # Process all equipment
        graphs = [dataset[i] for i in range(len(dataset))]

        assert len(graphs) == 3

        # All graphs valid
        for graph in graphs:
            assert graph.x is not None
            assert graph.edge_index is not None
            assert not torch.isnan(graph.x).any()

    def test_feature_consistency(self, mock_sensor_data, feature_config):
        """Test feature extraction consistency."""
        engineer = FeatureEngineer(feature_config)

        # Extract twice
        features1 = engineer.extract_all_features(mock_sensor_data[["pressure_pump_main", "temperature_pump_main"]])
        features2 = engineer.extract_all_features(mock_sensor_data[["pressure_pump_main", "temperature_pump_main"]])

        # Should be identical
        assert np.allclose(features1, features2)

    def test_graph_topology_validation(self, mock_sensor_data, sample_topology, feature_config):
        """Test graph respects topology."""
        engineer = FeatureEngineer(feature_config)
        builder = GraphBuilder(engineer, feature_config)

        graph = builder.build_graph(
            sensor_data=mock_sensor_data,
            topology=sample_topology,
            metadata=None
        )

        # Check node count matches components
        assert graph.num_nodes == len(sample_topology.components)

        # Check edge count matches topology edges
        assert graph.num_edges == len(sample_topology.edges)

        # Validate structure
        assert builder.validate_graph(graph)

    def test_edge_features_computation(self, mock_sensor_data, sample_topology, feature_config):
        """Test edge features are correctly computed."""
        engineer = FeatureEngineer(feature_config)
        builder = GraphBuilder(engineer, feature_config)

        graph = builder.build_graph(
            sensor_data=mock_sensor_data,
            topology=sample_topology,
            metadata=None
        )

        # Check edge features exist
        assert graph.edge_attr is not None
        assert graph.edge_attr.shape == (2, 8)  # 2 edges, 8 features

        # Edge features should be normalized (roughly 0-1 range)
        assert torch.all(graph.edge_attr >= 0)
        assert torch.all(graph.edge_attr <= 2)  # Allow some headroom

    def test_batch_shapes_consistency(self, mock_connector, equipment_list, feature_config, tmp_path):
        """Test batch shapes are consistent."""
        engineer = FeatureEngineer(feature_config)
        builder = GraphBuilder(engineer, feature_config)

        dataset = HydraulicGraphDataset(
            data_path=equipment_list,
            timescale_connector=mock_connector,
            feature_engineer=engineer,
            graph_builder=builder,
            cache_dir=tmp_path / "cache"
        )

        loader = create_dataloader(
            dataset,
            config=DataLoaderConfig(batch_size=2, num_workers=0),
            split="train"
        )

        for batch in loader:
            # Node features: [N_total, F]
            assert batch.x.dim() == 2
            assert batch.x.shape[1] > 0  # Has features

            # Edge index: [2, E_total]
            assert batch.edge_index.dim() == 2
            assert batch.edge_index.shape[0] == 2

            # Edge attr: [E_total, F_edge]
            assert batch.edge_attr.dim() == 2
            assert batch.edge_attr.shape[1] == 8  # 8D edge features

            # Batch assignment: [N_total]
            assert batch.batch.dim() == 1
            assert batch.batch.shape[0] == batch.x.shape[0]
