"""Model + Data integration tests.

Tests:
- UniversalTemporalGNN с HydraulicGraphDataset
- Forward pass с batched graphs
- Multi-task outputs
- Inference mode
- Gradient flow

Integrates:
- Issue #93 (schemas)
- Issue #94 (model)
- Issue #95 (data pipeline)
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import (
    DataLoaderConfig,
    FeatureConfig,
    FeatureEngineer,
    GraphBuilder,
    HydraulicGraphDataset,
    create_dataloader,
)
from src.models import UniversalTemporalGNN
from src.schemas import ComponentSpec, ComponentType, EdgeSpec, EdgeType, GraphTopology

# ==================== FIXTURES ====================

class MockTimescaleConnector:
    """Mock connector."""

    async def connect(self):
        pass

    async def close(self):
        pass

    async def fetch_sensor_data(self, equipment_id, time_window, sensors):
        """Return mock sensor data."""
        np.random.seed(42)
        n = 100

        data = {
            "timestamp": pd.date_range(start="2025-11-01", periods=n, freq="1min"),
            "pressure_pump_main": 150 + 20 * np.sin(np.linspace(0, 10, n)) + np.random.randn(n) * 3,
            "temperature_pump_main": 65 + 10 * np.sin(np.linspace(0, 10, n)) + np.random.randn(n) * 2,
            "vibration_pump_main": 2.5 + 0.5 * np.random.randn(n),
            "pressure_valve_control": 145 + 18 * np.sin(np.linspace(0, 10, n)) + np.random.randn(n) * 3,
            "temperature_valve_control": 60 + 8 * np.sin(np.linspace(0, 10, n)) + np.random.randn(n) * 2,
        }

        df = pd.DataFrame(data)
        cols = ["timestamp"] + [s for s in sensors if s in df.columns]
        return df[cols]


@pytest.fixture
def mock_connector():
    return MockTimescaleConnector()


@pytest.fixture
def sample_topology():
    """Sample topology с 3 nodes."""
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
            material="steel"
        ),
        EdgeSpec(
            source_id="valve_control",
            target_id="cylinder_main",
            edge_type=EdgeType.HOSE,
            diameter_mm=12.0,
            length_m=1.5,
            pressure_rating_bar=280,
            material="rubber"
        )
    ]

    return GraphTopology(components=components, edges=edges)


@pytest.fixture
def equipment_list(tmp_path):
    """Mock equipment list."""
    equipment = [
        {"equipment_id": "exc_001", "equipment_type": "excavator"},
        {"equipment_id": "exc_002", "equipment_type": "excavator"},
        {"equipment_id": "exc_003", "equipment_type": "excavator"},
        {"equipment_id": "exc_004", "equipment_type": "excavator"}
    ]

    list_path = tmp_path / "equipment.json"
    with open(list_path, "w") as f:
        json.dump(equipment, f)

    return list_path


@pytest.fixture
def feature_config():
    return FeatureConfig(
        use_statistical=True,
        use_frequency=True,
        use_temporal=True,
        use_hydraulic=True,
        num_frequencies=5
    )


# ==================== INTEGRATION TESTS ====================

class TestModelDataIntegration:
    """Integration tests для Model + Data."""

    def test_model_forward_with_dataloader(self, mock_connector, equipment_list, feature_config, tmp_path):
        """Test model forward pass с real DataLoader."""
        # Setup data pipeline
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

        # Initialize model
        in_channels = feature_config.total_features_per_sensor
        model = UniversalTemporalGNN(
            in_channels=in_channels,
            hidden_channels=32,  # Small для тестов
            num_heads=2,
            num_gat_layers=2,
            lstm_hidden=64,
            lstm_layers=1,
            use_compile=False  # Disable для тестов
        )

        model.eval()

        # Forward pass
        with torch.no_grad():
            for batch in loader:
                health, degradation, anomaly = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch
                )

                # Check outputs
                assert health.shape == (batch.num_graphs, 1)
                assert degradation.shape == (batch.num_graphs, 1)
                assert anomaly.shape == (batch.num_graphs, 9)  # 9 anomaly types

                # Check ranges
                assert torch.all((health >= 0) & (health <= 1))  # Sigmoid output
                assert torch.all((degradation >= 0) & (degradation <= 1))
                # Anomaly logits can be any value

                break  # Test first batch only

    def test_model_output_shapes(self, mock_connector, equipment_list, feature_config, tmp_path):
        """Test output shapes для different batch sizes."""
        engineer = FeatureEngineer(feature_config)
        builder = GraphBuilder(engineer, feature_config)

        dataset = HydraulicGraphDataset(
            data_path=equipment_list,
            timescale_connector=mock_connector,
            feature_engineer=engineer,
            graph_builder=builder,
            cache_dir=tmp_path / "cache"
        )

        in_channels = feature_config.total_features_per_sensor
        model = UniversalTemporalGNN(
            in_channels=in_channels,
            hidden_channels=32,
            num_heads=2,
            num_gat_layers=2,
            lstm_hidden=64,
            lstm_layers=1,
            use_compile=False
        )
        model.eval()

        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            loader = create_dataloader(
                dataset,
                config=DataLoaderConfig(batch_size=batch_size, num_workers=0),
                split="val"
            )

            with torch.no_grad():
                for batch in loader:
                    health, degradation, anomaly = model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch
                    )

                    # Outputs match batch size (or less for last batch)
                    assert health.shape[0] <= batch_size
                    assert health.shape[0] == batch.num_graphs

                    break  # Test first batch only

    def test_gradient_flow(self, mock_connector, equipment_list, feature_config, tmp_path):
        """Test backward pass (gradient flow)."""
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

        in_channels = feature_config.total_features_per_sensor
        model = UniversalTemporalGNN(
            in_channels=in_channels,
            hidden_channels=32,
            num_heads=2,
            num_gat_layers=2,
            lstm_hidden=64,
            lstm_layers=1,
            use_compile=False
        )
        model.train()

        # Forward + backward
        for batch in loader:
            health, degradation, anomaly = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch
            )

            # Dummy loss
            loss = health.mean() + degradation.mean() + anomaly.mean()

            # Backward
            loss.backward()

            # Check gradients exist
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"No gradient for {name}"

            break  # Test first batch only

    def test_inference_mode(self, mock_connector, equipment_list, feature_config, tmp_path):
        """Test inference mode (no gradients)."""
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
            split="test"
        )

        in_channels = feature_config.total_features_per_sensor
        model = UniversalTemporalGNN(
            in_channels=in_channels,
            hidden_channels=32,
            num_heads=2,
            num_gat_layers=2,
            lstm_hidden=64,
            lstm_layers=1,
            use_compile=False
        )
        model.eval()

        # Inference mode
        with torch.inference_mode():
            for batch in loader:
                health, degradation, anomaly = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch
                )

                # Outputs should not require grad
                assert not health.requires_grad
                assert not degradation.requires_grad
                assert not anomaly.requires_grad

                break

    def test_multi_task_outputs(self, mock_connector, equipment_list, feature_config, tmp_path):
        """Test all 3 task outputs are valid."""
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

        in_channels = feature_config.total_features_per_sensor
        model = UniversalTemporalGNN(
            in_channels=in_channels,
            hidden_channels=32,
            num_heads=2,
            num_gat_layers=2,
            lstm_hidden=64,
            lstm_layers=1,
            use_compile=False
        )
        model.eval()

        with torch.no_grad():
            for batch in loader:
                health, degradation, anomaly = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch
                )

                # Health: [B, 1] in [0, 1]
                assert health.dim() == 2
                assert health.shape[1] == 1
                assert torch.all((health >= 0) & (health <= 1))

                # Degradation: [B, 1] in [0, 1]
                assert degradation.dim() == 2
                assert degradation.shape[1] == 1
                assert torch.all((degradation >= 0) & (degradation <= 1))

                # Anomaly: [B, 9] logits (any value)
                assert anomaly.dim() == 2
                assert anomaly.shape[1] == 9
                assert not torch.isnan(anomaly).any()
                assert not torch.isinf(anomaly).any()

                break
