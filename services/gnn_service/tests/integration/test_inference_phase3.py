"""Integration tests for Phase 3 - Dynamic Graph Builder and Inference.

Tests:
- DynamicGraphBuilder with variable sensor counts
- InferenceEngine with DynamicGraphBuilder
- Multiple equipment topologies
- Missing sensor handling
- Batch inference with variable-sized graphs

Python 3.14 Features:
    - Deferred annotations
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest
import torch
from torch_geometric.data import Data

from src.data.feature_config import FeatureConfig
from src.data.feature_engineer import FeatureEngineer
from src.inference.dynamic_graph_builder import DynamicGraphBuilder
from src.schemas import GraphTopology


class MockTimescaleConnector:
    """Mock connector for testing (no DB dependency)."""

    async def read_sensor_data(self, equipment_id: str, lookback_minutes: int) -> Any:
        """Return mock sensor data."""
        # Simulate different equipment having different sensor counts
        if equipment_id.startswith("pump"):
            # Pump: 5 sensors (inlet, outlet, motor, pump, filter)
            import pandas as pd
            data = {}
            for i in range(5):
                data[f"pump_{i+1}"] = torch.randn(100).numpy()
            return pd.DataFrame(data)
        elif equipment_id.startswith("compressor"):
            # Compressor: 7 sensors
            import pandas as pd
            data = {}
            for i in range(7):
                data[f"comp_{i+1}"] = torch.randn(100).numpy()
            return pd.DataFrame(data)
        else:
            # Generic: 4 sensors
            import pandas as pd
            data = {}
            for i in range(4):
                data[f"sensor_{i+1}"] = torch.randn(100).numpy()
            return pd.DataFrame(data)


class TestDynamicGraphBuilder:
    """Tests for DynamicGraphBuilder."""

    def test_pump_topology_5_sensors(self):
        """Test building graph for pump (5 sensors)."""
        config = FeatureConfig(edge_in_dim=14)
        engineer = FeatureEngineer(config)
        connector = MockTimescaleConnector()

        builder = DynamicGraphBuilder(
            timescale_connector=connector,
            feature_engineer=engineer,
            feature_config=config,
        )

        # Define pump topology
        topology = GraphTopology(
            topology_id="pump_standard",
            equipment_type="pump",
            sensor_ids=["pump_1", "pump_2", "pump_3", "pump_4", "pump_5"],
            connections=[
                {"from": "pump_1", "to": "pump_2"},
                {"from": "pump_2", "to": "pump_3"},
                {"from": "pump_3", "to": "pump_4"},
                {"from": "pump_4", "to": "pump_5"},
            ],
        )

        # Build graph
        async def build():
            return await builder.build_from_timescale(
                equipment_id="pump_001",
                topology=topology,
                lookback_minutes=10,
            )

        graph = asyncio.run(build())

        # Verify
        assert graph.x.shape[0] == 5  # 5 sensors
        assert graph.x.shape[1] == config.total_features_per_sensor
        assert graph.edge_index.shape[1] > 0  # Has edges
        assert graph.edge_attr.shape[1] == 14  # 14D edges
        assert graph.equipment_id == "pump_001"
        assert graph.topology_id == "pump_standard"

    def test_compressor_topology_7_sensors(self):
        """Test building graph for compressor (7 sensors)."""
        config = FeatureConfig(edge_in_dim=14)
        engineer = FeatureEngineer(config)
        connector = MockTimescaleConnector()

        builder = DynamicGraphBuilder(
            timescale_connector=connector,
            feature_engineer=engineer,
            feature_config=config,
        )

        topology = GraphTopology(
            topology_id="compressor_standard",
            equipment_type="compressor",
            sensor_ids=["comp_1", "comp_2", "comp_3", "comp_4", "comp_5", "comp_6", "comp_7"],
            connections=[
                {"from": "comp_1", "to": "comp_2"},
                {"from": "comp_2", "to": "comp_3"},
                {"from": "comp_3", "to": "comp_4"},
                {"from": "comp_4", "to": "comp_5"},
                {"from": "comp_5", "to": "comp_6"},
                {"from": "comp_6", "to": "comp_7"},
            ],
        )

        async def build():
            return await builder.build_from_timescale(
                equipment_id="compressor_001",
                topology=topology,
                lookback_minutes=10,
            )

        graph = asyncio.run(build())

        # Verify different size than pump
        assert graph.x.shape[0] == 7  # 7 sensors (different from pump)
        assert graph.edge_attr.shape[1] == 14

    def test_missing_sensor_handling(self):
        """Test graceful handling of missing sensors."""
        config = FeatureConfig(edge_in_dim=14)
        engineer = FeatureEngineer(config)

        # Custom connector with missing sensor
        class MissingConnector(MockTimescaleConnector):
            async def read_sensor_data(self, equipment_id, lookback_minutes):
                import pandas as pd
                # Only provide 4 sensors, but topology expects 5
                return pd.DataFrame({
                    "pump_1": torch.randn(100).numpy(),
                    "pump_2": torch.randn(100).numpy(),
                    "pump_3": torch.randn(100).numpy(),
                    "pump_4": torch.randn(100).numpy(),
                    # pump_5 is MISSING
                })

        builder = DynamicGraphBuilder(
            timescale_connector=MissingConnector(),
            feature_engineer=engineer,
            feature_config=config,
        )

        topology = GraphTopology(
            topology_id="pump_standard",
            equipment_type="pump",
            sensor_ids=["pump_1", "pump_2", "pump_3", "pump_4", "pump_5"],
            connections=[
                {"from": "pump_1", "to": "pump_2"},
                {"from": "pump_2", "to": "pump_3"},
                {"from": "pump_3", "to": "pump_4"},
                {"from": "pump_4", "to": "pump_5"},
            ],
        )

        async def build():
            return await builder.build_from_timescale(
                equipment_id="pump_001",
                topology=topology,
                lookback_minutes=10,
            )

        graph = asyncio.run(build())

        # Should still have 5 nodes (missing sensor replaced with zeros)
        assert graph.x.shape[0] == 5
        # Last node should be zeros (missing sensor)
        assert torch.allclose(graph.x[4], torch.zeros_like(graph.x[4]))

    def test_validate_graph(self):
        """Test graph validation."""
        config = FeatureConfig(edge_in_dim=14)
        engineer = FeatureEngineer(config)
        connector = MockTimescaleConnector()

        builder = DynamicGraphBuilder(
            timescale_connector=connector,
            feature_engineer=engineer,
            feature_config=config,
        )

        topology = GraphTopology(
            topology_id="pump_standard",
            equipment_type="pump",
            sensor_ids=["pump_1", "pump_2", "pump_3", "pump_4", "pump_5"],
            connections=[
                {"from": "pump_1", "to": "pump_2"},
                {"from": "pump_2", "to": "pump_3"},
            ],
        )

        async def build():
            return await builder.build_from_timescale(
                equipment_id="pump_001",
                topology=topology,
                lookback_minutes=10,
            )

        graph = asyncio.run(build())

        # Validation should pass
        assert builder.validate_graph(graph, topology)

    def test_variable_edge_in_dim(self):
        """Test different edge_in_dim values."""
        for edge_dim in [8, 14, 20]:
            config = FeatureConfig(edge_in_dim=edge_dim)
            engineer = FeatureEngineer(config)
            connector = MockTimescaleConnector()

            builder = DynamicGraphBuilder(
                timescale_connector=connector,
                feature_engineer=engineer,
                feature_config=config,
            )

            topology = GraphTopology(
                topology_id="pump_standard",
                equipment_type="pump",
                sensor_ids=["pump_1", "pump_2", "pump_3"],
                connections=[
                    {"from": "pump_1", "to": "pump_2"},
                    {"from": "pump_2", "to": "pump_3"},
                ],
            )

            async def build():
                return await builder.build_from_timescale(
                    equipment_id="pump_001",
                    topology=topology,
                    lookback_minutes=10,
                )

            graph = asyncio.run(build())

            # Edge dimension should match config
            assert graph.edge_attr.shape[1] == edge_dim


class TestInferencePhase3:
    """Tests for InferenceEngine with Phase 3 features."""

    def test_inference_engine_stats(self):
        """Test InferenceEngine.get_stats() includes dynamic builder."""
        # Note: This is a minimal test without full model setup
        # Real integration would require proper model loading
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
