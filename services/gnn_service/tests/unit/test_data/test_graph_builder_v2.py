"""Unit tests for GraphBuilderV2 (Edge-Centric Graph Construction).

Tests for Phase 3.2 edge-centric architecture:
    - Node features (29D minimal)
    - Edge features (14-116D rich)
    - Graph construction from HybridInferenceRequest
    - DiagnosticScope support
    - Validation logic

Test Coverage:
    - Node feature extraction (5 tests)
    - Edge feature extraction (5 tests)
    - Graph construction (5 tests)
    - Validation (5 tests)
    - Helper methods (3 tests)

Total: 23 test methods covering all functionality.
"""

from datetime import datetime, UTC, date, timedelta

import pytest
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from src.data.graph_builder_v2 import (
    GraphBuilderV2,
    NODE_FEATURES_TOTAL_DIM,
    EDGE_STATIC_DIM,
    EDGE_DYNAMIC_INSTANT_DIM,
    COMPONENT_TYPE_MAPPING,
)
from src.schemas.requests import (
    HybridInferenceRequest,
    EdgeSensorReading,
    ComponentSensorReading,
    DiagnosticScope,
)
from src.schemas.topology import (
    ComponentConfiguration,
    EdgeConfiguration,
    TopologyConfig,
)
from src.data.feature_config import FeatureConfig
from src.data.feature_engineer import FeatureEngineer


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def feature_config_basic() -> FeatureConfig:
    """Fixture: Basic feature config with 14D edges."""
    return FeatureConfig(
        edge_in_dim=14,  # 8 static + 6 dynamic
        hidden_dim=64,
        output_dim=32,
    )


@pytest.fixture
def feature_config_with_timeseries() -> FeatureConfig:
    """Fixture: Advanced feature config with 48D edges (+ time-series)."""
    return FeatureConfig(
        edge_in_dim=48,  # 8 static + 6 dynamic + 34 timeseries
        hidden_dim=64,
        output_dim=32,
    )


@pytest.fixture
def builder_basic(feature_config_basic: FeatureConfig) -> GraphBuilderV2:
    """Fixture: GraphBuilderV2 with basic config (14D edges)."""
    feature_engineer = FeatureEngineer(feature_config_basic)
    return GraphBuilderV2(
        feature_engineer=feature_engineer,
        feature_config=feature_config_basic,
        use_edge_timeseries=False,
    )


@pytest.fixture
def builder_with_timeseries(feature_config_with_timeseries: FeatureConfig) -> GraphBuilderV2:
    """Fixture: GraphBuilderV2 with time-series support (48D edges)."""
    feature_engineer = FeatureEngineer(feature_config_with_timeseries)
    return GraphBuilderV2(
        feature_engineer=feature_engineer,
        feature_config=feature_config_with_timeseries,
        use_edge_timeseries=True,
    )


@pytest.fixture
def simple_topology() -> TopologyConfig:
    """Fixture: Simple pump-valve-cylinder topology."""
    return TopologyConfig(
        topology_id="test_simple",
        name="Simple Test Topology",
        version="v1.0",
        components=[
            ComponentConfiguration(
                component_id="pump_main",
                component_type="piston_pump",
                name="Main Piston Pump",
                nominal_pressure_bar=250.0,
                nominal_flow_lpm=150.0,
            ),
            ComponentConfiguration(
                component_id="valve_01",
                component_type="proportional_valve",
                name="Proportional Control Valve",
                nominal_pressure_bar=245.0,
                nominal_flow_lpm=150.0,
            ),
            ComponentConfiguration(
                component_id="cylinder_left",
                component_type="cylinder",
                name="Boom Cylinder Left",
                nominal_pressure_bar=240.0,
                nominal_flow_lpm=75.0,
            ),
        ],
        edges=[
            EdgeConfiguration(
                source_id="pump_main",
                target_id="valve_01",
                diameter_mm=25.0,
                length_m=5.0,
                material="steel",
                pressure_rating_bar=350.0,
            ),
            EdgeConfiguration(
                source_id="valve_01",
                target_id="cylinder_left",
                diameter_mm=20.0,
                length_m=3.0,
                material="rubber",
                pressure_rating_bar=300.0,
            ),
        ],
    )


@pytest.fixture
def hybrid_request_full(simple_topology: TopologyConfig) -> HybridInferenceRequest:
    """Fixture: HybridInferenceRequest for full topology."""
    timestamp = datetime.now(UTC)
    return HybridInferenceRequest(
        equipment_id="test_excavator_001",
        timestamp=timestamp,
        topology_id="test_simple",
        edge_readings={
            "pump_main__valve_01": EdgeSensorReading(
                edge_id="pump_main__valve_01",
                pressure_inlet_bar=250.2,
                pressure_outlet_bar=248.5,
                flow_rate_lpm=145.5,
                temperature_c=68.3,
                vibration_g=0.8,
                timestamp=timestamp,
            ),
            "valve_01__cylinder_left": EdgeSensorReading(
                edge_id="valve_01__cylinder_left",
                pressure_inlet_bar=248.0,
                pressure_outlet_bar=245.2,
                flow_rate_lpm=72.5,
                temperature_c=67.5,
                timestamp=timestamp,
            ),
        },
        component_readings={
            "pump_main": ComponentSensorReading(
                component_id="pump_main",
                rpm=1800,
                current_a=35.2,
                voltage_v=400,
                timestamp=timestamp,
            ),
            "valve_01": ComponentSensorReading(
                component_id="valve_01",
                position_percent=65.5,
                timestamp=timestamp,
            ),
            "cylinder_left": ComponentSensorReading(
                component_id="cylinder_left",
                position_percent=45.8,
                timestamp=timestamp,
            ),
        },
        diagnostic_scope=None,  # Full topology
    )


@pytest.fixture
def hybrid_request_focused(simple_topology: TopologyConfig) -> HybridInferenceRequest:
    """Fixture: HybridInferenceRequest with DiagnosticScope (focused on pump→valve)."""
    timestamp = datetime.now(UTC)
    return HybridInferenceRequest(
        equipment_id="test_excavator_001",
        timestamp=timestamp,
        topology_id="test_simple",
        edge_readings={
            "pump_main__valve_01": EdgeSensorReading(
                edge_id="pump_main__valve_01",
                pressure_inlet_bar=250.2,
                pressure_outlet_bar=248.5,
                flow_rate_lpm=145.5,
                timestamp=timestamp,
            ),
        },
        component_readings={
            "pump_main": ComponentSensorReading(
                component_id="pump_main",
                rpm=1800,
                timestamp=timestamp,
            ),
        },
        diagnostic_scope=DiagnosticScope(
            target_edges=["pump_main__valve_01"],
            include_context=False,
        ),
    )


# ============================================================================
# NODE FEATURE TESTS
# ============================================================================


class TestBuildNodeFeaturesV2:
    """Test cases for build_node_features_v2() method."""

    def test_piston_pump_with_rpm_and_current(self, builder_basic: GraphBuilderV2):
        """Test node features for piston pump with RPM and current."""
        # Arrange
        timestamp = datetime.now(UTC)
        reading = ComponentSensorReading(
            component_id="piston_pump_main",
            rpm=1450,  # 1450 / 3000 = 0.483
            current_a=25.5,  # 25.5 / 100 = 0.255
            voltage_v=400,  # 400 / 500 = 0.8
            timestamp=timestamp,
        )

        # Act
        features = builder_basic.build_node_features_v2(
            component_id="piston_pump_main",
            component_reading=reading,
            component_type="piston_pump",
        )

        # Assert
        assert features.shape == (NODE_FEATURES_TOTAL_DIM,), f"Expected {NODE_FEATURES_TOTAL_DIM}D, got {features.shape}"
        assert features.dtype == torch.float32

        # Internal sensors (first 4 values)
        assert abs(features[0].item() - 0.483) < 0.01, "RPM normalization incorrect"
        assert features[1].item() == 0.0, "Position should be 0 (not provided)"
        assert abs(features[2].item() - 0.255) < 0.01, "Current normalization incorrect"
        assert abs(features[3].item() - 0.8) < 0.01, "Voltage normalization incorrect"

        # Component type one-hot (piston_pump should be at index 2)
        type_index = COMPONENT_TYPE_MAPPING["piston_pump"]
        assert features[4 + type_index].item() == 1.0, "piston_pump one-hot encoding incorrect"
        assert features[4:].sum().item() == 1.0, "Only one component type should be hot"

    def test_proportional_valve_with_position(self, builder_basic: GraphBuilderV2):
        """Test node features for proportional valve with position."""
        # Arrange
        timestamp = datetime.now(UTC)
        reading = ComponentSensorReading(
            component_id="proportional_valve_01",
            position_percent=65.5,  # 65.5 / 100 = 0.655
            timestamp=timestamp,
        )

        # Act
        features = builder_basic.build_node_features_v2(
            component_id="proportional_valve_01",
            component_reading=reading,
            component_type="proportional_valve",
        )

        # Assert
        assert features[0].item() == 0.0, "RPM should be 0 (not provided)"
        assert abs(features[1].item() - 0.655) < 0.01, "Position normalization incorrect"
        assert features[2].item() == 0.0, "Current should be 0"
        assert features[3].item() == 0.0, "Voltage should be 0"

        # proportional_valve at index 7
        type_index = COMPONENT_TYPE_MAPPING["proportional_valve"]
        assert features[4 + type_index].item() == 1.0

    def test_passive_sensor_no_internal_sensors(self, builder_basic: GraphBuilderV2):
        """Test node features for passive sensor (no internal sensors)."""
        # Act
        features = builder_basic.build_node_features_v2(
            component_id="pressure_sensor_01",
            component_reading=None,
            component_type="pressure_sensor",
        )

        # Assert
        # All internal sensors should be 0
        assert features[:4].sum().item() == 0.0, "All internal sensor features should be 0"

        # pressure_sensor at index 22
        type_index = COMPONENT_TYPE_MAPPING["pressure_sensor"]
        assert features[4 + type_index].item() == 1.0

    def test_component_type_inference_from_id(self, builder_basic: GraphBuilderV2):
        """Test automatic component type inference from component_id."""
        # Test various IDs
        test_cases = [
            ("gear_pump_aux", "gear_pump"),
            ("servo_valve_steering", "servo_valve"),
            ("telescopic_cylinder_boom", "telescopic_cylinder"),
            ("flow_sensor_main", "flow_sensor"),
        ]

        for component_id, expected_type in test_cases:
            # Act
            features = builder_basic.build_node_features_v2(
                component_id=component_id,
                component_reading=None,
                component_type=None,  # Let it infer
            )

            # Assert
            type_index = COMPONENT_TYPE_MAPPING[expected_type]
            assert features[4 + type_index].item() == 1.0, (
                f"Type inference failed for '{component_id}': "
                f"expected '{expected_type}' at index {type_index}"
            )

    def test_normalization_clipping(self, builder_basic: GraphBuilderV2):
        """Test that sensor values are clipped to [0, 1] range."""
        # Arrange: Extreme values
        timestamp = datetime.now(UTC)
        reading = ComponentSensorReading(
            component_id="pump",
            rpm=5000,  # > 3000 max → should clip to 1.0
            position_percent=150,  # > 100 → should clip to 1.0
            current_a=200,  # > 100 → should clip to 1.0
            voltage_v=-50,  # < 0 → should clip to 0.0
            timestamp=timestamp,
        )

        # Act
        features = builder_basic.build_node_features_v2(
            component_id="pump",
            component_reading=reading,
            component_type="pump",
        )

        # Assert
        assert features[0].item() == 1.0, "RPM should clip to 1.0"
        assert features[1].item() == 1.0, "Position should clip to 1.0"
        assert features[2].item() == 1.0, "Current should clip to 1.0"
        assert features[3].item() == 0.0, "Negative voltage should clip to 0.0"


# ============================================================================
# EDGE FEATURE TESTS
# ============================================================================


class TestBuildEdgeFeaturesV2:
    """Test cases for build_edge_features_v2() method."""

    def test_static_features_only(self, builder_basic: GraphBuilderV2):
        """Test edge features with only static properties (no sensor reading)."""
        # Arrange
        edge_config = EdgeConfiguration(
            source_id="pump",
            target_id="valve",
            diameter_mm=25.0,
            length_m=5.0,
            material="steel",
            pressure_rating_bar=350.0,
        )

        # Act
        features = builder_basic.build_edge_features_v2(
            edge_spec=edge_config,
            edge_reading=None,  # No sensor data
            edge_history=None,
        )

        # Assert
        assert features.shape == (14,), "Expected 14D (8 static + 6 dynamic zeros)"
        assert features[:8].sum() > 0, "Static features should be non-zero"
        assert features[8:14].sum() == 0.0, "Dynamic features should be zero (no reading)"

    def test_dynamic_features_with_edge_reading(self, builder_basic: GraphBuilderV2):
        """Test edge features with EdgeSensorReading."""
        # Arrange
        timestamp = datetime.now(UTC)
        edge_config = EdgeConfiguration(
            source_id="pump",
            target_id="valve",
            diameter_mm=25.0,
            length_m=5.0,
            material="steel",
            pressure_rating_bar=350.0,
        )
        edge_reading = EdgeSensorReading(
            edge_id="pump__valve",
            pressure_inlet_bar=150.0,
            pressure_outlet_bar=148.0,
            flow_rate_lpm=115.5,
            temperature_c=65.0,
            vibration_g=0.8,
            timestamp=timestamp,
        )

        # Act
        features = builder_basic.build_edge_features_v2(
            edge_spec=edge_config,
            edge_reading=edge_reading,
            edge_history=None,
        )

        # Assert
        assert features.shape == (14,)
        assert features[8:14].sum() > 0, "Dynamic features should be non-zero"
        # Pressure drop should be computed
        pressure_drop_normalized = features[8].item()
        assert pressure_drop_normalized >= 0, "Pressure drop should be non-negative"

    def test_timeseries_features_with_history(self, builder_with_timeseries: GraphBuilderV2):
        """Test edge features with time-series history."""
        # Arrange
        timestamp = datetime.now(UTC)
        edge_config = EdgeConfiguration(
            source_id="pump",
            target_id="valve",
            diameter_mm=25.0,
            length_m=5.0,
            material="steel",
            pressure_rating_bar=350.0,
        )
        edge_reading = EdgeSensorReading(
            edge_id="pump__valve",
            pressure_inlet_bar=150.0,
            pressure_outlet_bar=148.0,
            timestamp=timestamp,
        )
        
        # Create mock time-series data
        edge_history = pd.DataFrame({
            "timestamp": pd.date_range(start=timestamp - timedelta(hours=1), periods=60, freq="1min"),
            "pressure": np.random.normal(150, 2, 60),
        })

        # Act
        features = builder_with_timeseries.build_edge_features_v2(
            edge_spec=edge_config,
            edge_reading=edge_reading,
            edge_history=edge_history,
        )

        # Assert
        assert features.shape == (48,), "Expected 48D (8 static + 6 dynamic + 34 timeseries)"
        assert features[14:].sum() > 0, "Time-series features should be non-zero"

    def test_padding_to_edge_in_dim(self, builder_basic: GraphBuilderV2):
        """Test that edge features are padded to edge_in_dim."""
        # Arrange: config has edge_in_dim=14, but we only have 8 static
        edge_config = EdgeConfiguration(
            source_id="pump",
            target_id="valve",
            diameter_mm=25.0,
            length_m=5.0,
            material="steel",
            pressure_rating_bar=350.0,
        )

        # Act
        features = builder_basic.build_edge_features_v2(
            edge_spec=edge_config,
            edge_reading=None,
            edge_history=None,
        )

        # Assert
        assert features.shape == (14,), "Should be padded to 14D"

    def test_material_encoding(self, builder_basic: GraphBuilderV2):
        """Test material one-hot encoding in static features."""
        materials = ["steel", "rubber", "composite"]
        expected_encodings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        for material, expected in zip(materials, expected_encodings):
            # Arrange
            edge_config = EdgeConfiguration(
                source_id="pump",
                target_id="valve",
                diameter_mm=25.0,
                length_m=5.0,
                material=material,
                pressure_rating_bar=350.0,
            )

            # Act
            features = builder_basic.build_edge_features_v2(
                edge_spec=edge_config,
                edge_reading=None,
            )

            # Assert (material encoding is at indices 5-7 in static features)
            material_encoding = features[5:8].tolist()
            assert material_encoding == expected, f"Material '{material}' encoding incorrect"


# ============================================================================
# GRAPH CONSTRUCTION TESTS
# ============================================================================


class TestBuildGraphHybrid:
    """Test cases for build_graph_hybrid() method."""

    def test_full_topology_graph(self, builder_basic: GraphBuilderV2, hybrid_request_full: HybridInferenceRequest, simple_topology: TopologyConfig):
        """Test successful graph construction for full topology."""
        # Act
        graph = builder_basic.build_graph_hybrid(hybrid_request_full, simple_topology)

        # Assert
        assert isinstance(graph, Data)
        assert graph.num_nodes == 3, "Expected 3 nodes (pump, valve, cylinder)"
        assert graph.num_edges == 2, "Expected 2 edges"
        assert graph.x.shape == (3, 29), "Node features should be [3, 29]"
        assert graph.edge_attr.shape == (2, 14), "Edge features should be [2, 14]"
        assert graph.edge_index.shape == (2, 2), "Edge index should be [2, 2]"

        # Check metadata
        assert graph.equipment_id == "test_excavator_001"
        assert graph.topology_id == "test_simple"
        assert graph.timestamp is not None

    def test_focused_diagnostics(self, builder_basic: GraphBuilderV2, hybrid_request_focused: HybridInferenceRequest, simple_topology: TopologyConfig):
        """Test DiagnosticScope with target_edges (focused diagnostics)."""
        # Act
        graph = builder_basic.build_graph_hybrid(hybrid_request_focused, simple_topology)

        # Assert
        assert graph.num_nodes == 3, "All nodes should be present"
        assert graph.num_edges == 1, "Only 1 target edge (pump→valve)"
        assert graph.edge_attr.shape == (1, 14)

    def test_diagnostic_scope_with_context(self, builder_basic: GraphBuilderV2, simple_topology: TopologyConfig):
        """Test DiagnosticScope with include_context=True."""
        # Arrange
        timestamp = datetime.now(UTC)
        request = HybridInferenceRequest(
            equipment_id="test_001",
            timestamp=timestamp,
            topology_id="test_simple",
            edge_readings={
                "pump_main__valve_01": EdgeSensorReading(
                    edge_id="pump_main__valve_01",
                    pressure_inlet_bar=250.0,
                    pressure_outlet_bar=248.0,
                    timestamp=timestamp,
                ),
            },
            component_readings={},
            diagnostic_scope=DiagnosticScope(
                target_edges=["pump_main__valve_01"],
                include_context=True,  # Should include valve→cylinder as context
            ),
        )

        # Act
        graph = builder_basic.build_graph_hybrid(request, simple_topology)

        # Assert
        assert graph.num_edges == 2, "Should include target + context edges"

    def test_edge_index_validity(self, builder_basic: GraphBuilderV2, hybrid_request_full: HybridInferenceRequest, simple_topology: TopologyConfig):
        """Test that edge_index references valid node indices."""
        # Act
        graph = builder_basic.build_graph_hybrid(hybrid_request_full, simple_topology)

        # Assert
        assert graph.edge_index.min() >= 0, "Edge index should not have negative values"
        assert graph.edge_index.max() < graph.num_nodes, "Edge index out of bounds"

    def test_no_nan_inf_in_features(self, builder_basic: GraphBuilderV2, hybrid_request_full: HybridInferenceRequest, simple_topology: TopologyConfig):
        """Test that graph features contain no NaN or Inf values."""
        # Act
        graph = builder_basic.build_graph_hybrid(hybrid_request_full, simple_topology)

        # Assert
        assert not torch.isnan(graph.x).any(), "Node features contain NaN"
        assert not torch.isinf(graph.x).any(), "Node features contain Inf"
        assert not torch.isnan(graph.edge_attr).any(), "Edge features contain NaN"
        assert not torch.isinf(graph.edge_attr).any(), "Edge features contain Inf"


# ============================================================================
# VALIDATION TESTS
# ============================================================================


class TestGraphValidation:
    """Test cases for graph validation logic."""

    def test_component_not_found_error(self, builder_basic: GraphBuilderV2, simple_topology: TopologyConfig):
        """Test error when edge references unknown component."""
        # Arrange: Request with invalid edge
        timestamp = datetime.now(UTC)
        request = HybridInferenceRequest(
            equipment_id="test_001",
            timestamp=timestamp,
            topology_id="test_simple",
            edge_readings={
                "pump_main__unknown_component": EdgeSensorReading(
                    edge_id="pump_main__unknown_component",
                    pressure_inlet_bar=250.0,
                    pressure_outlet_bar=248.0,
                    timestamp=timestamp,
                ),
            },
            component_readings={},
        )

        # Modify topology to have this invalid edge (for test)
        invalid_edge = EdgeConfiguration(
            source_id="pump_main",
            target_id="unknown_component",  # Does not exist!
            diameter_mm=25.0,
            length_m=5.0,
            material="steel",
            pressure_rating_bar=350.0,
        )
        # Add to edges list
        test_topology = TopologyConfig(
            topology_id="test_invalid",
            name="Invalid Topology",
            version="v1.0",
            components=simple_topology.components,
            edges=simple_topology.edges + [invalid_edge],
        )

        # Act & Assert
        with pytest.raises(ValueError, match="unknown component"):
            builder_basic.build_graph_hybrid(request, test_topology)

    def test_no_edges_error(self, builder_basic: GraphBuilderV2, simple_topology: TopologyConfig):
        """Test error when DiagnosticScope results in no edges."""
        # Arrange: Request with non-existent target edge
        timestamp = datetime.now(UTC)
        request = HybridInferenceRequest(
            equipment_id="test_001",
            timestamp=timestamp,
            topology_id="test_simple",
            edge_readings={},
            component_readings={},
            diagnostic_scope=DiagnosticScope(
                target_edges=["nonexistent__edge"],
                include_context=False,
            ),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="No edges to build graph"):
            builder_basic.build_graph_hybrid(request, simple_topology)

    def test_node_count_mismatch_error(self, builder_basic: GraphBuilderV2, simple_topology: TopologyConfig):
        """Test validation error when node count doesn't match topology."""
        # This test validates that _validate_graph_structure catches mismatches
        # In practice, build_graph_hybrid should always create correct node count
        # This is more of a safety check
        
        # Create valid graph first
        timestamp = datetime.now(UTC)
        request = HybridInferenceRequest(
            equipment_id="test_001",
            timestamp=timestamp,
            topology_id="test_simple",
            edge_readings={
                "pump_main__valve_01": EdgeSensorReading(
                    edge_id="pump_main__valve_01",
                    pressure_inlet_bar=250.0,
                    pressure_outlet_bar=248.0,
                    timestamp=timestamp,
                ),
            },
            component_readings={},
        )
        graph = builder_basic.build_graph_hybrid(request, simple_topology)
        
        # Manually corrupt the graph by adding extra node
        corrupted_x = torch.cat([graph.x, graph.x[:1]], dim=0)  # Add duplicate first node
        graph.x = corrupted_x
        
        # Act & Assert
        with pytest.raises(ValueError, match="Node count mismatch"):
            builder_basic._validate_graph_structure(graph, simple_topology)

    def test_edge_index_out_of_bounds_error(self, builder_basic: GraphBuilderV2, simple_topology: TopologyConfig):
        """Test validation error when edge_index contains out-of-bounds indices."""
        # Create valid graph
        timestamp = datetime.now(UTC)
        request = HybridInferenceRequest(
            equipment_id="test_001",
            timestamp=timestamp,
            topology_id="test_simple",
            edge_readings={
                "pump_main__valve_01": EdgeSensorReading(
                    edge_id="pump_main__valve_01",
                    pressure_inlet_bar=250.0,
                    pressure_outlet_bar=248.0,
                    timestamp=timestamp,
                ),
            },
            component_readings={},
        )
        graph = builder_basic.build_graph_hybrid(request, simple_topology)
        
        # Corrupt edge_index
        graph.edge_index[1, 0] = 999  # Invalid node index
        
        # Act & Assert
        with pytest.raises(ValueError, match="out of bounds"):
            builder_basic._validate_graph_structure(graph, simple_topology)

    def test_isolated_nodes_warning(self, builder_basic: GraphBuilderV2, simple_topology: TopologyConfig, caplog):
        """Test warning for isolated nodes (degree 0)."""
        # Arrange: Topology with isolated node
        isolated_topology = TopologyConfig(
            topology_id="test_isolated",
            name="Topology with Isolated Node",
            version="v1.0",
            components=[
                ComponentConfiguration(
                    component_id="pump",
                    component_type="pump",
                    name="Main Pump",
                    nominal_pressure_bar=250.0,
                    nominal_flow_lpm=150.0,
                ),
                ComponentConfiguration(
                    component_id="valve",
                    component_type="valve",
                    name="Control Valve",
                    nominal_pressure_bar=245.0,
                    nominal_flow_lpm=150.0,
                ),
                ComponentConfiguration(
                    component_id="isolated_sensor",
                    component_type="pressure_sensor",
                    name="Isolated Sensor",
                    nominal_pressure_bar=0.0,
                    nominal_flow_lpm=0.0,
                ),
            ],
            edges=[
                EdgeConfiguration(
                    source_id="pump",
                    target_id="valve",
                    diameter_mm=25.0,
                    length_m=5.0,
                    material="steel",
                    pressure_rating_bar=350.0,
                ),
                # No edges connected to isolated_sensor!
            ],
        )
        
        timestamp = datetime.now(UTC)
        request = HybridInferenceRequest(
            equipment_id="test_001",
            timestamp=timestamp,
            topology_id="test_isolated",
            edge_readings={
                "pump__valve": EdgeSensorReading(
                    edge_id="pump__valve",
                    pressure_inlet_bar=250.0,
                    pressure_outlet_bar=248.0,
                    timestamp=timestamp,
                ),
            },
            component_readings={},
        )
        
        # Act
        with caplog.at_level("WARNING"):
            graph = builder_basic.build_graph_hybrid(request, isolated_topology)
        
        # Assert
        assert "isolated nodes" in caplog.text.lower(), "Should warn about isolated nodes"


# ============================================================================
# HELPER METHOD TESTS
# ============================================================================


class TestHelperMethods:
    """Test cases for helper methods."""

    def test_build_component_index_map(self, builder_basic: GraphBuilderV2, simple_topology: TopologyConfig):
        """Test component_id → node_index mapping."""
        # Act
        index_map = builder_basic._build_component_index_map(simple_topology)

        # Assert
        assert len(index_map) == 3
        assert "pump_main" in index_map
        assert "valve_01" in index_map
        assert "cylinder_left" in index_map
        assert index_map["pump_main"] == 0
        assert index_map["valve_01"] == 1
        assert index_map["cylinder_left"] == 2

    def test_get_edge_id(self, builder_basic: GraphBuilderV2):
        """Test edge_id construction from source/target."""
        # Act
        edge_id = builder_basic._get_edge_id("pump_main", "valve_01")

        # Assert
        assert edge_id == "pump_main__valve_01"

    def test_infer_component_type(self, builder_basic: GraphBuilderV2):
        """Test component type inference from ID."""
        test_cases = [
            ("piston_pump_main", "piston_pump"),
            ("gear_pump_aux", "gear_pump"),
            ("proportional_valve_boom", "proportional_valve"),
            ("servo_valve_steering", "servo_valve"),
            ("telescopic_cylinder_left", "telescopic_cylinder"),
            ("pressure_sensor_01", "pressure_sensor"),
            ("unknown_component_xyz", "pump"),  # Default
        ]

        for component_id, expected_type in test_cases:
            # Act
            inferred_type = builder_basic._infer_component_type(component_id)

            # Assert
            assert inferred_type == expected_type, (
                f"Type inference failed for '{component_id}': "
                f"expected '{expected_type}', got '{inferred_type}'"
            )


# ============================================================================
# INTEGRATION TESTS (End-to-End)
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_end_to_end_full_topology(self, builder_basic: GraphBuilderV2, hybrid_request_full: HybridInferenceRequest, simple_topology: TopologyConfig):
        """Test complete workflow: HybridInferenceRequest → Graph."""
        # Act
        graph = builder_basic.build_graph_hybrid(hybrid_request_full, simple_topology)

        # Assert complete workflow
        assert isinstance(graph, Data)
        assert graph.num_nodes > 0
        assert graph.num_edges > 0
        assert graph.x.shape[1] == 29
        assert graph.edge_attr.shape[1] == 14
        assert not torch.isnan(graph.x).any()
        assert not torch.isnan(graph.edge_attr).any()
        assert graph.equipment_id == hybrid_request_full.equipment_id

    def test_end_to_end_focused_diagnostics(self, builder_basic: GraphBuilderV2, hybrid_request_focused: HybridInferenceRequest, simple_topology: TopologyConfig):
        """Test complete workflow with DiagnosticScope."""
        # Act
        graph = builder_basic.build_graph_hybrid(hybrid_request_focused, simple_topology)

        # Assert
        assert graph.num_edges == 1, "Only target edge should be included"
        assert graph.num_nodes == 3, "All nodes should be present"
