"""Unit tests for GraphBuilderV2 node features.

Week 1, Day 1: Testing build_node_features_v2() with 25 component types.

🏗️ Production-Grade Testing Architecture:
    - Layer 1 (Pydantic): Rejects "physically impossible" values (e.g., negative RPM, >100% position)
    - Layer 2 (Features): Clips "physically unlikely but real" values (e.g., sensor overshoot, calibration drift)
"""

import pytest
import torch
from pydantic import ValidationError

from src.data.graph_builder_v2 import (
    GraphBuilderV2,
    NODE_FEATURES_TOTAL_DIM,
    NODE_INTERNAL_SENSORS_DIM,
    NODE_COMPONENT_TYPE_DIM,
    COMPONENT_TYPE_MAPPING,
)


class TestGraphBuilderV2NodeFeatures:
    """Test suite for build_node_features_v2() method."""

    @pytest.fixture
    def builder(self):
        """Create GraphBuilderV2 instance."""
        return GraphBuilderV2()

    # ========================================================================
    # TEST: Dimension Validation
    # ========================================================================

    def test_node_features_dimensions(self, builder):
        """Test that node features have correct dimensions (29D)."""
        features = builder.build_node_features_v2(
            component_id="pump_main",
            component_reading=None,
            component_type="pump"
        )

        assert features.shape == (NODE_FEATURES_TOTAL_DIM,), (
            f"Expected shape ({NODE_FEATURES_TOTAL_DIM},), got {features.shape}"
        )
        assert NODE_FEATURES_TOTAL_DIM == 29, "Total dimension should be 29D"
        assert NODE_INTERNAL_SENSORS_DIM == 4, "Internal sensors should be 4D"
        assert NODE_COMPONENT_TYPE_DIM == 25, "Component type one-hot should be 25D"

    # ========================================================================
    # TEST: Internal Sensors (4D)
    # ========================================================================

    def test_internal_sensors_all_zeros_when_no_reading(self, builder):
        """Test that internal sensors are zeros when no component_reading provided."""
        features = builder.build_node_features_v2(
            component_id="pump_main",
            component_reading=None,
            component_type="pump"
        )

        internal_sensors = features[:NODE_INTERNAL_SENSORS_DIM]
        expected = torch.zeros(NODE_INTERNAL_SENSORS_DIM)

        assert torch.allclose(internal_sensors, expected), (
            f"Expected all zeros for internal sensors, got {internal_sensors}"
        )

    def test_rpm_normalization(self, builder):
        """Test RPM normalization (max 3000 RPM)."""
        from src.schemas.requests import ComponentSensorReading
        from datetime import datetime, UTC

        # Test cases: (rpm_input, expected_normalized)
        test_cases = [
            (0, 0.0),
            (1500, 0.5),
            (3000, 1.0),
            (4500, 1.0),  # Clipped to 1.0 by np.clip in build_node_features_v2
        ]

        for rpm_input, expected_norm in test_cases:
            reading = ComponentSensorReading(
                component_id="pump",
                rpm=rpm_input,
                timestamp=datetime.now(UTC)
            )
            features = builder.build_node_features_v2("pump", reading, "pump")
            rpm_feature = features[0].item()

            assert abs(rpm_feature - expected_norm) < 1e-6, (
                f"RPM {rpm_input} should normalize to {expected_norm}, got {rpm_feature}"
            )

    def test_position_normalization(self, builder):
        """Test position normalization (0-100%).

        🔧 Production Note:
            - Pydantic Field(ge=0, le=100) blocks values >100 at API layer
            - np.clip in build_node_features_v2 handles edge cases within valid range
            - Test only VALID Pydantic values (0-100%)
        """
        from src.schemas.requests import ComponentSensorReading
        from datetime import datetime, UTC

        # Only test valid Pydantic range [0, 100]
        test_cases = [
            (0.0, 0.0),
            (50.0, 0.5),
            (100.0, 1.0),
            # (150.0, 1.0) - REMOVED: Pydantic Field(le=100) blocks this at API layer
        ]

        for pos_input, expected_norm in test_cases:
            reading = ComponentSensorReading(
                component_id="valve",
                position_percent=pos_input,
                timestamp=datetime.now(UTC)
            )
            features = builder.build_node_features_v2("valve", reading, "valve")
            pos_feature = features[1].item()

            assert abs(pos_feature - expected_norm) < 1e-6, (
                f"Position {pos_input}% should normalize to {expected_norm}, got {pos_feature}"
            )

    def test_current_normalization(self, builder):
        """Test current normalization (max 100A)."""
        from src.schemas.requests import ComponentSensorReading
        from datetime import datetime, UTC

        test_cases = [
            (0.0, 0.0),
            (50.0, 0.5),
            (100.0, 1.0),
            (200.0, 1.0),  # Clipped to 1.0 by np.clip
        ]

        for current_input, expected_norm in test_cases:
            reading = ComponentSensorReading(
                component_id="motor",
                current_a=current_input,
                timestamp=datetime.now(UTC)
            )
            features = builder.build_node_features_v2("motor", reading, "motor")
            current_feature = features[2].item()

            assert abs(current_feature - expected_norm) < 1e-6, (
                f"Current {current_input}A should normalize to {expected_norm}, got {current_feature}"
            )

    def test_voltage_normalization(self, builder):
        """Test voltage normalization (max 500V)."""
        from src.schemas.requests import ComponentSensorReading
        from datetime import datetime, UTC

        test_cases = [
            (0.0, 0.0),
            (250.0, 0.5),
            (500.0, 1.0),
            (750.0, 1.0),  # Clipped to 1.0 by np.clip
        ]

        for voltage_input, expected_norm in test_cases:
            reading = ComponentSensorReading(
                component_id="motor",
                voltage_v=voltage_input,
                timestamp=datetime.now(UTC)
            )
            features = builder.build_node_features_v2("motor", reading, "motor")
            voltage_feature = features[3].item()

            assert abs(voltage_feature - expected_norm) < 1e-6, (
                f"Voltage {voltage_input}V should normalize to {expected_norm}, got {voltage_feature}"
            )

    # ========================================================================
    # TEST: Pydantic Validation (API Layer)
    # ========================================================================

    def test_pydantic_rejects_invalid_position(self):
        """Test that Pydantic rejects position_percent > 100.

        🏗️ Architecture: API layer (Pydantic) blocks "physically impossible" values.
        """
        from src.schemas.requests import ComponentSensorReading
        from datetime import datetime, UTC

        # Attempt to create invalid reading (position > 100%)
        with pytest.raises(ValidationError) as exc_info:
            ComponentSensorReading(
                component_id="valve",
                position_percent=150.0,  # INVALID: > Field(le=100)
                timestamp=datetime.now(UTC)
            )

        # Verify error message mentions boundary
        error_str = str(exc_info.value)
        assert "less than or equal to 100" in error_str, (
            f"Expected 'less than or equal to 100' in error, got: {error_str}"
        )

    def test_pydantic_rejects_negative_sensors(self):
        """Test that Pydantic rejects negative sensor values.

        🏗️ Architecture: API layer blocks "physically impossible" negative values.
        """
        from src.schemas.requests import ComponentSensorReading
        from datetime import datetime, UTC

        # Test negative RPM
        with pytest.raises(ValidationError) as exc_info:
            ComponentSensorReading(
                component_id="motor",
                rpm=-100,  # INVALID: < Field(ge=0)
                timestamp=datetime.now(UTC)
            )
        assert "greater than or equal to 0" in str(exc_info.value)

        # Test negative current
        with pytest.raises(ValidationError) as exc_info:
            ComponentSensorReading(
                component_id="motor",
                current_a=-50.0,  # INVALID
                timestamp=datetime.now(UTC)
            )
        assert "greater than or equal to 0" in str(exc_info.value)

        # Test negative voltage
        with pytest.raises(ValidationError) as exc_info:
            ComponentSensorReading(
                component_id="motor",
                voltage_v=-250.0,  # INVALID
                timestamp=datetime.now(UTC)
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_boundary_values(self, builder):
        """Test valid boundary values (0, max) pass Pydantic and normalize correctly.

        🏗️ Architecture: Valid boundaries work through both layers.
        """
        from src.schemas.requests import ComponentSensorReading
        from datetime import datetime, UTC

        # Boundary: all zeros (minimum)
        reading_min = ComponentSensorReading(
            component_id="motor",
            rpm=0,
            position_percent=0.0,
            current_a=0.0,
            voltage_v=0.0,
            timestamp=datetime.now(UTC)
        )
        features_min = builder.build_node_features_v2("motor", reading_min, "motor")
        assert features_min[0].item() == 0.0  # RPM
        assert features_min[1].item() == 0.0  # Position
        assert features_min[2].item() == 0.0  # Current
        assert features_min[3].item() == 0.0  # Voltage

        # Boundary: max valid values
        reading_max = ComponentSensorReading(
            component_id="motor",
            rpm=10000,  # Field(le=10000)
            position_percent=100.0,  # Field(le=100)
            current_a=1000.0,  # Field(le=1000)
            voltage_v=1000.0,  # Field(le=1000)
            timestamp=datetime.now(UTC)
        )
        features_max = builder.build_node_features_v2("motor", reading_max, "motor")
        # Normalized values depend on MAX_RPM=3000, MAX_CURRENT=100, MAX_VOLTAGE=500
        assert features_max[0].item() == 1.0  # RPM: 10000/3000 clipped to 1.0
        assert features_max[1].item() == 1.0  # Position: 100/100 = 1.0
        assert features_max[2].item() == 1.0  # Current: 1000/100 clipped to 1.0
        assert features_max[3].item() == 1.0  # Voltage: 1000/500 clipped to 1.0

    # ========================================================================
    # TEST: Component Type One-Hot (25D)
    # ========================================================================

    def test_component_type_onehot_structure(self, builder):
        """Test that one-hot encoding has exactly one 1.0 and rest 0.0."""
        features = builder.build_node_features_v2(
            component_id="pump_main",
            component_reading=None,
            component_type="pump"
        )

        onehot = features[NODE_INTERNAL_SENSORS_DIM:]
        assert onehot.sum().item() == 1.0, "One-hot should have exactly one 1.0"
        assert len(onehot) == NODE_COMPONENT_TYPE_DIM, f"One-hot should have {NODE_COMPONENT_TYPE_DIM} elements"

    @pytest.mark.parametrize("component_type,expected_idx", [
        ("pump", 0),
        ("gear_pump", 1),
        ("piston_pump", 2),
        ("proportional_valve", 7),
        ("servo_valve", 8),
        ("relief_valve", 9),
        ("cylinder", 14),
        ("telescopic_cylinder", 15),
        ("filter", 17),
        ("cooler", 18),
        ("pressure_sensor", 22),
        ("flow_sensor", 23),
        ("temperature_sensor", 24),
    ])
    def test_specific_component_types(self, builder, component_type, expected_idx):
        """Test that specific component types map to correct one-hot index."""
        features = builder.build_node_features_v2(
            component_id=f"{component_type}_test",
            component_reading=None,
            component_type=component_type
        )

        onehot = features[NODE_INTERNAL_SENSORS_DIM:]
        actual_idx = torch.argmax(onehot).item()

        assert actual_idx == expected_idx, (
            f"{component_type} should map to index {expected_idx}, got {actual_idx}"
        )
        assert onehot[expected_idx].item() == 1.0, (
            f"Index {expected_idx} should be 1.0 for {component_type}"
        )

    # ========================================================================
    # TEST: Component Type Inference
    # ========================================================================

    @pytest.mark.parametrize("component_id,expected_type", [
        ("piston_pump_main", "piston_pump"),
        ("gear_pump_aux", "gear_pump"),
        ("vane_pump_01", "vane_pump"),
        ("hydraulic_motor_drive", "hydraulic_motor"),
        ("orbital_motor_wheel", "orbital_motor"),
        ("proportional_valve_boom", "proportional_valve"),
        ("servo_valve_steering", "servo_valve"),
        ("relief_valve_main", "relief_valve"),
        ("check_valve_return", "check_valve"),
        ("flow_control_valve_01", "flow_control_valve"),
        ("directional_valve_02", "directional_valve"),
        ("telescopic_cylinder_boom", "telescopic_cylinder"),
        ("cylinder_arm", "cylinder"),
        ("pressure_sensor_inlet", "pressure_sensor"),
        ("flow_sensor_main", "flow_sensor"),
        ("temperature_sensor_oil", "temperature_sensor"),
        ("accumulator_main", "accumulator"),
        ("filter_suction", "filter"),
        ("cooler_oil", "cooler"),
        ("heater_tank", "heater"),
        ("reservoir_main", "reservoir"),
        ("manifold_control", "manifold"),
    ])
    def test_component_type_inference_from_id(self, builder, component_id, expected_type):
        """Test that _infer_component_type correctly identifies type from component_id."""
        inferred_type = builder._infer_component_type(component_id)
        assert inferred_type == expected_type, (
            f"Component ID '{component_id}' should infer type '{expected_type}', got '{inferred_type}'"
        )

    def test_component_type_inference_priority(self, builder):
        """Test that specific types take priority over general types."""
        # "piston_pump" should be inferred as "piston_pump", NOT "pump"
        inferred = builder._infer_component_type("piston_pump_main")
        assert inferred == "piston_pump", (
            f"Should prioritize 'piston_pump' over 'pump', got '{inferred}'"
        )

        # "proportional_valve" should be inferred, NOT "valve"
        inferred = builder._infer_component_type("proportional_valve_boom")
        assert inferred == "proportional_valve", (
            f"Should prioritize 'proportional_valve' over 'valve', got '{inferred}'"
        )

    def test_component_type_inference_unknown_defaults_to_pump(self, builder):
        """Test that unknown component types default to 'pump'."""
        unknown_ids = [
            "unknown_component",
            "random_xyz_123",
            "mysterious_device",
        ]

        for unknown_id in unknown_ids:
            inferred = builder._infer_component_type(unknown_id)
            assert inferred == "pump", (
                f"Unknown component '{unknown_id}' should default to 'pump', got '{inferred}'"
            )

    # ========================================================================
    # TEST: Integration (Full Feature Vector)
    # ========================================================================

    def test_full_feature_vector_with_sensors(self, builder):
        """Test complete feature vector with all sensors populated."""
        from src.schemas.requests import ComponentSensorReading
        from datetime import datetime, UTC

        reading = ComponentSensorReading(
            component_id="piston_pump_main",
            rpm=1500,
            position_percent=None,
            current_a=50.0,
            voltage_v=250.0,
            timestamp=datetime.now(UTC)
        )

        features = builder.build_node_features_v2(
            component_id="piston_pump_main",
            component_reading=reading,
            component_type="piston_pump"
        )

        # Check dimensions
        assert features.shape == (29,)

        # Check internal sensors
        assert abs(features[0].item() - 0.5) < 1e-6  # RPM: 1500/3000 = 0.5
        assert features[1].item() == 0.0  # Position: None → 0.0
        assert abs(features[2].item() - 0.5) < 1e-6  # Current: 50/100 = 0.5
        assert abs(features[3].item() - 0.5) < 1e-6  # Voltage: 250/500 = 0.5

        # Check one-hot (piston_pump at index 2)
        onehot = features[4:]
        assert torch.argmax(onehot).item() == 2
        assert onehot[2].item() == 1.0

    def test_feature_vector_passive_component(self, builder):
        """Test feature vector for passive component (no internal sensors)."""
        features = builder.build_node_features_v2(
            component_id="filter_suction",
            component_reading=None,
            component_type="filter"
        )

        # Internal sensors should be zeros
        internal_sensors = features[:4]
        assert torch.allclose(internal_sensors, torch.zeros(4))

        # One-hot should have filter at index 17
        onehot = features[4:]
        assert torch.argmax(onehot).item() == 17
        assert onehot[17].item() == 1.0

    # ========================================================================
    # TEST: Edge Cases
    # ========================================================================

    def test_component_type_case_insensitive(self, builder):
        """Test that component type matching is case-insensitive."""
        features_lower = builder.build_node_features_v2(
            "pump", None, "piston_pump"
        )
        features_upper = builder.build_node_features_v2(
            "pump", None, "PISTON_PUMP"
        )
        features_mixed = builder.build_node_features_v2(
            "pump", None, "Piston_Pump"
        )

        # All should map to same one-hot encoding
        assert torch.allclose(features_lower, features_upper)
        assert torch.allclose(features_lower, features_mixed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
