"""Unit tests for sensor coverage schemas.

Tests:
    - EdgeSensorCoverage: minimal, Level 3, Level 4 coverage
    - ComponentSensorCoverage: pump, valve monitoring
    - SensorCoverageConfig: validation, helper methods
"""

import pytest

from src.schemas.sensor_coverage import (
    ComponentSensorCoverage,
    EdgeSensorCoverage,
    SensorCoverageConfig,
)


class TestEdgeSensorCoverage:
    """Test EdgeSensorCoverage schema."""

    def test_minimal_coverage_no_sensors(self):
        """Test minimal edge coverage (no sensors installed)."""
        coverage = EdgeSensorCoverage()
        assert not coverage.has_any_sensor()
        assert not coverage.has_pressure_monitoring()
        assert coverage.sensor_count() == 0

    def test_single_sensor(self):
        """Test edge with single pressure sensor."""
        coverage = EdgeSensorCoverage(pressure_inlet=True)
        assert coverage.has_any_sensor()
        assert not coverage.has_pressure_monitoring()  # Need both inlet/outlet
        assert coverage.sensor_count() == 1

    def test_level3_coverage_pressure_and_flow(self):
        """Test Level 3 coverage (pressure drop monitoring + flow)."""
        coverage = EdgeSensorCoverage(
            pressure_inlet=True, pressure_outlet=True, flow_meter=True
        )
        assert coverage.has_any_sensor()
        assert coverage.has_pressure_monitoring()  # Both pressures!
        assert coverage.sensor_count() == 3

    def test_level4_coverage_full(self):
        """Test Level 4 full coverage (all sensors)."""
        coverage = EdgeSensorCoverage(
            pressure_inlet=True,
            pressure_outlet=True,
            flow_meter=True,
            temperature=True,
            vibration=True,
        )
        assert coverage.has_any_sensor()
        assert coverage.has_pressure_monitoring()
        assert coverage.sensor_count() == 5

    def test_defaults(self):
        """Test that all sensors default to False."""
        coverage = EdgeSensorCoverage()
        assert coverage.pressure_inlet is False
        assert coverage.pressure_outlet is False
        assert coverage.flow_meter is False
        assert coverage.temperature is False
        assert coverage.vibration is False


class TestComponentSensorCoverage:
    """Test ComponentSensorCoverage schema."""

    def test_minimal_coverage_no_sensors(self):
        """Test minimal component coverage (no sensors)."""
        coverage = ComponentSensorCoverage()
        assert not coverage.has_any_sensor()
        assert coverage.sensor_count() == 0

    def test_pump_monitoring_typical(self):
        """Test typical pump monitoring setup (RPM + current + vibration)."""
        coverage = ComponentSensorCoverage(rpm=True, current=True, vibration=True)
        assert coverage.has_any_sensor()
        assert coverage.sensor_count() == 3

    def test_valve_monitoring_proportional(self):
        """Test proportional valve monitoring (position + current)."""
        coverage = ComponentSensorCoverage(position=True, current=True)
        assert coverage.has_any_sensor()
        assert coverage.sensor_count() == 2

    def test_motor_monitoring_full(self):
        """Test electric motor full monitoring."""
        coverage = ComponentSensorCoverage(
            rpm=True, current=True, voltage=True, vibration=True
        )
        assert coverage.sensor_count() == 4

    def test_defaults(self):
        """Test that all sensors default to False."""
        coverage = ComponentSensorCoverage()
        assert coverage.rpm is False
        assert coverage.position is False
        assert coverage.current is False
        assert coverage.voltage is False
        assert coverage.vibration is False


class TestSensorCoverageConfig:
    """Test SensorCoverageConfig validation and helpers."""

    def test_empty_coverage_fails_validation(self):
        """Test that completely empty coverage raises ValueError."""
        with pytest.raises(ValueError, match="At least one sensor"):
            SensorCoverageConfig(
                equipment_id="test_equipment",
                topology_id="test_topology",
                installed_sensors={"edges": {}, "components": {}},
            )

    def test_minimal_valid_config_single_edge_sensor(self):
        """Test minimal valid config with single edge sensor."""
        config = SensorCoverageConfig(
            equipment_id="test_001",
            topology_id="minimal_circuit",
            installed_sensors={
                "edges": {"pump__valve": EdgeSensorCoverage(pressure_inlet=True)},
                "components": {},
            },
        )
        assert config.total_sensor_count() == 1
        assert config.has_edge_sensor("pump__valve")

    def test_level3_equipment_configuration(self):
        """Test Level 3 equipment configuration."""
        config = SensorCoverageConfig(
            equipment_id="excavator_001",
            topology_id="boom_circuit",
            installed_sensors={
                "edges": {
                    "pump_main__valve_boom": EdgeSensorCoverage(
                        pressure_inlet=True, pressure_outlet=True, flow_meter=True
                    ),
                    "valve_boom__cylinder": EdgeSensorCoverage(pressure_inlet=True),
                },
                "components": {
                    "pump_main": ComponentSensorCoverage(rpm=True, current=True)
                },
            },
        )

        assert config.total_sensor_count() == 6  # 3 + 1 + 2
        assert config.has_edge_sensor("pump_main__valve_boom")
        assert config.has_edge_sensor("valve_boom__cylinder")
        assert config.has_component_sensor("pump_main")
        assert not config.has_component_sensor("cylinder")  # Not monitored

    def test_get_edge_coverage_existing(self):
        """Test get_edge_coverage for existing edge."""
        config = SensorCoverageConfig(
            equipment_id="test",
            topology_id="test",
            installed_sensors={
                "edges": {
                    "edge_1": EdgeSensorCoverage(
                        pressure_inlet=True, pressure_outlet=True
                    )
                },
                "components": {},
            },
        )

        coverage = config.get_edge_coverage("edge_1")
        assert coverage is not None
        assert coverage.has_pressure_monitoring()

    def test_get_edge_coverage_missing(self):
        """Test get_edge_coverage for non-existent edge."""
        config = SensorCoverageConfig(
            equipment_id="test",
            topology_id="test",
            installed_sensors={"edges": {}, "components": {"pump": ComponentSensorCoverage(rpm=True)}},
        )

        coverage = config.get_edge_coverage("nonexistent_edge")
        assert coverage is None

    def test_get_component_coverage_existing(self):
        """Test get_component_coverage for existing component."""
        config = SensorCoverageConfig(
            equipment_id="test",
            topology_id="test",
            installed_sensors={
                "edges": {},
                "components": {"pump_1": ComponentSensorCoverage(rpm=True, current=True)},
            },
        )

        coverage = config.get_component_coverage("pump_1")
        assert coverage is not None
        assert coverage.sensor_count() == 2

    def test_version_defaults_to_v1_0(self):
        """Test that version defaults to v1.0."""
        config = SensorCoverageConfig(
            equipment_id="test",
            topology_id="test",
            installed_sensors={"edges": {}, "components": {"pump": ComponentSensorCoverage(rpm=True)}},
        )
        assert config.version == "v1.0"

    def test_version_pattern_validation(self):
        """Test that version must match vX.Y pattern."""
        # Valid versions
        config = SensorCoverageConfig(
            equipment_id="test",
            topology_id="test",
            version="v2.5",
            installed_sensors={"edges": {}, "components": {"pump": ComponentSensorCoverage(rpm=True)}},
        )
        assert config.version == "v2.5"

        # Invalid version should fail validation
        with pytest.raises(ValueError):
            SensorCoverageConfig(
                equipment_id="test",
                topology_id="test",
                version="invalid",
                installed_sensors={
                    "edges": {},
                    "components": {"pump": ComponentSensorCoverage(rpm=True)},
                },
            )
