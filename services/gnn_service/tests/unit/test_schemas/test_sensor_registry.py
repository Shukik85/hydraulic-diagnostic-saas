"""Unit tests for sensor registry schemas."""

import pytest
from datetime import date, datetime, UTC
from src.schemas.sensor_registry import (
    SensorDataSourceConfig,
    PhysicalSensor,
)


class TestSensorDataSourceConfig:
    """Test SensorDataSourceConfig validation."""
    
    def test_timescaledb_config(self):
        """Test TimescaleDB data source configuration."""
        config = SensorDataSourceConfig(
            source_type="timescaledb",
            tsdb_table="sensor_readings",
            sampling_rate_hz=10.0
        )
        assert config.source_type == "timescaledb"
        assert config.get_effective_sampling_rate() == 10.0
    
    def test_csv_config(self):
        """Test CSV data source configuration."""
        config = SensorDataSourceConfig(
            source_type="csv",
            csv_column_name="pressure_pump_bar"
        )
        assert config.source_type == "csv"
        assert config.csv_column_name == "pressure_pump_bar"
    
    def test_api_config(self):
        """Test REST API data source configuration."""
        config = SensorDataSourceConfig(
            source_type="api",
            api_endpoint="/sensors/pt_001/latest",
            api_field_path="data.pressure.value"
        )
        assert config.source_type == "api"
        assert config.api_endpoint == "/sensors/pt_001/latest"
    
    def test_scaling_and_offset(self):
        """Test scaling factor and offset defaults."""
        config = SensorDataSourceConfig(
            source_type="timescaledb",
            tsdb_table="test",
            scaling_factor=2.0,
            offset=10.0
        )
        assert config.scaling_factor == 2.0
        assert config.offset == 10.0
    
    def test_default_sampling_rate(self):
        """Test default sampling rate fallback."""
        config = SensorDataSourceConfig(
            source_type="csv",
            csv_column_name="test"
        )
        assert config.get_effective_sampling_rate(default=5.0) == 5.0


class TestPhysicalSensor:
    """Test PhysicalSensor schema."""
    
    def test_pressure_transducer(self):
        """Test pressure transducer configuration."""
        sensor = PhysicalSensor(
            sensor_id="pt_001",
            sensor_type="pressure",
            manufacturer="Bosch Rexroth",
            model="HM20-1X/400",
            measurement_range=(0.0, 400.0),
            unit="bar",
            location_description="Pump outlet",
            data_source_config=SensorDataSourceConfig(
                source_type="timescaledb",
                tsdb_table="sensor_readings"
            )
        )
        assert sensor.sensor_id == "pt_001"
        assert sensor.sensor_type == "pressure"
        assert sensor.get_full_scale_range() == 400.0
        assert sensor.get_absolute_accuracy() == 4.0  # 1% of 400
    
    def test_flow_meter(self):
        """Test flow meter configuration."""
        sensor = PhysicalSensor(
            sensor_id="fm_main",
            sensor_type="flow",
            manufacturer="Hydac",
            model="EVS3100",
            measurement_range=(0.0, 200.0),
            unit="L/min",
            accuracy_percent=0.5,
            location_description="Main supply line",
            data_source_config=SensorDataSourceConfig(
                source_type="csv",
                csv_column_name="flow_main_lpm"
            )
        )
        assert sensor.sensor_type == "flow"
        assert sensor.accuracy_percent == 0.5
        assert sensor.get_absolute_accuracy() == 1.0  # 0.5% of 200
    
    def test_temperature_sensor(self):
        """Test temperature sensor configuration."""
        sensor = PhysicalSensor(
            sensor_id="temp_tank",
            sensor_type="temperature",
            manufacturer="Omega",
            model="PT100-RTD",
            measurement_range=(-20.0, 150.0),
            unit="°C",
            accuracy_percent=0.1,
            location_description="Hydraulic tank",
            data_source_config=SensorDataSourceConfig(
                source_type="api",
                api_endpoint="/sensors/temp_tank"
            )
        )
        assert sensor.sensor_type == "temperature"
        assert sensor.get_full_scale_range() == 170.0
    
    def test_invalid_range_fails(self):
        """Test that invalid measurement range raises error."""
        with pytest.raises(ValueError, match="must be < max"):
            PhysicalSensor(
                sensor_id="invalid",
                sensor_type="pressure",
                manufacturer="Test",
                model="Test",
                measurement_range=(400.0, 0.0),  # Invalid: min > max
                unit="bar",
                location_description="Test",
                data_source_config=SensorDataSourceConfig(
                    source_type="timescaledb",
                    tsdb_table="test"
                )
            )
    
    def test_equal_range_fails(self):
        """Test that equal min/max range raises error."""
        with pytest.raises(ValueError, match="must be < max"):
            PhysicalSensor(
                sensor_id="invalid",
                sensor_type="pressure",
                manufacturer="Test",
                model="Test",
                measurement_range=(100.0, 100.0),  # Invalid: min == max
                unit="bar",
                location_description="Test",
                data_source_config=SensorDataSourceConfig(
                    source_type="csv",
                    csv_column_name="test"
                )
            )
    
    def test_calibration_due_check(self):
        """Test calibration due checking."""
        sensor = PhysicalSensor(
            sensor_id="pt_test",
            sensor_type="pressure",
            manufacturer="Test",
            model="Test",
            measurement_range=(0.0, 400.0),
            unit="bar",
            last_calibration_date=date(2024, 1, 1),
            calibration_interval_days=365,
            location_description="Test",
            data_source_config=SensorDataSourceConfig(
                source_type="csv",
                csv_column_name="test"
            )
        )
        
        # Check with date after calibration interval
        check_date = date(2025, 1, 2)
        assert sensor.is_calibration_due(check_date)
        
        # Check with date before calibration interval
        check_date = date(2024, 6, 1)
        assert not sensor.is_calibration_due(check_date)
    
    def test_never_calibrated(self):
        """Test that sensor with no calibration date is always due."""
        sensor = PhysicalSensor(
            sensor_id="pt_new",
            sensor_type="pressure",
            manufacturer="Test",
            model="Test",
            measurement_range=(0.0, 400.0),
            unit="bar",
            location_description="Test",
            data_source_config=SensorDataSourceConfig(
                source_type="csv",
                csv_column_name="test"
            )
        )
        assert sensor.is_calibration_due()
    
    def test_serial_number_optional(self):
        """Test that serial_number is optional."""
        sensor = PhysicalSensor(
            sensor_id="pt_test",
            sensor_type="pressure",
            manufacturer="Test",
            model="Test",
            measurement_range=(0.0, 400.0),
            unit="bar",
            location_description="Test",
            data_source_config=SensorDataSourceConfig(
                source_type="csv",
                csv_column_name="test"
            )
        )
        assert sensor.serial_number is None


# TODO: Add more tests on Day 2
# - Test data source validation (required fields per source_type)
# - Test sensor_id uniqueness in registry
# - Test measurement unit validation
# - Test notes max length
