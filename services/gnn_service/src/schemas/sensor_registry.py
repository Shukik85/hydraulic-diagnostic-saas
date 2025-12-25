"""Physical sensor registration schemas.

Defines physical sensors installed on equipment and their data sources.
Minimal implementation with extension points for Phase 2.

Python 3.14 features:
    - Native union types (T | None)
    - PEP 692: TypedDict improvements
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "SensorDataSourceConfig",
    "PhysicalSensor",
]


class SensorDataSourceConfig(BaseModel):
    """Configuration for sensor data source.
    
    Minimal implementation supporting 3 source types:
    - timescaledb: Time-series database (primary)
    - csv: CSV files (testing/offline)
    - api: REST API endpoints (external systems)
    
    Phase 2 will add: modbus, opcua, mqtt
    
    Attributes:
        source_type: Type of data source
        
        # TimescaleDB fields
        tsdb_table: Table name in TimescaleDB
        tsdb_sensor_id_column: Column name for sensor ID
        tsdb_value_column: Column name for sensor value
        tsdb_timestamp_column: Column name for timestamp
        
        # CSV fields
        csv_column_name: Column name in CSV file
        csv_timestamp_column: Timestamp column in CSV
        
        # REST API fields
        api_endpoint: API endpoint path
        api_field_path: JSON path to value (e.g., "data.pressure.value")
        
        # Common fields
        sampling_rate_hz: Data sampling frequency (Hz)
        scaling_factor: Multiply raw value by this
        offset: Add this to scaled value
    """
    
    source_type: Literal["timescaledb", "csv", "api"] = Field(
        ...,
        description="Data source type (timescaledb, csv, api)"
    )
    
    # === TIMESCALEDB ===
    tsdb_table: str | None = Field(
        None,
        min_length=1,
        max_length=200,
        description="TimescaleDB table name"
    )
    tsdb_sensor_id_column: str = Field(
        default="sensor_id",
        description="Column containing sensor ID"
    )
    tsdb_value_column: str = Field(
        default="value",
        description="Column containing sensor value"
    )
    tsdb_timestamp_column: str = Field(
        default="timestamp",
        description="Column containing timestamp"
    )
    
    # === CSV ===
    csv_column_name: str | None = Field(
        None,
        description="CSV column name containing sensor values"
    )
    csv_timestamp_column: str = Field(
        default="timestamp",
        description="CSV column containing timestamps"
    )
    
    # === REST API ===
    api_endpoint: str | None = Field(
        None,
        description="REST API endpoint path (e.g., '/sensors/pt_001/latest')"
    )
    api_field_path: str | None = Field(
        None,
        description="JSON path to value (e.g., 'data.pressure.value')"
    )
    
    # === COMMON ===
    sampling_rate_hz: float | None = Field(
        None,
        gt=0,
        description="Data sampling frequency in Hz"
    )
    scaling_factor: float = Field(
        default=1.0,
        description="Scaling factor: final_value = (raw_value * scaling_factor) + offset"
    )
    offset: float = Field(
        default=0.0,
        description="Offset: final_value = (raw_value * scaling_factor) + offset"
    )
    
    # EXTENSION POINT: Phase 2 will add modbus/opcua/mqtt fields
    # modbus_slave_id: int | None = None
    # modbus_register: int | None = None
    # opcua_node_id: str | None = None
    # mqtt_topic: str | None = None
    
    def get_effective_sampling_rate(self, default: float = 1.0) -> float:
        """Get sampling rate with fallback to default."""
        return self.sampling_rate_hz or default


class PhysicalSensor(BaseModel):
    """Physical sensor installed on equipment.
    
    Minimal implementation for sensor registry.
    Describes physical hardware and how to read its data.
    
    Attributes:
        sensor_id: Unique sensor identifier (used in data sources)
        sensor_type: Type of sensor (pressure, flow, temperature, etc)
        manufacturer: Sensor manufacturer
        model: Sensor model number
        serial_number: Physical serial number (optional)
        measurement_range: (min, max) valid measurement range
        unit: Measurement unit (bar, L/min, °C, etc)
        accuracy_percent: Sensor accuracy as % of full scale
        install_date: Installation date (optional)
        last_calibration_date: Last calibration date (optional)
        calibration_interval_days: Days between calibrations
        data_source_config: Configuration for reading sensor data
        location_description: Human-readable installation location
        notes: Additional notes (optional)
    """
    
    # === IDENTIFICATION ===
    sensor_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique sensor identifier in data source"
    )
    
    sensor_type: Literal[
        "pressure",
        "flow",
        "temperature",
        "vibration",
        "rpm",
        "position",
        "current",
        "voltage"
    ] = Field(
        ...,
        description="Type of physical measurement"
    )
    
    # === PHYSICAL PROPERTIES ===
    manufacturer: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Sensor manufacturer name"
    )
    
    model: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Sensor model number"
    )
    
    serial_number: str | None = Field(
        None,
        max_length=200,
        description="Physical serial number (if available)"
    )
    
    # === MEASUREMENT SPECS ===
    measurement_range: tuple[float, float] = Field(
        ...,
        description="(min, max) valid measurement range"
    )
    
    unit: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Measurement unit (bar, L/min, °C, RPM, etc)"
    )
    
    accuracy_percent: float = Field(
        default=1.0,
        ge=0,
        le=100,
        description="Sensor accuracy as % of full scale"
    )
    
    # === MAINTENANCE ===
    install_date: date | None = Field(
        None,
        description="Sensor installation date"
    )
    
    last_calibration_date: date | None = Field(
        None,
        description="Last calibration date"
    )
    
    calibration_interval_days: int = Field(
        default=365,
        ge=1,
        description="Days between required calibrations"
    )
    
    # === DATA SOURCE ===
    data_source_config: SensorDataSourceConfig = Field(
        ...,
        description="Configuration for reading sensor data"
    )
    
    # === METADATA ===
    location_description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable installation location"
    )
    
    notes: str | None = Field(
        None,
        max_length=2000,
        description="Additional notes about sensor"
    )
    
    @field_validator("measurement_range")
    @classmethod
    def validate_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate measurement range min < max."""
        min_val, max_val = v
        if min_val >= max_val:
            raise ValueError(
                f"measurement_range min ({min_val}) must be < max ({max_val})"
            )
        return v
    
    def is_calibration_due(self, check_date: date | None = None) -> bool:
        """Check if sensor calibration is due.
        
        Args:
            check_date: Date to check (default: today)
        
        Returns:
            True if calibration is due or overdue
        """
        if self.last_calibration_date is None:
            return True  # Never calibrated
        
        if check_date is None:
            check_date = date.today()
        
        days_since_calibration = (check_date - self.last_calibration_date).days
        return days_since_calibration >= self.calibration_interval_days
    
    def get_full_scale_range(self) -> float:
        """Get full scale range (max - min)."""
        return self.measurement_range[1] - self.measurement_range[0]
    
    def get_absolute_accuracy(self) -> float:
        """Get absolute accuracy in measurement units."""
        return self.get_full_scale_range() * (self.accuracy_percent / 100.0)
