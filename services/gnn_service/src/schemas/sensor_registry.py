"""Physical sensor registry schemas (STUB for Phase 2).

TODO Phase 2:
    - Full PhysicalSensor implementation
    - SensorDataSourceConfig for all source types
    - SensorToTopologyMapping
    - Validation logic

Currently: Minimal stub for testing TimescaleDBMockAdapter.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["PhysicalSensor", "SensorDataSourceConfig"]


class SensorDataSourceConfig(BaseModel):
    """Sensor data source configuration (STUB)."""

    source_type: str = "timescaledb"


class PhysicalSensor(BaseModel):
    """Physical sensor configuration (STUB for testing).

    TODO Phase 2: Add full implementation with:
        - manufacturer, model, serial_number
        - installation date, calibration info
        - complete data source configs (Modbus, OPC-UA, etc)
        - validation
    """

    sensor_id: str = Field(..., min_length=1)
    sensor_type: str
    measurement_range: tuple[float, float]
    unit: str = "unknown"
    data_source_config: SensorDataSourceConfig = Field(
        default_factory=SensorDataSourceConfig
    )
