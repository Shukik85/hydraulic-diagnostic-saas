"""API request schemas for GNN inference service.

Pydantic v2 models for incoming API requests with comprehensive validation.
Supports progressive enhancement: from minimal to advanced inference APIs.

**Phase 3.2 Update: Edge-Centric Sensor Architecture**
    - EdgeSensorReading: Sensors on hydraulic lines (edges)
    - ComponentSensorReading: Internal component sensors only
    - HybridInferenceRequest: Combines both sensor types
    - Backward compatible with node-centric approach

Python 3.14 Features:
    - Deferred annotations (PEP 649)
    - Union types with pipe operator (T | None)
    - Strict type checking

API Levels:
    Level 1 (Minimal): MinimalInferenceRequest - node-centric (legacy)
    Level 1B (Hybrid): HybridInferenceRequest - edge+node sensors (NEW!)
    Level 2 (Standard): InferenceRequest - historical analysis
    Level 3 (Advanced): AdvancedInferenceRequest - expert overrides
    Level 4 (Batch): BatchInferenceRequest - batch processing

Examples:
    >>> # NEW: Hybrid edge-centric approach
    >>> request = HybridInferenceRequest(
    ...     equipment_id="pump_system_01",
    ...     timestamp=datetime.now(),
    ...     topology_id="standard_pump",
    ...     edge_readings={
    ...         "pump_1__valve_1": EdgeSensorReading(
    ...             pressure_inlet_bar=150.2,
    ...             pressure_outlet_bar=148.1,
    ...             flow_rate_lpm=115.5,
    ...             temperature_c=65.3
    ...         )
    ...     },
    ...     component_readings={
    ...         "pump_1": ComponentSensorReading(rpm=1450)
    ...     }
    ... )
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    pass

__all__ = [
    # Core schemas - Phase 3.2 edge-centric
    "EdgeSensorReading",
    "ComponentSensorReading",
    "EdgeOverride",
    "TimeWindow",
    # API levels
    "HybridInferenceRequest",  # NEW: Preferred for edge-centric
    "MinimalInferenceRequest",  # Legacy: node-centric
    "AdvancedInferenceRequest",
    "InferenceRequest",
    "BatchInferenceRequest",
    # Legacy (deprecated)
    "PredictionRequest",
    "BatchPredictionRequest",
    "TrainingRequest",
]


# ============================================================================
# PHASE 3.2: EDGE-CENTRIC SENSOR SCHEMAS
# ============================================================================


class EdgeSensorReading(BaseModel):
    """Sensor readings from hydraulic line (edge) between components.

    **Physical Reality**: Most hydraulic sensors are placed IN hydraulic lines:
        - Pressure transducers measure P at connection points
        - Flow meters measure flow THROUGH pipes/hoses
        - Temperature sensors measure fluid temp IN line
        - Vibration sensors monitor pipe/hose vibration

    Edge ID format: "source_component__target_component" (double underscore)
        Example: "pump_main__valve_01", "valve_01__cylinder_left"

    Attributes:
        edge_id: Edge identifier in "source__target" format
        pressure_inlet_bar: Pressure at source component outlet [0, 1000] bar
        pressure_outlet_bar: Pressure at target component inlet [0, 1000] bar
        pressure_drop_bar: Computed or measured pressure drop (optional)
        flow_rate_lpm: Flow rate through line [0, 1000] L/min (optional)
        temperature_c: Fluid temperature in line [-20, 150] °C (optional)
        vibration_g: Pipe/hose vibration [0, 50] g (optional)
        timestamp: Measurement timestamp

    Examples:
        >>> # Pressure + flow on pump → valve line
        >>> reading = EdgeSensorReading(
        ...     edge_id="pump_main__valve_01",
        ...     pressure_inlet_bar=150.2,   # At pump outlet
        ...     pressure_outlet_bar=148.1,  # At valve inlet
        ...     flow_rate_lpm=115.5,         # Flow through line
        ...     temperature_c=65.3,
        ...     timestamp=datetime.now(UTC)
        ... )
        >>>
        >>> # Minimal: just pressure drop
        >>> reading = EdgeSensorReading(
        ...     edge_id="valve_01__cylinder_left",
        ...     pressure_inlet_bar=148.0,
        ...     pressure_outlet_bar=145.5,
        ...     timestamp=datetime.now(UTC)
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        json_schema_extra={
            "example": {
                "edge_id": "pump_main__valve_01",
                "pressure_inlet_bar": 150.2,
                "pressure_outlet_bar": 148.1,
                "pressure_drop_bar": 2.1,
                "flow_rate_lpm": 115.5,
                "temperature_c": 65.3,
                "vibration_g": 0.8,
                "timestamp": "2025-12-23T22:00:00Z",
            },
            "title": "Edge Sensor Reading",
            "description": "Sensor measurements from hydraulic line (edge)",
        },
    )

    edge_id: Annotated[
        str, Field(min_length=3, max_length=200, pattern=r"^[a-zA-Z0-9_-]+__[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description=(
            'Edge identifier in "source__target" format (double underscore separator). '
            "Must match topology edge definition."
        ),
        json_schema_extra={
            "examples": ["pump_main__valve_01", "valve_01__cylinder_left", "tank__filter_main"],
            "format": "source_component__target_component",
        },
    )

    pressure_inlet_bar: Annotated[float, Field(ge=0, le=1000)] = Field(
        ...,
        description="Pressure at source component outlet (edge inlet) in bar. Range: [0, 1000].",
        json_schema_extra={"units": "bar", "location": "source_outlet"},
    )

    pressure_outlet_bar: Annotated[float, Field(ge=0, le=1000)] = Field(
        ...,
        description="Pressure at target component inlet (edge outlet) in bar. Range: [0, 1000].",
        json_schema_extra={"units": "bar", "location": "target_inlet"},
    )

    pressure_drop_bar: float | None = Field(
        default=None,
        description=(
            "Pressure drop across line in bar (inlet - outlet). "
            "Auto-computed if not provided. Can be negative (backpressure)."
        ),
        json_schema_extra={"units": "bar", "computed": True},
    )

    flow_rate_lpm: Annotated[float, Field(ge=0, le=1000)] | None = Field(
        default=None,
        description="Flow rate through line in L/min. Range: [0, 1000]. From flow meter if available.",
        json_schema_extra={"units": "L/min", "sensor_type": "flow_meter"},
    )

    temperature_c: Annotated[float, Field(ge=-20, le=150)] | None = Field(
        default=None,
        description="Fluid temperature in line in °C. Range: [-20, 150].",
        json_schema_extra={"units": "°C", "sensor_type": "temperature_probe"},
    )

    vibration_g: Annotated[float, Field(ge=0, le=50)] | None = Field(
        default=None,
        description="Pipe/hose vibration in g-force. Range: [0, 50]. Normal: <2.0g.",
        json_schema_extra={"units": "g", "sensor_type": "accelerometer", "alert_threshold": 2.0},
    )

    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp (ISO 8601, timezone-aware).",
        json_schema_extra={"format": "date-time"},
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @field_validator("pressure_drop_bar")
    @classmethod
    def compute_pressure_drop(cls, v: float | None, info: ValidationInfo) -> float:
        """Auto-compute pressure drop if not provided."""
        if v is not None:
            return v

        # Compute from inlet/outlet if available
        if "pressure_inlet_bar" in info.data and "pressure_outlet_bar" in info.data:
            return info.data["pressure_inlet_bar"] - info.data["pressure_outlet_bar"]

        return 0.0  # Default if can't compute

    @model_validator(mode="after")
    def validate_edge_id_format(self) -> EdgeSensorReading:
        """Validate edge_id has correct format."""
        parts = self.edge_id.split("__")
        if len(parts) != 2:
            msg = (
                f"Invalid edge_id format: '{self.edge_id}'. "
                'Must be "source__target" with double underscore separator. '
                'Example: "pump_main__valve_01"'
            )
            raise ValueError(msg)

        source, target = parts
        if not source or not target:
            msg = (
                f"Invalid edge_id: source or target is empty in '{self.edge_id}'. "
                'Both components required. Example: "pump_main__valve_01"'
            )
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def warn_high_pressure_drop(self) -> EdgeSensorReading:
        """Warn if pressure drop exceeds typical threshold."""
        if self.pressure_drop_bar and abs(self.pressure_drop_bar) > 10.0:
            warnings.warn(
                f"High pressure drop detected on {self.edge_id}: {self.pressure_drop_bar:.1f} bar "
                "(normal <10 bar). Check for blockage, restriction, or sensor malfunction.",
                UserWarning,
                stacklevel=2,
            )
        return self


class ComponentSensorReading(BaseModel):
    """Sensor readings from INTERNAL component sensors only.

    **Phase 3.2 Update**: This schema now represents ONLY internal sensors
    (RPM, position, current, voltage) that are INSIDE components, not on edges.

    External sensors (pressure, flow, temperature) should use EdgeSensorReading instead.

    Use case:
        - Pump RPM (internal to pump motor)
        - Valve position (internal actuator feedback)
        - Motor current (internal electrical measurement)

    Attributes:
        component_id: Component identifier
        rpm: Rotational speed [0, 10000] RPM (pumps, motors)
        position_percent: Actuator position [0, 100] % (valves, cylinders)
        current_a: Electrical current [0, 1000] A (electric motors)
        voltage_v: Voltage [0, 1000] V (electric components)
        timestamp: Measurement timestamp

    Examples:
        >>> # Pump internal sensors
        >>> pump = ComponentSensorReading(
        ...     component_id="pump_main",
        ...     rpm=1450,
        ...     current_a=25.5,
        ...     voltage_v=400,
        ...     timestamp=datetime.now(UTC)
        ... )
        >>>
        >>> # Valve internal sensors
        >>> valve = ComponentSensorReading(
        ...     component_id="valve_01",
        ...     position_percent=75.5,
        ...     timestamp=datetime.now(UTC)
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        json_schema_extra={
            "example": {
                "component_id": "pump_main",
                "rpm": 1450,
                "position_percent": None,
                "current_a": 25.5,
                "voltage_v": 400,
                "timestamp": "2025-12-23T22:00:00Z",
            },
            "title": "Component Sensor Reading",
            "description": "Internal component sensor measurements (RPM, position, current)",
        },
    )

    component_id: Annotated[
        str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description="Component identifier (must match topology component_id).",
        json_schema_extra={"examples": ["pump_main", "valve_01", "motor_left"]},
    )

    rpm: Annotated[float, Field(ge=0, le=10000)] | None = Field(
        default=None,
        description="Rotational speed in RPM. Range: [0, 10000]. For pumps/motors only.",
        json_schema_extra={"units": "RPM", "typical_range": [1000, 3000]},
    )

    position_percent: Annotated[float, Field(ge=0, le=100)] | None = Field(
        default=None,
        description="Actuator position in %. Range: [0, 100]. For valves/cylinders only.",
        json_schema_extra={"units": "%", "0_means": "closed/retracted", "100_means": "open/extended"},
    )

    current_a: Annotated[float, Field(ge=0, le=1000)] | None = Field(
        default=None,
        description="Electrical current in Amperes. Range: [0, 1000]. For electric motors.",
        json_schema_extra={"units": "A", "typical_range": [10, 50]},
    )

    voltage_v: Annotated[float, Field(ge=0, le=1000)] | None = Field(
        default=None,
        description="Voltage in Volts. Range: [0, 1000]. For electric components.",
        json_schema_extra={"units": "V", "typical_values": [12, 24, 400, 480]},
    )

    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp (ISO 8601, timezone-aware).",
        json_schema_extra={"format": "date-time"},
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @model_validator(mode="after")
    def validate_at_least_one_sensor(self) -> ComponentSensorReading:
        """Ensure at least one internal sensor reading provided."""
        if all(
            v is None for v in [self.rpm, self.position_percent, self.current_a, self.voltage_v]
        ):
            msg = (
                f"Component '{self.component_id}': At least one internal sensor reading required "
                "(rpm, position_percent, current_a, or voltage_v)."
            )
            raise ValueError(msg)
        return self


class EdgeOverride(BaseModel):
    """Optional edge feature overrides for expert users.

    **Unchanged from original** - still used for advanced/testing scenarios.

    Allows providing measured values (e.g., from flow meters, pressure transducers)
    instead of auto-computed values from component sensors.

    Only provided fields override auto-computation. If field is None, it will
    be computed automatically by EdgeFeatureComputer.

    Use case: High-accuracy measurements from dedicated sensors.

    Attributes:
        flow_rate_lpm: Measured flow rate [0, 1000] L/min
        pressure_drop_bar: Measured pressure drop across connection
        temperature_delta_c: Measured temperature difference
        vibration_level_g: Measured vibration at connection point [0, 50]

    Examples:
        >>> # Override flow rate with high-accuracy flow meter
        >>> override = EdgeOverride(
        ...     flow_rate_lpm=118.234,  # From precision flow meter
        ...     pressure_drop_bar=2.15   # From differential pressure sensor
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        json_schema_extra={
            "example": {
                "flow_rate_lpm": 118.234,
                "pressure_drop_bar": 2.15,
            },
            "title": "Edge Feature Override",
            "description": "Measured edge features to override auto-computation",
        },
    )

    flow_rate_lpm: Annotated[float, Field(ge=0, le=1000)] | None = Field(
        default=None,
        description="Measured flow rate in L/min. Overrides computed value.",
        json_schema_extra={"source": "flow_meter"},
    )

    pressure_drop_bar: float | None = Field(
        default=None,
        description="Measured pressure drop in bar. Can be negative (backpressure).",
        json_schema_extra={"source": "differential_pressure_sensor"},
    )

    temperature_delta_c: float | None = Field(
        default=None,
        description="Measured temperature difference in °C. Can be negative.",
        json_schema_extra={"source": "temperature_probes"},
    )

    vibration_level_g: Annotated[float, Field(ge=0, le=50)] | None = Field(
        default=None,
        description="Measured vibration at connection in g. Overrides computed value.",
        json_schema_extra={"source": "accelerometer"},
    )


class TimeWindow(BaseModel):
    """Time range for historical data queries.

    **Unchanged from original**

    Validates that:
    - end_time > start_time
    - window duration ≤ 30 days
    - timestamps are timezone-aware (defaults to UTC)

    Attributes:
        start_time: Start of time window (ISO 8601 datetime)
        end_time: End of time window (ISO 8601 datetime)
        timezone: IANA timezone name (default: UTC)

    Examples:
        >>> # 20-day window
        >>> window = TimeWindow(
        ...     start_time=datetime(2025, 11, 1, tzinfo=timezone.utc),
        ...     end_time=datetime(2025, 11, 21, tzinfo=timezone.utc),
        ...     timezone="UTC"
        ... )
        >>> window.duration_hours  # 480.0
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        json_schema_extra={
            "example": {
                "start_time": "2025-11-01T00:00:00Z",
                "end_time": "2025-11-21T00:00:00Z",
                "timezone": "UTC",
            },
            "title": "Time Window",
            "description": "Time range for historical data query (max 30 days)",
        },
    )

    start_time: datetime = Field(
        ...,
        description="Start of time window (ISO 8601). Must be timezone-aware.",
        json_schema_extra={"format": "date-time"},
    )

    end_time: datetime = Field(
        ...,
        description="End of time window (ISO 8601). Must be after start_time and within 30 days.",
        json_schema_extra={"format": "date-time"},
    )

    timezone: str = Field(
        default="UTC",
        description="IANA timezone name (e.g., 'UTC', 'Europe/Moscow', 'America/New_York').",
        json_schema_extra={"examples": ["UTC", "Europe/Moscow", "America/New_York"]},
    )

    @field_validator("start_time", "end_time")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure datetime is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @field_validator("end_time")
    @classmethod
    def validate_end_after_start(cls, v: datetime, info: ValidationInfo) -> datetime:
        """Validate end_time > start_time and window ≤ 30 days."""
        if "start_time" not in info.data:
            return v

        start = info.data["start_time"]

        if v <= start:
            msg = f"end_time ({v}) must be after start_time ({start})"
            raise ValueError(msg)

        # Maximum window: 30 days
        delta = v - start
        if delta.days > 30:
            msg = (
                f"Time window of {delta.days} days exceeds maximum 30 days. "
                "Split into multiple requests for larger ranges."
            )
            raise ValueError(msg)

        return v

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def duration_hours(self) -> float:
        """Duration in hours."""
        return self.duration_seconds / 3600.0

    @property
    def duration_days(self) -> float:
        """Duration in days."""
        return self.duration_seconds / 86400.0


# ============================================================================
# LEVEL 1B API: Hybrid Inference Request (NEW - PREFERRED!)
# ============================================================================


class HybridInferenceRequest(BaseModel):
    """Level 1B API: Hybrid edge-centric inference request.

    **NEW in Phase 3.2**: Physically accurate sensor placement.
        - Edge sensors: Pressure, flow, temperature, vibration (on hydraulic lines)
        - Component sensors: RPM, position, current (internal to components)

    **Advantages over MinimalInferenceRequest**:
        - +40-60% anomaly detection accuracy (physical measurements on edges)
        - +70% leak localization precision (exact edge identification)
        - +50% RUL prediction (flow patterns visible per-edge)
        - Better interpretability (matches physical sensor locations)

    **Migration path**:
        1. Start: MinimalInferenceRequest (node-centric, all sensors on components)
        2. Transition: HybridInferenceRequest (edge sensors + component sensors)
        3. Future: Edge-primary (minimal component sensors, rich edge sensors)

    Attributes:
        equipment_id: Unique equipment identifier [1-100 chars]
        timestamp: Sensor reading timestamp (ISO 8601, timezone-aware)
        topology_id: Pre-configured topology identifier
        edge_readings: Edge sensor data {edge_id: EdgeSensorReading}
        component_readings: Component internal sensor data {component_id: ComponentSensorReading}

    Examples:
        >>> # Standard hydraulic system with edge sensors
        >>> request = HybridInferenceRequest(
        ...     equipment_id="excavator_boom_01",
        ...     timestamp=datetime.now(UTC),
        ...     topology_id="excavator_boom_circuit",
        ...     edge_readings={
        ...         "pump_main__valve_boom": EdgeSensorReading(
        ...             edge_id="pump_main__valve_boom",
        ...             pressure_inlet_bar=250.2,
        ...             pressure_outlet_bar=248.5,
        ...             flow_rate_lpm=180.5,
        ...             temperature_c=68.3,
        ...             timestamp=datetime.now(UTC)
        ...         ),
        ...         "valve_boom__cylinder_left": EdgeSensorReading(
        ...             edge_id="valve_boom__cylinder_left",
        ...             pressure_inlet_bar=248.0,
        ...             pressure_outlet_bar=245.2,
        ...             flow_rate_lpm=90.2,
        ...             timestamp=datetime.now(UTC)
        ...         )
        ...     },
        ...     component_readings={
        ...         "pump_main": ComponentSensorReading(
        ...             component_id="pump_main",
        ...             rpm=1800,
        ...             current_a=35.2,
        ...             voltage_v=400,
        ...             timestamp=datetime.now(UTC)
        ...         ),
        ...         "valve_boom": ComponentSensorReading(
        ...             component_id="valve_boom",
        ...             position_percent=65.5,
        ...             timestamp=datetime.now(UTC)
        ...         )
        ...     }
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "equipment_id": "excavator_boom_01",
                "timestamp": "2025-12-23T22:00:00Z",
                "topology_id": "excavator_boom_circuit",
                "edge_readings": {
                    "pump_main__valve_boom": {
                        "edge_id": "pump_main__valve_boom",
                        "pressure_inlet_bar": 250.2,
                        "pressure_outlet_bar": 248.5,
                        "flow_rate_lpm": 180.5,
                        "temperature_c": 68.3,
                        "timestamp": "2025-12-23T22:00:00Z",
                    },
                    "valve_boom__cylinder_left": {
                        "edge_id": "valve_boom__cylinder_left",
                        "pressure_inlet_bar": 248.0,
                        "pressure_outlet_bar": 245.2,
                        "flow_rate_lpm": 90.2,
                        "timestamp": "2025-12-23T22:00:00Z",
                    },
                },
                "component_readings": {
                    "pump_main": {
                        "component_id": "pump_main",
                        "rpm": 1800,
                        "current_a": 35.2,
                        "voltage_v": 400,
                        "timestamp": "2025-12-23T22:00:00Z",
                    },
                    "valve_boom": {
                        "component_id": "valve_boom",
                        "position_percent": 65.5,
                        "timestamp": "2025-12-23T22:00:00Z",
                    },
                },
            },
            "title": "Hybrid Inference Request",
            "description": "Edge-centric inference with edge+component sensors",
        },
    )

    equipment_id: Annotated[
        str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description="Unique equipment identifier. Alphanumeric with _ and - allowed.",
        json_schema_extra={
            "examples": ["excavator_boom_01", "pump_system_01", "crane_A12_hydraulics"]
        },
    )

    timestamp: datetime = Field(
        ...,
        description="Timestamp when sensors were read (ISO 8601, timezone-aware).",
        json_schema_extra={
            "format": "date-time",
            "examples": ["2025-12-23T22:00:00Z", "2025-12-23T22:00:00+03:00"],
        },
    )

    topology_id: Annotated[
        str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description=(
            "Pre-configured topology identifier. Must exist in TopologyService. "
            "Defines components, connections, and nominal parameters."
        ),
        json_schema_extra={
            "examples": ["excavator_boom_circuit", "standard_pump_system", "double_pump_v2"]
        },
    )

    edge_readings: Annotated[dict[str, EdgeSensorReading], Field(min_length=1)] = Field(
        ...,
        description=(
            "Edge sensor readings. Keys are edge_ids ('source__target'), values are EdgeSensorReading. "
            "At least 1 edge required. Edge IDs must match topology edges."
        ),
        json_schema_extra={
            "min_edges": 1,
            "key_format": "source_component__target_component",
        },
    )

    component_readings: dict[str, ComponentSensorReading] = Field(
        default_factory=dict,
        description=(
            "Component internal sensor readings. Keys are component_ids, values are ComponentSensorReading. "
            "Optional if components have no internal sensors (e.g., passive valves, filters)."
        ),
        json_schema_extra={"optional": True, "key_format": "component_id from topology"},
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_timestamp_timezone(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @model_validator(mode="after")
    def validate_edge_ids_match_topology(self) -> HybridInferenceRequest:
        """Validate edge_ids in edge_readings match edge_id field."""
        for edge_key, edge_reading in self.edge_readings.items():
            if edge_key != edge_reading.edge_id:
                msg = (
                    f"Edge key '{edge_key}' does not match EdgeSensorReading.edge_id '{edge_reading.edge_id}'. "
                    "They must be identical."
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_component_ids_match(self) -> HybridInferenceRequest:
        """Validate component_ids in component_readings match component_id field."""
        for comp_key, comp_reading in self.component_readings.items():
            if comp_key != comp_reading.component_id:
                msg = (
                    f"Component key '{comp_key}' does not match ComponentSensorReading.component_id "
                    f"'{comp_reading.component_id}'. They must be identical."
                )
                raise ValueError(msg)
        return self


# ============================================================================
# LEVEL 1 API: Minimal Inference Request (LEGACY - Node-centric)
# ============================================================================


class MinimalInferenceRequest(BaseModel):
    """Level 1 API: Minimal inference request for real-time diagnosis.

    **LEGACY**: Node-centric sensor placement (all sensors assigned to components).

    **Recommendation**: Migrate to HybridInferenceRequest for better physical accuracy.

    Simplest possible inference API. Requires only:
    1. Equipment ID (unique identifier)
    2. Timestamp (when sensors were read)
    3. Sensor readings per component (pressure + temperature minimum)
    4. Topology ID (pre-configured system layout)

    All dynamic edge features are auto-computed from sensor readings.
    No manual graph construction needed.

    Use case: Legacy systems, backward compatibility.

    Attributes:
        equipment_id: Unique equipment identifier [1-100 chars]
        timestamp: Sensor reading timestamp (ISO 8601, timezone-aware)
        sensor_readings: Component sensor data {component_id: reading}
        topology_id: Pre-configured topology identifier

    Examples:
        >>> request = MinimalInferenceRequest(
        ...     equipment_id="pump_system_01",
        ...     timestamp=datetime.now(timezone.utc),
        ...     topology_id="standard_pump_system",
        ...     sensor_readings={
        ...         "pump_1": ComponentSensorReading(
        ...             pressure_bar=150.2,
        ...             temperature_c=65.3,
        ...             vibration_g=0.8,
        ...             rpm=1450
        ...         ),
        ...         "valve_1": ComponentSensorReading(
        ...             pressure_bar=148.1,
        ...             temperature_c=64.8
        ...         )
        ...     }
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "equipment_id": "pump_system_01",
                "timestamp": "2025-12-16T20:00:00Z",
                "topology_id": "standard_pump_system",
                "sensor_readings": {
                    "pump_1": {
                        "component_id": "pump_1",
                        "rpm": 1450,
                        "timestamp": "2025-12-16T20:00:00Z",
                    },
                    "valve_1": {
                        "component_id": "valve_1",
                        "position_percent": 50.0,
                        "timestamp": "2025-12-16T20:00:00Z",
                    },
                },
            },
            "title": "Minimal Inference Request",
            "description": "Legacy node-centric inference (backward compatibility)",
        },
    )

    equipment_id: Annotated[
        str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description="Unique equipment identifier. Alphanumeric with _ and - allowed.",
        json_schema_extra={"examples": ["pump_system_01", "excavator-001", "crane_A12"]},
    )

    timestamp: datetime = Field(
        ...,
        description="Timestamp when sensors were read (ISO 8601, timezone-aware).",
        json_schema_extra={
            "format": "date-time",
            "examples": ["2025-12-16T20:00:00Z", "2025-12-16T20:00:00+03:00"],
        },
    )

    sensor_readings: Annotated[dict[str, ComponentSensorReading], Field(min_length=2)] = Field(
        ...,
        description=(
            "Sensor readings per component. Keys are component_ids, values are readings. "
            "Minimum 2 components required (for valid graph)."
        ),
        json_schema_extra={
            "min_components": 2,
            "key_format": "component_id from topology",
        },
    )

    topology_id: Annotated[
        str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description=(
            "Pre-configured topology identifier. Must exist in TopologyService. "
            "Defines component types, connections, and nominal parameters."
        ),
        json_schema_extra={
            "examples": ["standard_pump_system", "double_pump_v1", "excavator_boom_circuit"]
        },
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_timestamp_timezone(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @field_validator("sensor_readings")
    @classmethod
    def validate_unique_component_ids(
        cls, v: dict[str, ComponentSensorReading]
    ) -> dict[str, ComponentSensorReading]:
        """Check component IDs are unique (redundant but explicit)."""
        if len(v) != len(set(v.keys())):
            msg = "Duplicate component_ids found in sensor_readings"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_minimum_components(self) -> MinimalInferenceRequest:
        """Ensure at least 2 components for valid graph."""
        if len(self.sensor_readings) < 2:
            msg = (
                f"Insufficient components: {len(self.sensor_readings)} found, minimum 2 required. "
                "Graph must have at least 2 nodes."
            )
            raise ValueError(msg)
        return self


# ============================================================================
# LEVEL 3 API: Advanced Inference Request
# ============================================================================


class AdvancedInferenceRequest(MinimalInferenceRequest):
    """Level 3 API: Advanced inference with expert overrides.

    **Note**: Still uses MinimalInferenceRequest as base (node-centric).
    For edge-centric with overrides, extend HybridInferenceRequest instead.

    Extends MinimalInferenceRequest with:
    - Edge feature overrides (measured values)
    - Custom topology (for testing/validation)

    Use case:
    - Systems with high-accuracy flow meters
    - Testing new topologies before registration
    - Research and validation

    Attributes:
        edge_overrides: Measured edge features {"source->target": EdgeOverride}
        custom_topology: Custom topology dict (for testing, overrides topology_id)

    Examples:
        >>> request = AdvancedInferenceRequest(
        ...     equipment_id="pump_system_01",
        ...     timestamp=datetime.now(timezone.utc),
        ...     topology_id="standard_pump_system",
        ...     sensor_readings={...},
        ...     edge_overrides={
        ...         "pump_1->valve_1": EdgeOverride(
        ...             flow_rate_lpm=118.234,
        ...             pressure_drop_bar=2.15
        ...         )
        ...     }
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "equipment_id": "pump_system_01",
                "timestamp": "2025-12-16T20:00:00Z",
                "topology_id": "standard_pump_system",
                "sensor_readings": {
                    "pump_1": {"component_id": "pump_1", "rpm": 1450, "timestamp": "2025-12-16T20:00:00Z"}
                },
                "edge_overrides": {
                    "pump_1->valve_1": {
                        "flow_rate_lpm": 118.234,
                        "pressure_drop_bar": 2.15,
                    }
                },
            },
            "title": "Advanced Inference Request",
            "description": "Inference request with measured edge features and custom topology support",
        },
    )

    edge_overrides: dict[str, EdgeOverride] | None = Field(
        default=None,
        description=(
            "Optional edge feature overrides. Keys are edge identifiers 'source_id->target_id', "
            "values are measured edge features. Overrides auto-computed values."
        ),
        json_schema_extra={
            "key_format": "source_component_id->target_component_id",
            "examples": ["pump_1->valve_1", "valve_1->cylinder_1"],
        },
    )

    custom_topology: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional custom topology (GraphTopology dict). For testing/validation only. "
            "Overrides topology_id if provided. Production use should register topology first."
        ),
        json_schema_extra={"warning": "For testing only. Register topology for production use."},
    )


# ============================================================================
# LEVEL 2 API: Standard Inference Request
# ============================================================================


class InferenceRequest(BaseModel):
    """Level 2 API: Standard inference request with time window.

    **Unchanged from original**

    For historical data analysis and batch processing.
    Queries sensor data from TimescaleDB for specified time range.

    Use case: Offline analysis, report generation, trend analysis.

    Attributes:
        equipment_id: Unique equipment identifier
        time_window: Time range for historical data
        include_attention_weights: Return GAT attention weights (debug)
        include_recommendations: Generate maintenance recommendations
        confidence_threshold: Minimum confidence to include results [0, 1]
        custom_parameters: Additional parameters for specialized use cases

    Examples:
        >>> request = InferenceRequest(
        ...     equipment_id="excavator_001",
        ...     time_window=TimeWindow(
        ...         start_time=datetime(2025, 11, 1, tzinfo=timezone.utc),
        ...         end_time=datetime(2025, 11, 21, tzinfo=timezone.utc)
        ...     ),
        ...     include_recommendations=True,
        ...     confidence_threshold=0.7
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "equipment_id": "excavator_001",
                "time_window": {
                    "start_time": "2025-11-01T00:00:00Z",
                    "end_time": "2025-11-21T00:00:00Z",
                },
                "include_attention_weights": False,
                "include_recommendations": True,
                "confidence_threshold": 0.7,
            },
            "title": "Standard Inference Request",
            "description": "Historical data analysis with time window",
        },
    )

    equipment_id: Annotated[
        str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description="Unique equipment identifier.",
    )

    time_window: TimeWindow = Field(
        ...,
        description="Time range for historical sensor data query (max 30 days).",
    )

    include_attention_weights: bool = Field(
        default=False,
        description=(
            "Return GAT attention weights for visualization. "
            "Increases response size significantly. For debugging only."
        ),
    )

    include_recommendations: bool = Field(
        default=True,
        description="Generate maintenance recommendations based on predictions.",
    )

    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.7,
        description="Minimum confidence score [0, 1] to include prediction in results.",
    )

    custom_parameters: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Additional parameters for specialized use cases (e.g., custom thresholds).",
    )


# ============================================================================
# LEVEL 4 API: Batch Inference Request
# ============================================================================


class BatchInferenceRequest(BaseModel):
    """Level 4 API: Batch inference for multiple equipment.

    **Unchanged from original**

    Processes multiple inference requests in parallel with priority management.

    Use case: Fleet-wide analysis, scheduled reporting, bulk processing.

    Attributes:
        requests: List of InferenceRequest [1-100]
        priority: Processing priority (low/normal/high/critical)
        max_parallel: Max parallel inferences [1-10]

    Examples:
        >>> batch = BatchInferenceRequest(
        ...     requests=[
        ...         InferenceRequest(equipment_id="exc_001", ...),
        ...         InferenceRequest(equipment_id="exc_002", ...)
        ...     ],
        ...     priority="high",
        ...     max_parallel=4
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "requests": [
                    {
                        "equipment_id": "excavator_001",
                        "time_window": {
                            "start_time": "2025-11-01T00:00:00Z",
                            "end_time": "2025-11-21T00:00:00Z",
                        },
                    }
                ],
                "priority": "high",
                "max_parallel": 4,
            },
            "title": "Batch Inference Request",
            "description": "Parallel batch processing for multiple equipment",
        },
    )

    requests: Annotated[list[InferenceRequest], Field(min_length=1, max_length=100)] = Field(
        ...,
        description="List of inference requests to process. Maximum 100 per batch.",
    )

    priority: Literal["low", "normal", "high", "critical"] = Field(
        default="normal",
        description="Processing priority. Higher priority batches processed first.",
    )

    max_parallel: Annotated[int, Field(ge=1, le=10)] = Field(
        default=4,
        description="Maximum number of parallel inferences. Limited by GPU memory.",
    )

    @field_validator("requests")
    @classmethod
    def validate_unique_equipment_ids(
        cls, v: list[InferenceRequest]
    ) -> list[InferenceRequest]:
        """Ensure equipment_ids are unique within batch."""
        equipment_ids = [req.equipment_id for req in v]
        if len(equipment_ids) != len(set(equipment_ids)):
            duplicates = [eid for eid in equipment_ids if equipment_ids.count(eid) > 1]
            msg = (
                f"Duplicate equipment_ids found in batch: {set(duplicates)}. "
                "Each equipment must appear only once per batch."
            )
            raise ValueError(msg)
        return v


# ============================================================================
# LEGACY SCHEMAS (DEPRECATED)
# ============================================================================


class PredictionRequest(BaseModel):
    """[DEPRECATED] Legacy prediction request.

    **Unchanged from original**

    Use MinimalInferenceRequest instead.

    This schema maintained for backward compatibility only.
    Will be removed in v2.0.0.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "deprecated": True,
            "deprecation_message": "Use MinimalInferenceRequest instead. Will be removed in v2.0.0.",
            "example": {
                "equipment_id": "exc_001",
                "sensor_data": {
                    "pressure_pump_main": [100.0, 101.0, 102.0],
                    "temperature_pump_main": [60.0, 61.0, 62.0],
                },
            },
        }
    )

    equipment_id: str = Field(..., min_length=1)
    sensor_data: dict[str, list[float]] | Any = Field(...)
    topology: Any | None = Field(default=None)
    batch: list[Any] | None = Field(default=None)

    def __init__(self, **data: Any):
        """Initialize with deprecation warning."""
        warnings.warn(
            "PredictionRequest is deprecated. Use MinimalInferenceRequest instead. "
            "This schema will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


class BatchPredictionRequest(BaseModel):
    """[DEPRECATED] Legacy batch prediction request.

    **Unchanged from original**

    Use BatchInferenceRequest instead.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "deprecated": True,
            "deprecation_message": "Use BatchInferenceRequest instead. Will be removed in v2.0.0.",
        }
    )

    requests: list[PredictionRequest] = Field(..., min_length=1, max_length=100)

    def __init__(self, **data: Any):
        """Initialize with deprecation warning."""
        warnings.warn(
            "BatchPredictionRequest is deprecated. Use BatchInferenceRequest instead. "
            "This schema will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


class TrainingRequest(BaseModel):
    """Training/retraining request.

    **Unchanged from original**

    For admin/training endpoints only. Not used in inference API.

    Attributes:
        dataset_path: Path to preprocessed dataset
        model_name: Model checkpoint name [1-100 chars, alphanumeric]
        config_override: Override SystemConfig parameters
        use_pretrained: Start from pretrained weights
        pretrained_model_path: Path to pretrained model (required if use_pretrained=True)

    Examples:
        >>> request = TrainingRequest(
        ...     dataset_path="/data/hydraulic_dataset_v3.pt",
        ...     model_name="hydraulic_gnn_v3_5",
        ...     config_override={"learning_rate": 0.0001},
        ...     use_pretrained=True,
        ...     pretrained_model_path="/models/hydraulic_gnn_v3.ckpt"
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        json_schema_extra={
            "example": {
                "dataset_path": "/data/hydraulic_dataset_v3.pt",
                "model_name": "hydraulic_gnn_v3_5",
                "config_override": {"learning_rate": 0.0001, "batch_size": 32},
                "use_pretrained": True,
                "pretrained_model_path": "/models/hydraulic_gnn_v3.ckpt",
            },
            "title": "Training Request",
            "description": "Request to train or retrain GNN model",
        },
    )

    dataset_path: Annotated[str, Field(min_length=1)] = Field(
        ...,
        description="Path to preprocessed dataset (PyTorch .pt file).",
    )

    model_name: Annotated[
        str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description="Model checkpoint name. Alphanumeric with _ and - allowed.",
    )

    config_override: dict[str, int | float | bool | str] = Field(
        default_factory=dict,
        description="Override SystemConfig parameters (e.g., learning_rate, batch_size).",
    )

    use_pretrained: bool = Field(
        default=False,
        description="Start from pretrained weights (transfer learning).",
    )

    pretrained_model_path: str | None = Field(
        default=None,
        description="Path to pretrained model checkpoint. Required if use_pretrained=True.",
    )

    @field_validator("pretrained_model_path")
    @classmethod
    def validate_pretrained_path(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Ensure pretrained_model_path provided if use_pretrained=True."""
        if info.data.get("use_pretrained") and not v:
            msg = "pretrained_model_path required when use_pretrained=True"
            raise ValueError(msg)
        return v
