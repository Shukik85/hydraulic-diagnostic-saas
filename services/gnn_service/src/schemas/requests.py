"""API request schemas for GNN inference service.

Pydantic v2 models for incoming API requests with comprehensive validation.
Supports progressive enhancement: from minimal to advanced inference APIs.

**Phase 3.2 Update: Edge-Centric Sensor Architecture**
    - EdgeSensorReading: Sensors on hydraulic lines (edges)
    - ComponentSensorReading: Internal component sensors only
    - HybridInferenceRequest: Combines both sensor types
    - Backward compatible with node-centric approach

**Day 2 Addition: FlexibleInferenceRequest**
    - ALL sensor fields optional (except equipment metadata)
    - Input for ValueSubstitutionEngine
    - Supports ANY sensor coverage (1 sensor → full coverage)
    - Physics-based estimation fills missing values

Python 3.14 Features:
    - Deferred annotations (PEP 649)
    - Union types with pipe operator (T | None)
    - Strict type checking

API Levels:
    Level 0 (Flexible): FlexibleInferenceRequest - partial sensor data (NEW!)
    Level 1 (Minimal): MinimalInferenceRequest - node-centric (legacy)
    Level 1B (Hybrid): HybridInferenceRequest - edge+node sensors
    Level 2 (Standard): InferenceRequest - historical analysis
    Level 3 (Advanced): AdvancedInferenceRequest - expert overrides
    Level 4 (Batch): BatchInferenceRequest - batch processing

Examples:
    >>> # NEW: Flexible partial sensor data
    >>> request = FlexibleInferenceRequest(
    ...     equipment_id="pump_system_01",
    ...     timestamp=datetime.now(),
    ...     topology_id="standard_pump",
    ...     edge_readings={
    ...         "pump_1__valve_1": FlexibleEdgeSensorReading(
    ...             pressure_inlet_bar=150.2,  # Only 1 field!
    ...             timestamp=datetime.now()
    ...         )
    ...     }
    ... )
    >>> # ValueSubstitutionEngine fills missing flow_rate, temperature, etc.
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
    # NEW: Flexible schemas for partial sensor data
    "FlexibleEdgeSensorReading",
    "FlexibleComponentSensorReading",
    "FlexibleInferenceRequest",
    # API levels
    "HybridInferenceRequest",  # Preferred for edge-centric
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
# PHASE 3.3: FLEXIBLE SENSOR SCHEMAS (Day 2 - Value Substitution)
# ============================================================================


class FlexibleEdgeSensorReading(BaseModel):
    """Edge sensor reading with ALL fields optional (except edge_id, timestamp).

    **Day 2 Addition**: Input for ValueSubstitutionEngine.

    Allows ANY sensor coverage:
        - Minimal: Just 1 pressure sensor
        - Partial: Pressure + flow (no temperature)
        - Full: All sensors

    Missing values filled by ValueSubstitutionEngine using:
        1. Physics-based estimation (Darcy-Weisbach, conservation laws)
        2. Nominal values from topology
        3. Reasonable defaults

    Use case:
        >>> # Client has only inlet pressure
        >>> reading = FlexibleEdgeSensorReading(
        ...     edge_id="pump__valve",
        ...     pressure_inlet_bar=150.2,
        ...     timestamp=datetime.now(UTC)
        ... )
        >>> # Engine estimates: outlet, flow, temperature

    Attributes:
        edge_id: Edge identifier (REQUIRED)
        timestamp: Measurement timestamp (REQUIRED)
        pressure_inlet_bar: Inlet pressure (optional)
        pressure_outlet_bar: Outlet pressure (optional)
        pressure_drop_bar: Pressure drop (optional, auto-computed)
        flow_rate_lpm: Flow rate (optional)
        temperature_c: Temperature (optional)
        vibration_g: Vibration (optional)
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        json_schema_extra={
            "example": {
                "edge_id": "pump_main__valve_01",
                "pressure_inlet_bar": 150.2,
                "timestamp": "2025-12-26T22:00:00Z",
            },
            "title": "Flexible Edge Sensor Reading",
            "description": "Edge reading with optional fields (for partial sensor coverage)",
        },
    )

    edge_id: Annotated[
        str, Field(min_length=3, max_length=200, pattern=r"^[a-zA-Z0-9_-]+__[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description='Edge identifier in "source__target" format (REQUIRED).',
    )

    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp (REQUIRED, ISO 8601, timezone-aware).",
    )

    pressure_inlet_bar: Annotated[float, Field(ge=0, le=1000)] | None = Field(
        default=None,
        description="Pressure at source outlet in bar (OPTIONAL). Will be estimated if missing.",
    )

    pressure_outlet_bar: Annotated[float, Field(ge=0, le=1000)] | None = Field(
        default=None,
        description="Pressure at target inlet in bar (OPTIONAL). Will be estimated if missing.",
    )

    pressure_drop_bar: float | None = Field(
        default=None,
        description="Pressure drop in bar (OPTIONAL). Auto-computed from inlet/outlet.",
    )

    flow_rate_lpm: Annotated[float, Field(ge=0, le=1000)] | None = Field(
        default=None,
        description="Flow rate in L/min (OPTIONAL). Will be estimated if missing.",
    )

    temperature_c: Annotated[float, Field(ge=-20, le=150)] | None = Field(
        default=None,
        description="Fluid temperature in °C (OPTIONAL). Will be estimated if missing.",
    )

    vibration_g: Annotated[float, Field(ge=0, le=50)] | None = Field(
        default=None,
        description="Vibration in g (OPTIONAL). Will use default if missing.",
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
    def compute_pressure_drop(cls, v: float | None, info: ValidationInfo) -> float | None:
        """Auto-compute pressure drop if inlet/outlet available."""
        if v is not None:
            return v

        # Only compute if BOTH inlet and outlet provided
        if "pressure_inlet_bar" in info.data and "pressure_outlet_bar" in info.data:
            inlet = info.data["pressure_inlet_bar"]
            outlet = info.data["pressure_outlet_bar"]
            if inlet is not None and outlet is not None:
                return inlet - outlet

        return None  # Will be estimated by ValueSubstitutionEngine


class FlexibleComponentSensorReading(BaseModel):
    """Component sensor reading with ALL internal sensor fields optional.

    **Day 2 Addition**: Input for ValueSubstitutionEngine.

    Supports partial internal sensor coverage:
        - Pump with only RPM (no current/voltage)
        - Valve with only position (no other sensors)
        - Component with no internal sensors at all

    Use case:
        >>> # Pump with only RPM
        >>> reading = FlexibleComponentSensorReading(
        ...     component_id="pump_main",
        ...     rpm=1450,
        ...     timestamp=datetime.now(UTC)
        ... )

    Attributes:
        component_id: Component identifier (REQUIRED)
        timestamp: Measurement timestamp (REQUIRED)
        rpm: Rotational speed (optional)
        position_percent: Actuator position (optional)
        current_a: Electrical current (optional)
        voltage_v: Voltage (optional)
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        json_schema_extra={
            "example": {
                "component_id": "pump_main",
                "rpm": 1450,
                "timestamp": "2025-12-26T22:00:00Z",
            },
            "title": "Flexible Component Sensor Reading",
            "description": "Component reading with optional fields (for partial sensor coverage)",
        },
    )

    component_id: Annotated[
        str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description="Component identifier (REQUIRED).",
    )

    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp (REQUIRED, ISO 8601, timezone-aware).",
    )

    rpm: Annotated[float, Field(ge=0, le=10000)] | None = Field(
        default=None,
        description="Rotational speed in RPM (OPTIONAL).",
    )

    position_percent: Annotated[float, Field(ge=0, le=100)] | None = Field(
        default=None,
        description="Actuator position in % (OPTIONAL).",
    )

    current_a: Annotated[float, Field(ge=0, le=1000)] | None = Field(
        default=None,
        description="Electrical current in A (OPTIONAL).",
    )

    voltage_v: Annotated[float, Field(ge=0, le=1000)] | None = Field(
        default=None,
        description="Voltage in V (OPTIONAL).",
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class FlexibleInferenceRequest(BaseModel):
    """Level 0 API: Flexible inference request with partial sensor data.

    **Day 2 Addition**: Input for ValueSubstitutionEngine.

    Supports ANY sensor coverage:
        - Minimal: 1 pressure sensor on 1 edge
        - Partial: Some edges with pressure, some with flow
        - Full: All sensors on all edges

    **Workflow**:
        1. Client sends FlexibleInferenceRequest (partial data)
        2. ValueSubstitutionEngine.substitute_missing_values()
        3. Returns complete HybridInferenceRequest
        4. GNN inference proceeds normally

    **Advantages**:
        - ✅ Flexible sensor deployment (add sensors incrementally)
        - ✅ Cost savings (fewer sensors needed)
        - ✅ Backward compatible with existing systems
        - ✅ Physics-based estimation → higher accuracy than naive imputation

    Use case:
        >>> # Minimal: 1 pressure sensor
        >>> request = FlexibleInferenceRequest(
        ...     equipment_id="pump_system_01",
        ...     timestamp=datetime.now(UTC),
        ...     topology_id="standard_pump",
        ...     edge_readings={
        ...         "pump__valve": FlexibleEdgeSensorReading(
        ...             edge_id="pump__valve",
        ...             pressure_inlet_bar=150.2,
        ...             timestamp=datetime.now(UTC)
        ...         )
        ...     }
        ... )
        >>> # ValueSubstitutionEngine estimates:
        >>> # - pressure_outlet (via Darcy-Weisbach)
        >>> # - flow_rate (via conservation of mass)
        >>> # - temperature (via thermal model)

    Attributes:
        equipment_id: Unique equipment identifier (REQUIRED)
        timestamp: Sensor reading timestamp (REQUIRED)
        topology_id: Pre-configured topology identifier (REQUIRED)
        edge_readings: Partial edge sensor data (OPTIONAL)
        component_readings: Partial component sensor data (OPTIONAL)
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "equipment_id": "pump_system_01",
                "timestamp": "2025-12-26T22:00:00Z",
                "topology_id": "standard_pump",
                "edge_readings": {
                    "pump__valve": {
                        "edge_id": "pump__valve",
                        "pressure_inlet_bar": 150.2,
                        "timestamp": "2025-12-26T22:00:00Z",
                    }
                },
                "component_readings": {
                    "pump": {
                        "component_id": "pump",
                        "rpm": 1450,
                        "timestamp": "2025-12-26T22:00:00Z",
                    }
                },
            },
            "title": "Flexible Inference Request",
            "description": "Partial sensor data input for ValueSubstitutionEngine",
        },
    )

    equipment_id: Annotated[
        str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description="Unique equipment identifier (REQUIRED).",
    )

    timestamp: datetime = Field(
        ...,
        description="Sensor reading timestamp (REQUIRED, ISO 8601, timezone-aware).",
    )

    topology_id: Annotated[
        str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    ] = Field(
        ...,
        description="Pre-configured topology identifier (REQUIRED).",
    )

    edge_readings: dict[str, FlexibleEdgeSensorReading] = Field(
        default_factory=dict,
        description=(
            "Partial edge sensor data (OPTIONAL). Keys are edge_ids, values are readings. "
            "Missing fields will be estimated by ValueSubstitutionEngine."
        ),
    )

    component_readings: dict[str, FlexibleComponentSensorReading] = Field(
        default_factory=dict,
        description=(
            "Partial component sensor data (OPTIONAL). Keys are component_ids, values are readings. "
            "Missing fields will use defaults."
        ),
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_timestamp_timezone(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @model_validator(mode="after")
    def validate_at_least_one_reading(self) -> FlexibleInferenceRequest:
        """Ensure at least ONE sensor reading provided."""
        if not self.edge_readings and not self.component_readings:
            msg = (
                "At least ONE sensor reading required (edge_readings or component_readings). "
                "Cannot infer system state with zero measurements."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_edge_ids_match(self) -> FlexibleInferenceRequest:
        """Validate edge_ids in dict keys match FlexibleEdgeSensorReading.edge_id."""
        for edge_key, edge_reading in self.edge_readings.items():
            if edge_key != edge_reading.edge_id:
                msg = (
                    f"Edge key '{edge_key}' does not match FlexibleEdgeSensorReading.edge_id "
                    f"'{edge_reading.edge_id}'. They must be identical."
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_component_ids_match(self) -> FlexibleInferenceRequest:
        """Validate component_ids in dict keys match FlexibleComponentSensorReading.component_id."""
        for comp_key, comp_reading in self.component_readings.items():
            if comp_key != comp_reading.component_id:
                msg = (
                    f"Component key '{comp_key}' does not match FlexibleComponentSensorReading.component_id "
                    f"'{comp_reading.component_id}'. They must be identical."
                )
                raise ValueError(msg)
        return self


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


# ... (rest of the file remains unchanged - EdgeOverride, TimeWindow, etc.)
