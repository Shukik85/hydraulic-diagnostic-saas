"""API request schemas for GNN inference service.

Pydantic v2 models for incoming API requests with comprehensive validation.
Supports progressive enhancement: from minimal to advanced inference APIs.

Python 3.14 Features:
    - Deferred annotations (PEP 649)
    - Union types with pipe operator (T | None)
    - Strict type checking

API Levels:
    Level 1 (Minimal): MinimalInferenceRequest - только essential данные
    Level 2 (Standard): InferenceRequest - полный контроль
    Level 3 (Advanced): AdvancedInferenceRequest - с overrides
    Level 4 (Batch): BatchInferenceRequest - batch processing

Examples:
    >>> # Level 1: Minimal API
    >>> request = MinimalInferenceRequest(
    ...     equipment_id="pump_system_01",
    ...     timestamp=datetime.now(),
    ...     topology_id="standard_pump",
    ...     sensor_readings={
    ...         "pump_1": ComponentSensorReading(
    ...             pressure_bar=150.2,
    ...             temperature_c=65.3
    ...         )
    ...     }
    ... )
    >>> 
    >>> # Level 4: Batch API
    >>> batch = BatchInferenceRequest(
    ...     requests=[request1, request2],
    ...     priority="high",
    ...     max_parallel=4
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
    # Core schemas
    "ComponentSensorReading",
    "EdgeOverride",
    "TimeWindow",
    # API levels
    "MinimalInferenceRequest",
    "AdvancedInferenceRequest",
    "InferenceRequest",
    "BatchInferenceRequest",
    # Legacy (deprecated)
    "PredictionRequest",
    "BatchPredictionRequest",
    "TrainingRequest",
]


# ============================================================================
# CORE BUILDING BLOCKS
# ============================================================================


class ComponentSensorReading(BaseModel):
    """Sensor readings for a single hydraulic component.

    Contains minimal required sensor data. All fields have validation
    for physical constraints (pressure, temperature, vibration ranges).

    Dynamic edge features are auto-computed from these readings by
    EdgeFeatureComputer when edge_overrides not provided.

    Attributes:
        pressure_bar: Pressure reading in bar [0, 1000]
        temperature_c: Temperature in °C [-20, 150]
        vibration_g: Vibration level in g [0, 50] (optional)
        flow_rate_lpm: Flow rate in L/min [0, 1000] (optional)
        rpm: RPM for rotating equipment [0, 10000] (optional)

    Examples:
        >>> # Pump reading
        >>> pump_reading = ComponentSensorReading(
        ...     pressure_bar=150.2,
        ...     temperature_c=65.3,
        ...     vibration_g=0.8,
        ...     rpm=1450
        ... )
        >>>
        >>> # Valve reading (no rotation)
        >>> valve_reading = ComponentSensorReading(
        ...     pressure_bar=148.1,
        ...     temperature_c=64.8
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,  # Immutable for safety
        json_schema_extra={
            "example": {
                "pressure_bar": 150.2,
                "temperature_c": 65.3,
                "vibration_g": 0.8,
                "flow_rate_lpm": 115.5,
                "rpm": 1450,
            },
            "title": "Component Sensor Reading",
            "description": "Sensor data from hydraulic component",
        },
    )

    pressure_bar: Annotated[float, Field(ge=0, le=1000)] = Field(
        ...,
        description="Pressure in bar. Range: [0, 1000]. Typical: 50-350 for mobile hydraulics.",
        json_schema_extra={"units": "bar", "typical_range": [50, 350]},
    )

    temperature_c: Annotated[float, Field(ge=-20, le=150)] = Field(
        ...,
        description="Temperature in °C. Range: [-20, 150]. Optimal: 40-80°C.",
        json_schema_extra={"units": "°C", "optimal_range": [40, 80]},
    )

    vibration_g: Annotated[float, Field(ge=0, le=50)] | None = Field(
        default=None,
        description="Vibration level in g-force. Range: [0, 50]. Normal: <2.0g. Optional.",
        json_schema_extra={"units": "g", "alert_threshold": 2.0},
    )

    flow_rate_lpm: Annotated[float, Field(ge=0, le=1000)] | None = Field(
        default=None,
        description="Flow rate in liters/min. Range: [0, 1000]. From flow meter if available. Optional.",
        json_schema_extra={"units": "L/min", "typical_range": [10, 300]},
    )

    rpm: Annotated[float, Field(ge=0, le=10000)] | None = Field(
        default=None,
        description="Rotational speed in RPM. Range: [0, 10000]. For pumps/motors only. Optional.",
        json_schema_extra={"units": "RPM", "typical_range": [1000, 3000]},
    )

    @field_validator("vibration_g")
    @classmethod
    def warn_high_vibration(cls, v: float | None) -> float | None:
        """Warn if vibration exceeds safe threshold."""
        if v is not None and v > 2.0:
            warnings.warn(
                f"High vibration detected: {v}g (normal <2.0g). "
                "Component may require immediate inspection.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @field_validator("temperature_c")
    @classmethod
    def warn_temperature_range(cls, v: float) -> float:
        """Warn if temperature outside optimal range."""
        if v < 40:
            warnings.warn(
                f"Low temperature: {v}°C (optimal 40-80°C). "
                "System may not be warmed up.",
                UserWarning,
                stacklevel=2,
            )
        elif v > 80:
            warnings.warn(
                f"High temperature: {v}°C (optimal 40-80°C). "
                "Check cooling system and fluid condition.",
                UserWarning,
                stacklevel=2,
            )
        return v


class EdgeOverride(BaseModel):
    """Optional edge feature overrides for expert users.

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
        >>>
        >>> # Invalid: window too large
        >>> window = TimeWindow(
        ...     start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ...     end_time=datetime(2025, 3, 1, tzinfo=timezone.utc)  # 59 days
        ... )
        ValidationError: Time window cannot exceed 30 days
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
            # Default to UTC if naive
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
# LEVEL 1 API: Minimal Inference Request
# ============================================================================


class MinimalInferenceRequest(BaseModel):
    """Level 1 API: Minimal inference request for real-time diagnosis.

    Simplest possible inference API. Requires only:
    1. Equipment ID (unique identifier)
    2. Timestamp (when sensors were read)
    3. Sensor readings per component (pressure + temperature minimum)
    4. Topology ID (pre-configured system layout)

    All dynamic edge features are auto-computed from sensor readings.
    No manual graph construction needed.

    Use case: Real-time monitoring dashboards, IoT edge devices.

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
        ...         ),
        ...         "filter_1": ComponentSensorReading(
        ...             pressure_bar=145.0,
        ...             temperature_c=66.0
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
                        "pressure_bar": 150.2,
                        "temperature_c": 65.3,
                        "vibration_g": 0.8,
                        "rpm": 1450,
                    },
                    "valve_1": {
                        "pressure_bar": 148.1,
                        "temperature_c": 64.8,
                    },
                    "filter_1": {
                        "pressure_bar": 145.0,
                        "temperature_c": 66.0,
                    },
                },
            },
            "title": "Minimal Inference Request",
            "description": "Real-time diagnosis request with auto-computed edge features",
        },
    )

    equipment_id: Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")] = Field(
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

    topology_id: Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")] = Field(
        ...,  # NOW REQUIRED (was default="default")
        description=(
            "Pre-configured topology identifier. Must exist in TopologyService. "
            "Defines component types, connections, and nominal parameters."
        ),
        json_schema_extra={
            "examples": ["standard_pump_system", "double_pump_v1", "excavator_boom_circuit"],
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
    def validate_unique_component_ids(cls, v: dict[str, ComponentSensorReading]) -> dict[str, ComponentSensorReading]:
        """Check component IDs are unique (redundant but explicit)."""
        # Dict keys are always unique, but explicit check for clarity
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
        ...             flow_rate_lpm=118.234,  # From precision flow meter
        ...             pressure_drop_bar=2.15   # From differential sensor
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
                    "pump_1": {"pressure_bar": 150.2, "temperature_c": 65.3}
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

    equipment_id: Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")] = Field(
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
                    },
                    {
                        "equipment_id": "excavator_002",
                        "time_window": {
                            "start_time": "2025-11-01T00:00:00Z",
                            "end_time": "2025-11-21T00:00:00Z",
                        },
                    },
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
    topology: Any | None = Field(default=None)  # Added for compatibility
    batch: list[Any] | None = Field(default=None)  # Added for compatibility

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

    model_name: Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")] = Field(
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
