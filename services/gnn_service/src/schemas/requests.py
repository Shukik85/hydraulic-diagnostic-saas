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

**Day 3 Addition: HybridInferenceRequest + DiagnosticScope**
    - Complete inference request (output from ValueSubstitutionEngine)
    - HYBRID validation: strict for critical, flexible for optional
    - Flexible topology: diagnose full system or focus on subsystems
    - Supports partial sensor deployment with physics-based estimation

Python 3.14 Features:
    - Deferred annotations (PEP 649)
    - Union types with pipe operator (T | None)
    - Strict type checking

API Levels:
    Level 0 (Flexible): FlexibleInferenceRequest - partial sensor data
    Level 1 (Hybrid): HybridInferenceRequest - complete validated data (NEW!)
    Level 2 (Standard): InferenceRequest - historical analysis (future)
    Level 3 (Advanced): AdvancedInferenceRequest - expert overrides (future)
    Level 4 (Batch): BatchInferenceRequest - batch processing (future)

Examples:
    >>> # Day 2: Flexible partial sensor data
    >>> flex_request = FlexibleInferenceRequest(
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
    >>> 
    >>> # Day 3: Complete validated data for GNN
    >>> hybrid_request = engine.substitute_missing_values(flex_request)
    >>> # All fields filled, ready for inference!
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated

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

from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    # Phase 3.2: Edge-centric sensor schemas
    "EdgeSensorReading",
    "ComponentSensorReading",
    # Day 2: Flexible schemas for partial sensor data
    "FlexibleEdgeSensorReading",
    "FlexibleComponentSensorReading",
    "FlexibleInferenceRequest",
    # Day 3: Complete validated schemas + diagnostic scope
    "DiagnosticScope",
    "HybridInferenceRequest",
]


# ============================================================================
# DAY 3: DIAGNOSTIC SCOPE (Flexible Topology Support)
# ============================================================================


class DiagnosticScope(BaseModel):
    """Define which parts of topology to diagnose.

    **Day 3 Addition**: Support flexible sensor deployment and focused diagnostics.

    Allows two modes:
    1. **Full system diagnosis** (default: scope=None)
       - All topology edges required
       - Complete system health assessment

    2. **Focused subsystem diagnosis** (scope specified)
       - Only target_edges required to have sensor data
       - Other edges use nominal values for GNN context
       - Cost-effective incremental sensor deployment

    Use cases:
        >>> # Focus on pump subsystem only
        >>> scope = DiagnosticScope(
        ...     target_components=["pump_main", "valve_01"],
        ...     target_edges=["pump_main__valve_01", "valve_01__cylinder"],
        ...     include_context=True  # Use nominal for neighboring edges
        ... )
        >>>
        >>> # Full system (default)
        >>> scope = None  # All edges required

    Attributes:
        target_components: Components to diagnose (None = all components)
        target_edges: Edges requiring sensor data (None = all edges)
        include_context: Include neighboring edges with nominal values for GNN context
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        json_schema_extra={
            "example": {
                "target_components": ["pump_main", "valve_01"],
                "target_edges": ["pump_main__valve_01", "valve_01__cylinder"],
                "include_context": True,
            },
            "title": "Diagnostic Scope",
            "description": "Define focus area for flexible diagnostics",
        },
    )

    target_components: list[str] | None = Field(
        default=None,
        description=(
            "Components to diagnose. None = all components in topology. "
            "Use for focused diagnostics on specific subsystems."
        ),
        json_schema_extra={"examples": [["pump_main", "valve_01"], ["cylinder_left", "cylinder_right"]]},
    )

    target_edges: list[str] | None = Field(
        default=None,
        description=(
            "Edges requiring sensor data. None = all edges in topology. "
            "Only these edges need real/estimated measurements."
        ),
        json_schema_extra={
            "examples": [["pump_main__valve_01"], ["valve_01__cylinder", "cylinder__tank"]]
        },
    )

    include_context: bool = Field(
        default=True,
        description=(
            "Include neighboring edges with nominal values for GNN context. "
            "Recommended: True (GNN benefits from graph structure)."
        ),
    )


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

    **Day 3 Update**: Added diagnostic_scope for focused diagnostics.

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
        - ✅ NEW: Focused subsystem diagnostics (Day 3)

    Use case:
        >>> # Minimal: 1 pressure sensor, focus on pump subsystem
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
        ...     },
        ...     diagnostic_scope=DiagnosticScope(
        ...         target_edges=["pump__valve"],
        ...         include_context=True
        ...     )
        ... )
        >>> # ValueSubstitutionEngine estimates:
        >>> # - pressure_outlet (via Darcy-Weisbach)
        >>> # - flow_rate (via conservation of mass)
        >>> # - temperature (via thermal model)
        >>> # - Other edges use nominal for context

    Attributes:
        equipment_id: Unique equipment identifier (REQUIRED)
        timestamp: Sensor reading timestamp (REQUIRED)
        topology_id: Pre-configured topology identifier (REQUIRED)
        edge_readings: Partial edge sensor data (OPTIONAL)
        component_readings: Partial component sensor data (OPTIONAL)
        diagnostic_scope: Focus area for diagnostics (OPTIONAL, Day 3)
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
                "diagnostic_scope": {
                    "target_edges": ["pump__valve"],
                    "include_context": True,
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

    diagnostic_scope: DiagnosticScope | None = Field(
        default=None,
        description=(
            "Diagnostic focus area (OPTIONAL, Day 3). "
            "None = diagnose full topology. "
            "Specified = focus on target components/edges only."
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
# DAY 3: HYBRID INFERENCE REQUEST (Complete Validated Data for GNN)
# ============================================================================


class HybridInferenceRequest(BaseModel):
    """Level 1 API: Complete inference request with validated sensor data.

    **Day 3 Addition**: Output from ValueSubstitutionEngine, input for GNN inference.

    ALL critical fields are guaranteed filled (no None in critical fields).

    **HYBRID Validation Strategy**:
        TIER 1 (CRITICAL): ≥1 pressure per required edge (STRICT)
        TIER 2 (RECOMMENDED): flow_rate_lpm (WARN if missing)
        TIER 3 (OPTIONAL): temperature_c, vibration_g (INFO if missing)

    **Flexible Topology Support**:
        - diagnostic_scope=None: ALL topology edges required
        - diagnostic_scope specified: Only target_edges required
        - Other edges use nominal values for GNN context

    **Quality Guarantees**:
        ✅ At least ONE pressure measurement per required edge
        ✅ All critical fields filled (measured/estimated/nominal/default)
        ✅ No NaN values in tensor conversion
        ✅ Timestamps consistent across readings

    **Workflow**:
        FlexibleInferenceRequest (partial)
            ↓
        ValueSubstitutionEngine.substitute_missing_values()
            ↓
        HybridInferenceRequest (complete) ← YOU ARE HERE
            ↓
        GraphBuilder.build_from_hybrid_request()
            ↓
        GNN Inference

    Use case:
        >>> # Created by ValueSubstitutionEngine
        >>> hybrid = HybridInferenceRequest(
        ...     equipment_id="pump_system_01",
        ...     timestamp=datetime.now(UTC),
        ...     topology_id="standard_pump",
        ...     edge_readings={
        ...         "pump__valve": EdgeSensorReading(
        ...             edge_id="pump__valve",
        ...             pressure_inlet_bar=150.2,   # ✅ Measured
        ...             pressure_outlet_bar=148.1,  # ✅ Estimated
        ...             flow_rate_lpm=115.5,         # ✅ Estimated
        ...             temperature_c=65.3,          # ✅ Nominal
        ...             vibration_g=0.7,             # ✅ Default
        ...             timestamp=datetime.now(UTC)
        ...         )
        ...     }
        ... )
        >>> # ALL fields filled, ready for GNN! ✅

    Attributes:
        equipment_id: Unique equipment identifier (REQUIRED)
        timestamp: Sensor reading timestamp (REQUIRED)
        topology_id: Pre-configured topology identifier (REQUIRED)
        edge_readings: COMPLETE edge sensor data (ALL fields filled)
        component_readings: Component sensor data (OPTIONAL)
        diagnostic_scope: Diagnostic focus area (OPTIONAL)
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
                        "pressure_outlet_bar": 148.1,
                        "flow_rate_lpm": 115.5,
                        "temperature_c": 65.3,
                        "vibration_g": 0.7,
                        "timestamp": "2025-12-26T22:00:00Z",
                    }
                },
                "component_readings": {
                    "pump": {
                        "component_id": "pump",
                        "rpm": 1450,
                        "current_a": 25.5,
                        "voltage_v": 400,
                        "timestamp": "2025-12-26T22:00:00Z",
                    }
                },
                "diagnostic_scope": None,
            },
            "title": "Hybrid Inference Request",
            "description": "Complete validated sensor data for GNN inference",
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

    edge_readings: dict[str, EdgeSensorReading] = Field(
        ...,
        description=(
            "COMPLETE edge sensor data (REQUIRED). ALL critical fields must be filled. "
            "Created by ValueSubstitutionEngine with physics-based estimation."
        ),
    )

    component_readings: dict[str, ComponentSensorReading] = Field(
        default_factory=dict,
        description=(
            "Component sensor data (OPTIONAL). Internal sensors only. "
            "Empty dict if no internal sensors available."
        ),
    )

    diagnostic_scope: DiagnosticScope | None = Field(
        default=None,
        description=(
            "Diagnostic focus area (OPTIONAL). "
            "None = full topology diagnosis. "
            "Specified = focused subsystem diagnosis."
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
    def validate_at_least_one_edge(self) -> HybridInferenceRequest:
        """Ensure at least ONE edge reading provided."""
        if not self.edge_readings:
            msg = (
                "At least ONE edge reading required in edge_readings. "
                "Cannot perform GNN inference with zero edges."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_edge_data_quality(self) -> HybridInferenceRequest:
        """Validate edge data quality with HYBRID approach.

        TIER 1 (CRITICAL): At least ONE pressure per required edge (FAIL if missing)
        TIER 2 (RECOMMENDED): flow_rate_lpm (WARN if missing)
        TIER 3 (OPTIONAL): temperature_c, vibration_g (INFO if missing)

        Required edges determined by diagnostic_scope:
        - scope=None: ALL edges required
        - scope.target_edges: Only target edges required
        """
        # Determine required edges
        if self.diagnostic_scope and self.diagnostic_scope.target_edges:
            required_edges = set(self.diagnostic_scope.target_edges)
            logger.info(
                f"Focused diagnostics: validating {len(required_edges)} target edges "
                f"(out of {len(self.edge_readings)} total edges)"
            )
        else:
            # Full topology: all edges required
            required_edges = set(self.edge_readings.keys())
            logger.info(f"Full system diagnostics: validating all {len(required_edges)} edges")

        # Validate each required edge
        for edge_id in required_edges:
            if edge_id not in self.edge_readings:
                msg = (
                    f"Required edge '{edge_id}' missing from edge_readings. "
                    f"Diagnostic scope requires this edge."
                )
                raise ValueError(msg)

            reading = self.edge_readings[edge_id]

            # TIER 1 (CRITICAL): At least ONE pressure
            has_inlet = reading.pressure_inlet_bar is not None and reading.pressure_inlet_bar >= 0
            has_outlet = (
                reading.pressure_outlet_bar is not None and reading.pressure_outlet_bar >= 0
            )

            if not has_inlet and not has_outlet:
                msg = (
                    f"CRITICAL: Required edge '{edge_id}' has NO pressure data. "
                    f"At least ONE pressure (inlet OR outlet) required per edge. "
                    f"Current: inlet={reading.pressure_inlet_bar}, outlet={reading.pressure_outlet_bar}"
                )
                raise ValueError(msg)

            # TIER 2 (RECOMMENDED): Flow rate
            if reading.flow_rate_lpm is None or reading.flow_rate_lpm < 0:
                logger.warning(
                    f"Edge '{edge_id}': flow_rate_lpm missing or invalid ({reading.flow_rate_lpm}). "
                    "GNN may have reduced accuracy without flow data."
                )

            # TIER 3 (OPTIONAL): Temperature
            if reading.temperature_c is None:
                logger.info(
                    f"Edge '{edge_id}': temperature_c missing, likely using nominal value. "
                    "This is acceptable for basic diagnostics."
                )

            # TIER 3 (OPTIONAL): Vibration
            if reading.vibration_g is None:
                logger.info(
                    f"Edge '{edge_id}': vibration_g missing, likely using default heuristic. "
                    "This is acceptable for basic diagnostics."
                )

        # Warn about optional (context) edges
        optional_edges = set(self.edge_readings.keys()) - required_edges
        if optional_edges:
            logger.info(
                f"Context edges included: {len(optional_edges)} edges "
                f"(using nominal values for GNN context)"
            )

        return self

    @model_validator(mode="after")
    def validate_edge_ids_match(self) -> HybridInferenceRequest:
        """Validate edge_ids in dict keys match EdgeSensorReading.edge_id."""
        for edge_key, edge_reading in self.edge_readings.items():
            if edge_key != edge_reading.edge_id:
                msg = (
                    f"Edge key '{edge_key}' does not match EdgeSensorReading.edge_id "
                    f"'{edge_reading.edge_id}'. They must be identical."
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_component_ids_match(self) -> HybridInferenceRequest:
        """Validate component_ids in dict keys match ComponentSensorReading.component_id."""
        for comp_key, comp_reading in self.component_readings.items():
            if comp_key != comp_reading.component_id:
                msg = (
                    f"Component key '{comp_key}' does not match ComponentSensorReading.component_id "
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
