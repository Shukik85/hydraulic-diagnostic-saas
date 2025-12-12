"""Pydantic schemas for API requests and responses.

Defines:
- Request models (input data)
- Response models (output data)
- Data validation
- OpenAPI documentation

Python 3.14 Features:
    - Deferred annotations
    - Pydantic v2 models
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class SensorData(BaseModel):
    """Sensor readings for prediction.
    
    Attributes:
        equipment_id: Equipment identifier (e.g., 'pump_001')
        sensor_readings: Time series data per sensor
        topology_id: Optional equipment topology ID
        lookback_minutes: Time window for analysis (1-60 minutes)
    
    Examples:
        >>> data = SensorData(
        ...     equipment_id="pump_001",
        ...     sensor_readings={
        ...         "PS1": [100.5, 101.2, 100.8, ...],  # Pressure
        ...         "TS1": [45.3, 45.5, 45.4, ...],     # Temperature
        ...         "FS1": [8.5, 8.6, 8.5, ...]         # Flow
        ...     },
        ...     lookback_minutes=10
        ... )
    """

    equipment_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Equipment identifier",
        example="pump_001"
    )

    sensor_readings: dict[str, list[float]] = Field(
        ...,
        description="Sensor time series data",
        example={
            "PS1": [100.5, 101.2, 100.8],
            "TS1": [45.3, 45.5, 45.4],
            "FS1": [8.5, 8.6, 8.5]
        }
    )

    topology_id: str | None = Field(
        None,
        description="Equipment topology identifier",
        example="pump_5sensor_v1"
    )

    lookback_minutes: int = Field(
        10,
        ge=1,
        le=60,
        description="Time window for analysis",
        example=10
    )

    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "pump_001",
                "sensor_readings": {
                    "PS1": [100.5, 101.2, 100.8],
                    "TS1": [45.3, 45.5, 45.4],
                    "FS1": [8.5, 8.6, 8.5]
                },
                "lookback_minutes": 10
            }
        }


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class ComponentPrediction(BaseModel):
    """Prediction for single hydraulic component.
    
    Attributes:
        component_name: Component identifier (cooler, valve, pump, accumulator)
        health_score: 0-1 health rating
        severity_grade: Qualitative severity (optimal, degraded, failure)
        confidence: Model confidence in prediction (0-1)
        contributing_sensors: Top sensors influencing prediction
    
    Examples:
        >>> pred = ComponentPrediction(
        ...     component_name="pump",
        ...     health_score=0.85,
        ...     severity_grade="optimal",
        ...     confidence=0.92,
        ...     contributing_sensors=["PS1", "FS1"]
        ... )
    """

    component_name: str = Field(
        ...,
        description="Component name",
        example="pump"
    )

    health_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Health score (0=failure, 1=optimal)",
        example=0.85
    )

    severity_grade: str = Field(
        ...,
        description="Severity grade",
        example="optimal"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence",
        example=0.92
    )

    contributing_sensors: list[str] = Field(
        default_factory=list,
        description="Top contributing sensors",
        example=["PS1", "FS1"]
    )


class DiagnosticResponse(BaseModel):
    """Complete diagnostic response for equipment.
    
    Attributes:
        equipment_id: Equipment identifier
        timestamp: Prediction timestamp
        overall_health: Overall equipment health (0-1)
        components: Per-component predictions
        recommendations: Maintenance recommendations
        model_version: Model version used
        inference_time_ms: Inference latency
    
    Examples:
        >>> response = DiagnosticResponse(
        ...     equipment_id="pump_001",
        ...     timestamp="2025-12-12T12:00:00Z",
        ...     overall_health=0.85,
        ...     components=[...],
        ...     recommendations=["✅ All components operating optimally"],
        ...     model_version="mock-v0.1.0",
        ...     inference_time_ms=52.34
        ... )
    """

    equipment_id: str = Field(
        ...,
        description="Equipment identifier",
        example="pump_001"
    )

    timestamp: str = Field(
        ...,
        description="Prediction timestamp (ISO 8601)",
        example="2025-12-12T12:00:00Z"
    )

    overall_health: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall health score",
        example=0.85
    )

    components: list[ComponentPrediction] = Field(
        ...,
        description="Per-component predictions",
        min_items=1
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description="Maintenance recommendations",
        example=["✅ All components operating optimally"]
    )

    model_version: str = Field(
        ...,
        description="Model version used",
        example="mock-v0.1.0"
    )

    inference_time_ms: float = Field(
        ...,
        ge=0,
        description="Inference time in milliseconds",
        example=52.34
    )


class HealthResponse(BaseModel):
    """Service health check response.
    
    Attributes:
        status: Service status (healthy, degraded, unhealthy)
        model_loaded: Whether inference model is loaded
        cuda_available: Whether CUDA/GPU is available
        version: Service version
    """

    status: str = Field(
        ...,
        description="Service status",
        example="healthy"
    )

    model_loaded: bool = Field(
        ...,
        description="Model loaded status",
        example=True
    )

    cuda_available: bool = Field(
        ...,
        description="CUDA availability",
        example=False
    )

    version: str = Field(
        ...,
        description="Service version",
        example="mock-v0.1.0"
    )
