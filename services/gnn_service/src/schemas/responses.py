"""Response schemas for GNN service.

Pydantic v2 schemas for API responses.

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ComponentPrediction(BaseModel):
    """Component-level (per-node) predictions.
    
    Attributes:
        component_id: Unique component identifier
        component_type: Type of component (pump, valve, cylinder, etc.)
        health: Health score [0, 1] (1=perfect, 0=critical)
        anomalies: Anomaly probabilities for each type
    
    Examples:
        >>> comp = ComponentPrediction(
        ...     component_id="pump_01",
        ...     component_type="hydraulic_pump",
        ...     health=0.85,
        ...     anomalies={
        ...         "cavitation": 0.12,
        ...         "leakage": 0.05,
        ...         "overheating": 0.08
        ...     }
        ... )
    """

    model_config = ConfigDict(frozen=False)

    component_id: str = Field(
        ...,
        description="Unique component identifier",
        examples=["pump_01", "valve_23", "cylinder_07"]
    )

    component_type: str = Field(
        ...,
        description="Type of hydraulic component",
        examples=["pump", "valve", "cylinder", "filter", "motor"]
    )

    health: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Component health score (1.0=perfect, 0.0=critical)"
    )

    anomalies: dict[str, float] = Field(
        default_factory=dict,
        description="Anomaly probabilities per type"
    )


class GraphPrediction(BaseModel):
    """Graph-level (entire equipment) predictions.
    
    Attributes:
        health: Overall system health [0, 1]
        degradation: Degradation rate [0, 1]
        rul_hours: Remaining useful life in hours
        anomalies: System-level anomaly probabilities
    
    Examples:
        >>> graph = GraphPrediction(
        ...     health=0.78,
        ...     degradation=0.22,
        ...     rul_hours=450.5,
        ...     anomalies={"pressure_drop": 0.15, "contamination": 0.08}
        ... )
    """

    model_config = ConfigDict(frozen=False)

    health: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall system health score (1.0=perfect, 0.0=critical)"
    )

    degradation: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="System degradation rate (0.0=stable, 1.0=rapid degradation)"
    )

    rul_hours: float = Field(
        ...,
        ge=0.0,
        description="Remaining useful life in hours until predicted failure"
    )

    anomalies: dict[str, float] = Field(
        default_factory=dict,
        description="System-level anomaly probabilities per type"
    )


class PredictionResponse(BaseModel):
    """Multi-level prediction response.
    
    Nested structure with component-level and graph-level predictions.
    
    Attributes:
        request_id: Unique request identifier
        equipment_id: Equipment identifier
        timestamp: Prediction timestamp
        component: Component-level predictions
        graph: Graph-level predictions
        model_version: Model version used
        inference_time_ms: Inference latency
    
    Examples:
        >>> response = PredictionResponse(
        ...     request_id="req_123",
        ...     equipment_id="excavator_42",
        ...     timestamp=datetime.now(),
        ...     component=[
        ...         ComponentPrediction(
        ...             component_id="pump_01",
        ...             component_type="pump",
        ...             health=0.85,
        ...             anomalies={"cavitation": 0.12}
        ...         )
        ...     ],
        ...     graph=GraphPrediction(
        ...         health=0.78,
        ...         degradation=0.22,
        ...         rul_hours=450.5,
        ...         anomalies={"pressure_drop": 0.15}
        ...     ),
        ...     model_version="v2.0.0",
        ...     inference_time_ms=120.5
        ... )
    """

    model_config = ConfigDict(frozen=False)

    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )

    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp (UTC)"
    )

    component: list[ComponentPrediction] = Field(
        default_factory=list,
        description="Component-level predictions (per-node)"
    )

    graph: GraphPrediction = Field(
        ...,
        description="Graph-level predictions (entire equipment)"
    )

    model_version: str = Field(
        ...,
        description="Model version used for inference",
        examples=["v2.0.0", "v2.1.0"]
    )

    inference_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference latency in milliseconds"
    )

    # Optional metadata
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall prediction confidence"
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings or notes about the prediction"
    )


class ModelInfo(BaseModel):
    """Model information and metadata.
    
    Attributes:
        version: Model version
        architecture: Model architecture name
        num_parameters: Total trainable parameters
        input_features: Expected input feature dimension
        output_structure: Output structure description
        training_date: When model was trained
        metrics: Model performance metrics
    """

    model_config = ConfigDict(frozen=False)

    version: str = Field(
        ...,
        description="Model version (semantic versioning)",
        examples=["v2.0.0"]
    )

    architecture: str = Field(
        ...,
        description="Model architecture name",
        examples=["UniversalTemporalGNN"]
    )

    num_parameters: int = Field(
        ...,
        ge=0,
        description="Total trainable parameters"
    )

    input_features: int = Field(
        ...,
        ge=1,
        description="Expected input node feature dimension"
    )

    output_structure: str = Field(
        ...,
        description="Output structure description",
        examples=["multi-level: component + graph"]
    )

    training_date: datetime | None = Field(
        default=None,
        description="Model training completion date"
    )

    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Model performance metrics"
    )


class HealthResponse(BaseModel):
    """Service health check response.
    
    Attributes:
        status: Service status (healthy/degraded/unhealthy)
        timestamp: Health check timestamp
        model_loaded: Whether model is loaded
        gpu_available: Whether GPU is available
        memory_usage_mb: Memory usage in MB
        uptime_seconds: Service uptime
    """

    model_config = ConfigDict(frozen=False)

    status: str = Field(
        ...,
        description="Service status",
        examples=["healthy", "degraded", "unhealthy"]
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp (UTC)"
    )

    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded and ready"
    )

    gpu_available: bool = Field(
        ...,
        description="Whether GPU is available for inference"
    )

    memory_usage_mb: float = Field(
        ...,
        ge=0.0,
        description="Current memory usage in MB"
    )

    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="Service uptime in seconds"
    )

    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional health check details"
    )
