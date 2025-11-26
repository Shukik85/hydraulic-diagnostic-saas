"""API response schemas.

Pydantic модели для исходящих API responses.

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from typing import Dict, List, Literal, Annotated
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


# ==================== ENUMS ====================

class ComponentStatus(str, Enum):
    """Статус компонента."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class AnomalyType(str, Enum):
    """Тип аномалии."""
    PRESSURE_DROP = "pressure_drop"
    OVERHEATING = "overheating"
    CAVITATION = "cavitation"
    LEAKAGE = "leakage"
    VIBRATION_ANOMALY = "vibration_anomaly"
    FLOW_RESTRICTION = "flow_restriction"
    CONTAMINATION = "contamination"
    SEAL_DEGRADATION = "seal_degradation"
    VALVE_STICTION = "valve_stiction"


# ==================== HEALTH CHECKS ====================

class HealthCheckResponse(BaseModel):
    """Basic health check response.
    
    Attributes:
        status: Service status
        version: Service version
        model_loaded: Whether model is loaded
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "2.0.0",
                "model_loaded": True
            }
        }
    )
    
    status: Literal["healthy", "unhealthy"] = Field(..., description="Health status")
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Model loaded status")


class ComponentHealth(BaseModel):
    """Component health status."""
    
    name: str = Field(..., description="Component name")
    status: ComponentStatus = Field(..., description="Health status")
    message: str | None = Field(default=None, description="Status message")
    last_check: datetime = Field(
        default_factory=datetime.now,
        description="Last health check timestamp"
    )


class DetailedHealthResponse(BaseModel):
    """Detailed health check response.
    
    Attributes:
        status: Overall health status
        version: Service version
        uptime_seconds: Service uptime
        components: Component health statuses
        system_metrics: System resource usage
        model_info: Model information
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "2.0.0",
                "uptime_seconds": 3600.0,
                "components": {
                    "inference_engine": {
                        "name": "inference_engine",
                        "status": "healthy",
                        "message": "Ready"
                    },
                    "model_manager": {
                        "name": "model_manager",
                        "status": "healthy",
                        "message": "Model loaded"
                    }
                },
                "system_metrics": {
                    "cpu_percent": 25.3,
                    "memory_percent": 45.7,
                    "disk_percent": 35.2
                },
                "model_info": {
                    "loaded": True,
                    "device": "cuda:0",
                    "parameters": 2500000
                }
            }
        }
    )
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall health status"
    )
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., ge=0, description="Service uptime")
    components: Dict[str, ComponentHealth] = Field(
        ...,
        description="Component health statuses"
    )
    system_metrics: Dict[str, float] = Field(
        ...,
        description="System resource metrics"
    )
    model_info: Dict[str, bool | str | int] = Field(
        ...,
        description="Model information"
    )


# ==================== PREDICTIONS ====================

class HealthPrediction(BaseModel):
    """Прогноз здоровья оборудования."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "score": 0.87,
                "status": "healthy"
            }
        }
    )
    
    score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Health score (0-1, higher=better)"
    )
    status: Literal["healthy", "warning", "critical"] = Field(
        default="healthy",
        description="Health status category"
    )


class DegradationPrediction(BaseModel):
    """Прогноз деградации."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rate": 0.12,
                "time_to_failure_hours": 733.3
            }
        }
    )
    
    rate: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Degradation rate (0-1, higher=faster degradation)"
    )
    time_to_failure_hours: float | None = Field(
        default=None,
        ge=0,
        description="Estimated time to failure (hours)"
    )


class Anomaly(BaseModel):
    """Обнаруженная аномалия."""
    
    anomaly_type: AnomalyType = Field(..., description="Anomaly type")
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Detection confidence"
    )
    severity: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Anomaly severity"
    )


class AnomalyPrediction(BaseModel):
    """Прогноз аномалий."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": {
                    "pressure_drop": 0.05,
                    "overheating": 0.03,
                    "cavitation": 0.02,
                    "leakage": 0.01,
                    "vibration_anomaly": 0.01,
                    "flow_restriction": 0.01,
                    "contamination": 0.01,
                    "seal_degradation": 0.01,
                    "valve_stiction": 0.01
                },
                "detected_anomalies": []
            }
        }
    )
    
    predictions: Dict[str, float] = Field(
        ...,
        description="Anomaly probabilities (9 types)"
    )
    detected_anomalies: List[Anomaly] = Field(
        default_factory=list,
        description="Detected anomalies (threshold exceeded)"
    )


class PredictionResponse(BaseModel):
    """Ответ на прогноз для одной единицы оборудования."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "equipment_id": "exc_001",
                "health": {"score": 0.87, "status": "healthy"},
                "degradation": {"rate": 0.12, "time_to_failure_hours": 733.3},
                "anomaly": {
                    "predictions": {
                        "pressure_drop": 0.05,
                        "overheating": 0.03
                    },
                    "detected_anomalies": []
                },
                "inference_time_ms": 45.3
            }
        }
    )
    
    equipment_id: str = Field(..., description="Equipment ID")
    health: HealthPrediction = Field(..., description="Health prediction")
    degradation: DegradationPrediction = Field(..., description="Degradation prediction")
    anomaly: AnomalyPrediction = Field(..., description="Anomaly predictions")
    inference_time_ms: float = Field(..., ge=0, description="Inference time (ms)")


class BatchPredictionResponse(BaseModel):
    """Ответ на batch прогноз."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [],
                "total_count": 10,
                "total_time_ms": 234.5
            }
        }
    )
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions"
    )
    total_count: int = Field(..., ge=0, description="Total predictions")
    total_time_ms: float = Field(..., ge=0, description="Total time (ms)")


# ==================== LEGACY RESPONSES ====================

class InferenceResponse(BaseModel):
    """Ответ на inference запрос."""
    
    model_config = ConfigDict(strict=True)
    
    equipment_id: str = Field(..., description="Equipment ID")
    health_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Health score"
    )
    degradation_rate: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Degradation rate"
    )
    anomalies: List[str] = Field(
        default_factory=list,
        description="Detected anomalies"
    )
    inference_time_ms: float = Field(..., ge=0, description="Inference time")


class TrainingResponse(BaseModel):
    """Ответ на training запрос."""
    
    model_config = ConfigDict(strict=True)
    
    model_path: str = Field(..., description="Path to trained model")
    final_train_loss: float = Field(..., description="Final training loss")
    final_val_loss: float = Field(..., description="Final validation loss")
    epochs_completed: int = Field(..., ge=0, description="Epochs completed")
    training_time_seconds: float = Field(..., ge=0, description="Training time")
