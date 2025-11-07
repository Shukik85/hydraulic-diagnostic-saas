from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator


class BaseResponse(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SensorReading(BaseModel):
    timestamp: str  # ISO format
    sensor_type: str
    value: float
    component_id: str | None = None
    unit: str | None = None


class SensorDataBatch(BaseModel):
    system_id: str  # changed from UUID to str for flexibility
    readings: list[SensorReading]
    metadata: dict[str, Any] | None = None


class FeatureVector(BaseModel):
    features: dict[str, float]
    feature_names: list[str]
    extraction_time_ms: float
    data_quality_score: float = Field(ge=0.0, le=1.0)


class ModelPrediction(BaseModel):
    ml_model: str
    version: str
    prediction_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time_ms: float = Field(ge=0.0)
    features_used: int = Field(ge=1)


class AnomalyPrediction(BaseModel):
    is_anomaly: bool
    anomaly_score: float = Field(ge=0.0, le=1.0)
    severity: str = Field(pattern=r"^(normal|warning|critical)$")
    confidence: float = Field(ge=0.0, le=1.0)
    affected_components: list[str] = Field(default_factory=list)
    anomaly_type: str | None = None


class TwoStageInfo(BaseModel):
    """Information about two-stage classification results"""

    stage1_score: float = Field(ge=0.0, le=1.0, description="Binary anomaly probability")
    stage2_confidence: float = Field(ge=0.0, le=1.0, description="Multiclass confidence")
    fault_class: int = Field(ge=0, description="Predicted fault class (0=normal, 1-3=fault types)")
    processing_time_ms: float = Field(ge=0.0, description="Two-stage processing time")


class PredictionRequest(BaseModel):
    sensor_data: SensorDataBatch
    use_cache: bool = True
    feature_groups: list[str] | None = None


class PredictionResponse(BaseResponse):
    system_id: str
    prediction: AnomalyPrediction
    ml_predictions: list[ModelPrediction]
    ensemble_score: float = Field(ge=0.0, le=1.0)
    total_processing_time_ms: float = Field(ge=0.0)
    features_extracted: int = Field(ge=1)
    cache_hit: bool = False
    # Adaptive thresholds fields
    threshold_used: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_source: str | None = None
    baseline_context: dict[str, Any] | None = None
    # Two-stage enhancement fields
    two_stage_info: TwoStageInfo | None = None


class ErrorResponse(BaseResponse):
    error: str
    error_code: str
    details: dict[str, Any] | None = None


class BatchPredictionRequest(BaseModel):
    requests: list[PredictionRequest]
    parallel_processing: bool = True


class BatchPredictionResponse(BaseResponse):
    results: list[PredictionResponse | ErrorResponse]
    total_processing_time_ms: float = Field(ge=0.0)
    successful_predictions: int = Field(ge=0)
    failed_predictions: int = Field(ge=0)


class FeatureExtractionRequest(BaseModel):
    sensor_data: SensorDataBatch
    feature_groups: list[str] | None = None


class FeatureExtractionResponse(BaseResponse):
    system_id: str
    feature_vector: FeatureVector


class ModelStatusResponse(BaseResponse):
    models: dict[str, Any]
    ensemble_ready: bool
    total_predictions: int
    cache_hit_rate: float


class ConfigResponse(BaseResponse):
    prediction_threshold: float
    ensemble_weights: list[float]
    cache_enabled: bool
    # Adaptive threshold config
    adaptive_thresholds_enabled: bool | None = None
    threshold_adaptation_rate: float | None = None
    target_fpr: float | None = None
    # Two-stage config
    two_stage_enabled: bool | None = None


class ConfigUpdateRequest(BaseModel):
    prediction_threshold: float | None = Field(None, ge=0.0, le=1.0)
    ensemble_weights: list[float] | None = None
    cache_predictions: bool | None = None
    # Adaptive threshold updates
    adaptive_thresholds_enabled: bool | None = None
    threshold_adaptation_rate: float | None = Field(None, ge=0.001, le=0.5)
    target_fpr: float | None = Field(None, ge=0.01, le=0.50)
    # Two-stage updates
    two_stage_enabled: bool | None = None

    @field_validator("ensemble_weights")
    @classmethod
    def validate_weights(cls, v):
        if v is not None and (len(v) != 4 or not all(0.0 <= w <= 1.0 for w in v)):
            raise ValueError("ensemble_weights must be 4 values between 0.0 and 1.0")
        return v


class MetricsResponse(BaseResponse):
    system_metrics: dict[str, Any]
    model_metrics: dict[str, Any]
    performance_metrics: dict[str, Any]
