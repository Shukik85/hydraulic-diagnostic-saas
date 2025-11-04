from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class BaseResponse(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SensorReading(BaseModel):
    timestamp: str  # ISO format
    sensor_type: str
    value: float
    component_id: Optional[str] = None
    unit: Optional[str] = None


class SensorDataBatch(BaseModel):
    system_id: str  # changed from UUID to str for flexibility
    readings: List[SensorReading]
    metadata: Optional[Dict[str, Any]] = None


class FeatureVector(BaseModel):
    features: Dict[str, float]
    feature_names: List[str]
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
    severity: str = Field(pattern=r'^(normal|warning|critical)$')
    confidence: float = Field(ge=0.0, le=1.0)
    affected_components: List[str] = Field(default_factory=list)
    anomaly_type: Optional[str] = None


class PredictionRequest(BaseModel):
    sensor_data: SensorDataBatch
    use_cache: bool = True
    feature_groups: Optional[List[str]] = None


class PredictionResponse(BaseResponse):
    system_id: str
    prediction: AnomalyPrediction
    ml_predictions: List[ModelPrediction]
    ensemble_score: float = Field(ge=0.0, le=1.0)
    total_processing_time_ms: float = Field(ge=0.0)
    features_extracted: int = Field(ge=1)
    cache_hit: bool = False
    # Новые поля для adaptive thresholds
    threshold_used: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    threshold_source: Optional[str] = None
    baseline_context: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseResponse):
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None


class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]
    parallel_processing: bool = True


class BatchPredictionResponse(BaseResponse):
    results: List[Union[PredictionResponse, ErrorResponse]]
    total_processing_time_ms: float = Field(ge=0.0)
    successful_predictions: int = Field(ge=0)
    failed_predictions: int = Field(ge=0)


class FeatureExtractionRequest(BaseModel):
    sensor_data: SensorDataBatch
    feature_groups: Optional[List[str]] = None


class FeatureExtractionResponse(BaseResponse):
    system_id: str
    feature_vector: FeatureVector


class ModelStatusResponse(BaseResponse):
    models: Dict[str, Any]
    ensemble_ready: bool
    total_predictions: int
    cache_hit_rate: float


class ConfigResponse(BaseResponse):
    prediction_threshold: float
    ensemble_weights: List[float]
    cache_enabled: bool
    # Adaptive threshold config
    adaptive_thresholds_enabled: Optional[bool] = None
    threshold_adaptation_rate: Optional[float] = None
    target_fpr: Optional[float] = None


class ConfigUpdateRequest(BaseModel):
    prediction_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    ensemble_weights: Optional[List[float]] = None
    cache_predictions: Optional[bool] = None
    # Adaptive threshold updates
    adaptive_thresholds_enabled: Optional[bool] = None
    threshold_adaptation_rate: Optional[float] = Field(None, ge=0.001, le=0.5)
    target_fpr: Optional[float] = Field(None, ge=0.01, le=0.50)

    @field_validator('ensemble_weights')
    @classmethod
    def validate_weights(cls, v):
        if v is not None and (len(v) != 4 or not all(0.0 <= w <= 1.0 for w in v)):
            raise ValueError('ensemble_weights must be 4 values between 0.0 and 1.0')
        return v


class MetricsResponse(BaseResponse):
    system_metrics: Dict[str, Any]
    model_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]