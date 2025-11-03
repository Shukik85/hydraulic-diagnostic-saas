"""
Pydantic schemas for ML Inference API
Enterprise схемы для гидравлической диагностики
"""

from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


# Base Models
class BaseResponse(BaseModel):
    """Базовая схема ответа."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: Optional[str] = None
    
    class Config:
        # Убираем protected namespaces для полей вида ml_*
        protected_namespaces = ()


class ErrorResponse(BaseResponse):
    """Схема ошибки."""

    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Sensor Data Models
class SensorReading(BaseModel):
    """Одно показание с датчика."""

    timestamp: datetime
    sensor_type: str = Field(..., description="Тип датчика: pressure, temperature, flow, vibration")
    value: float = Field(..., description="Значение с датчика")
    unit: str = Field(..., description="Единица измерения")
    component_id: Optional[UUID] = Field(None, description="ID компонента")

    @validator("sensor_type")
    def validate_sensor_type(cls, v):
        allowed_types = ["pressure", "temperature", "flow", "vibration"]
        if v not in allowed_types:
            raise ValueError(f"sensor_type must be one of {allowed_types}")
        return v


class SensorDataBatch(BaseModel):
    """Пакет данных с датчиков."""

    system_id: UUID = Field(..., description="ID гидравлической системы")
    readings: List[SensorReading] = Field(..., min_items=1, max_items=1000)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Дополнительные метаданные")

    @validator("readings")
    def validate_readings_timespan(cls, v):
        if len(v) > 1:
            timestamps = [r.timestamp for r in v]
            time_span = max(timestamps) - min(timestamps)
            if time_span.total_seconds() > 3600:  # 1 час
                raise ValueError("Readings span cannot exceed 1 hour")
        return v


# Prediction Models
class PredictionRequest(BaseModel):
    """Запрос на предсказание."""

    sensor_data: SensorDataBatch
    prediction_type: str = Field(default="anomaly", description="Тип предсказания")
    use_cache: bool = Field(default=True, description="Использовать кеш")
    ml_models: Optional[List[str]] = Field(None, description="Конкретные модели")  # было model_names

    @validator("prediction_type")
    def validate_prediction_type(cls, v):
        allowed_types = ["anomaly", "classification", "regression"]
        if v not in allowed_types:
            raise ValueError(f"prediction_type must be one of {allowed_types}")
        return v
    
    class Config:
        protected_namespaces = ()


class ModelPrediction(BaseModel):
    """Предсказание одной модели."""

    ml_model: str          # было model_name
    version: str           # было model_version
    prediction_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float
    features_used: int
    
    class Config:
        protected_namespaces = ()


class AnomalyPrediction(BaseModel):
    """Предсказание аномалий."""

    is_anomaly: bool
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Оценка аномальности")
    severity: str = Field(..., description="Уровень серьезности: normal, warning, critical")
    confidence: float = Field(..., ge=0.0, le=1.0)
    affected_components: List[str] = Field(default_factory=list)
    anomaly_type: Optional[str] = Field(None, description="Тип аномалии")

    @validator("severity")
    def validate_severity(cls, v):
        allowed_severities = ["normal", "warning", "critical"]
        if v not in allowed_severities:
            raise ValueError(f"severity must be one of {allowed_severities}")
        return v


class PredictionResponse(BaseResponse):
    """Ответ с предсказанием."""

    system_id: UUID
    prediction: AnomalyPrediction
    ml_predictions: List[ModelPrediction]  # было model_predictions
    ensemble_score: float = Field(..., ge=0.0, le=1.0)
    total_processing_time_ms: float
    features_extracted: int
    cache_hit: bool = False


# Batch Processing
class BatchPredictionRequest(BaseModel):
    """Пакетный запрос на предсказание."""

    requests: List[PredictionRequest] = Field(..., min_items=1, max_items=32)
    parallel_processing: bool = Field(default=True)
    
    class Config:
        protected_namespaces = ()


class BatchPredictionResponse(BaseResponse):
    """Ответ на пакетный запрос."""

    results: List[Union[PredictionResponse, ErrorResponse]]
    total_processing_time_ms: float
    successful_predictions: int
    failed_predictions: int


# Model Management
class ModelInfo(BaseModel):
    """Информация о модели."""

    name: str
    version: str
    description: str
    accuracy: float = Field(..., ge=0.0, le=1.0)
    last_trained: datetime
    size_mb: float         # было model_size_mb
    features_count: int
    is_loaded: bool
    load_time_ms: Optional[float] = None
    
    class Config:
        protected_namespaces = ()


class ModelStatusResponse(BaseResponse):
    """Статус моделей."""

    models: List[ModelInfo]
    ensemble_weights: List[float]
    total_models_loaded: int
    memory_usage_mb: float


# Feature Engineering
class FeatureExtractionRequest(BaseModel):
    """Запрос на извлечение признаков."""

    sensor_data: SensorDataBatch
    feature_groups: List[str] = Field(
        default=["sensor_features", "derived_features", "window_features"],
        description="Группы признаков",
    )


class FeatureVector(BaseModel):
    """Вектор признаков."""

    features: Dict[str, float]
    feature_names: List[str]
    extraction_time_ms: float
    data_quality_score: float = Field(..., ge=0.0, le=1.0)


class FeatureExtractionResponse(BaseResponse):
    """Ответ с признаками."""

    system_id: UUID
    feature_vector: FeatureVector


# Health and Monitoring
class HealthStatus(BaseModel):
    """Статус здоровья сервиса."""

    healthy: bool
    status: str
    checks: Dict[str, bool]
    metrics: Dict[str, float]
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Метрики производительности."""

    predictions_total: int
    predictions_per_second: float
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate: float
    cache_hit_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float


# Configuration
class ConfigUpdateRequest(BaseModel):
    """Обновление конфигурации."""

    ensemble_weights: Optional[List[float]] = None
    prediction_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    cache_ttl_seconds: Optional[int] = Field(None, ge=0)

    @validator("ensemble_weights")
    def validate_ensemble_weights(cls, v):
        if v is not None:
            if len(v) != 3:
                raise ValueError("Ensemble weights must have exactly 3 values")
            if abs(sum(v) - 1.0) > 0.01:
                raise ValueError("Ensemble weights must sum to 1.0")
        return v


class ConfigResponse(BaseResponse):
    """Текущая конфигурация."""

    ensemble_weights: List[float]
    prediction_threshold: float
    max_inference_time_ms: int
    cache_enabled: bool
    cache_ttl_seconds: int
    models_loaded: List[str]
