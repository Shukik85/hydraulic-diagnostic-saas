"""API response schemas.

Pydantic модели для ответов GNN сервиса.

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from typing import List, Dict, Annotated, Literal
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, computed_field


class ComponentStatus(str, Enum):
    """Статус компонента."""
    
    HEALTHY = "healthy"  # Health score ≥ 0.7
    WARNING = "warning"  # 0.5 ≤ health score < 0.7
    CRITICAL = "critical"  # Health score < 0.5
    UNKNOWN = "unknown"  # Недостаточно данных


class AnomalyType(str, Enum):
    """Типы аномалий в гидравлической системе."""
    
    PRESSURE_DROP = "pressure_drop"  # Падение давления
    OVERHEATING = "overheating"  # Перегрев
    CAVITATION = "cavitation"  # Кавитация
    LEAKAGE = "leakage"  # Утечка
    VIBRATION_ANOMALY = "vibration_anomaly"  # Аномальная вибрация
    FLOW_RESTRICTION = "flow_restriction"  # Ограничение потока
    CONTAMINATION = "contamination"  # Загрязнение жидкости
    SEAL_DEGRADATION = "seal_degradation"  # Износ уплотнений
    VALVE_STICTION = "valve_stiction"  # Залипание клапана


# ==================== API Response Schemas ====================

class HealthCheckResponse(BaseModel):
    """Health check response.
    
    Attributes:
        status: Service status
        version: Service version
        model_loaded: Model loaded flag
    
    Examples:
        >>> response = HealthCheckResponse(
        ...     status="healthy",
        ...     version="2.0.0",
        ...     model_loaded=True
        ... )
    """
    
    status: Literal["healthy", "unhealthy"] = Field(
        ...,
        description="Service health status"
    )
    version: str = Field(
        ...,
        description="Service version"
    )
    model_loaded: bool = Field(
        ...,
        description="Model loaded flag"
    )


class HealthPrediction(BaseModel):
    """Health prediction result.
    
    Attributes:
        score: Health score [0, 1]
        status: Categorical status
    
    Examples:
        >>> health = HealthPrediction(score=0.87)
        >>> print(health.status)  # 'healthy'
    """
    
    score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Health score: 1.0 = excellent, 0.0 = critical"
    )
    
    @computed_field
    @property
    def status(self) -> str:
        """Categorical status based on score."""
        if self.score >= 0.7:
            return "healthy"
        elif self.score >= 0.5:
            return "warning"
        else:
            return "critical"


class DegradationPrediction(BaseModel):
    """Degradation prediction result.
    
    Attributes:
        rate: Degradation rate [0, 1]
        time_to_failure_hours: Estimated time to failure (hours)
    
    Examples:
        >>> degradation = DegradationPrediction(rate=0.12)
        >>> print(degradation.time_to_failure_hours)
    """
    
    rate: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Degradation rate: 0.0 = stable, 1.0 = rapid degradation"
    )
    
    @computed_field
    @property
    def time_to_failure_hours(self) -> float | None:
        """Estimated time to failure (hours)."""
        if self.rate > 0.01:  # Significant degradation
            # Simple linear estimate: TTF = (1 - current_health) / degradation_rate
            # Assuming current health ~= 1 - rate for simplicity
            return ((1.0 - self.rate) / self.rate) * 100  # Scale factor
        return None  # Stable, no immediate concern


class AnomalyPrediction(BaseModel):
    """Anomaly prediction result.
    
    Attributes:
        predictions: Dict of anomaly_type -> probability
        detected_anomalies: List of detected anomaly types (prob > 0.5)
    
    Examples:
        >>> anomaly = AnomalyPrediction(
        ...     predictions={
        ...         "pressure_drop": 0.05,
        ...         "overheating": 0.03,
        ...         "cavitation": 0.02,
        ...         "leakage": 0.01,
        ...         "vibration_anomaly": 0.01,
        ...         "flow_restriction": 0.01,
        ...         "contamination": 0.01,
        ...         "seal_degradation": 0.01,
        ...         "valve_stiction": 0.01
        ...     }
        ... )
        >>> print(anomaly.detected_anomalies)  # []
    """
    
    predictions: Dict[str, float] = Field(
        ...,
        description="Anomaly type -> probability mapping"
    )
    
    @computed_field
    @property
    def detected_anomalies(self) -> list[str]:
        """List of detected anomaly types (prob > 0.5)."""
        return [
            anomaly_type
            for anomaly_type, prob in self.predictions.items()
            if prob > 0.5
        ]


class PredictionResponse(BaseModel):
    """Single equipment prediction response.
    
    Attributes:
        equipment_id: Equipment ID
        health: Health prediction
        degradation: Degradation prediction
        anomaly: Anomaly prediction
        inference_time_ms: Inference time (milliseconds)
    
    Examples:
        >>> response = PredictionResponse(
        ...     equipment_id="exc_001",
        ...     health=HealthPrediction(score=0.87),
        ...     degradation=DegradationPrediction(rate=0.12),
        ...     anomaly=AnomalyPrediction(predictions={...}),
        ...     inference_time_ms=45.3
        ... )
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "equipment_id": "exc_001",
                "health": {"score": 0.87, "status": "healthy"},
                "degradation": {"rate": 0.12, "time_to_failure_hours": 733.3},
                "anomaly": {
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
                },
                "inference_time_ms": 45.3
            }
        }
    )
    
    equipment_id: str = Field(
        ...,
        description="Equipment ID"
    )
    health: HealthPrediction = Field(
        ...,
        description="Health prediction"
    )
    degradation: DegradationPrediction = Field(
        ...,
        description="Degradation prediction"
    )
    anomaly: AnomalyPrediction = Field(
        ...,
        description="Anomaly predictions"
    )
    inference_time_ms: float = Field(
        ...,
        ge=0,
        description="Inference time (milliseconds)"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response.
    
    Attributes:
        predictions: List of predictions
        total_count: Total number of predictions
        total_time_ms: Total inference time (milliseconds)
    
    Examples:
        >>> response = BatchPredictionResponse(
        ...     predictions=[pred1, pred2, pred3],
        ...     total_count=3,
        ...     total_time_ms=123.4
        ... )
    """
    
    predictions: list[PredictionResponse] = Field(
        ...,
        description="List of predictions"
    )
    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of predictions"
    )
    total_time_ms: float = Field(
        ...,
        ge=0,
        description="Total inference time (milliseconds)"
    )


# ==================== Legacy Schemas (keep for backward compatibility) ====================

class ComponentHealth(BaseModel):
    """Оценка здоровья отдельного компонента.
    
    Attributes:
        component_id: ID компонента
        component_type: Тип компонента
        health_score: Оценка здоровья [0, 1], 1 = отличное
        degradation_rate: Скорость деградации [0, 1]
        confidence: Уверенность модели в prediction
        status: Категориальный статус
        last_updated: Timestamp обновления
        time_to_failure_hours: Прогноз времени до отказа (часы)
    
    Examples:
        >>> health = ComponentHealth(
        ...     component_id="pump_001",
        ...     component_type="piston_pump",
        ...     health_score=0.87,
        ...     degradation_rate=0.02,
        ...     confidence=0.94
        ... )
    """
    
    model_config = ConfigDict(
        strict=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "component_id": "pump_main_001",
                "component_type": "piston_pump",
                "health_score": 0.87,
                "degradation_rate": 0.02,
                "confidence": 0.94,
                "status": "healthy",
                "last_updated": "2025-11-21T20:00:00Z",
                "time_to_failure_hours": 2400.0
            }
        }
    )
    
    component_id: str = Field(..., description="ID компонента")
    component_type: str = Field(..., description="Тип компонента")
    
    health_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Оценка здоровья: 1.0 = отличное, 0.0 = критическое"
    )
    
    degradation_rate: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Скорость деградации: 0.0 = стабильно, 1.0 = быстрая деградация"
    )
    
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Уверенность модели в prediction [0, 1]"
    )
    
    status: ComponentStatus = Field(
        ...,
        description="Категориальный статус компонента"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp последнего обновления"
    )
    
    time_to_failure_hours: float | None = Field(
        default=None,
        ge=0,
        description="Прогноз времени до отказа (часы), None если не применимо"
    )
    
    @computed_field
    @property
    def requires_immediate_action(self) -> bool:
        """Требуется ли немедленное действие."""
        return (
            self.status == ComponentStatus.CRITICAL or
            (self.time_to_failure_hours is not None and self.time_to_failure_hours < 24)
        )


class Anomaly(BaseModel):
    """Обнаруженная аномалия в системе.
    
    Attributes:
        anomaly_id: Уникальный ID аномалии
        anomaly_type: Тип аномалии
        severity: Серьёзность (low/medium/high/critical)
        confidence: Уверенность в обнаружении
        affected_components: Список затронутых компонентов
        description: Человеко-читаемое описание
        detected_at: Timestamp обнаружения
        related_sensor_data: Релевантные данные сенсоров
    
    Examples:
        >>> anomaly = Anomaly(
        ...     anomaly_id="anom_001",
        ...     anomaly_type=AnomalyType.PRESSURE_DROP,
        ...     severity="medium",
        ...     confidence=0.89,
        ...     affected_components=["valve_001"],
        ...     description="Unusual pressure fluctuation detected"
        ... )
    """
    
    model_config = ConfigDict(
        strict=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "anomaly_id": "anom_20251121_001",
                "anomaly_type": "pressure_drop",
                "severity": "medium",
                "confidence": 0.89,
                "affected_components": ["valve_001", "cylinder_002"],
                "description": "Unusual pressure drop detected in valve_001 circuit",
                "detected_at": "2025-11-21T14:23:00Z",
                "related_sensor_data": {
                    "pressure_valve_001": 180.5,
                    "pressure_cylinder_002": 175.2,
                    "expected_pressure": 280.0
                }
            }
        }
    )
    
    anomaly_id: str = Field(
        ...,
        min_length=1,
        description="Уникальный ID аномалии"
    )
    
    anomaly_type: AnomalyType = Field(
        ...,
        description="Тип обнаруженной аномалии"
    )
    
    severity: Literal["low", "medium", "high", "critical"] = Field(
        ...,
        description="Серьёзность аномалии"
    )
    
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Уверенность модели в обнаружении аномалии"
    )
    
    affected_components: List[str] = Field(
        ...,
        min_length=1,
        description="Список ID затронутых компонентов"
    )
    
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Человеко-читаемое описание аномалии"
    )
    
    detected_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp обнаружения аномалии"
    )
    
    related_sensor_data: Dict[str, float] = Field(
        default_factory=dict,
        description="Релевантные показания сенсоров на момент обнаружения"
    )


class InferenceResponse(BaseModel):
    """Результат inference для оборудования.
    
    Полный ответ с оценками здоровья компонентов, обнаруженными аномалиями,
    рекомендациями и метаданными inference.
    
    Attributes:
        request_id: ID запроса (для трассировки)
        equipment_id: ID оборудования
        overall_health_score: Общая оценка здоровья системы
        component_health: Оценки для каждого компонента
        anomalies: Список обнаруженных аномалий
        recommendations: Рекомендации по обслуживанию
        attention_weights: Attention weights (optional, для debug)
        metadata: Метаданные inference (timing, model version, etc.)
    
    Examples:
        >>> response = InferenceResponse(
        ...     request_id="req_1732212000000",
        ...     equipment_id="excavator_001",
        ...     overall_health_score=0.85,
        ...     component_health=[
        ...         ComponentHealth(component_id="pump_001", health_score=0.92, ...)
        ...     ],
        ...     anomalies=[],
        ...     recommendations=["Schedule routine maintenance"]
        ... )
    """
    
    model_config = ConfigDict(
        strict=True,
        json_schema_extra={
            "example": {
                "request_id": "req_1732212000000",
                "equipment_id": "excavator_001",
                "overall_health_score": 0.85,
                "component_health": [
                    {
                        "component_id": "pump_001",
                        "component_type": "piston_pump",
                        "health_score": 0.92,
                        "degradation_rate": 0.02,
                        "confidence": 0.95,
                        "status": "healthy"
                    }
                ],
                "anomalies": [],
                "recommendations": ["System operating normally", "Schedule routine maintenance in 30 days"],
                "metadata": {
                    "inference_time_ms": 387.5,
                    "model_version": "2.0.0",
                    "timestamp": "2025-11-21T20:00:00Z",
                    "device": "cuda:0"
                }
            }
        }
    )
    
    request_id: str = Field(
        ...,
        description="Уникальный ID запроса для трассировки"
    )
    
    equipment_id: str = Field(
        ...,
        description="ID оборудования"
    )
    
    overall_health_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Общая оценка здоровья всей системы (weighted average)"
    )
    
    component_health: List[ComponentHealth] = Field(
        ...,
        description="Оценки здоровья для каждого компонента"
    )
    
    anomalies: List[Anomaly] = Field(
        default_factory=list,
        description="Список обнаруженных аномалий"
    )
    
    recommendations: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Рекомендации по обслуживанию и действиям"
    )
    
    attention_weights: Dict[str, List[float]] | None = Field(
        default=None,
        description="GAT attention weights (если запрошены)"
    )
    
    metadata: Dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Метаданные inference: timing, model version, device, etc."
    )
    
    @computed_field
    @property
    def has_critical_issues(self) -> bool:
        """Есть ли критические проблемы."""
        critical_components = [
            c for c in self.component_health 
            if c.status == ComponentStatus.CRITICAL
        ]
        critical_anomalies = [
            a for a in self.anomalies 
            if a.severity == "critical"
        ]
        return len(critical_components) > 0 or len(critical_anomalies) > 0
    
    @computed_field
    @property
    def num_warnings(self) -> int:
        """Количество предупреждений."""
        warning_components = [
            c for c in self.component_health 
            if c.status == ComponentStatus.WARNING
        ]
        warning_anomalies = [
            a for a in self.anomalies 
            if a.severity in ["medium", "high"]
        ]
        return len(warning_components) + len(warning_anomalies)
    
    def get_critical_components(self) -> List[ComponentHealth]:
        """Получить список критических компонентов."""
        return [
            c for c in self.component_health
            if c.status == ComponentStatus.CRITICAL
        ]
    
    def get_high_severity_anomalies(self) -> List[Anomaly]:
        """Получить аномалии высокой серьёзности."""
        return [
            a for a in self.anomalies
            if a.severity in ["high", "critical"]
        ]


class TrainingResponse(BaseModel):
    """Результат обучения модели.
    
    Attributes:
        training_id: ID training job
        model_name: Имя обученной модели
        status: Статус обучения
        epochs_completed: Количество завершённых эпох
        best_validation_loss: Лучший validation loss
        best_epoch: Эпоха с лучшим validation loss
        checkpoint_path: Путь к сохранённому checkpoint
        training_time_seconds: Время обучения (секунды)
        final_metrics: Финальные метрики
        error_message: Сообщение об ошибке (если status=failed)
    """
    
    model_config = ConfigDict(
        strict=True,
        json_schema_extra={
            "example": {
                "training_id": "train_20251121_001",
                "model_name": "gnn_v2_excavator",
                "status": "completed",
                "epochs_completed": 100,
                "best_validation_loss": 0.0234,
                "best_epoch": 87,
                "checkpoint_path": "models/checkpoints/gnn_v2_excavator_epoch87.ckpt",
                "training_time_seconds": 14325.6,
                "final_metrics": {
                    "health_mae": 0.045,
                    "degradation_mae": 0.038,
                    "anomaly_f1": 0.89
                }
            }
        }
    )
    
    training_id: str = Field(..., description="ID training job")
    model_name: str = Field(..., description="Имя обученной модели")
    
    status: Literal["running", "completed", "failed", "stopped"] = Field(
        ...,
        description="Статус обучения"
    )
    
    epochs_completed: Annotated[int, Field(ge=0)] = Field(
        ...,
        description="Количество завершённых эпох"
    )
    
    best_validation_loss: float | None = Field(
        default=None,
        description="Лучший validation loss"
    )
    
    best_epoch: int | None = Field(
        default=None,
        description="Эпоха с лучшим validation loss"
    )
    
    checkpoint_path: str | None = Field(
        default=None,
        description="Путь к сохранённому checkpoint"
    )
    
    training_time_seconds: Annotated[float, Field(ge=0)] = Field(
        ...,
        description="Время обучения (секунды)"
    )
    
    final_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Финальные метрики на validation set"
    )
    
    error_message: str | None = Field(
        default=None,
        description="Сообщение об ошибке (если training failed)"
    )
    
    @computed_field
    @property
    def training_time_hours(self) -> float:
        """Время обучения (часы)."""
        return self.training_time_seconds / 3600.0
