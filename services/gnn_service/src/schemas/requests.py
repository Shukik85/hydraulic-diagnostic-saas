"""API request schemas.

Pydantic модели для входящих API запросов.

Python 3.14 Features:
    - Deferred annotations
    - Union types с pipe operator
"""

from __future__ import annotations

from typing import List, Annotated, Literal
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict, field_validator


class TimeWindow(BaseModel):
    """Временное окно для запроса данных.
    
    Attributes:
        start_time: Начало окна (ISO 8601)
        end_time: Конец окна (ISO 8601)
        timezone: Временная зона (по умолчанию UTC)
    
    Examples:
        >>> window = TimeWindow(
        ...     start_time=datetime(2025, 11, 1, 0, 0, 0),
        ...     end_time=datetime(2025, 11, 21, 0, 0, 0)
        ... )
        >>> window.duration_hours  # 480.0
    """
    
    model_config = ConfigDict(
        strict=True,
        json_schema_extra={
            "example": {
                "start_time": "2025-11-01T00:00:00Z",
                "end_time": "2025-11-21T00:00:00Z",
                "timezone": "UTC"
            }
        }
    )
    
    start_time: datetime = Field(
        ...,
        description="Начало временного окна (ISO 8601)"
    )
    
    end_time: datetime = Field(
        ...,
        description="Конец временного окна (ISO 8601)"
    )
    
    timezone: str = Field(
        default="UTC",
        description="Временная зона (IANA timezone name)"
    )
    
    @field_validator("end_time")
    @classmethod
    def validate_end_after_start(cls, v: datetime, info) -> datetime:
        """Проверка корректности временного окна."""
        if "start_time" in info.data:
            if v <= info.data["start_time"]:
                raise ValueError("end_time must be after start_time")
            
            # Maximum window: 30 days
            delta = v - info.data["start_time"]
            if delta.days > 30:
                raise ValueError("Time window cannot exceed 30 days")
        
        return v
    
    @property
    def duration_seconds(self) -> float:
        """Длительность окна (секунды)."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def duration_hours(self) -> float:
        """Длительность окна (часы)."""
        return self.duration_seconds / 3600.0


class InferenceRequest(BaseModel):
    """Запрос на inference для одной единицы оборудования.
    
    Attributes:
        equipment_id: ID оборудования для диагностики
        time_window: Временное окно для анализа
        include_attention_weights: Вернуть attention weights (для debug)
        include_recommendations: Вернуть рекомендации по обслуживанию
        confidence_threshold: Минимальная confidence для включения в результат
        custom_parameters: Дополнительные параметры
    
    Examples:
        >>> request = InferenceRequest(
        ...     equipment_id="excavator_001",
        ...     time_window=TimeWindow(
        ...         start_time=datetime(2025, 11, 1),
        ...         end_time=datetime(2025, 11, 21)
        ...     ),
        ...     include_recommendations=True
        ... )
    """
    
    model_config = ConfigDict(
        strict=True,
        json_schema_extra={
            "example": {
                "equipment_id": "excavator_001",
                "time_window": {
                    "start_time": "2025-11-01T00:00:00Z",
                    "end_time": "2025-11-21T00:00:00Z"
                },
                "include_attention_weights": False,
                "include_recommendations": True,
                "confidence_threshold": 0.7
            }
        }
    )
    
    equipment_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Уникальный ID оборудования"
    )
    
    time_window: TimeWindow = Field(
        ...,
        description="Временное окно для анализа данных сенсоров"
    )
    
    include_attention_weights: bool = Field(
        default=False,
        description="Вернуть attention weights для визуализации (увеличивает размер ответа)"
    )
    
    include_recommendations: bool = Field(
        default=True,
        description="Вернуть рекомендации по обслуживанию"
    )
    
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.7,
        description="Минимальная confidence для включения результатов"
    )
    
    custom_parameters: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Дополнительные параметры для специфических use cases"
    )


class BatchInferenceRequest(BaseModel):
    """Batch inference запрос для нескольких единиц оборудования.
    
    Attributes:
        requests: Список индивидуальных inference запросов
        priority: Приоритет batch (high обрабатывается первым)
        max_parallel: Максимальное количество параллельных inference
    
    Examples:
        >>> batch_request = BatchInferenceRequest(
        ...     requests=[
        ...         InferenceRequest(equipment_id="exc_001", ...),
        ...         InferenceRequest(equipment_id="exc_002", ...)
        ...     ],
        ...     priority="normal"
        ... )
    """
    
    model_config = ConfigDict(
        strict=True,
        json_schema_extra={
            "example": {
                "requests": [
                    {
                        "equipment_id": "excavator_001",
                        "time_window": {
                            "start_time": "2025-11-01T00:00:00Z",
                            "end_time": "2025-11-21T00:00:00Z"
                        }
                    },
                    {
                        "equipment_id": "excavator_002",
                        "time_window": {
                            "start_time": "2025-11-01T00:00:00Z",
                            "end_time": "2025-11-21T00:00:00Z"
                        }
                    }
                ],
                "priority": "normal",
                "max_parallel": 4
            }
        }
    )
    
    requests: List[InferenceRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Список inference запросов"
    )
    
    priority: Literal["low", "normal", "high", "critical"] = Field(
        default="normal",
        description="Приоритет обработки batch"
    )
    
    max_parallel: Annotated[int, Field(ge=1, le=10)] = Field(
        default=4,
        description="Максимальное количество параллельных inference"
    )
    
    @field_validator("requests")
    @classmethod
    def validate_unique_equipment_ids(cls, v: List[InferenceRequest]) -> List[InferenceRequest]:
        """Проверка уникальности equipment_id в batch."""
        equipment_ids = [req.equipment_id for req in v]
        if len(equipment_ids) != len(set(equipment_ids)):
            raise ValueError("equipment_id must be unique within batch")
        return v


class TrainingRequest(BaseModel):
    """Запрос на обучение/переобучение модели.
    
    Attributes:
        dataset_path: Путь к dataset
        model_name: Имя модели для сохранения
        config_override: Override для SystemConfig параметров
        use_pretrained: Использовать ли pretrained weights
        pretrained_model_path: Путь к pretrained модели
    """
    
    model_config = ConfigDict(strict=True)
    
    dataset_path: str = Field(
        ...,
        min_length=1,
        description="Путь к preprocessed dataset"
    )
    
    model_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Имя модели для сохранения checkpoint"
    )
    
    config_override: dict[str, int | float | bool | str] = Field(
        default_factory=dict,
        description="Override параметров из SystemConfig"
    )
    
    use_pretrained: bool = Field(
        default=False,
        description="Начать с pretrained weights"
    )
    
    pretrained_model_path: str | None = Field(
        default=None,
        description="Путь к pretrained модели (если use_pretrained=True)"
    )
    
    @field_validator("pretrained_model_path")
    @classmethod
    def validate_pretrained_path(cls, v: str | None, info) -> str | None:
        """Проверка наличия пути при use_pretrained=True."""
        if info.data.get("use_pretrained") and not v:
            raise ValueError("pretrained_model_path required when use_pretrained=True")
        return v
