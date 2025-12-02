"""API request schemas.

Pydantic модели для входящих API запросов.

Python 3.14 Features:
    - Deferred annotations
    - Union types с pipe operator
"""

from __future__ import annotations

from typing import List, Dict, Any, Annotated, Literal, Optional
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


# ============================================================================
# Level 1 API: Minimal Inference Request (Progressive Enhancement)
# ============================================================================

class ComponentSensorReading(BaseModel):
    """Minimal sensor readings per component.
    
    Only essential sensor data required. Dynamic edge features are
    auto-computed from these readings.
    
    Attributes:
        pressure_bar: Pressure in bar (required)
        temperature_c: Temperature in °C (required)
        vibration_g: Vibration in g (optional)
        flow_rate_lpm: Flow rate in L/min (optional, if flow meter available)
        rpm: RPM for pumps/motors (optional)
    
    Examples:
        >>> reading = ComponentSensorReading(
        ...     pressure_bar=150.2,
        ...     temperature_c=65.3,
        ...     vibration_g=0.8
        ... )
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pressure_bar": 150.2,
                "temperature_c": 65.3,
                "vibration_g": 0.8,
                "rpm": 1450
            }
        }
    )
    
    pressure_bar: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Pressure in bar"
    )
    
    temperature_c: float = Field(
        ...,
        ge=-20,
        le=150,
        description="Temperature in °C"
    )
    
    vibration_g: Optional[float] = Field(
        default=None,
        ge=0,
        le=50,
        description="Vibration level in g (optional)"
    )
    
    flow_rate_lpm: Optional[float] = Field(
        default=None,
        ge=0,
        le=1000,
        description="Flow rate in L/min (optional, from flow meter if available)"
    )
    
    rpm: Optional[float] = Field(
        default=None,
        ge=0,
        le=10000,
        description="RPM for pumps/motors (optional)"
    )


class EdgeOverride(BaseModel):
    """Optional edge feature overrides for advanced users.
    
    Allows expert users to provide measured values instead of auto-computed.
    Only provided fields override auto-computation.
    
    Attributes:
        flow_rate_lpm: Override flow rate (from flow meter)
        pressure_drop_bar: Override pressure drop
        temperature_delta_c: Override temperature delta
        vibration_level_g: Override vibration level
    
    Examples:
        >>> override = EdgeOverride(
        ...     flow_rate_lpm=118.2  # From flow meter
        ... )
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "flow_rate_lpm": 118.2
            }
        }
    )
    
    flow_rate_lpm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Override flow rate with measured value"
    )
    
    pressure_drop_bar: Optional[float] = Field(
        default=None,
        description="Override pressure drop"
    )
    
    temperature_delta_c: Optional[float] = Field(
        default=None,
        description="Override temperature delta"
    )
    
    vibration_level_g: Optional[float] = Field(
        default=None,
        ge=0,
        description="Override vibration level"
    )


class MinimalInferenceRequest(BaseModel):
    """Level 1 API: Minimal inference request.
    
    Simplest possible inference API. Only requires:
    1. Equipment ID
    2. Timestamp
    3. Sensor readings per component
    4. Topology ID (pre-configured)
    
    All dynamic edge features are auto-computed from sensor readings.
    
    Attributes:
        equipment_id: Unique equipment identifier
        timestamp: Timestamp of sensor readings
        sensor_readings: Dict of {component_id: ComponentSensorReading}
        topology_id: Pre-configured topology identifier
    
    Examples:
        >>> request = MinimalInferenceRequest(
        ...     equipment_id="pump_system_01",
        ...     timestamp=datetime.now(),
        ...     sensor_readings={
        ...         "pump_1": ComponentSensorReading(
        ...             pressure_bar=150.2,
        ...             temperature_c=65.3,
        ...             vibration_g=0.8
        ...         ),
        ...         "valve_1": ComponentSensorReading(
        ...             pressure_bar=148.1,
        ...             temperature_c=64.8
        ...         )
        ...     },
        ...     topology_id="standard_pump_system"
        ... )
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "equipment_id": "pump_system_01",
                "timestamp": "2025-12-03T00:00:00Z",
                "sensor_readings": {
                    "pump_1": {
                        "pressure_bar": 150.2,
                        "temperature_c": 65.3,
                        "vibration_g": 0.8,
                        "rpm": 1450
                    },
                    "valve_1": {
                        "pressure_bar": 148.1,
                        "temperature_c": 64.8
                    },
                    "filter_1": {
                        "pressure_bar": 145.0,
                        "temperature_c": 66.0
                    }
                },
                "topology_id": "standard_pump_system"
            }
        }
    )
    
    equipment_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique equipment identifier"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Timestamp of sensor readings (ISO 8601)"
    )
    
    sensor_readings: Dict[str, ComponentSensorReading] = Field(
        ...,
        min_length=1,
        description="Sensor readings per component {component_id: reading}"
    )
    
    topology_id: str = Field(
        default="default",
        description="Pre-configured topology identifier"
    )


class AdvancedInferenceRequest(MinimalInferenceRequest):
    """Level 3 API: Advanced inference request with overrides.
    
    Extends MinimalInferenceRequest with optional overrides for expert users.
    Allows providing measured values (e.g., from flow meters) instead of
    auto-computed values.
    
    Attributes:
        edge_overrides: Optional edge feature overrides
        custom_topology: Optional custom topology (for testing)
    
    Examples:
        >>> request = AdvancedInferenceRequest(
        ...     equipment_id="pump_system_01",
        ...     timestamp=datetime.now(),
        ...     sensor_readings={...},
        ...     topology_id="standard_pump_system",
        ...     edge_overrides={
        ...         "pump_1->valve_1": EdgeOverride(
        ...             flow_rate_lpm=118.2  # From flow meter
        ...         )
        ...     }
        ... )
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "equipment_id": "pump_system_01",
                "timestamp": "2025-12-03T00:00:00Z",
                "sensor_readings": {
                    "pump_1": {"pressure_bar": 150.2, "temperature_c": 65.3}
                },
                "topology_id": "standard_pump_system",
                "edge_overrides": {
                    "pump_1->valve_1": {
                        "flow_rate_lpm": 118.2
                    }
                }
            }
        }
    )
    
    edge_overrides: Optional[Dict[str, EdgeOverride]] = Field(
        default=None,
        description="Optional edge feature overrides {edge_id: override}"
    )
    
    custom_topology: Optional[dict] = Field(
        default=None,
        description="Optional custom topology (for testing/advanced use)"
    )


# ============================================================================
# Legacy API Request Schemas (Backward Compatibility)
# ============================================================================

class PredictionRequest(BaseModel):
    """Single equipment prediction request.
    
    Attributes:
        equipment_id: Equipment ID
        sensor_data: Sensor data (dict or pandas DataFrame compatible)
    
    Examples:
        >>> request = PredictionRequest(
        ...     equipment_id="exc_001",
        ...     sensor_data={
        ...         "pressure_pump_main": [100.0, 101.0, ...],
        ...         "temperature_pump_main": [60.0, 61.0, ...],
        ...         "vibration_pump_main": [2.5, 2.6, ...]
        ...     }
        ... )
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "equipment_id": "exc_001",
                "sensor_data": {
                    "pressure_pump_main": [100.0, 101.0, 102.0],
                    "temperature_pump_main": [60.0, 61.0, 62.0],
                    "vibration_pump_main": [2.5, 2.6, 2.7]
                }
            }
        }
    )
    
    equipment_id: str = Field(
        ...,
        min_length=1,
        description="Equipment ID"
    )
    
    sensor_data: Dict[str, List[float]] | Any = Field(
        ...,
        description="Sensor data (dict or pandas DataFrame)"
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request.
    
    Attributes:
        requests: List of prediction requests
    
    Examples:
        >>> batch_request = BatchPredictionRequest(
        ...     requests=[
        ...         PredictionRequest(equipment_id="exc_001", sensor_data={...}),
        ...         PredictionRequest(equipment_id="exc_002", sensor_data={...})
        ...     ]
        ... )
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requests": [
                    {
                        "equipment_id": "exc_001",
                        "sensor_data": {
                            "pressure_pump_main": [100.0, 101.0, 102.0],
                            "temperature_pump_main": [60.0, 61.0, 62.0]
                        }
                    },
                    {
                        "equipment_id": "exc_002",
                        "sensor_data": {
                            "pressure_pump_main": [105.0, 106.0, 107.0],
                            "temperature_pump_main": [62.0, 63.0, 64.0]
                        }
                    }
                ]
            }
        }
    )
    
    requests: List[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of prediction requests"
    )


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
        description="Дополнительные параметры для специфичных use cases"
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
