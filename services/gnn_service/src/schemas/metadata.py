"""Equipment metadata schemas.

Метаданные оборудования, конфигурации сенсоров и системные параметры.

Python 3.14 Features:
    - Deferred annotations для forward references
    - Union types с pipe operator
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class TimeWindow(BaseModel):
    """Time window for data queries.
    
    Represents a time range [start_time, end_time].
    
    Attributes:
        start_time: Start of time window
        end_time: End of time window
    
    Examples:
        >>> window = TimeWindow(
        ...     start_time=datetime(2025, 11, 1),
        ...     end_time=datetime(2025, 11, 21)
        ... )
        >>> window.duration_minutes
        28800.0
    """

    model_config = ConfigDict(frozen=False)

    start_time: datetime = Field(
        ...,
        description="Start of time window (inclusive)"
    )

    end_time: datetime = Field(
        ...,
        description="End of time window (inclusive)"
    )

    @field_validator("end_time")
    @classmethod
    def validate_end_after_start(cls, v: datetime, info) -> datetime:
        """Validate end_time > start_time."""
        if "start_time" in info.data:
            if v <= info.data["start_time"]:
                raise ValueError("end_time must be greater than start_time")
        return v

    @computed_field
    @property
    def duration_minutes(self) -> float:
        """Duration in minutes."""
        delta = self.end_time - self.start_time
        return delta.total_seconds() / 60.0


class SensorType(str, Enum):
    """Типы сенсоров гидравлической системы."""

    # Давление
    PRESSURE = "pressure"
    PRESSURE_DIFFERENTIAL = "pressure_differential"

    # Температура
    TEMPERATURE = "temperature"
    TEMPERATURE_FLUID = "temperature_fluid"

    # Расход
    FLOW_RATE = "flow_rate"
    FLOW_VELOCITY = "flow_velocity"

    # Механика
    VIBRATION = "vibration"
    POSITION = "position"
    SPEED = "speed"
    FORCE = "force"

    # Качество жидкости
    CONTAMINATION = "contamination"
    VISCOSITY = "viscosity"
    HUMIDITY = "humidity"

    # Акустика
    ACOUSTIC_EMISSION = "acoustic_emission"
    ULTRASONIC = "ultrasonic"


class SensorConfig(BaseModel):
    """Конфигурация отдельного сенсора.
    
    Attributes:
        sensor_id: Уникальный ID сенсора
        sensor_type: Тип сенсора
        component_id: ID компонента, к которому подключен
        unit: Единица измерения
        sampling_rate_hz: Частота дискретизации (Гц)
        accuracy_percent: Точность измерения (%)
        range_min: Минимальное значение диапазона
        range_max: Максимальное значение диапазона
        calibration_date: Дата последней калибровки
        is_active: Активен ли сенсор
    
    Examples:
        >>> sensor = SensorConfig(
        ...     sensor_id="pressure_pump_out",
        ...     sensor_type=SensorType.PRESSURE,
        ...     component_id="pump_001",
        ...     unit="bar",
        ...     sampling_rate_hz=100.0,
        ...     accuracy_percent=0.5,
        ...     range_min=0.0,
        ...     range_max=400.0
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "sensor_id": "pressure_pump_out",
                "sensor_type": "pressure",
                "component_id": "pump_001",
                "unit": "bar",
                "sampling_rate_hz": 100.0,
                "accuracy_percent": 0.5,
                "range_min": 0.0,
                "range_max": 400.0,
                "calibration_date": "2024-06-15T10:00:00Z",
                "is_active": True
            }
        }
    )

    sensor_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Уникальный идентификатор сенсора"
    )

    sensor_type: SensorType = Field(
        ...,
        description="Тип сенсора"
    )

    component_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="ID компонента, к которому подключен сенсор"
    )

    unit: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Единица измерения (bar, °C, L/min, mm, etc.)"
    )

    sampling_rate_hz: Annotated[float, Field(gt=0, le=10000)] = Field(
        ...,
        description="Частота дискретизации (Гц)"
    )

    accuracy_percent: Annotated[float, Field(gt=0, le=100)] = Field(
        ...,
        description="Точность измерения (% от full scale)"
    )

    range_min: float = Field(
        ...,
        description="Минимальное значение измерительного диапазона"
    )

    range_max: float = Field(
        ...,
        description="Максимальное значение измерительного диапазона"
    )

    calibration_date: datetime | None = Field(
        default=None,
        description="Дата последней калибровки (ISO 8601)"
    )

    is_active: bool = Field(
        default=True,
        description="Активен ли сенсор в данный момент"
    )

    @field_validator("range_min", "range_max")
    @classmethod
    def validate_range(cls, v: float, info) -> float:
        """Валидация корректности диапазона."""
        if info.field_name == "range_max" and "range_min" in info.data:
            if v <= info.data["range_min"]:
                raise ValueError("range_max must be greater than range_min")
        return v

    @computed_field
    @property
    def measurement_range(self) -> float:
        """Ширина измерительного диапазона."""
        return self.range_max - self.range_min

    @computed_field
    @property
    def absolute_accuracy(self) -> float:
        """Абсолютная точность в единицах измерения."""
        return (self.accuracy_percent / 100.0) * self.measurement_range


class EquipmentMetadata(BaseModel):
    """Метаданные оборудования.
    
    Полная информация об оборудовании, включая технические характеристики,
    историю обслуживания и текущие параметры работы.
    
    Attributes:
        equipment_id: Уникальный ID оборудования
        equipment_type: Тип (excavator, loader, crane, etc.)
        manufacturer: Производитель
        model: Модель
        serial_number: Серийный номер
        manufacture_year: Год выпуска
        installation_date: Дата ввода в эксплуатацию
        operating_hours: Наработка (моточасы)
        last_maintenance_date: Дата последнего ТО
        next_maintenance_due: Дата следующего ТО
        hydraulic_system_type: Тип гидросистемы (open/closed loop)
        fluid_type: Тип гидравлической жидкости
        fluid_viscosity_grade: Класс вязкости (ISO VG)
        tank_capacity_liters: Объём бака (литры)
        max_working_pressure_bar: Максимальное рабочее давление
        sensors: Конфигурации всех сенсоров
        location: Географическое расположение
        custom_metadata: Дополнительные поля
    """

    model_config = ConfigDict(
        strict=True,
        json_schema_extra={
            "example": {
                "equipment_id": "excavator_001",
                "equipment_type": "hydraulic_excavator",
                "manufacturer": "Caterpillar",
                "model": "320D",
                "serial_number": "CAT0320DEJRE02456",
                "manufacture_year": 2022,
                "installation_date": "2022-03-15T00:00:00Z",
                "operating_hours": 3450.5,
                "last_maintenance_date": "2024-10-01T00:00:00Z",
                "hydraulic_system_type": "closed_loop",
                "fluid_type": "ISO VG 46",
                "tank_capacity_liters": 180.0,
                "max_working_pressure_bar": 350
            }
        }
    )

    equipment_id: str = Field(..., min_length=1, max_length=100)
    equipment_type: str = Field(..., min_length=1, max_length=50)
    manufacturer: str = Field(..., min_length=1, max_length=100)
    model: str = Field(..., min_length=1, max_length=100)
    serial_number: str = Field(..., min_length=1, max_length=100)

    manufacture_year: Annotated[int, Field(ge=1950, le=2030)] = Field(
        ...,
        description="Год выпуска оборудования"
    )

    installation_date: datetime = Field(
        ...,
        description="Дата ввода в эксплуатацию"
    )

    operating_hours: Annotated[float, Field(ge=0, le=100000)] = Field(
        ...,
        description="Общая наработка (моточасы)"
    )

    last_maintenance_date: datetime | None = Field(
        default=None,
        description="Дата последнего технического обслуживания"
    )

    next_maintenance_due: datetime | None = Field(
        default=None,
        description="Дата планового ТО"
    )

    hydraulic_system_type: Literal["open_loop", "closed_loop", "hybrid"] = Field(
        default="closed_loop",
        description="Тип гидравлической системы"
    )

    fluid_type: str = Field(
        ...,
        description="Тип гидравлической жидкости (ISO VG grade)"
    )

    fluid_viscosity_grade: Annotated[int, Field(ge=10, le=1000)] = Field(
        default=46,
        description="Класс вязкости по ISO VG"
    )

    tank_capacity_liters: Annotated[float, Field(gt=0, le=5000)] = Field(
        ...,
        description="Объём гидробака (литры)"
    )

    max_working_pressure_bar: Annotated[int, Field(gt=0, le=1000)] = Field(
        ...,
        description="Максимальное рабочее давление (бар)"
    )

    sensors: list[SensorConfig] = Field(
        default_factory=list,
        description="Конфигурации всех установленных сенсоров"
    )

    location: dict[str, str | float] = Field(
        default_factory=dict,
        description="Географическое расположение (latitude, longitude, site_name)"
    )

    custom_metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Дополнительные пользовательские поля"
    )

    @computed_field
    @property
    def age_years(self) -> float:
        """Возраст оборудования (лет)."""
        now = datetime.now()
        delta = now - self.installation_date
        return delta.days / 365.25

    @computed_field
    @property
    def hours_since_maintenance(self) -> float | None:
        """Часы с последнего ТО."""
        if self.last_maintenance_date is None:
            return None
        now = datetime.now()
        delta = now - self.last_maintenance_date
        # Approximate (assumes continuous operation)
        return delta.days * 10  # ~10 hours/day average


class SystemConfig(BaseModel):
    """Конфигурация всей системы диагностики.
    
    Глобальные параметры для inference, training и monitoring.
    
    Attributes:
        inference_config: Параметры inference
        training_config: Параметры обучения
        monitoring_config: Параметры мониторинга
    """

    model_config = ConfigDict(strict=True)

    # Inference parameters
    inference_batch_size: Annotated[int, Field(ge=1, le=256)] = Field(
        default=32,
        description="Размер батча для inference"
    )

    inference_timeout_seconds: Annotated[int, Field(ge=1, le=300)] = Field(
        default=30,
        description="Таймаут inference запроса (секунды)"
    )

    use_gpu: bool = Field(
        default=True,
        description="Использовать ли GPU для inference"
    )

    gpu_device_id: Annotated[int, Field(ge=0, le=7)] = Field(
        default=0,
        description="ID GPU устройства"
    )

    use_amp: bool = Field(
        default=True,
        description="Использовать Automatic Mixed Precision"
    )

    use_compile: bool = Field(
        default=True,
        description="Использовать torch.compile (PyTorch 2.8)"
    )

    # Training parameters
    training_epochs: Annotated[int, Field(ge=1, le=1000)] = Field(
        default=100,
        description="Количество эпох обучения"
    )

    training_batch_size: Annotated[int, Field(ge=1, le=256)] = Field(
        default=16,
        description="Размер батча для обучения"
    )

    learning_rate: Annotated[float, Field(gt=0, le=1.0)] = Field(
        default=0.001,
        description="Learning rate"
    )

    use_float8_training: bool = Field(
        default=False,
        description="Использовать float8 training (PyTorch 2.8, требует A100/H100)"
    )

    use_distributed: bool = Field(
        default=False,
        description="Использовать distributed training (DDP)"
    )

    num_gpus: Annotated[int, Field(ge=1, le=8)] = Field(
        default=1,
        description="Количество GPU для distributed training"
    )

    # Model parameters
    hidden_channels: Annotated[int, Field(ge=32, le=512)] = Field(
        default=128,
        description="Размерность скрытых слоёв GNN"
    )

    num_attention_heads: Annotated[int, Field(ge=1, le=16)] = Field(
        default=8,
        description="Количество attention heads в GAT"
    )

    num_gat_layers: Annotated[int, Field(ge=1, le=10)] = Field(
        default=3,
        description="Количество GAT layers"
    )

    lstm_hidden: Annotated[int, Field(ge=64, le=1024)] = Field(
        default=256,
        description="Размерность LSTM hidden state"
    )

    lstm_layers: Annotated[int, Field(ge=1, le=5)] = Field(
        default=2,
        description="Количество LSTM layers"
    )

    dropout: Annotated[float, Field(ge=0.0, le=0.9)] = Field(
        default=0.3,
        description="Dropout rate"
    )

    # Data parameters
    sequence_length: Annotated[int, Field(ge=5, le=100)] = Field(
        default=10,
        description="Длина временной последовательности"
    )

    window_minutes: Annotated[int, Field(ge=5, le=1440)] = Field(
        default=60,
        description="Размер временного окна (минуты)"
    )

    # Health thresholds
    health_threshold_warning: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.7,
        description="Порог health score для warning"
    )

    health_threshold_critical: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5,
        description="Порог health score для critical"
    )

    degradation_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.1,
        description="Порог degradation rate для alert"
    )

    anomaly_confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.8,
        description="Минимальная confidence для anomaly alert"
    )

    # Monitoring
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Уровень логирования"
    )

    enable_metrics: bool = Field(
        default=True,
        description="Включить Prometheus metrics"
    )

    metrics_port: Annotated[int, Field(ge=1024, le=65535)] = Field(
        default=9090,
        description="Порт для Prometheus metrics"
    )

    @field_validator("health_threshold_critical")
    @classmethod
    def validate_critical_less_than_warning(cls, v: float, info) -> float:
        """Critical threshold должен быть ниже warning."""
        if "health_threshold_warning" in info.data:
            if v >= info.data["health_threshold_warning"]:
                raise ValueError(
                    "health_threshold_critical must be less than health_threshold_warning"
                )
        return v
