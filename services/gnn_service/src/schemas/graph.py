"""Graph topology schemas for hydraulic system GNN.

Представление гидравлических систем в виде графов с компонентами (nodes)
и соединениями (edges). Поддержка edge features для GATv2.

Python 3.14 Features:
    - Deferred annotations (PEP 649) для динамической типизации
    - Union types с pipe operator (str | int | float)
"""

from __future__ import annotations  # PEP 649: Deferred annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class ComponentType(str, Enum):
    """Типы компонентов гидравлической системы."""

    # Насосы
    HYDRAULIC_PUMP = "hydraulic_pump"
    GEAR_PUMP = "gear_pump"
    PISTON_PUMP = "piston_pump"
    VANE_PUMP = "vane_pump"
    PUMP = "pump"  # Generic alias

    # Клапаны
    HYDRAULIC_VALVE = "hydraulic_valve"
    DIRECTIONAL_VALVE = "directional_valve"
    PRESSURE_RELIEF_VALVE = "pressure_relief_valve"
    FLOW_CONTROL_VALVE = "flow_control_valve"
    CHECK_VALVE = "check_valve"
    VALVE = "valve"  # Generic alias

    # Приводы
    HYDRAULIC_CYLINDER = "hydraulic_cylinder"
    HYDRAULIC_MOTOR = "hydraulic_motor"
    ROTARY_ACTUATOR = "rotary_actuator"
    CYLINDER = "cylinder"  # Generic alias

    # Прочие
    ACCUMULATOR = "accumulator"
    FILTER = "filter"
    RESERVOIR = "reservoir"
    HEAT_EXCHANGER = "heat_exchanger"
    MANIFOLD = "manifold"


class EdgeMaterial(str, Enum):
    """Материалы соединений."""

    STEEL = "steel"
    RUBBER = "rubber"
    COMPOSITE = "composite"
    THERMOPLASTIC = "thermoplastic"


class EdgeType(str, Enum):
    """Типы соединений между компонентами."""

    HYDRAULIC_LINE = "hydraulic_line"  # Стандартная гидролиния
    HIGH_PRESSURE_HOSE = "high_pressure_hose"  # Высокое давление
    LOW_PRESSURE_RETURN = "low_pressure_return"  # Обратная линия
    PILOT_LINE = "pilot_line"  # Управляющая линия
    DRAIN_LINE = "drain_line"  # Дренажная линия
    MANIFOLD_CONNECTION = "manifold_connection"  # Через коллектор


class EdgeSpec(BaseModel):
    """Edge specification для GATv2 edge-conditioned attention.
    
    Содержит статические и динамические характеристики соединения между 
    компонентами. Динамические признаки (flow, pressure drop, etc.) могут быть
    заполнены в runtime или вычислены автоматически из sensor data.
    
    Static Attributes (from topology):
        source_id: ID исходного компонента
        target_id: ID целевого компонента
        edge_type: Тип соединения
        diameter_mm: Диаметр гидролинии (мм)
        length_m: Длина соединения (метры)
        pressure_rating_bar: Номинальное давление (бар)
        flow_direction: Направление потока
        has_quick_disconnect: Наличие быстроразъёмного соединения
        material: Материал (steel, rubber, composite)
    
    Dynamic Attributes (computed at inference):
        flow_rate_lpm: Real-time flow rate (L/min)
        pressure_drop_bar: Pressure drop across connection (bar)
        temperature_delta_c: Temperature difference (°C)
        vibration_level_g: Average vibration level (g)
        age_hours: Operating hours since installation
        last_maintenance_date: Date of last maintenance
    
    Examples:
        >>> # Static configuration (from topology)
        >>> edge = EdgeSpec(
        ...     source_id="pump_001",
        ...     target_id="valve_001",
        ...     edge_type=EdgeType.HIGH_PRESSURE_HOSE,
        ...     diameter_mm=16.0,
        ...     length_m=2.5,
        ...     pressure_rating_bar=350,
        ...     material=EdgeMaterial.STEEL
        ... )
        >>> 
        >>> # With dynamic features (at inference)
        >>> edge.flow_rate_lpm = 115.3
        >>> edge.pressure_drop_bar = 2.1
        >>> edge.temperature_delta_c = 1.5
    """

    model_config = ConfigDict(
        strict=True,
        frozen=False,  # Changed from True to allow dynamic field updates
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "source_id": "pump_001",
                "target_id": "valve_001",
                "edge_type": "high_pressure_hose",
                "diameter_mm": 16.0,
                "length_m": 2.5,
                "pressure_rating_bar": 350,
                "flow_direction": "bidirectional",
                "has_quick_disconnect": False,
                "material": "steel",
                # Dynamic fields (optional)
                "flow_rate_lpm": 115.3,
                "pressure_drop_bar": 2.1,
                "temperature_delta_c": 1.5,
                "vibration_level_g": 0.3,
                "age_hours": 12500.0,
                "last_maintenance_date": "2024-06-01"
            }
        }
    )

    # ========================================================================
    # STATIC FIELDS (from topology configuration)
    # ========================================================================

    source_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="ID исходного компонента"
    )

    target_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="ID целевого компонента"
    )

    edge_type: EdgeType = Field(
        ...,
        description="Тип соединения между компонентами"
    )

    diameter_mm: float = Field(
        ...,
        gt=0,
        le=500,
        description="Внутренний диаметр гидролинии (мм)"
    )

    length_m: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Длина соединения (метры)"
    )

    pressure_rating_bar: Annotated[float, Field(gt=0, le=1000)] = Field(
        ...,
        description="Номинальное рабочее давление (бар)"
    )

    flow_direction: Literal["unidirectional", "bidirectional"] = Field(
        default="unidirectional",
        description="Направление потока"
    )

    has_quick_disconnect: bool = Field(
        default=False,
        description="Наличие быстроразъёмного соединения"
    )

    material: EdgeMaterial = Field(
        default=EdgeMaterial.STEEL,
        description="Материал гидролинии"
    )

    # ========================================================================
    # DYNAMIC FIELDS (computed at inference time)
    # ========================================================================

    flow_rate_lpm: float | None = Field(
        default=None,
        ge=0,
        description="Real-time flow rate (L/min). Computed from sensors if not provided."
    )

    pressure_drop_bar: float | None = Field(
        default=None,
        description="Pressure drop across connection (bar). Computed from ΔP = P_source - P_target."
    )

    temperature_delta_c: float | None = Field(
        default=None,
        description="Temperature difference across connection (°C). Computed from ΔT = T_source - T_target."
    )

    vibration_level_g: float | None = Field(
        default=None,
        ge=0,
        description="Average vibration level at connection (g). Computed from adjacent sensors."
    )

    age_hours: float | None = Field(
        default=None,
        ge=0,
        description="Connection age in operating hours. Computed from install_date if available."
    )

    last_maintenance_date: date | None = Field(
        default=None,
        description="Date of last maintenance. Used to compute maintenance_score [0, 1]."
    )

    # ========================================================================
    # COMPUTED PROPERTIES (static)
    # ========================================================================

    @computed_field
    @property
    def cross_section_area_mm2(self) -> float:
        """Площадь поперечного сечения (мм²)."""
        return np.pi * (self.diameter_mm / 2) ** 2

    @computed_field
    @property
    def pressure_loss_coefficient(self) -> float:
        """Упрощённый коэффициент потерь давления.
        
        Зависит от длины, диаметра и материала.
        """
        material_factors = {
            "steel": 1.0,
            "rubber": 1.2,
            "composite": 1.1,
            "thermoplastic": 1.15
        }
        factor = material_factors.get(self.material, 1.0)
        return factor * self.length_m / (self.diameter_mm ** 4)

    # ========================================================================
    # DYNAMIC METHODS
    # ========================================================================

    def get_age_hours(self, current_time: datetime) -> float:
        """Get connection age in hours.
        
        Returns age_hours if set, otherwise 0 (unknown age).
        For topology-based age computation, see EdgeConfiguration.get_age_hours().
        
        Args:
            current_time: Current timestamp
        
        Returns:
            Age in hours (0 if unknown)
        """
        return self.age_hours if self.age_hours is not None else 0.0

    def get_maintenance_score(self, current_date: date | datetime) -> float:
        """Compute maintenance score [0, 1].
        
        Score decays linearly from 1.0 (just maintained) to 0.0 (365+ days ago).
        Returns 0.5 if no maintenance history (neutral).
        
        Args:
            current_date: Current date/datetime
        
        Returns:
            Maintenance score in [0, 1]
        
        Examples:
            >>> edge = EdgeSpec(..., last_maintenance_date=date(2024, 6, 1))
            >>> edge.get_maintenance_score(date(2024, 7, 1))  # 30 days ago
            0.918  # ≈ 1.0 - 30/365
            >>> edge.get_maintenance_score(date(2025, 6, 1))  # 365 days ago
            0.0
        """
        if not self.last_maintenance_date:
            return 0.5  # Unknown = neutral

        if isinstance(current_date, datetime):
            current_date = current_date.date()

        days_since = (current_date - self.last_maintenance_date).days

        # Linear decay over 365 days
        score = max(0.0, 1.0 - days_since / 365.0)
        return score


class ComponentSpec(BaseModel):
    """Спецификация компонента гидравлической системы.
    
    Представляет один компонент (node) в графе с его характеристиками,
    подключенными сенсорами и метаданными.
    
    Attributes:
        component_id: Уникальный идентификатор компонента
        component_type: Тип компонента (pump, valve, cylinder, etc.)
        sensors: Список ID подключенных сенсоров
        feature_dim: Размерность вектора признаков
        nominal_pressure_bar: Номинальное рабочее давление (бар)
        nominal_flow_lpm: Номинальный расход (л/мин)
        rated_power_kw: Номинальная мощность (кВт)
        metadata: Дополнительные метаданные
    
    Examples:
        >>> component = ComponentSpec(
        ...     component_id="pump_main_001",
        ...     component_type=ComponentType.PISTON_PUMP,
        ...     sensors=["pressure_in", "pressure_out", "temperature", "vibration"],
        ...     feature_dim=12,
        ...     nominal_pressure_bar=280,
        ...     nominal_flow_lpm=120,
        ...     rated_power_kw=45.0
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "component_id": "pump_main_001",
                "component_type": "piston_pump",
                "sensors": ["pressure_in", "pressure_out", "temperature", "vibration"],
                "feature_dim": 12,
                "nominal_pressure_bar": 280,
                "nominal_flow_lpm": 120,
                "rated_power_kw": 45.0,
                "metadata": {
                    "manufacturer": "Bosch Rexroth",
                    "model": "A10VSO",
                    "serial_number": "12345678",
                    "installation_date": "2023-01-15"
                }
            }
        }
    )

    component_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Уникальный идентификатор компонента"
    )

    component_type: ComponentType = Field(
        ...,
        description="Тип компонента (pump, valve, cylinder, etc.)"
    )

    sensors: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Список ID подключенных сенсоров"
    )

    feature_dim: Annotated[int, Field(gt=0, le=256)] = Field(
        ...,
        description="Размерность вектора признаков для этого компонента"
    )

    nominal_pressure_bar: Annotated[float, Field(gt=0, le=1000)] = Field(
        ...,
        description="Номинальное рабочее давление (бар)"
    )

    nominal_flow_lpm: Annotated[float, Field(gt=0, le=1000)] = Field(
        ...,
        description="Номинальный расход жидкости (литры/мин)"
    )

    rated_power_kw: Annotated[float, Field(ge=0, le=500)] = Field(
        default=0.0,
        description="Номинальная мощность (кВт), 0 для пассивных компонентов"
    )

    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Дополнительные метаданные (manufacturer, model, serial, etc.)"
    )

    @field_validator("sensors")
    @classmethod
    def validate_unique_sensors(cls, v: list[str]) -> list[str]:
        """Проверка уникальности sensor IDs."""
        if len(v) != len(set(v)):
            raise ValueError("Sensor IDs must be unique")
        return v

    @computed_field
    @property
    def power_density(self) -> float:
        """Плотность мощности (кВт на л/мин)."""
        if self.nominal_flow_lpm > 0:
            return self.rated_power_kw / self.nominal_flow_lpm
        return 0.0


class GraphTopology(BaseModel):
    """Топология гидравлического графа.
    
    Определяет структуру графа: компоненты (nodes), соединения (edges)
    и их характеристики. Используется для построения PyTorch Geometric Data.
    
    Attributes:
        equipment_id: ID оборудования
        components: Словарь {component_id: ComponentSpec}
        edges: Список соединений между компонентами с характеристиками
        topology_version: Версия топологии (для tracking изменений)
    
    Examples:
        >>> topology = GraphTopology(
        ...     equipment_id="excavator_001",
        ...     components={
        ...         "pump_001": ComponentSpec(...),
        ...         "valve_001": ComponentSpec(...)
        ...     },
        ...     edges=[
        ...         EdgeSpec(source_id="pump_001", target_id="valve_001", ...)
        ...     ],
        ...     topology_version="v1.0"
        ... )
        >>> topology.validate_connectivity()  # True if graph is connected
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "equipment_id": "excavator_001",
                "components": {
                    "pump_001": {"component_id": "pump_001", "component_type": "piston_pump", "sensors": ["pressure"], "feature_dim": 8},
                    "valve_001": {"component_id": "valve_001", "component_type": "directional_valve", "sensors": ["position"], "feature_dim": 6}
                },
                "edges": [
                    {"source_id": "pump_001", "target_id": "valve_001", "edge_type": "high_pressure_hose", "diameter_mm": 16, "length_m": 2.5, "pressure_rating_bar": 350}
                ],
                "topology_version": "v1.0"
            }
        }
    )

    equipment_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Уникальный идентификатор оборудования"
    )

    components: dict[str, ComponentSpec] = Field(
        ...,
        min_length=2,
        description="Словарь компонентов {component_id: ComponentSpec}"
    )

    edges: list[EdgeSpec] = Field(
        ...,
        min_length=1,
        description="Список соединений между компонентами с характеристиками"
    )

    topology_version: str = Field(
        default="v1.0",
        pattern=r"^v\d+\.\d+$",
        description="Версия топологии (для tracking)"
    )

    @field_validator("components")
    @classmethod
    def validate_component_ids_match(cls, v: dict[str, ComponentSpec]) -> dict[str, ComponentSpec]:
        """Проверка соответствия ключей и component_id."""
        for key, component in v.items():
            if key != component.component_id:
                raise ValueError(
                    f"Key '{key}' does not match component_id '{component.component_id}'"
                )
        return v

    @field_validator("edges")
    @classmethod
    def validate_edges_reference_components(cls, v: list[EdgeSpec], info) -> list[EdgeSpec]:
        """Проверка, что все edges ссылаются на существующие компоненты."""
        if "components" in info.data:
            component_ids = set(info.data["components"].keys())
            for edge in v:
                if edge.source_id not in component_ids:
                    raise ValueError(
                        f"Edge source_id '{edge.source_id}' not found in components"
                    )
                if edge.target_id not in component_ids:
                    raise ValueError(
                        f"Edge target_id '{edge.target_id}' not found in components"
                    )
        return v

    def validate_connectivity(self) -> bool:
        """Проверка связности графа (все компоненты достижимы).
        
        Returns:
            True если граф связный, False иначе
        """
        if not self.components or not self.edges:
            return False

        # Build adjacency list
        adj: dict[str, list[str]] = {cid: [] for cid in self.components}
        for edge in self.edges:
            adj[edge.source_id].append(edge.target_id)
            # Bidirectional edges
            if edge.flow_direction == "bidirectional":
                adj[edge.target_id].append(edge.source_id)

        # DFS from first component
        visited = set()
        stack = [next(iter(self.components.keys()))]

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(adj[node])

        return len(visited) == len(self.components)

    @computed_field
    @property
    def num_components(self) -> int:
        """Количество компонентов в графе."""
        return len(self.components)

    @computed_field
    @property
    def num_edges(self) -> int:
        """Количество соединений в графе."""
        return len(self.edges)

    @computed_field
    @property
    def avg_degree(self) -> float:
        """Средняя степень узлов графа."""
        if self.num_components == 0:
            return 0.0
        return 2 * self.num_edges / self.num_components

    def get_component_types_distribution(self) -> dict[ComponentType, int]:
        """Распределение типов компонентов.
        
        Returns:
            Словарь {ComponentType: count}
        """
        distribution: dict[ComponentType, int] = {}
        for component in self.components.values():
            comp_type = component.component_type
            distribution[comp_type] = distribution.get(comp_type, 0) + 1
        return distribution

    def get_edge_types_distribution(self) -> dict[EdgeType, int]:
        """Распределение типов соединений.
        
        Returns:
            Словарь {EdgeType: count}
        """
        distribution: dict[EdgeType, int] = {}
        for edge in self.edges:
            edge_type = edge.edge_type
            distribution[edge_type] = distribution.get(edge_type, 0) + 1
        return distribution

    def to_pyg_format(self) -> tuple[dict[str, int], list[tuple[int, int]]]:
        """Конвертация в формат PyTorch Geometric.
        
        Returns:
            node_mapping: {component_id: node_index}
            edge_list: [(source_idx, target_idx), ...]
        """
        # Create node mapping
        node_mapping = {cid: idx for idx, cid in enumerate(self.components.keys())}

        # Create edge list with indices
        edge_list = [
            (node_mapping[edge.source_id], node_mapping[edge.target_id])
            for edge in self.edges
        ]

        return node_mapping, edge_list
