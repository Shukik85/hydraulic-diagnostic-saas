"""Pydantic v2 schemas для GNN сервиса.

Экспорт всех schemas для удобного импорта:

    from src.schemas import (
        GraphTopology,
        ComponentSpec,
        EdgeSpec,
        InferenceRequest,
        InferenceResponse,
        ComponentHealth,
        Anomaly
    )

Python 3.14 Features:
    - Deferred annotations (PEP 649)
    - All type hints evaluated lazily
"""

from __future__ import annotations

# Graph schemas
from src.schemas.graph import (
    ComponentType,
    EdgeType,
    EdgeSpec,
    ComponentSpec,
    GraphTopology,
)

# Metadata schemas
from src.schemas.metadata import (
    SensorType,
    SensorConfig,
    EquipmentMetadata,
    SystemConfig,
)

# Request schemas
from src.schemas.requests import (
    TimeWindow,
    InferenceRequest,
    BatchInferenceRequest,
    TrainingRequest,
)

# Response schemas
from src.schemas.responses import (
    ComponentStatus,
    AnomalyType,
    ComponentHealth,
    Anomaly,
    InferenceResponse,
    TrainingResponse,
)

__all__ = [
    # Enums
    "ComponentType",
    "EdgeType",
    "SensorType",
    "ComponentStatus",
    "AnomalyType",
    # Graph
    "EdgeSpec",
    "ComponentSpec",
    "GraphTopology",
    # Metadata
    "SensorConfig",
    "EquipmentMetadata",
    "SystemConfig",
    # Requests
    "TimeWindow",
    "InferenceRequest",
    "BatchInferenceRequest",
    "TrainingRequest",
    # Responses
    "ComponentHealth",
    "Anomaly",
    "InferenceResponse",
    "TrainingResponse",
]
