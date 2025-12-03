"""Schema exports.

All Pydantic schemas for GNN service.
"""

# Requests
# Graph (removed ComponentRole - doesn't exist)
from .graph import (
    ComponentSpec,
    ComponentType,
    EdgeMaterial,
    EdgeSpec,
    EdgeType,
    GraphTopology,
)

# Metadata (now with TimeWindow!)
from .metadata import (
    EquipmentMetadata,
    SensorConfig,
    SensorType,
    SystemConfig,
    TimeWindow,
)
from .requests import (
    ComponentSensorReading,
    InferenceRequest,
    MinimalInferenceRequest,
)

# Responses
from .responses import (
    ComponentPrediction,
    GraphPrediction,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)

# Topology
from .topology import (
    TopologyTemplate,
)

__all__ = [
    # Requests
    "InferenceRequest",
    "ComponentSensorReading",
    "MinimalInferenceRequest",
    # Responses
    "PredictionResponse",
    "ComponentPrediction",
    "GraphPrediction",
    "ModelInfo",
    "HealthResponse",
    # Metadata
    "TimeWindow",
    "EquipmentMetadata",
    "SensorConfig",
    "SensorType",
    "SystemConfig",
    # Graph (no ComponentRole)
    "GraphTopology",
    "ComponentSpec",
    "EdgeSpec",
    "EdgeType",
    "EdgeMaterial",
    "ComponentType",
    # Topology
    "TopologyTemplate",
]
