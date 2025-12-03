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
    "ComponentPrediction",
    "ComponentSensorReading",
    "ComponentSpec",
    "ComponentType",
    "EdgeMaterial",
    "EdgeSpec",
    "EdgeType",
    "EquipmentMetadata",
    "GraphPrediction",
    # Graph (no ComponentRole)
    "GraphTopology",
    "HealthResponse",
    # Requests
    "InferenceRequest",
    "MinimalInferenceRequest",
    "ModelInfo",
    # Responses
    "PredictionResponse",
    "SensorConfig",
    "SensorType",
    "SystemConfig",
    # Metadata
    "TimeWindow",
    # Topology
    "TopologyTemplate",
]
