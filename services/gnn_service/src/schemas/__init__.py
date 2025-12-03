"""Schema exports.

All Pydantic schemas for GNN service.
"""

# Requests
from .requests import (
    InferenceRequest,
    ComponentSensorReading,
    MinimalInferenceRequest,
)

# Responses
from .responses import (
    PredictionResponse,
    ComponentPrediction,
    GraphPrediction,
    ModelInfo,
    HealthResponse,
)

# Metadata (now with TimeWindow!)
from .metadata import (
    TimeWindow,
    EquipmentMetadata,
    SensorConfig,
    SensorType,
    SystemConfig,
)

# Graph
from .graph import (
    GraphTopology,
    ComponentSpec,
    EdgeSpec,
    EdgeType,
    EdgeMaterial,
    ComponentType,
    ComponentRole,
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
    # Graph
    "GraphTopology",
    "ComponentSpec",
    "EdgeSpec",
    "EdgeType",
    "EdgeMaterial",
    "ComponentType",
    "ComponentRole",
    # Topology
    "TopologyTemplate",
]
