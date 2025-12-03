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

# Metadata
from .metadata import (
    TimeWindow,
    EquipmentMetadata,
    SensorConfig,
    ComponentType,
    ComponentRole,
)

# Graph
from .graph import (
    GraphTopology,
    ComponentSpec,
    EdgeSpec,
    EdgeType,
    EdgeMaterial,
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
    "ComponentType",
    "ComponentRole",
    # Graph
    "GraphTopology",
    "ComponentSpec",
    "EdgeSpec",
    "EdgeType",
    "EdgeMaterial",
    # Topology
    "TopologyTemplate",
]
