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
    PredictionRequest,
)

# Responses
from .responses import (
    AnomalyPrediction,
    ComponentPrediction,
    DegradationPrediction,
    GraphPrediction,
    HealthPrediction,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)

# Topology
from .topology import (
    TopologyTemplate,
)

__all__ = [
    # Responses (NEW: prediction schemas)
    "AnomalyPrediction",
    "ComponentPrediction",
    "DegradationPrediction",
    "GraphPrediction",
    "HealthPrediction",
    "HealthResponse",
    "ModelInfo",
    "PredictionResponse",
    # Graph
    "ComponentSpec",
    "ComponentType",
    "EdgeMaterial",
    "EdgeSpec",
    "EdgeType",
    "GraphTopology",
    # Requests
    "ComponentSensorReading",
    "InferenceRequest",
    "MinimalInferenceRequest",
    "PredictionRequest",
    # Metadata
    "EquipmentMetadata",
    "SensorConfig",
    "SensorType",
    "SystemConfig",
    "TimeWindow",
    # Topology
    "TopologyTemplate",
]
