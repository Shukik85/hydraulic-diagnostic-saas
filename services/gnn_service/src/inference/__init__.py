"""Inference package.

Provides production-ready inference engine with:
- Dynamic batching
- Multi-model A/B testing
- Prometheus metrics
- Topology caching
- Request tracking
"""

from .batching import BatchItem
from .cache import AsyncTopologyCache, TopologyCache
from .exceptions import (
    GPUOutOfMemoryError,
    GraphBuildError,
    InferenceEngineError,
    InferenceError,
    ModelLoadError,
    TensorValidationError,
    TopologyNotFoundError,
)
from .inference_engine import InferenceConfig, InferenceEngine
from .model_registry import ModelConfig, ModelRegistry
from .request_context import (
    ensure_request_id,
    generate_request_id,
    get_request_id,
    set_request_id,
)
from .validation import TensorValidator

__all__ = [
    # Main classes
    "InferenceEngine",
    "InferenceConfig",
    # Model management
    "ModelConfig",
    "ModelRegistry",
    # Caching
    "TopologyCache",
    "AsyncTopologyCache",
    # Batching
    "BatchItem",
    # Validation
    "TensorValidator",
    # Request tracking
    "get_request_id",
    "set_request_id",
    "generate_request_id",
    "ensure_request_id",
    # Exceptions
    "InferenceEngineError",
    "ModelLoadError",
    "GraphBuildError",
    "InferenceError",
    "TopologyNotFoundError",
    "GPUOutOfMemoryError",
    "TensorValidationError",
]
