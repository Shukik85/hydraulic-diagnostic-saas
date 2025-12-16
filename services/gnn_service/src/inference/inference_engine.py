"""Production-Ready Inference Engine with Advanced Features.

Enterprise-grade inference engine featuring:
- Dynamic batching with automatic flushing
- Multi-model support with A/B testing
- Prometheus metrics for observability
- LRU topology caching with TTL
- Comprehensive tensor validation

Python 3.14:
    - PEP 649 deferred annotations
    - Pipe operator for unions (T | None)
    - Enhanced asyncio support

Architecture:
    ┌─────────────────┐
    │  FastAPI        │
    │  /v1/diagnose   │
    └────────┬────────┘
             │
    ┌────────▼────────────────────┐
    │  InferenceEngine            │
    │  ├─ Dynamic batching        │
    │  ├─ Multi-model routing     │
    │  ├─ Topology cache          │
    │  └─ Tensor validation       │
    └────────┬────────────────────┘
             │
    ┌────────▼────────┐  ┌──────────────┐
    │  ModelRegistry  │  │  BatchQueue  │
    │  - v1: 80%      │  │  - Timeout   │
    │  - v2: 20%      │  │  - Auto-flush│
    └─────────────────┘  └──────────────┘

Usage:
    >>> # Single model with batching
    >>> config = InferenceConfig(
    ...     model_path="models/v2.0.0.ckpt",
    ...     enable_dynamic_batching=True,
    ...     batch_size=32,
    ...     max_wait_ms=50.0
    ... )
    >>> async with InferenceEngine(config) as engine:
    ...     response = await engine.predict_minimal(request)
    >>>
    >>> # Multi-model A/B testing
    >>> config = InferenceConfig(
    ...     model_versions={
    ...         "v1": ModelConfig(path="v1.ckpt", traffic=0.8),
    ...         "v2": ModelConfig(path="v2.ckpt", traffic=0.2)
    ...     }
    ... )
    >>> # Monitoring
    >>> stats = engine.get_stats()
    >>> print(f"Cache hit rate: {stats['topology_cache_hit_rate']:.2%}")
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from prometheus_client import Counter, Gauge, Histogram
from torch_geometric.data import Data

from src.schemas import (
    AnomalyPrediction,
    DegradationPrediction,
    GraphTopology,
    HealthPrediction,
    PredictionResponse,
)
from src.schemas.requests import MinimalInferenceRequest

if TYPE_CHECKING:
    from src.data import FeatureConfig
    from src.data.timescale_connector import TimescaleConnector

logger = logging.getLogger(__name__)


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Counters
INFERENCE_REQUESTS_TOTAL = Counter(
    "gnn_inference_requests_total",
    "Total inference requests",
    ["model_version", "status"],
)

INFERENCE_ERRORS_TOTAL = Counter(
    "gnn_inference_errors_total",
    "Total inference errors",
    ["error_type"],
)

# Histograms
INFERENCE_DURATION_SECONDS = Histogram(
    "gnn_inference_duration_seconds",
    "Inference duration in seconds",
    ["model_version"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

INFERENCE_BATCH_SIZE = Histogram(
    "gnn_inference_batch_size",
    "Actual batch size",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128],
)

# Gauges
MODEL_GPU_MEMORY_BYTES = Gauge(
    "gnn_model_gpu_memory_bytes",
    "GPU memory allocated",
    ["model_version", "device"],
)

REQUEST_QUEUE_SIZE = Gauge(
    "gnn_request_queue_size",
    "Request queue size",
)

TOPOLOGY_CACHE_SIZE = Gauge(
    "gnn_topology_cache_size",
    "Cached topologies",
)


# ============================================================================
# EXCEPTIONS
# ============================================================================


class InferenceEngineError(Exception):
    """Base exception."""


class ModelLoadError(InferenceEngineError):
    """Model load failed."""


class GraphBuildError(InferenceEngineError):
    """Graph build failed."""


class InferenceError(InferenceEngineError):
    """Inference failed."""


class TopologyNotFoundError(InferenceEngineError):
    """Topology not found."""


class GPUOutOfMemoryError(InferenceError):
    """GPU OOM."""


class TensorValidationError(InferenceError):
    """Tensor validation failed."""


# ============================================================================
# MODEL REGISTRY
# ============================================================================


@dataclass
class ModelConfig:
    """Model version configuration.

    Attributes:
        path: Model checkpoint path
        version: Version identifier
        traffic: Traffic split [0, 1]
        device: Device override
        enabled: Active flag
    """

    path: str
    version: str
    traffic: float = 1.0
    device: str | None = None
    enabled: bool = True

    def __post_init__(self):
        """Validate."""
        if not 0.0 <= self.traffic <= 1.0:
            msg = f"traffic must be [0,1], got {self.traffic}"
            raise ValueError(msg)
        if not Path(self.path).exists():
            msg = f"Model not found: {self.path}"
            raise FileNotFoundError(msg)


class ModelRegistry:
    """Multi-model registry with A/B testing.

    Examples:
        >>> registry = ModelRegistry()
        >>> registry.register("v1", ModelConfig(...))
        >>> version = registry.select_model("req_123")  # Consistent routing
    """

    def __init__(self):
        """Initialize."""
        self._models: dict[str, ModelConfig] = {}
        self._loaded_models: dict[str, Any] = {}

    def register(self, version: str, config: ModelConfig) -> None:
        """Register model."""
        self._models[version] = config
        logger.info(
            f"✅ Registered model '{version}'",
            extra={"traffic": config.traffic},
        )

    def select_model(self, request_id: str) -> str:
        """Select model by traffic split (consistent hashing)."""
        enabled = {v: c for v, c in self._models.items() if c.enabled}
        if not enabled:
            msg = "No enabled models"
            raise ValueError(msg)
        if len(enabled) == 1:
            return next(iter(enabled))

        # Consistent hashing
        hash_val = hash(request_id) % 100
        cumulative = 0.0
        for version, config in enabled.items():
            cumulative += config.traffic * 100
            if hash_val < cumulative:
                return version
        return next(iter(enabled))

    def get_model(self, version: str) -> Any:
        """Get loaded model."""
        if version not in self._loaded_models:
            msg = f"Model '{version}' not loaded"
            raise KeyError(msg)
        return self._loaded_models[version]

    def set_loaded_model(self, version: str, model: Any) -> None:
        """Store loaded model."""
        self._loaded_models[version] = model

    def get_all_versions(self) -> list[str]:
        """Get all versions."""
        return list(self._models.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get stats."""
        return {
            "total_versions": len(self._models),
            "enabled_versions": sum(1 for c in self._models.values() if c.enabled),
            "loaded_versions": len(self._loaded_models),
            "traffic_split": {v: c.traffic for v, c in self._models.items() if c.enabled},
        }


# ============================================================================
# TOPOLOGY CACHE
# ============================================================================


class TopologyCache:
    """LRU cache with TTL.

    Examples:
        >>> cache = TopologyCache(max_size=100, ttl_seconds=300)
        >>> cache.put("topo_001", topology)
        >>> topo = cache.get("topo_001")  # Hit
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 300.0):
        """Initialize."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[GraphTopology, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> GraphTopology | None:
        """Get from cache."""
        if key not in self._cache:
            self._misses += 1
            TOPOLOGY_CACHE_SIZE.set(len(self._cache))
            return None

        topology, timestamp = self._cache[key]

        # Check TTL
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            self._misses += 1
            TOPOLOGY_CACHE_SIZE.set(len(self._cache))
            return None

        # LRU
        self._cache.move_to_end(key)
        self._hits += 1
        return topology

    def put(self, key: str, topology: GraphTopology) -> None:
        """Put in cache."""
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (topology, time.time())
        self._cache.move_to_end(key)
        TOPOLOGY_CACHE_SIZE.set(len(self._cache))

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        TOPOLOGY_CACHE_SIZE.set(0)

    def get_stats(self) -> dict[str, Any]:
        """Get stats."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "ttl_seconds": self.ttl_seconds,
        }


# ============================================================================
# TENSOR VALIDATOR
# ============================================================================


class TensorValidator:
    """Tensor validation before inference.

    Examples:
        >>> validator = TensorValidator(node_dim=34, edge_dim=14)
        >>> validator.validate_graph(graph)  # Raises if invalid
    """

    def __init__(
        self,
        expected_node_dim: int,
        expected_edge_dim: int,
        allowed_devices: list[str] | None = None,
    ):
        """Initialize."""
        self.expected_node_dim = expected_node_dim
        self.expected_edge_dim = expected_edge_dim
        self.allowed_devices = allowed_devices

    def validate_graph(self, graph: Data) -> None:
        """Validate graph.

        Raises:
            TensorValidationError: If invalid
        """
        # Node features
        if graph.x.shape[1] != self.expected_node_dim:
            raise TensorValidationError(
                f"Node dim mismatch: expected {self.expected_node_dim}, "
                f"got {graph.x.shape[1]}"
            )

        # Edge features
        if graph.edge_attr is not None and graph.edge_attr.shape[1] != self.expected_edge_dim:
            raise TensorValidationError(
                f"Edge dim mismatch: expected {self.expected_edge_dim}, "
                f"got {graph.edge_attr.shape[1]}"
            )

        # NaN/Inf
        if torch.isnan(graph.x).any():
            raise TensorValidationError("NaN in node features")
        if torch.isinf(graph.x).any():
            raise TensorValidationError("Inf in node features")

        if graph.edge_attr is not None:
            if torch.isnan(graph.edge_attr).any():
                raise TensorValidationError("NaN in edge features")
            if torch.isinf(graph.edge_attr).any():
                raise TensorValidationError("Inf in edge features")

        # Device
        if self.allowed_devices:
            device_str = str(graph.x.device)
            if not any(allowed in device_str for allowed in self.allowed_devices):
                raise TensorValidationError(
                    f"Invalid device: {device_str}. Allowed: {self.allowed_devices}"
                )

        # Dtype
        if graph.x.dtype != torch.float32:
            warnings.warn(
                f"Non-float32 nodes: {graph.x.dtype}",
                UserWarning,
                stacklevel=2,
            )


# ============================================================================
# BATCH ITEM
# ============================================================================


@dataclass
class BatchItem:
    """Batch queue item.

    Attributes:
        request: Request
        future: Result future
        enqueue_time: Timestamp
        model_version: Selected model
    """

    request: MinimalInferenceRequest
    future: asyncio.Future[PredictionResponse]
    model_version: str
    enqueue_time: float = field(default_factory=time.time)


# ============================================================================
# INFERENCE CONFIG
# ============================================================================


@dataclass
class InferenceConfig:
    """Inference configuration.

    Examples:
        >>> # Single model
        >>> config = InferenceConfig(
        ...     model_path="models/v2.ckpt",
        ...     enable_dynamic_batching=True
        ... )
        >>> # Multi-model
        >>> config = InferenceConfig(
        ...     model_versions={
        ...         "v1": ModelConfig(path="v1.ckpt", traffic=0.8),
        ...         "v2": ModelConfig(path="v2.ckpt", traffic=0.2)
        ...     }
        ... )
    """

    model_path: str | None = None
    device: Literal["cpu", "cuda", "auto"] = "auto"
    batch_size: int = 32
    max_queue_size: int = 100
    max_wait_ms: float = 50.0
    inference_timeout_s: float = 30.0
    enable_dynamic_batching: bool = False
    use_dynamic_features: bool = True
    use_dynamic_builder: bool = True
    topology_templates_path: Path | None = None
    enable_compile: bool = True
    fallback_to_cpu: bool = True
    pin_memory: bool = True

    # Multi-model
    model_versions: dict[str, ModelConfig] | None = None

    # Cache
    topology_cache_size: int = 100
    topology_cache_ttl_s: float = 300.0

    # Validation
    validate_tensors: bool = True

    # Stats
    _total_inferences: int = field(default=0, init=False, repr=False)
    _total_errors: int = field(default=0, init=False, repr=False)
    _total_inference_time_s: float = field(default=0.0, init=False, repr=False)
    _total_batch_items: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Validate."""
        if not self.model_path and not self.model_versions:
            msg = "model_path or model_versions required"
            raise ValueError(msg)
        if self.batch_size < 1:
            msg = f"batch_size >= 1 required, got {self.batch_size}"
            raise ValueError(msg)
        if self.max_wait_ms < 0:
            msg = f"max_wait_ms >= 0 required, got {self.max_wait_ms}"
            raise ValueError(msg)

        # Single model validation
        if self.model_path and not self.model_versions and not Path(self.model_path).exists():
            msg = f"Model not found: {self.model_path}"
            raise FileNotFoundError(msg)

        # Multi-model traffic validation
        if self.model_versions:
            total_traffic = sum(c.traffic for c in self.model_versions.values() if c.enabled)
            if not 0.99 <= total_traffic <= 1.01:
                warnings.warn(
                    f"Traffic sums to {total_traffic:.2f} (expected 1.0)",
                    UserWarning,
                    stacklevel=2,
                )


# ============================================================================
# INFERENCE ENGINE
# ============================================================================


class InferenceEngine:
    """Production-ready inference engine.

    Features:
    - Dynamic batching
    - Multi-model A/B testing
    - Prometheus metrics
    - Topology caching
    - Tensor validation

    Examples:
        >>> async with InferenceEngine(config) as engine:
        ...     response = await engine.predict_minimal(request)
    """

    def __init__(
        self,
        config: InferenceConfig,
        timescale_connector: TimescaleConnector | None = None,
        feature_config: FeatureConfig | None = None,
    ):
        """Initialize engine.

        Args:
            config: Configuration
            timescale_connector: Database connector
            feature_config: Feature config

        Raises:
            ModelLoadError: If model load fails
        """
        self.config = config
        self._shutdown = False

        # Import here to avoid circular deps
        from src.data import FeatureConfig, FeatureEngineer, GraphBuilder
        from src.data.edge_features import create_edge_feature_computer
        from src.data.normalization import create_edge_feature_normalizer
        from src.inference.dynamic_graph_builder import DynamicGraphBuilder
        from src.inference.model_manager import ModelManager
        from src.services.topology_service import get_topology_service

        self.feature_config = feature_config or FeatureConfig()
        self.timescale_connector = timescale_connector

        logger.info(
            "Initializing InferenceEngine",
            extra={
                "device": config.device,
                "batch_size": config.batch_size,
                "dynamic_batching": config.enable_dynamic_batching,
            },
        )

        try:
            # Components
            self.model_manager = ModelManager()
            self.feature_engineer = FeatureEngineer(self.feature_config)

            # Dynamic builder
            self.dynamic_builder = None
            if config.use_dynamic_builder and timescale_connector:
                self.dynamic_builder = DynamicGraphBuilder(
                    timescale_connector=timescale_connector,
                    feature_engineer=self.feature_engineer,
                    feature_config=self.feature_config,
                )
                logger.info("✅ DynamicGraphBuilder enabled")
            elif config.use_dynamic_builder:
                logger.warning("⚠️  DynamicGraphBuilder requested but no connector")

            # Edge features
            self.edge_feature_computer = create_edge_feature_computer()
            self.edge_normalizer = self._load_normalizer(
                config.model_path or next(iter(config.model_versions.values())).path
            )

            # Graph builder
            self.graph_builder = GraphBuilder(
                feature_engineer=self.feature_engineer,
                feature_config=self.feature_config,
                edge_feature_computer=self.edge_feature_computer,
                edge_normalizer=self.edge_normalizer,
                use_dynamic_features=config.use_dynamic_features,
            )

            # Topology service
            self.topology_service = get_topology_service(config.topology_templates_path)

            # Topology cache
            self.topology_cache = TopologyCache(
                max_size=config.topology_cache_size,
                ttl_seconds=config.topology_cache_ttl_s,
            )

            # Tensor validator
            self.tensor_validator = None
            if config.validate_tensors:
                self.tensor_validator = TensorValidator(
                    expected_node_dim=self.feature_config.node_feature_dim,
                    expected_edge_dim=14,  # Dynamic edges
                )

            # Multi-model registry
            self.model_registry = None
            if config.model_versions:
                self.model_registry = ModelRegistry()
                for version, model_config in config.model_versions.items():
                    self.model_registry.register(version, model_config)
                    # Load each model
                    model = self._load_model_safe(model_config.path, version)
                    self.model_registry.set_loaded_model(version, model)
            else:
                # Single model
                self.model = self._load_model_safe(config.model_path, "default")

            # Dynamic batching
            self._batch_queue = None
            self._batch_processor_task = None
            if config.enable_dynamic_batching:
                self._batch_queue = asyncio.Queue(maxsize=config.max_queue_size)
                self._batch_processor_task = asyncio.create_task(
                    self._batch_processor_loop()
                )
                logger.info("✅ Dynamic batching enabled")

            logger.info("✅ InferenceEngine initialized")

        except Exception as e:
            logger.error("❌ Initialization failed", exc_info=True)
            raise ModelLoadError(f"Init failed: {e}") from e

    def _load_normalizer(self, checkpoint_path: str) -> Any:
        """Load normalizer from checkpoint."""
        # Import already available from __init__ imports
        from src.data.normalization import create_edge_feature_normalizer

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "normalizer_stats" in checkpoint:
                normalizer = create_edge_feature_normalizer()
                normalizer.load_stats(checkpoint["normalizer_stats"])
                logger.info("✅ Loaded normalizer")
                return normalizer
            logger.warning("⚠️  No normalizer stats, using defaults")
            return create_edge_feature_normalizer()
        except Exception:
            logger.warning("⚠️  Normalizer load failed, using defaults")
            return create_edge_feature_normalizer()

    def _load_model_safe(self, model_path: str, version: str) -> Any:
        """Load model with error handling."""
        try:
            model = self.model_manager.load_model(
                model_path=model_path,
                device=self.config.device,
                use_compile=self.config.enable_compile,
            )
            self.model_manager.warmup(model_path, batch_size=self.config.batch_size)
            return model
        except torch.cuda.OutOfMemoryError as e:
            if self.config.fallback_to_cpu:
                logger.warning(f"⚠️  GPU OOM for {version}, falling back to CPU")
                model = self.model_manager.load_model(
                    model_path=model_path,
                    device="cpu",
                    use_compile=False,
                )
                return model
            logger.error("❌ GPU OOM and fallback_to_cpu=False")
            raise GPUOutOfMemoryError(f"GPU OOM loading {version}") from e
        except Exception as e:
            logger.error(f"❌ Model load failed: {version}", exc_info=True)
            raise ModelLoadError(f"Load failed: {e}") from e

    # ========================================================================
    # PREDICTION
    # ========================================================================

    async def predict_minimal(self, request: MinimalInferenceRequest) -> PredictionResponse:
        """Predict with minimal request.

        Args:
            request: Request

        Returns:
            Response

        Raises:
            InferenceError: If prediction fails
        """
        start_time = time.time()

        # Select model version
        model_version = "default"
        if self.model_registry:
            model_version = self.model_registry.select_model(request.equipment_id)

        try:
            # Dynamic batching path
            if self.config.enable_dynamic_batching and self._batch_queue is not None:
                future: asyncio.Future[PredictionResponse] = asyncio.Future()
                item = BatchItem(
                    request=request,
                    future=future,
                    model_version=model_version,
                )
                await self._batch_queue.put(item)
                REQUEST_QUEUE_SIZE.set(self._batch_queue.qsize())

                # Wait for result
                response = await asyncio.wait_for(
                    future,
                    timeout=self.config.inference_timeout_s,
                )
            else:
                # Direct inference
                response = await asyncio.wait_for(
                    self._predict_minimal_impl(request, model_version),
                    timeout=self.config.inference_timeout_s,
                )

            # Stats
            self.config._total_inferences += 1
            self.config._total_inference_time_s += time.time() - start_time

            # Metrics
            INFERENCE_REQUESTS_TOTAL.labels(
                model_version=model_version,
                status="success",
            ).inc()

            return response

        except TimeoutError as e:
            self.config._total_errors += 1
            INFERENCE_ERRORS_TOTAL.labels(error_type="timeout").inc()
            INFERENCE_REQUESTS_TOTAL.labels(
                model_version=model_version,
                status="timeout",
            ).inc()
            logger.error(f"⏱️  Timeout: {request.equipment_id}")
            raise InferenceError(f"Timeout after {self.config.inference_timeout_s}s") from e
        except Exception as e:
            self.config._total_errors += 1
            error_type = type(e).__name__
            INFERENCE_ERRORS_TOTAL.labels(error_type=error_type).inc()
            INFERENCE_REQUESTS_TOTAL.labels(
                model_version=model_version,
                status="error",
            ).inc()
            logger.error(f"❌ Inference failed: {request.equipment_id}", exc_info=True)
            raise

    async def _predict_minimal_impl(
        self, request: MinimalInferenceRequest, model_version: str
    ) -> PredictionResponse:
        """Implementation (no timeout wrapper)."""
        start_time = time.time()

        # Get topology (with caching)
        topology = self.topology_cache.get(request.topology_id)
        if topology is None:
            template = self.topology_service.get_template(request.topology_id)
            if not template:
                raise TopologyNotFoundError(
                    f"Topology '{request.topology_id}' not found. "
                    f"Available: {list(self.topology_service.get_all_templates().keys())}"
                )
            topology = template.to_graph_topology(request.equipment_id)
            self.topology_cache.put(request.topology_id, topology)

        # Build graph
        try:
            if self.config.use_dynamic_builder and self.dynamic_builder:
                graph = await self.dynamic_builder.build_from_timescale(
                    equipment_id=request.equipment_id,
                    topology=topology,
                    lookback_minutes=10,
                )
            else:
                graph = self._preprocess_minimal(request, topology)
        except Exception as e:
            raise GraphBuildError(f"Graph build failed: {e}") from e

        # Validate
        if self.tensor_validator:
            self.tensor_validator.validate_graph(graph)

        # Inference
        try:
            with INFERENCE_DURATION_SECONDS.labels(model_version=model_version).time():
                health, degradation, anomaly = self._inference_single(
                    graph, model_version
                )
        except torch.cuda.OutOfMemoryError as e:
            if self.config.fallback_to_cpu:
                logger.warning(f"⚠️  GPU OOM, retrying on CPU: {request.equipment_id}")
                # Move model to CPU
                if self.model_registry:
                    model = self.model_registry.get_model(model_version)
                else:
                    model = self.model
                model = model.cpu()
                health, degradation, anomaly = self._inference_single(graph, model_version)
            else:
                raise GPUOutOfMemoryError("GPU OOM during inference") from e
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e

        # Postprocess
        response = self._postprocess(
            equipment_id=request.equipment_id,
            health=health,
            degradation=degradation,
            anomaly=anomaly,
            inference_time=time.time() - start_time,
        )

        logger.info(
            "✅ Prediction complete",
            extra={
                "equipment_id": request.equipment_id,
                "health_score": response.health.score,
                "inference_time_ms": response.inference_time_ms,
            },
        )

        return response

    def _preprocess_minimal(
        self, request: MinimalInferenceRequest, topology: GraphTopology
    ) -> Data:
        """Preprocess minimal request."""
        import pandas as pd

        try:
            if not request.sensor_readings:
                raise GraphBuildError("No sensor readings")

            # Convert to DataFrame
            sensor_records = []
            for component_id, readings in request.sensor_readings.items():
                readings_dict = (
                    readings.model_dump() if hasattr(readings, "model_dump") else readings
                )
                for sensor_name, value in readings_dict.items():
                    if value is None:
                        continue
                    sensor_records.append(
                        {
                            "component_id": component_id,
                            "sensor_name": sensor_name,
                            "value": float(value),
                            "timestamp": request.timestamp,
                        }
                    )

            if not sensor_records:
                raise GraphBuildError("No valid readings")

            sensor_df = pd.DataFrame(sensor_records)

            # Build
            graph = self.graph_builder.build_graph(
                sensor_data=sensor_df,
                topology=topology,
                sensor_readings=request.sensor_readings,
                current_time=request.timestamp,
            )

            return graph

        except Exception as e:
            raise GraphBuildError(f"Preprocess failed: {e}") from e

    def _inference_single(self, graph: Data, model_version: str) -> tuple:
        """Single inference."""
        try:
            if self.model_registry:
                model = self.model_registry.get_model(model_version)
            else:
                model = self.model

            device = next(model.parameters()).device
            graph = graph.to(device)
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)

            with torch.inference_mode():
                health, degradation, anomaly = model(
                    x=graph.x,
                    edge_index=graph.edge_index,
                    edge_attr=graph.edge_attr,
                    batch=batch,
                )

            return health, degradation, anomaly

        except Exception as e:
            raise InferenceError(f"Single inference failed: {e}") from e

    def _postprocess(
        self,
        equipment_id: str,
        health: torch.Tensor,
        degradation: torch.Tensor,
        anomaly: torch.Tensor,
        inference_time: float,
    ) -> PredictionResponse:
        """Postprocess outputs."""
        health_score = float(health.squeeze().cpu().item())
        degradation_rate = float(degradation.squeeze().cpu().item())
        anomaly_logits = anomaly.squeeze().cpu().numpy()
        anomaly_probs = torch.sigmoid(torch.from_numpy(anomaly_logits)).numpy()

        anomaly_types = [
            "pressure_drop",
            "overheating",
            "cavitation",
            "leakage",
            "vibration_anomaly",
            "flow_restriction",
            "contamination",
            "seal_degradation",
            "valve_stiction",
        ]

        anomaly_predictions = {
            atype: float(prob) for atype, prob in zip(anomaly_types, anomaly_probs, strict=True)
        }

        # Warnings
        if health_score < 0.3:
            logger.warning(f"⚠️  Low health: {health_score:.2f} for {equipment_id}")
        if degradation_rate > 0.8:
            logger.warning(f"⚠️  High degradation: {degradation_rate:.2f} for {equipment_id}")

        return PredictionResponse(
            equipment_id=equipment_id,
            health=HealthPrediction(score=health_score),
            degradation=DegradationPrediction(rate=degradation_rate),
            anomaly=AnomalyPrediction(predictions=anomaly_predictions),
            inference_time_ms=inference_time * 1000,
        )

    # ========================================================================
    # DYNAMIC BATCHING
    # ========================================================================

    async def _batch_processor_loop(self) -> None:
        """Background batch processor."""
        logger.info("🚀 Batch processor started")
        while not self._shutdown:
            try:
                batch_items = await self._collect_batch()
                if batch_items:
                    await self._process_batch(batch_items)
            except Exception:
                logger.error("❌ Batch processor error", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info("🛑 Batch processor stopped")

    async def _collect_batch(self) -> list[BatchItem]:
        """Collect batch items."""
        if self._batch_queue is None:
            return []

        batch_items: list[BatchItem] = []
        deadline = time.time() + (self.config.max_wait_ms / 1000.0)

        while len(batch_items) < self.config.batch_size:
            timeout = max(0.0, deadline - time.time())
            try:
                item = await asyncio.wait_for(self._batch_queue.get(), timeout=timeout)
                batch_items.append(item)
            except TimeoutError:
                break

        REQUEST_QUEUE_SIZE.set(self._batch_queue.qsize())
        return batch_items

    async def _process_batch(self, batch_items: list[BatchItem]) -> None:
        """Process batch."""
        if not batch_items:
            return

        logger.debug(f"Processing batch: {len(batch_items)} items")
        INFERENCE_BATCH_SIZE.observe(len(batch_items))
        self.config._total_batch_items += len(batch_items)

        for item in batch_items:
            try:
                response = await self._predict_minimal_impl(
                    item.request, item.model_version
                )
                item.future.set_result(response)
            except Exception as e:
                item.future.set_exception(e)

    # ========================================================================
    # CLEANUP
    # ========================================================================

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up InferenceEngine")
        self._shutdown = True

        # Stop batch processor
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._batch_processor_task

        # Clear queue
        if self._batch_queue:
            while not self._batch_queue.empty():
                try:
                    self._batch_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Unload models
        if self.model_registry:
            for version in self.model_registry.get_all_versions():
                with suppress(KeyError):
                    model = self.model_registry.get_model(version)
                    del model
        elif hasattr(self, "model"):
            del self.model

        # Clear GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("✅ Cleanup complete")

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()

    # ========================================================================
    # STATS
    # ========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get statistics."""
        total = self.config._total_inferences
        error_rate = self.config._total_errors / total if total > 0 else 0.0
        avg_time_ms = (
            (self.config._total_inference_time_s / total) * 1000 if total > 0 else 0.0
        )
        avg_batch_size = (
            self.config._total_batch_items / total if total > 0 else 0.0
        )

        stats = {
            # Config
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "enable_dynamic_batching": self.config.enable_dynamic_batching,
            # Stats
            "total_inferences": total,
            "total_errors": self.config._total_errors,
            "error_rate": error_rate,
            "avg_inference_time_ms": avg_time_ms,
            "avg_batch_size": avg_batch_size,
            # Cache
            "topology_cache": self.topology_cache.get_stats(),
        }

        # Multi-model
        if self.model_registry:
            stats["model_registry"] = self.model_registry.get_stats()

        # Queue
        if self._batch_queue:
            stats["queue_size"] = self._batch_queue.qsize()

        # GPU
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

        return stats
