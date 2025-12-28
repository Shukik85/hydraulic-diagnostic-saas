"""Production-Ready Inference Engine.

Enterprise-grade inference engine featuring:
- Dynamic batching with automatic flushing
- Multi-model support with A/B testing
- Prometheus metrics for observability
- LRU topology caching with TTL
- Comprehensive tensor validation
- Request ID tracking

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
    >>> from src.inference import InferenceEngine, InferenceConfig
    >>> config = InferenceConfig(model_path="models/v2.0.0.ckpt")
    >>> async with InferenceEngine(config) as engine:
    ...     response = await engine.predict_minimal(request)
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import sys
import time
import warnings
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from configs.config import inference_config
from torch_geometric.data import Data

from src.schemas import (
    AnomalyPrediction,
    DegradationPrediction,
    GraphTopology,
    HealthPrediction,
    PredictionResponse,
)
from src.schemas.requests import HybridInferenceRequest, MinimalInferenceRequest

# Import from new modular architecture
from .batching import BatchItem
from .cache import TopologyCache
from .exceptions import (
    GPUOutOfMemoryError,
    GraphBuildError,
    InferenceError,
    ModelLoadError,
    TopologyNotFoundError,
)
from .metrics import (
    INFERENCE_BATCH_SIZE,
    INFERENCE_DURATION_SECONDS,
    INFERENCE_ERRORS_TOTAL,
    INFERENCE_REQUESTS_TOTAL,
    REQUEST_QUEUE_SIZE,
)
from .model_registry import ModelConfig, ModelRegistry
from .request_context import ensure_request_id, get_request_id
from .validation import TensorValidator

if TYPE_CHECKING:
    from src.data import FeatureConfig
    from src.data.timescale_connector import TimescaleConnector

logger = logging.getLogger(__name__)

# Check Python version for TaskGroup support
PYTHON_311_PLUS = sys.version_info >= (3, 11)


# ============================================================================
# INFERENCE CONFIG
# ============================================================================


@dataclass
class InferenceConfig:
    """Inference engine configuration.

    Examples:
        >>> # Single model
        >>> config = InferenceConfig(
        ...     model_path="models/v2.ckpt",
        ...     enable_dynamic_batching=True
        ... )
        >>> # Multi-model A/B testing
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
        """Validate configuration."""
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
        if (
            self.model_path
            and not self.model_versions
            and not Path(self.model_path).exists()
        ):
            msg = f"Model not found: {self.model_path}"
            raise FileNotFoundError(msg)

        # Multi-model traffic validation
        if self.model_versions:
            total_traffic = sum(
                c.traffic for c in self.model_versions.values() if c.enabled
            )
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
    - Dynamic batching with race condition protection
    - Multi-model A/B testing with traffic splitting
    - Prometheus metrics for observability
    - LRU+TTL topology caching
    - Comprehensive tensor validation
    - Request ID propagation

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
            config: Engine configuration
            timescale_connector: Database connector (optional)
            feature_config: Feature configuration (optional)

        Raises:
            ModelLoadError: If model loading fails
        """
        self.config = config
        self._shutdown = False

        # Deferred imports to avoid circular dependencies
        from src.data import FeatureConfig, FeatureEngineer, GraphBuilder
        from src.data.edge_features import create_edge_feature_computer
        from src.data.graph_builder_v2 import GraphBuilderV2
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
            # Core components
            self.model_manager = ModelManager()
            self.feature_engineer = FeatureEngineer(self.feature_config)

            # Dynamic graph builder
            self.dynamic_builder = None
            if config.use_dynamic_builder and timescale_connector:
                self.dynamic_builder = DynamicGraphBuilder(
                    timescale_connector=timescale_connector,
                    feature_engineer=self.feature_engineer,
                    feature_config=self.feature_config,
                )
                logger.info("DynamicGraphBuilder enabled")
            elif config.use_dynamic_builder:
                logger.warning("DynamicGraphBuilder requested but no connector")

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

            # Graph builder v2 (edge-centric)
            self.graph_builder_v2 = GraphBuilderV2(
                feature_engineer=self.feature_engineer,
                feature_config=self.feature_config,
                edge_feature_computer=self.edge_feature_computer,
                edge_normalizer=self.edge_normalizer,
                use_edge_timeseries=False,
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
            self.tensor_validator_v2 = None
            if config.validate_tensors:
                self.tensor_validator = TensorValidator(
                    expected_node_dim=self.feature_config.node_feature_dim,
                    expected_edge_dim=14,
                )
                self.tensor_validator_v2 = TensorValidator(
                    expected_node_dim=29,
                    expected_edge_dim=self.feature_config.edge_in_dim,
                )

            # Multi-model registry
            self.model_registry = None
            if config.model_versions:
                self.model_registry = ModelRegistry()
                for version, model_config in config.model_versions.items():
                    self.model_registry.register(version, model_config)
                    model = self._load_model_safe(model_config.path, version)
                    self.model_registry.set_loaded_model(version, model)
            else:
                self.model = self._load_model_safe(config.model_path, "default")

            # Dynamic batching
            self._batch_queue = None
            self._batch_processor_task = None
            if config.enable_dynamic_batching:
                self._batch_queue = asyncio.Queue(maxsize=config.max_queue_size)
                self._batch_processor_task = asyncio.create_task(
                    self._batch_processor_loop()
                )
                logger.info("Dynamic batching enabled")

            logger.info("InferenceEngine initialized")

        except Exception as e:
            logger.error("Initialization failed", exc_info=True)
            raise ModelLoadError(f"Init failed: {e}") from e

    def _load_normalizer(self, checkpoint_path: str) -> Any:
        """Load normalizer from checkpoint.

        Security: Uses weights_only=True to prevent RCE.
        """
        from src.data.normalization import create_edge_feature_normalizer

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if "normalizer_stats" in checkpoint:
                normalizer = create_edge_feature_normalizer()
                normalizer.load_stats(checkpoint["normalizer_stats"])
                logger.info("Loaded normalizer from checkpoint")
                return normalizer
            logger.warning("No normalizer stats in checkpoint, using defaults")
            return create_edge_feature_normalizer()
        except (RuntimeError, pickle.UnpicklingError) as e:
            logger.error(
                f"Failed to load checkpoint with weights_only=True: {checkpoint_path}. "
                "Ensure checkpoint was saved with PyTorch 2.0+",
                exc_info=True,
            )
            raise ModelLoadError(
                f"Checkpoint incompatible with weights_only=True. "
                f"Re-save with PyTorch 2.0+: {e}"
            ) from e
        except Exception:
            logger.warning("Normalizer load failed, using defaults", exc_info=True)
            return create_edge_feature_normalizer()

    def _load_model_safe(self, model_path: str, version: str) -> Any:
        """Load model with error handling and CPU fallback."""
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
                logger.warning(f"GPU OOM for {version}, falling back to CPU")
                model = self.model_manager.load_model(
                    model_path=model_path,
                    device="cpu",
                    use_compile=False,
                )
                return model
            logger.error("GPU OOM and fallback_to_cpu=False")
            raise GPUOutOfMemoryError(f"GPU OOM loading {version}") from e
        except Exception as e:
            logger.error(f"Model load failed: {version}", exc_info=True)
            raise ModelLoadError(f"Load failed: {e}") from e

    # ========================================================================
    # PREDICTION
    # ========================================================================

    async def predict_minimal(
        self, request: MinimalInferenceRequest
    ) -> PredictionResponse:
        """Predict with minimal request.

        Args:
            request: Inference request

        Returns:
            Prediction response

        Raises:
            InferenceError: If prediction fails
        """
        start_time = time.time()

        # Ensure request ID for tracking
        request_id = ensure_request_id()

        # Select model version
        model_version = "default"
        if self.model_registry:
            model_version = self.model_registry.select_model(request.equipment_id)

        try:
            if self.config.enable_dynamic_batching and self._batch_queue is not None:
                future: asyncio.Future[PredictionResponse] = asyncio.Future()
                item = BatchItem(
                    request=request,
                    future=future,
                    model_version=model_version,
                    request_id=request_id,
                )

                try:
                    await asyncio.wait_for(
                        self._batch_queue.put(item),
                        timeout=inference_config.queue_put_timeout_s,
                    )
                    REQUEST_QUEUE_SIZE.set(self._batch_queue.qsize())
                except TimeoutError as e:
                    INFERENCE_ERRORS_TOTAL.labels(error_type="queue_timeout").inc()
                    logger.error(
                        "Queue full, cannot accept request",
                        extra={
                            "equipment_id": request.equipment_id,
                            "request_id": request_id,
                            "queue_size": self._batch_queue.qsize(),
                        },
                    )
                    raise InferenceError("Request queue full, please retry") from e

                try:
                    response = await asyncio.wait_for(
                        future,
                        timeout=self.config.inference_timeout_s,
                    )
                except TimeoutError as e:
                    if not future.done():
                        future.cancel()
                    logger.error(
                        "Inference timeout",
                        extra={
                            "equipment_id": request.equipment_id,
                            "request_id": request_id,
                            "timeout_s": self.config.inference_timeout_s,
                        },
                    )
                    raise InferenceError(
                        f"Inference timeout after {self.config.inference_timeout_s}s"
                    ) from e
            else:
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
            logger.error(
                f"Timeout: {request.equipment_id}",
                extra={"request_id": request_id},
            )
            raise InferenceError(
                f"Timeout after {self.config.inference_timeout_s}s"
            ) from e
        except Exception as e:
            self.config._total_errors += 1
            error_type = type(e).__name__
            INFERENCE_ERRORS_TOTAL.labels(error_type=error_type).inc()
            INFERENCE_REQUESTS_TOTAL.labels(
                model_version=model_version,
                status="error",
            ).inc()
            logger.error(
                f"Inference failed: {request.equipment_id}",
                extra={"request_id": request_id},
                exc_info=True,
            )
            raise

    async def predict_hybrid(self, request: HybridInferenceRequest) -> PredictionResponse:
        """Predict using edge-centric HybridInferenceRequest.

        This is the preferred production inference path for Day 6.

        Note:
            Dynamic batching currently supports MinimalInferenceRequest only.
            If dynamic batching is enabled, this method still runs per-request.
        """
        start_time = time.time()

        request_id = ensure_request_id()

        # Select model version
        model_version = "default"
        if self.model_registry:
            model_version = self.model_registry.select_model(request.equipment_id)

        try:
            if self.config.enable_dynamic_batching:
                logger.warning(
                    "Dynamic batching not supported for HybridInferenceRequest yet; running single request",
                    extra={
                        "equipment_id": request.equipment_id,
                        "request_id": request_id,
                    },
                )

            response = await asyncio.wait_for(
                self._predict_hybrid_impl(request, model_version),
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
            logger.error(
                f"Timeout: {request.equipment_id}",
                extra={"request_id": request_id},
            )
            raise InferenceError(
                f"Timeout after {self.config.inference_timeout_s}s"
            ) from e
        except Exception as e:
            self.config._total_errors += 1
            error_type = type(e).__name__
            INFERENCE_ERRORS_TOTAL.labels(error_type=error_type).inc()
            INFERENCE_REQUESTS_TOTAL.labels(
                model_version=model_version,
                status="error",
            ).inc()
            logger.error(
                f"Inference failed: {request.equipment_id}",
                extra={"request_id": request_id},
                exc_info=True,
            )
            raise

    async def _predict_hybrid_impl(
        self, request: HybridInferenceRequest, model_version: str
    ) -> PredictionResponse:
        """Core hybrid prediction implementation."""
        start_time = time.time()

        # Get topology config (cached).
        # Keep a separate namespace to avoid mixing with legacy GraphTopology objects.
        topology_cache_key = f"v2::{request.topology_id}"
        topology_config = self.topology_cache.get(topology_cache_key)

        if topology_config is None:
            topology_config = self.topology_service.get_config(
                template_id=request.topology_id,
                topology_id=request.topology_id,
            )
            if topology_config is None:
                raise TopologyNotFoundError(
                    f"Topology '{request.topology_id}' not found. "
                    f"Available: {list(self.topology_service.get_all_templates().keys())}"
                )
            self.topology_cache.put(topology_cache_key, topology_config)

        # Build graph (edge-centric)
        try:
            # Day 6: edge_history integration will be connected to TimescaleDB later.
            graph = self.graph_builder_v2.build_graph_hybrid(
                request=request,
                topology=topology_config,
                edge_history=None,
            )
        except Exception as e:
            raise GraphBuildError(f"Graph build failed: {e}") from e

        # Validate
        if self.tensor_validator_v2:
            self.tensor_validator_v2.validate_graph(graph)

        # Inference
        try:
            with INFERENCE_DURATION_SECONDS.labels(model_version=model_version).time():
                health, degradation, anomaly = self._inference_single(
                    graph, model_version
                )
        except torch.cuda.OutOfMemoryError as e:
            if self.config.fallback_to_cpu:
                logger.warning(
                    "GPU OOM detected, falling back to CPU",
                    extra={"equipment_id": request.equipment_id},
                )
                if self.model_registry:
                    model = self.model_registry.get_model(model_version)
                    model.to("cpu")
                else:
                    self.model.to("cpu")
                graph = graph.to("cpu")
                health, degradation, anomaly = self._inference_single(
                    graph, model_version
                )
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
            "Prediction complete",
            extra={
                "equipment_id": request.equipment_id,
                "request_id": get_request_id(),
                "health_score": response.health.score,
                "inference_time_ms": response.inference_time_ms,
            },
        )

        return response

    async def _predict_minimal_impl(
        self, request: MinimalInferenceRequest, model_version: str
    ) -> PredictionResponse:
        """Core prediction implementation."""
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
                logger.warning(
                    "GPU OOM detected, falling back to CPU",
                    extra={"equipment_id": request.equipment_id},
                )
                if self.model_registry:
                    model = self.model_registry.get_model(model_version)
                    model.to("cpu")
                else:
                    self.model.to("cpu")
                graph = graph.to("cpu")
                health, degradation, anomaly = self._inference_single(
                    graph, model_version
                )
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
            "Prediction complete",
            extra={
                "equipment_id": request.equipment_id,
                "request_id": get_request_id(),
                "health_score": response.health.score,
                "inference_time_ms": response.inference_time_ms,
            },
        )

        return response

    def _preprocess_minimal(
        self, request: MinimalInferenceRequest, topology: GraphTopology
    ) -> Data:
        """Preprocess request into graph.

        PERFORMANCE FIX: Uses polars instead of pandas.
        Polars is async-friendly and doesn't block GIL.
        """
        try:
            # FIX 1: Use polars instead of pandas
            import polars as pl

            if not request.sensor_readings:
                raise GraphBuildError("No sensor readings")

            sensor_records = []
            for component_id, readings in request.sensor_readings.items():
                readings_dict = (
                    readings.model_dump()
                    if hasattr(readings, "model_dump")
                    else readings
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

            # Use polars DataFrame (async-friendly, faster)
            sensor_df = pl.DataFrame(sensor_records)

            # Convert to pandas for compatibility with graph_builder
            # TODO: Update graph_builder to support polars natively
            sensor_df_pd = sensor_df.to_pandas()

            graph = self.graph_builder.build_graph(
                sensor_data=sensor_df_pd,
                topology=topology,
                sensor_readings=request.sensor_readings,
                current_time=request.timestamp,
            )

            return graph

        except Exception as e:
            raise GraphBuildError(f"Preprocess failed: {e}") from e

    def _inference_single(self, graph: Data, model_version: str) -> tuple:
        """Single graph inference."""
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
        """Postprocess model outputs."""
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
            atype: float(prob)
            for atype, prob in zip(anomaly_types, anomaly_probs, strict=True)
        }

        if health_score < 0.3:
            logger.warning(f"Low health: {health_score:.2f} for {equipment_id}")
        if degradation_rate > 0.8:
            logger.warning(
                f"High degradation: {degradation_rate:.2f} for {equipment_id}"
            )

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
        """Background batch processor.

        IMPROVEMENT: Uses TaskGroup (Python 3.11+) for better task management.
        Falls back to traditional approach for Python 3.10.
        """
        logger.info("Batch processor started")

        # FIX 3: Use TaskGroup if available (Python 3.11+)
        if PYTHON_311_PLUS:
            await self._batch_processor_loop_taskgroup()
        else:
            await self._batch_processor_loop_legacy()

    async def _batch_processor_loop_taskgroup(self) -> None:
        """Batch processor with TaskGroup (Python 3.11+)."""
        while not self._shutdown:
            try:
                async with asyncio.TaskGroup() as tg:
                    batch_items = await self._collect_batch()
                    if batch_items:
                        tg.create_task(self._process_batch(batch_items))
            except* Exception as eg:
                for exc in eg.exceptions:
                    logger.error("Batch processor error", exc_info=exc)
                await asyncio.sleep(0.1)

        logger.info("Batch processor stopped")

    async def _batch_processor_loop_legacy(self) -> None:
        """Batch processor legacy (Python 3.10)."""
        while not self._shutdown:
            try:
                batch_items = await self._collect_batch()
                if batch_items:
                    await self._process_batch(batch_items)
            except Exception:
                logger.error("Batch processor error", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info("Batch processor stopped")

    async def _collect_batch(self) -> list[BatchItem]:
        """Collect batch items with timeout."""
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
        """Process batch with race condition guards."""
        if not batch_items:
            return

        logger.debug(f"Processing batch: {len(batch_items)} items")
        INFERENCE_BATCH_SIZE.observe(len(batch_items))
        self.config._total_batch_items += len(batch_items)

        for item in batch_items:
            if item.future.done():
                logger.warning(
                    "Future already completed, skipping",
                    extra={
                        "equipment_id": item.request.equipment_id,
                        "request_id": item.request_id,
                    },
                )
                continue

            try:
                response = await self._predict_minimal_impl(
                    item.request, item.model_version
                )
                if not item.future.done():
                    item.future.set_result(response)
                else:
                    logger.warning(
                        "Future completed during processing",
                        extra={
                            "equipment_id": item.request.equipment_id,
                            "request_id": item.request_id,
                        },
                    )
            except Exception as e:
                if not item.future.done():
                    item.future.set_exception(e)
                else:
                    logger.error(
                        "Cannot set exception, future already done",
                        extra={
                            "equipment_id": item.request.equipment_id,
                            "request_id": item.request_id,
                            "error": str(e),
                        },
                        exc_info=True,
                    )

    # ========================================================================
    # CLEANUP
    # ========================================================================

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up InferenceEngine")
        self._shutdown = True

        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._batch_processor_task

        if self._batch_queue:
            while not self._batch_queue.empty():
                try:
                    self._batch_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        if self.model_registry:
            for version in self.model_registry.get_all_versions():
                with suppress(KeyError):
                    model = self.model_registry.get_model(version)
                    del model
            self.model_registry._loaded_models.clear()
        elif hasattr(self, "model"):
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cleanup complete")

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
        """Get engine statistics."""
        total = self.config._total_inferences
        error_rate = self.config._total_errors / total if total > 0 else 0.0
        avg_time_ms = (
            (self.config._total_inference_time_s / total) * 1000 if total > 0 else 0.0
        )
        avg_batch_size = self.config._total_batch_items / total if total > 0 else 0.0

        stats = {
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "enable_dynamic_batching": self.config.enable_dynamic_batching,
            "total_inferences": total,
            "total_errors": self.config._total_errors,
            "error_rate": error_rate,
            "avg_inference_time_ms": avg_time_ms,
            "avg_batch_size": avg_batch_size,
            "topology_cache": self.topology_cache.get_stats(),
        }

        if self.model_registry:
            stats["model_registry"] = self.model_registry.get_stats()

        if self._batch_queue:
            stats["queue_size"] = self._batch_queue.qsize()  # FIX: Added missing `]`

        if torch.cuda.is_available():
            stats["gpu_memory_allocated_mb"] = (
                torch.cuda.memory_allocated() / 1024 / 1024
            )
            stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

        return stats
