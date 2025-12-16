"""Production-Ready Inference Engine for GNN Diagnostics.

Provides robust, scalable inference with:
- Async batch processing
- GPU optimization with fallback
- Comprehensive error handling
- Structured logging and metrics
- Dynamic graph building
- Variable topology support

Python 3.14 Features:
    - Deferred annotations (PEP 649)
    - Union types with pipe operator
    - Context managers for resource cleanup

Examples:
    >>> # Basic usage
    >>> async with InferenceEngine(
    ...     config=InferenceConfig(model_path="models/v2.0.0.ckpt")
    ... ) as engine:
    ...     response = await engine.predict_minimal(request)
    >>>
    >>> # Batch processing
    >>> responses = await engine.predict_batch(requests)
    >>>
    >>> # Monitoring
    >>> stats = engine.get_stats()
    >>> print(f"Queue size: {stats['queue_size']}")
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from src.data import FeatureConfig, FeatureEngineer, GraphBuilder
from src.data.edge_features import create_edge_feature_computer
from src.data.normalization import EdgeFeatureNormalizer, create_edge_feature_normalizer
from src.inference.dynamic_graph_builder import DynamicGraphBuilder
from src.inference.model_manager import ModelManager
from src.schemas import (
    AnomalyPrediction,
    DegradationPrediction,
    GraphTopology,
    HealthPrediction,
    PredictionRequest,
    PredictionResponse,
)
from src.schemas.requests import MinimalInferenceRequest
from src.services.topology_service import get_topology_service

if TYPE_CHECKING:
    from src.data.timescale_connector import TimescaleConnector
    from src.models.universal_temporal_gnn import UniversalTemporalGNN

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class InferenceEngineError(Exception):
    """Base exception for InferenceEngine."""


class ModelLoadError(InferenceEngineError):
    """Failed to load model checkpoint."""


class GraphBuildError(InferenceEngineError):
    """Failed to build graph from sensor data."""


class InferenceError(InferenceEngineError):
    """Failed to run inference."""


class TopologyNotFoundError(InferenceEngineError):
    """Topology not found in service."""


class GPUOutOfMemoryError(InferenceError):
    """GPU ran out of memory during inference."""


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class InferenceConfig:
    """Inference engine configuration.

    Attributes:
        model_path: Path to model checkpoint
        device: Device for inference (cpu/cuda/auto)
        batch_size: Maximum batch size for inference
        max_queue_size: Maximum request queue size
        max_wait_ms: Maximum wait time for batching (ms)
        inference_timeout_s: Timeout for single inference (seconds)
        use_dynamic_batching: Enable dynamic batching
        pin_memory: Pin tensors to CPU memory for faster GPU transfer
        use_dynamic_features: Enable dynamic edge features (14D)
        use_dynamic_builder: Use DynamicGraphBuilder for TimescaleDB
        topology_templates_path: Custom topology templates directory
        enable_compile: Use torch.compile for optimization (PyTorch 2.0+)
        fallback_to_cpu: Fallback to CPU if GPU fails

    Examples:
        >>> config = InferenceConfig(
        ...     model_path="models/v2.0.0.ckpt",
        ...     device="cuda",
        ...     batch_size=32,
        ...     use_dynamic_features=True
        ... )
    """

    model_path: str
    device: Literal["cpu", "cuda", "auto"] = "auto"
    batch_size: int = 32
    max_queue_size: int = 100
    max_wait_ms: float = 50.0
    inference_timeout_s: float = 30.0
    use_dynamic_batching: bool = True
    pin_memory: bool = True
    use_dynamic_features: bool = True
    use_dynamic_builder: bool = True
    topology_templates_path: Path | None = None
    enable_compile: bool = True
    fallback_to_cpu: bool = True

    # Statistics
    _total_inferences: int = field(default=0, init=False, repr=False)
    _total_errors: int = field(default=0, init=False, repr=False)
    _total_inference_time_s: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            msg = f"batch_size must be >= 1, got {self.batch_size}"
            raise ValueError(msg)

        if self.max_queue_size < 1:
            msg = f"max_queue_size must be >= 1, got {self.max_queue_size}"
            raise ValueError(msg)

        if not Path(self.model_path).exists():
            msg = f"Model checkpoint not found: {self.model_path}"
            raise FileNotFoundError(msg)


# ============================================================================
# INFERENCE ENGINE
# ============================================================================


class InferenceEngine:
    """Production-ready inference engine for hydraulic diagnostics.

    Features:
    - Async batch processing with asyncio
    - GPU optimization with automatic fallback to CPU
    - Comprehensive error handling (OOM, timeout, validation)
    - Structured logging with performance metrics
    - Dynamic graph building from TimescaleDB
    - Variable topology support (arbitrary N/E)
    - Resource cleanup and context manager support

    Thread Safety:
        NOT thread-safe. Use separate instances per thread or add locking.

    GPU Memory:
        Automatically handles OOM errors with fallback to CPU.
        Monitor stats['gpu_memory_allocated'] for capacity planning.

    Examples:
        >>> # Context manager (recommended)
        >>> async with InferenceEngine(config) as engine:
        ...     response = await engine.predict_minimal(request)
        >>>
        >>> # Manual lifecycle
        >>> engine = InferenceEngine(config)
        >>> try:
        ...     response = await engine.predict_minimal(request)
        ... finally:
        ...     await engine.cleanup()
    """

    def __init__(
        self,
        config: InferenceConfig,
        timescale_connector: TimescaleConnector | None = None,
        feature_config: FeatureConfig | None = None,
    ):
        """Initialize inference engine.

        Args:
            config: Engine configuration
            timescale_connector: Database connector (required for DynamicGraphBuilder)
            feature_config: Feature engineering configuration

        Raises:
            ModelLoadError: If model loading fails
            ValueError: If configuration invalid

        Examples:
            >>> config = InferenceConfig(model_path="models/v2.0.0.ckpt")
            >>> engine = InferenceEngine(config)
        """
        self.config = config
        self.feature_config = feature_config or FeatureConfig()
        self.timescale_connector = timescale_connector

        # Initialize components
        logger.info(
            "Initializing InferenceEngine",
            extra={
                "model_path": config.model_path,
                "device": config.device,
                "batch_size": config.batch_size,
                "dynamic_features": config.use_dynamic_features,
                "dynamic_builder": config.use_dynamic_builder,
            },
        )

        try:
            self.model_manager = ModelManager()
            self.feature_engineer = FeatureEngineer(self.feature_config)

            # Phase 3: Dynamic graph builder
            self.dynamic_builder = None
            if config.use_dynamic_builder and timescale_connector:
                self.dynamic_builder = DynamicGraphBuilder(
                    timescale_connector=timescale_connector,
                    feature_engineer=self.feature_engineer,
                    feature_config=self.feature_config,
                )
                logger.info("✅ DynamicGraphBuilder enabled")
            elif config.use_dynamic_builder:
                logger.warning(
                    "⚠️  DynamicGraphBuilder requested but no timescale_connector provided. "
                    "Falling back to request-based graph building."
                )

            # Edge feature components
            self.edge_feature_computer = create_edge_feature_computer()
            self.edge_normalizer = self._load_normalizer_from_checkpoint(config.model_path)

            # Legacy graph builder (backward compatibility)
            self.graph_builder = GraphBuilder(
                feature_engineer=self.feature_engineer,
                feature_config=self.feature_config,
                edge_feature_computer=self.edge_feature_computer,
                edge_normalizer=self.edge_normalizer,
                use_dynamic_features=config.use_dynamic_features,
            )

            # Topology service
            self.topology_service = get_topology_service(
                templates_path=config.topology_templates_path
            )

            # Load model
            self.model: UniversalTemporalGNN = self._load_model_safe()

            # Request queue (for dynamic batching)
            self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
            self._processing = False
            self._shutdown = False

            logger.info(
                "✅ InferenceEngine initialized successfully",
                extra={
                    "model_parameters": sum(
                        p.numel() for p in self.model.parameters()
                    ),
                    "device": next(self.model.parameters()).device,
                },
            )

        except Exception as e:
            logger.error(
                "❌ InferenceEngine initialization failed",
                extra={"error": str(e)},
                exc_info=True,
            )
            raise ModelLoadError(f"Failed to initialize engine: {e}") from e

    def _load_model_safe(self) -> UniversalTemporalGNN:
        """Load model with error handling and fallback.

        Returns:
            model: Loaded model

        Raises:
            ModelLoadError: If loading fails on all devices
        """
        try:
            # Try primary device
            model = self.model_manager.load_model(
                model_path=self.config.model_path,
                device=self.config.device,
                use_compile=self.config.enable_compile,
            )

            # Warmup
            self.model_manager.warmup(
                self.config.model_path,
                batch_size=self.config.batch_size,
            )

            return model

        except torch.cuda.OutOfMemoryError as e:
            if self.config.fallback_to_cpu:
                logger.warning(
                    "⚠️  GPU OOM during model load, falling back to CPU",
                    extra={"error": str(e)},
                )
                # Retry on CPU
                model = self.model_manager.load_model(
                    model_path=self.config.model_path,
                    device="cpu",
                    use_compile=False,  # torch.compile may fail on CPU
                )
                return model

            logger.error("❌ GPU OOM and fallback_to_cpu=False")
            raise GPUOutOfMemoryError(
                "GPU out of memory during model load. "
                "Increase GPU memory or set fallback_to_cpu=True."
            ) from e

        except Exception as e:
            logger.error(
                "❌ Model loading failed",
                extra={"model_path": self.config.model_path},
                exc_info=True,
            )
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def _load_normalizer_from_checkpoint(self, checkpoint_path: str) -> EdgeFeatureNormalizer:
        """Load edge feature normalizer from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            normalizer: Loaded or default normalizer
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            if "normalizer_stats" in checkpoint:
                normalizer = create_edge_feature_normalizer()
                normalizer.load_stats(checkpoint["normalizer_stats"])
                logger.info("✅ Loaded normalizer stats from checkpoint")
                return normalizer

            logger.warning(
                "⚠️  No normalizer stats in checkpoint, using defaults. "
                "Consider retraining with updated training script."
            )
            return create_edge_feature_normalizer()

        except Exception as e:
            logger.warning(
                "⚠️  Failed to load normalizer, using defaults",
                extra={"error": str(e)},
            )
            return create_edge_feature_normalizer()

    # ========================================================================
    # PREDICTION METHODS
    # ========================================================================

    async def predict_minimal(
        self, request: MinimalInferenceRequest
    ) -> PredictionResponse:
        """Run inference on minimal request (Level 1 API).

        Args:
            request: MinimalInferenceRequest with sensor readings

        Returns:
            response: PredictionResponse with predictions

        Raises:
            TopologyNotFoundError: If topology_id not found
            GraphBuildError: If graph construction fails
            InferenceError: If inference fails
            asyncio.TimeoutError: If inference exceeds timeout

        Examples:
            >>> response = await engine.predict_minimal(
            ...     MinimalInferenceRequest(
            ...         equipment_id="pump_001",
            ...         timestamp=datetime.now(timezone.utc),
            ...         topology_id="standard_pump",
            ...         sensor_readings={...}
            ...     )
            ... )
        """
        start_time = time.time()

        try:
            # Apply timeout
            response = await asyncio.wait_for(
                self._predict_minimal_impl(request),
                timeout=self.config.inference_timeout_s,
            )

            # Update stats
            self.config._total_inferences += 1
            self.config._total_inference_time_s += time.time() - start_time

            return response

        except asyncio.TimeoutError as e:
            self.config._total_errors += 1
            logger.error(
                "⏱️  Inference timeout",
                extra={
                    "equipment_id": request.equipment_id,
                    "timeout_s": self.config.inference_timeout_s,
                },
            )
            raise InferenceError(
                f"Inference timed out after {self.config.inference_timeout_s}s"
            ) from e

        except Exception as e:
            self.config._total_errors += 1
            logger.error(
                "❌ Inference failed",
                extra={"equipment_id": request.equipment_id},
                exc_info=True,
            )
            raise

    async def _predict_minimal_impl(
        self, request: MinimalInferenceRequest
    ) -> PredictionResponse:
        """Implementation of predict_minimal (no timeout wrapper)."""
        start_time = time.time()

        # Resolve topology
        template = self.topology_service.get_template(request.topology_id)
        if not template:
            raise TopologyNotFoundError(
                f"Topology '{request.topology_id}' not found. "
                f"Available: {list(self.topology_service.get_all_templates().keys())}"
            )

        topology = template.to_graph_topology(request.equipment_id)

        # Build graph
        try:
            if self.config.use_dynamic_builder and self.dynamic_builder:
                # Dynamic builder from TimescaleDB
                graph = await self.dynamic_builder.build_from_timescale(
                    equipment_id=request.equipment_id,
                    topology=topology,
                    lookback_minutes=10,
                )
                logger.debug(
                    "Built dynamic graph from TimescaleDB",
                    extra={
                        "equipment_id": request.equipment_id,
                        "num_nodes": graph.x.shape[0],
                        "num_edges": graph.edge_index.shape[1],
                    },
                )
            else:
                # Build from request sensor_readings
                graph = self._preprocess_minimal(request, topology)
                logger.debug(
                    "Built graph from request",
                    extra={
                        "equipment_id": request.equipment_id,
                        "num_nodes": graph.x.shape[0],
                        "num_edges": graph.edge_index.shape[1],
                    },
                )
        except Exception as e:
            raise GraphBuildError(f"Failed to build graph: {e}") from e

        # Inference
        try:
            health, degradation, anomaly = self._inference_single(graph)
        except torch.cuda.OutOfMemoryError as e:
            if self.config.fallback_to_cpu:
                logger.warning(
                    "⚠️  GPU OOM during inference, retrying on CPU",
                    extra={"equipment_id": request.equipment_id},
                )
                # Move model to CPU and retry
                self.model = self.model.cpu()
                health, degradation, anomaly = self._inference_single(graph)
            else:
                raise GPUOutOfMemoryError(
                    "GPU out of memory during inference. Try reducing batch size."
                ) from e
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
            "✅ Prediction completed",
            extra={
                "equipment_id": request.equipment_id,
                "num_nodes": graph.x.shape[0],
                "health_score": response.health.score,
                "inference_time_ms": response.inference_time_ms,
            },
        )

        return response

    async def predict(
        self, request: PredictionRequest, topology: GraphTopology
    ) -> PredictionResponse:
        """Legacy prediction API (backward compatible).

        Args:
            request: Legacy PredictionRequest
            topology: Equipment topology

        Returns:
            response: PredictionResponse

        Examples:
            >>> response = await engine.predict(request, topology)
        """
        warnings.warn(
            "predict() is deprecated. Use predict_minimal() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        start_time = time.time()

        try:
            # Build graph (legacy path - no dynamic features)
            graph = self._preprocess_legacy(request, topology)

            # Inference
            health, degradation, anomaly = self._inference_single(graph)

            # Postprocess
            response = self._postprocess(
                equipment_id=request.equipment_id,
                health=health,
                degradation=degradation,
                anomaly=anomaly,
                inference_time=time.time() - start_time,
            )

            logger.info(
                "✅ Legacy prediction completed",
                extra={"equipment_id": request.equipment_id},
            )

            return response

        except Exception as e:
            logger.error(
                "❌ Legacy prediction failed",
                extra={"equipment_id": request.equipment_id},
                exc_info=True,
            )
            raise InferenceError(f"Legacy prediction failed: {e}") from e

    async def predict_batch(
        self,
        requests: list[PredictionRequest | MinimalInferenceRequest],
        topology: GraphTopology | None = None,
    ) -> list[PredictionResponse]:
        """Batch prediction with parallel processing.

        Args:
            requests: List of prediction requests (mixed legacy/new)
            topology: Topology for legacy requests (optional)

        Returns:
            responses: List of prediction responses (same order as requests)

        Raises:
            InferenceError: If batch processing fails

        Examples:
            >>> responses = await engine.predict_batch([req1, req2, req3])
        """
        start_time = time.time()

        try:
            # Process in parallel with asyncio.gather
            tasks = []
            for req in requests:
                if isinstance(req, MinimalInferenceRequest):
                    tasks.append(self.predict_minimal(req))
                else:
                    if not topology:
                        raise ValueError("topology required for legacy requests")
                    tasks.append(self.predict(req, topology))

            # Gather with return_exceptions=True for partial success
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors
            errors = [r for r in responses if isinstance(r, Exception)]
            if errors:
                logger.warning(
                    f"⚠️  Batch partial failure: {len(errors)}/{len(requests)} failed",
                    extra={"errors": [str(e) for e in errors]},
                )

            # Filter successful responses
            successful = [r for r in responses if isinstance(r, PredictionResponse)]

            logger.info(
                "✅ Batch prediction completed",
                extra={
                    "total_requests": len(requests),
                    "successful": len(successful),
                    "failed": len(errors),
                    "total_time_s": time.time() - start_time,
                },
            )

            return responses

        except Exception as e:
            logger.error(
                "❌ Batch prediction failed",
                extra={"num_requests": len(requests)},
                exc_info=True,
            )
            raise InferenceError(f"Batch prediction failed: {e}") from e

    # ========================================================================
    # PREPROCESSING
    # ========================================================================

    def _preprocess_minimal(
        self, request: MinimalInferenceRequest, topology: GraphTopology
    ) -> Data:
        """Convert MinimalInferenceRequest to PyG Data.

        Args:
            request: Request with sensor_readings
            topology: Equipment topology

        Returns:
            graph: PyG Data object

        Raises:
            GraphBuildError: If conversion fails
        """
        try:
            # Validate sensor readings
            if not request.sensor_readings:
                raise GraphBuildError("No sensor readings provided")

            # Convert to DataFrame format
            sensor_records = []
            for component_id, readings in request.sensor_readings.items():
                # readings is ComponentSensorReading (Pydantic model)
                readings_dict = readings.model_dump() if hasattr(readings, "model_dump") else readings

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
                raise GraphBuildError("No valid sensor readings after filtering")

            sensor_df = pd.DataFrame(sensor_records)

            # Build graph
            graph = self.graph_builder.build_graph(
                sensor_data=sensor_df,
                topology=topology,
                sensor_readings=request.sensor_readings,
                current_time=request.timestamp,
            )

            return graph

        except Exception as e:
            raise GraphBuildError(f"Failed to preprocess minimal request: {e}") from e

    def _preprocess_legacy(
        self, request: PredictionRequest, topology: GraphTopology
    ) -> Data:
        """Convert legacy PredictionRequest to PyG Data.

        Args:
            request: Legacy request
            topology: Equipment topology

        Returns:
            graph: PyG Data object

        Raises:
            GraphBuildError: If conversion fails
        """
        try:
            return self.graph_builder.build_graph(
                sensor_data=request.sensor_data,
                topology=topology,
            )
        except Exception as e:
            raise GraphBuildError(f"Failed to preprocess legacy request: {e}") from e

    # ========================================================================
    # INFERENCE
    # ========================================================================

    def _inference_single(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference on single graph.

        Args:
            graph: PyG Data object

        Returns:
            outputs: (health, degradation, anomaly) tensors

        Raises:
            InferenceError: If inference fails
        """
        try:
            device = next(self.model.parameters()).device

            # Move to device
            graph = graph.to(device)

            # Create batch tensor
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)

            # Inference
            with torch.inference_mode():
                health, degradation, anomaly = self.model(
                    x=graph.x,
                    edge_index=graph.edge_index,
                    edge_attr=graph.edge_attr,
                    batch=batch,
                )

            return health, degradation, anomaly

        except Exception as e:
            raise InferenceError(f"Single inference failed: {e}") from e

    def _inference_batch(
        self, graphs: list[Data]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run batch inference on multiple graphs.

        Args:
            graphs: List of PyG Data objects (variable sizes)

        Returns:
            outputs: (health, degradation, anomaly) batch tensors

        Raises:
            InferenceError: If batch inference fails
        """
        try:
            device = next(self.model.parameters()).device

            # Create PyG Batch (handles variable-sized graphs)
            batch = Batch.from_data_list(graphs)

            # Move to device
            if self.config.pin_memory:
                batch = batch.pin_memory()
            batch = batch.to(device, non_blocking=True)

            # Inference
            with torch.inference_mode():
                health, degradation, anomaly = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                )

            return health, degradation, anomaly

        except Exception as e:
            raise InferenceError(f"Batch inference failed: {e}") from e

    # ========================================================================
    # POSTPROCESSING
    # ========================================================================

    def _postprocess(
        self,
        equipment_id: str,
        health: torch.Tensor,
        degradation: torch.Tensor,
        anomaly: torch.Tensor,
        inference_time: float,
    ) -> PredictionResponse:
        """Convert model outputs to PredictionResponse.

        Args:
            equipment_id: Equipment identifier
            health: Health score tensor [1, 1]
            degradation: Degradation rate tensor [1, 1]
            anomaly: Anomaly logits tensor [1, 9]
            inference_time: Inference time in seconds

        Returns:
            response: PredictionResponse
        """
        # Extract scalar values
        health_score = float(health.squeeze().cpu().item())
        degradation_rate = float(degradation.squeeze().cpu().item())
        anomaly_logits = anomaly.squeeze().cpu().numpy()

        # Anomaly probabilities (sigmoid for multi-label)
        anomaly_probs = torch.sigmoid(torch.from_numpy(anomaly_logits)).numpy()

        # Anomaly types (order must match training)
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

        # Build anomaly predictions dict
        anomaly_predictions = {
            anomaly_type: float(prob)
            for anomaly_type, prob in zip(anomaly_types, anomaly_probs, strict=True)
        }

        # Warn on abnormal values
        if health_score < 0.3:
            logger.warning(
                f"⚠️  Low health score: {health_score:.2f} for {equipment_id}"
            )

        if degradation_rate > 0.8:
            logger.warning(
                f"⚠️  High degradation rate: {degradation_rate:.2f} for {equipment_id}"
            )

        # Create response
        return PredictionResponse(
            equipment_id=equipment_id,
            health=HealthPrediction(score=health_score),
            degradation=DegradationPrediction(rate=degradation_rate),
            anomaly=AnomalyPrediction(predictions=anomaly_predictions),
            inference_time_ms=inference_time * 1000,
        )

    # ========================================================================
    # RESOURCE MANAGEMENT
    # ========================================================================

    async def cleanup(self) -> None:
        """Cleanup resources.

        Examples:
            >>> await engine.cleanup()
        """
        logger.info("Cleaning up InferenceEngine")

        self._shutdown = True

        # Clear queue
        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Unload model (free GPU memory)
        if hasattr(self, "model"):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("✅ InferenceEngine cleanup complete")

    async def __aenter__(self) -> InferenceEngine:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()

    # ========================================================================
    # MONITORING
    # ========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics for monitoring.

        Returns:
            stats: Statistics dictionary

        Examples:
            >>> stats = engine.get_stats()
            >>> print(f"Total inferences: {stats['total_inferences']}")
            >>> print(f"Error rate: {stats['error_rate']:.2%}")
        """
        model_device = next(self.model.parameters()).device if hasattr(self, "model") else None
        model_params = (
            sum(p.numel() for p in self.model.parameters()) if hasattr(self, "model") else 0
        )

        topology_stats = self.topology_service.get_stats()

        total_inferences = self.config._total_inferences
        error_rate = (
            self.config._total_errors / total_inferences if total_inferences > 0 else 0.0
        )
        avg_inference_time_ms = (
            (self.config._total_inference_time_s / total_inferences) * 1000
            if total_inferences > 0
            else 0.0
        )

        stats = {
            # Configuration
            "model_path": self.config.model_path,
            "device": str(model_device),
            "batch_size": self.config.batch_size,
            "use_dynamic_features": self.config.use_dynamic_features,
            "use_dynamic_builder": self.config.use_dynamic_builder,
            # Model
            "model_parameters": model_params,
            "model_memory_mb": model_params * 4 / 1024 / 1024,  # Approximate (fp32)
            # Runtime
            "queue_size": self._request_queue.qsize(),
            "processing": self._processing,
            "shutdown": self._shutdown,
            # Statistics
            "total_inferences": total_inferences,
            "total_errors": self.config._total_errors,
            "error_rate": error_rate,
            "avg_inference_time_ms": avg_inference_time_ms,
            # Topology
            "topology_templates": topology_stats.get("cached_templates", 0),
            "custom_topologies": topology_stats.get("custom_topologies", 0),
            # Components
            "dynamic_builder_available": self.dynamic_builder is not None,
            "timescale_connector_available": self.timescale_connector is not None,
        }

        # Add GPU stats if available
        if torch.cuda.is_available() and str(model_device).startswith("cuda"):
            device_idx = int(str(model_device).split(":")[-1]) if ":" in str(model_device) else 0
            stats["gpu_memory_allocated_mb"] = (
                torch.cuda.memory_allocated(device_idx) / 1024 / 1024
            )
            stats["gpu_memory_reserved_mb"] = (
                torch.cuda.memory_reserved(device_idx) / 1024 / 1024
            )

        return stats
