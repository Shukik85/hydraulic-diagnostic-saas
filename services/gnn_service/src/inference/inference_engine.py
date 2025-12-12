"""Inference Engine for production predictions.

Features:
- Single and batch prediction
- GPU optimization
- Dynamic batching
- Request queueing
- Preprocessing/postprocessing
- Phase 3: Dynamic graph builder + variable topologies
- Phase 3: Support arbitrary sensor counts

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
    from pathlib import Path

    from src.data.timescale_connector import TimescaleConnector

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Inference configuration."""

    model_path: str
    device: Literal["cpu", "cuda", "auto"] = "auto"
    batch_size: int = 32
    max_queue_size: int = 100
    max_wait_ms: float = 50.0  # Max latency tolerance
    use_dynamic_batching: bool = True
    pin_memory: bool = True
    use_dynamic_features: bool = True  # Phase 3.1
    use_dynamic_builder: bool = True  # Phase 3: Use DynamicGraphBuilder
    topology_templates_path: Path | None = None  # Optional custom templates


class InferenceEngine:
    """Production inference engine.

    Features:
    - Batch inference
    - GPU optimization
    - Dynamic batching
    - Request queueing
    - Phase 3: Dynamic graph builder (variable topologies)
    - Phase 3: Arbitrary sensor counts
    - Phase 3: Graceful missing sensor handling

    Examples:
        >>> engine = InferenceEngine(
        ...     config=InferenceConfig(
        ...         model_path="models/v2.0.0.ckpt",
        ...         device="cuda",
        ...         batch_size=32,
        ...         use_dynamic_builder=True
        ...     ),
        ...     timescale_connector=connector
        ... )
        >>>
        >>> # Single prediction from TimescaleDB
        >>> response = await engine.predict_minimal(
        ...     request=MinimalInferenceRequest(...)
        ... )
        >>>
        >>> # Batch prediction
        >>> responses = await engine.predict_batch(...)
    """

    def __init__(
        self,
        config: InferenceConfig,
        timescale_connector: TimescaleConnector | None = None,
        feature_config: FeatureConfig | None = None,
    ):
        """Initialize engine.

        Args:
            config: Inference configuration
            timescale_connector: Database connector (for DynamicGraphBuilder)
            feature_config: Feature engineering config
        """
        self.config = config
        self.feature_config = feature_config or FeatureConfig()
        self.timescale_connector = timescale_connector

        # Initialize components
        self.model_manager = ModelManager()
        self.feature_engineer = FeatureEngineer(self.feature_config)

        # Phase 3: Dynamic graph builder (if connector provided)
        self.dynamic_builder = None
        if config.use_dynamic_builder and timescale_connector:
            self.dynamic_builder = DynamicGraphBuilder(
                timescale_connector=timescale_connector,
                feature_engineer=self.feature_engineer,
                feature_config=self.feature_config,
            )
            logger.info("DynamicGraphBuilder enabled for variable topologies")

        # Phase 3.1: Edge feature components
        self.edge_feature_computer = create_edge_feature_computer()

        # Load normalizer from checkpoint (if available)
        self.edge_normalizer = self._load_normalizer_from_checkpoint(config.model_path)

        # Initialize legacy graph builder (backward compatibility)
        self.graph_builder = GraphBuilder(
            feature_engineer=self.feature_engineer,
            feature_config=self.feature_config,
            edge_feature_computer=self.edge_feature_computer,
            edge_normalizer=self.edge_normalizer,
            use_dynamic_features=config.use_dynamic_features,
        )

        # Initialize topology service
        self.topology_service = get_topology_service(templates_path=config.topology_templates_path)

        # Load model
        self.model = self.model_manager.load_model(
            model_path=config.model_path, device=config.device, use_compile=True
        )

        # Warmup
        self.model_manager.warmup(config.model_path, batch_size=config.batch_size)

        # Request queue (для dynamic batching)
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self._processing = False

        logger.info(
            f"InferenceEngine initialized "
            f"(device={config.device}, batch_size={config.batch_size}, "
            f"dynamic_builder={config.use_dynamic_builder}, "
            f"dynamic_features={config.use_dynamic_features})"
        )

    def _load_normalizer_from_checkpoint(self, checkpoint_path: str) -> EdgeFeatureNormalizer:
        """Load normalizer stats from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            EdgeFeatureNormalizer with loaded stats
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Check for normalizer stats in checkpoint
            if "normalizer_stats" in checkpoint:
                normalizer = create_edge_feature_normalizer()
                normalizer.load_stats(checkpoint["normalizer_stats"])
                logger.info("Loaded normalizer stats from checkpoint")
                return normalizer
            logger.warning("No normalizer stats in checkpoint, using defaults")
            return create_edge_feature_normalizer()

        except Exception as e:
            logger.exception(f"Failed to load normalizer from checkpoint: {e}")
            return create_edge_feature_normalizer()

    async def predict_minimal(self, request: MinimalInferenceRequest) -> PredictionResponse:
        """Single prediction with minimal request (Phase 3 API).

        Uses DynamicGraphBuilder to read arbitrary sensors from TimescaleDB.

        Args:
            request: MinimalInferenceRequest

        Returns:
            response: PredictionResponse

        Examples:
            >>> response = await engine.predict_minimal(
            ...     MinimalInferenceRequest(
            ...         equipment_id="pump_001",
            ...         timestamp=datetime.now(),
            ...         topology_id="standard_pump_system"
            ...     )
            ... )
        """
        start_time = time.time()

        try:
            # Resolve topology
            template = self.topology_service.get_template(request.topology_id)
            if not template:
                msg = f"Topology not found: {request.topology_id}"
                raise ValueError(msg)

            topology = template.to_graph_topology(request.equipment_id)

            # Phase 3: Use DynamicGraphBuilder if available
            if self.config.use_dynamic_builder and self.dynamic_builder:
                graph = await self.dynamic_builder.build_from_timescale(
                    equipment_id=request.equipment_id,
                    topology=topology,
                    lookback_minutes=10,  # TODO: Make configurable
                )
                logger.debug(
                    f"Built dynamic graph: {graph.x.shape[0]} nodes, "
                    f"{graph.edge_attr.shape[0]} edges"
                )
            else:
                # Fallback: legacy path (sensor_readings provided in request)
                graph = self._preprocess_minimal(request=request, topology=topology)
                logger.debug("Built legacy graph from request")

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
                f"Prediction complete: {request.equipment_id} "
                f"({graph.x.shape[0]} nodes, "
                f"time={response.inference_time_ms:.1f}ms)"
            )

            return response

        except Exception as e:
            logger.exception(f"Prediction failed for {request.equipment_id}: {e}")
            raise

    async def predict(
        self, request: PredictionRequest, topology: GraphTopology
    ) -> PredictionResponse:
        """Single prediction (legacy API - backward compatible).

        Args:
            request: Prediction request
            topology: Equipment topology

        Returns:
            response: Prediction response

        Examples:
            >>> response = await engine.predict(
            ...     request=PredictionRequest(...),
            ...     topology=topology
            ... )
        """
        start_time = time.time()

        try:
            # Preprocess (legacy path - no dynamic features)
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
                f"Prediction complete (legacy): {request.equipment_id} "
                f"(time={response.inference_time_ms:.1f}ms)"
            )

            return response

        except Exception as e:
            logger.exception(f"Prediction failed for {request.equipment_id}: {e}")
            raise

    async def predict_batch(
        self,
        requests: list[PredictionRequest | MinimalInferenceRequest],
        topology: GraphTopology | None = None,
    ) -> list[PredictionResponse]:
        """Batch prediction.

        Args:
            requests: List of prediction requests (legacy or new)
            topology: Equipment topology (for legacy requests)

        Returns:
            responses: List of prediction responses

        Examples:
            >>> responses = await engine.predict_batch(
            ...     requests=[req1, req2, req3]
            ... )
        """
        start_time = time.time()

        try:
            # Preprocess all
            graphs = []
            for req in requests:
                if isinstance(req, MinimalInferenceRequest):
                    # Resolve topology for minimal request
                    template = self.topology_service.get_template(req.topology_id)
                    if not template:
                        msg = f"Topology not found: {req.topology_id}"
                        raise ValueError(msg)
                    req_topology = template.to_graph_topology(req.equipment_id)

                    # Use DynamicGraphBuilder if available
                    if self.config.use_dynamic_builder and self.dynamic_builder:
                        graph = await self.dynamic_builder.build_from_timescale(
                            equipment_id=req.equipment_id,
                            topology=req_topology,
                            lookback_minutes=10,
                        )
                    else:
                        graph = self._preprocess_minimal(req, req_topology)
                else:
                    # Legacy request
                    if not topology:
                        msg = "Topology required for legacy requests"
                        raise ValueError(msg)
                    graph = self._preprocess_legacy(req, topology)

                graphs.append(graph)

            # Batch inference
            health_batch, degradation_batch, anomaly_batch = self._inference_batch(graphs)

            # Postprocess all
            responses = [
                self._postprocess(
                    equipment_id=req.equipment_id,
                    health=health_batch[i],
                    degradation=degradation_batch[i],
                    anomaly=anomaly_batch[i],
                    inference_time=(time.time() - start_time) / len(requests),
                )
                for i, req in enumerate(requests)
            ]

            logger.info(
                f"Batch prediction complete: {len(requests)} requests "
                f"(time={time.time() - start_time:.3f}s)"
            )

            return responses

        except Exception as e:
            logger.exception(f"Batch prediction failed: {e}")
            raise

    def _preprocess_minimal(
        self, request: MinimalInferenceRequest, topology: GraphTopology
    ) -> Data:
        """Preprocess minimal request → graph (Phase 3.1 API).

        Converts sensor readings from request to graph with dynamic edge features.

        Args:
            request: MinimalInferenceRequest
                - sensor_readings: Dict[component_id, Dict[sensor_name, float]]
                - timestamp: Current measurement time
            topology: Equipment topology (components and connections)

        Returns:
            graph: PyG Data object with 14D edges

        Raises:
            ValueError: If sensor data is invalid or components missing
        """
        try:
            # Validate request has sensor data
            if not request.sensor_readings:
                msg = f"No sensor readings provided for {request.equipment_id}"
                raise ValueError(msg)

            # Convert sensor readings to DataFrame format for graph builder
            # Structure: component_id, sensor_name, value
            sensor_records = []
            for component_id, sensors in request.sensor_readings.items():
                if not isinstance(sensors, dict):
                    logger.warning(
                        f"Invalid sensor format for {component_id}: "
                        f"expected dict, got {type(sensors)}"
                    )
                    continue

                for sensor_name, value in sensors.items():
                    if value is None:
                        logger.warning(
                            f"Missing sensor {sensor_name} for {component_id}"
                        )
                        continue

                    sensor_records.append({
                        "component_id": component_id,
                        "sensor_name": sensor_name,
                        "value": float(value),
                        "timestamp": request.timestamp,
                    })

            if not sensor_records:
                msg = f"No valid sensor readings for {request.equipment_id}"
                raise ValueError(msg)

            sensor_df = pd.DataFrame(sensor_records)
            logger.debug(
                f"Converted {len(sensor_records)} sensor readings "
                f"from {len(request.sensor_readings)} components"
            )

            # Build graph with dynamic features
            # sensor_readings passed for edge feature computation
            graph = self.graph_builder.build_graph(
                sensor_data=sensor_df,
                topology=topology,
                metadata=None,
                sensor_readings=request.sensor_readings,  # For 14D dynamic edges
                current_time=request.timestamp,
            )

            logger.debug(
                f"Built graph: {graph.x.shape[0]} nodes, "
                f"{graph.edge_index.shape[1]} edges, "
                f"edge_attr shape: {graph.edge_attr.shape if graph.edge_attr is not None else 'None'}"
            )

            return graph

        except Exception as e:
            logger.error(
                f"Failed to preprocess minimal request for {request.equipment_id}: {e}"
            )
            raise

    def _preprocess_legacy(self, request: PredictionRequest, topology: GraphTopology) -> Data:
        """Preprocess legacy request → graph (no dynamic features).

        Args:
            request: PredictionRequest
            topology: Equipment topology

        Returns:
            graph: PyG Data object (14D edges with zeros)
        """
        # Build graph without dynamic features
        return self.graph_builder.build_graph(
            sensor_data=request.sensor_data,  # Should be DataFrame
            topology=topology,
            metadata=None,  # Not needed for inference
        )

    def _inference_single(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single graph inference.

        Args:
            graph: PyG Data object (variable N/E)

        Returns:
            outputs: (health, degradation, anomaly) tensors
        """
        device = next(self.model.parameters()).device

        # Move to device
        graph = graph.to(device)

        # Create batch tensor for single graph
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

    def _inference_batch(
        self, graphs: list[Data]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch inference with variable-sized graphs.

        Args:
            graphs: List of PyG Data objects (variable N/E each)

        Returns:
            outputs: (health_batch, degradation_batch, anomaly_batch)
        """
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

    def _postprocess(
        self,
        equipment_id: str,
        health: torch.Tensor,
        degradation: torch.Tensor,
        anomaly: torch.Tensor,
        inference_time: float,
    ) -> PredictionResponse:
        """Postprocess outputs → response.

        Args:
            equipment_id: Equipment identifier
            health: Health tensor [1, 1]
            degradation: Degradation tensor [1, 1]
            anomaly: Anomaly logits [1, 9]
            inference_time: Inference time (seconds)

        Returns:
            response: PredictionResponse
        """
        # Extract values
        health_score = float(health.squeeze().cpu().item())
        degradation_rate = float(degradation.squeeze().cpu().item())
        anomaly_logits = anomaly.squeeze().cpu().numpy()

        # Anomaly probabilities (sigmoid for multi-label)
        anomaly_probs = torch.sigmoid(torch.from_numpy(anomaly_logits)).numpy()

        # Anomaly types
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

        # Build anomaly predictions
        anomaly_predictions = {
            anomaly_type: float(prob)
            for anomaly_type, prob in zip(anomaly_types, anomaly_probs, strict=False)
        }

        # Create response
        return PredictionResponse(
            equipment_id=equipment_id,
            health=HealthPrediction(score=health_score),
            degradation=DegradationPrediction(rate=degradation_rate),
            anomaly=AnomalyPrediction(predictions=anomaly_predictions),
            inference_time_ms=inference_time * 1000,
        )

    def get_stats(self) -> dict:
        """Get inference statistics.

        Returns:
            stats: Statistics dict

        Examples:
            >>> stats = engine.get_stats()
            >>> print(stats["model_device"])
        """
        model_info = self.model_manager.get_model_info(self.config.model_path)
        topology_stats = self.topology_service.get_stats()

        return {
            "model_path": self.config.model_path,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "model_device": model_info["device"] if model_info else None,
            "model_parameters": model_info["num_parameters"] if model_info else None,
            "queue_size": self._request_queue.qsize(),
            "processing": self._processing,
            "use_dynamic_builder": self.config.use_dynamic_builder,
            "dynamic_builder_available": self.dynamic_builder is not None,
            "use_dynamic_features": self.config.use_dynamic_features,
            "topology_templates": topology_stats["cached_templates"],
            "custom_topologies": topology_stats["custom_topologies"],
        }
