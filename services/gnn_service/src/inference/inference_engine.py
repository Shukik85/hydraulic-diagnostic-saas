"""Inference Engine for production predictions.

Features:
- Single and batch prediction
- GPU optimization
- Dynamic batching
- Request queueing
- Preprocessing/postprocessing

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import torch
import logging
import asyncio
import time
from typing import Literal
from dataclasses import dataclass
import numpy as np

from torch_geometric.data import Data, Batch

from src.inference.model_manager import ModelManager
from src.data import FeatureEngineer, GraphBuilder, FeatureConfig
from src.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthPrediction,
    DegradationPrediction,
    AnomalyPrediction,
    GraphTopology
)

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
    

class InferenceEngine:
    """Production inference engine.
    
    Features:
    - Batch inference
    - GPU optimization
    - Dynamic batching
    - Request queueing
    
    Examples:
        >>> engine = InferenceEngine(
        ...     config=InferenceConfig(
        ...         model_path="models/best.ckpt",
        ...         device="cuda",
        ...         batch_size=32
        ...     )
        ... )
        >>> 
        >>> # Single prediction
        >>> response = await engine.predict(
        ...     request=PredictionRequest(...),
        ...     topology=topology
        ... )
        >>> 
        >>> # Batch prediction
        >>> responses = await engine.predict_batch(
        ...     requests=[req1, req2, req3],
        ...     topology=topology
        ... )
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        feature_config: FeatureConfig | None = None
    ):
        """Initialize engine.
        
        Args:
            config: Inference configuration
            feature_config: Feature engineering config
        """
        self.config = config
        self.feature_config = feature_config or FeatureConfig()
        
        # Initialize components
        self.model_manager = ModelManager()
        self.feature_engineer = FeatureEngineer(self.feature_config)
        self.graph_builder = GraphBuilder(
            self.feature_engineer,
            self.feature_config
        )
        
        # Load model
        self.model = self.model_manager.load_model(
            model_path=config.model_path,
            device=config.device,
            use_compile=True
        )
        
        # Warmup
        self.model_manager.warmup(config.model_path, batch_size=config.batch_size)
        
        # Request queue (для dynamic batching)
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self._processing = False
        
        logger.info(f"InferenceEngine initialized (device={config.device}, batch_size={config.batch_size})")
    
    async def predict(
        self,
        request: PredictionRequest,
        topology: GraphTopology
    ) -> PredictionResponse:
        """Single prediction.
        
        Args:
            request: Prediction request
            topology: Equipment topology
        
        Returns:
            response: Prediction response
        
        Examples:
            >>> response = await engine.predict(
            ...     request=PredictionRequest(
            ...         equipment_id="exc_001",
            ...         sensor_data={...}
            ...     ),
            ...     topology=topology
            ... )
            >>> print(response.health.score)
        """
        start_time = time.time()
        
        try:
            # Preprocess
            graph = self._preprocess(request, topology)
            
            # Inference
            health, degradation, anomaly = self._inference_single(graph)
            
            # Postprocess
            response = self._postprocess(
                request=request,
                health=health,
                degradation=degradation,
                anomaly=anomaly,
                inference_time=time.time() - start_time
            )
            
            logger.info(
                f"Prediction complete: {request.equipment_id} "
                f"(time={response.inference_time_ms:.1f}ms)"
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Prediction failed for {request.equipment_id}: {e}")
            raise
    
    async def predict_batch(
        self,
        requests: list[PredictionRequest],
        topology: GraphTopology
    ) -> list[PredictionResponse]:
        """Batch prediction.
        
        Args:
            requests: List of prediction requests
            topology: Equipment topology
        
        Returns:
            responses: List of prediction responses
        
        Examples:
            >>> responses = await engine.predict_batch(
            ...     requests=[req1, req2, req3],
            ...     topology=topology
            ... )
            >>> for resp in responses:
            ...     print(resp.equipment_id, resp.health.score)
        """
        start_time = time.time()
        
        try:
            # Preprocess all
            graphs = [self._preprocess(req, topology) for req in requests]
            
            # Batch inference
            health_batch, degradation_batch, anomaly_batch = self._inference_batch(graphs)
            
            # Postprocess all
            responses = [
                self._postprocess(
                    request=req,
                    health=health_batch[i],
                    degradation=degradation_batch[i],
                    anomaly=anomaly_batch[i],
                    inference_time=(time.time() - start_time) / len(requests)
                )
                for i, req in enumerate(requests)
            ]
            
            logger.info(
                f"Batch prediction complete: {len(requests)} requests "
                f"(time={time.time() - start_time:.3f}s)"
            )
            
            return responses
        
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def _preprocess(
        self,
        request: PredictionRequest,
        topology: GraphTopology
    ) -> Data:
        """Preprocess request → graph.
        
        Args:
            request: Prediction request
            topology: Equipment topology
        
        Returns:
            graph: PyG Data object
        """
        # Build graph from sensor data
        graph = self.graph_builder.build_graph(
            sensor_data=request.sensor_data,  # Should be DataFrame
            topology=topology,
            metadata=None  # Not needed for inference
        )
        
        return graph
    
    def _inference_single(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single graph inference.
        
        Args:
            graph: PyG Data object
        
        Returns:
            outputs: (health, degradation, anomaly) tensors
        """
        device = next(self.model.parameters()).device
        
        # Move to device
        graph = graph.to(device)
        
        # Inference
        with torch.inference_mode():
            health, degradation, anomaly = self.model(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                batch=torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)  # Single graph
            )
        
        return health, degradation, anomaly
    
    def _inference_batch(self, graphs: list[Data]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch inference.
        
        Args:
            graphs: List of PyG Data objects
        
        Returns:
            outputs: (health_batch, degradation_batch, anomaly_batch)
        """
        device = next(self.model.parameters()).device
        
        # Create PyG Batch
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
                batch=batch.batch
            )
        
        return health, degradation, anomaly
    
    def _postprocess(
        self,
        request: PredictionRequest,
        health: torch.Tensor,
        degradation: torch.Tensor,
        anomaly: torch.Tensor,
        inference_time: float
    ) -> PredictionResponse:
        """Postprocess outputs → response.
        
        Args:
            request: Original request
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
            "valve_stiction"
        ]
        
        # Build anomaly predictions
        anomaly_predictions = {
            anomaly_type: float(prob)
            for anomaly_type, prob in zip(anomaly_types, anomaly_probs)
        }
        
        # Create response
        response = PredictionResponse(
            equipment_id=request.equipment_id,
            health=HealthPrediction(score=health_score),
            degradation=DegradationPrediction(rate=degradation_rate),
            anomaly=AnomalyPrediction(predictions=anomaly_predictions),
            inference_time_ms=inference_time * 1000
        )
        
        return response
    
    def get_stats(self) -> dict:
        """Get inference statistics.
        
        Returns:
            stats: Statistics dict
        
        Examples:
            >>> stats = engine.get_stats()
            >>> print(stats["model_device"])
        """
        model_info = self.model_manager.get_model_info(self.config.model_path)
        
        return {
            "model_path": self.config.model_path,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "model_device": model_info["device"] if model_info else None,
            "model_parameters": model_info["num_parameters"] if model_info else None,
            "queue_size": self._request_queue.qsize(),
            "processing": self._processing
        }
