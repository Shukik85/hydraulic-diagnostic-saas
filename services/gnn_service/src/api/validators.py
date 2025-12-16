"""Request validation utilities for FastAPI endpoints.

Validates inference requests to ensure compatibility between:
- Topology configuration and model architecture
- Sensor readings and component requirements
- Batch sizes and GPU memory limits
- Feature dimensions and model expectations

Used by main.py endpoints to provide clear error messages before inference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from src.inference.inference_engine import InferenceEngine
    from src.schemas.graph import GraphTopology
    from src.schemas.requests import MinimalInferenceRequest, PredictionRequest

logger = logging.getLogger(__name__)


class RequestValidator:
    """Validates inference requests for compatibility and correctness.
    
    Provides early validation before inference to:
    - Prevent cryptic PyTorch errors
    - Give clear HTTP error messages
    - Protect against OOM and resource exhaustion
    
    Attributes:
        inference_engine: Engine instance to check model compatibility
        max_batch_size: Maximum allowed batch size
        max_graph_size: Maximum allowed nodes per graph
    
    Examples:
        >>> validator = RequestValidator(engine, max_batch_size=32)
        >>> 
        >>> # Validate diagnosis request
        >>> topology = await validator.validate_diagnosis_request(request)
        >>> 
        >>> # Validate prediction batch
        >>> validator.validate_batch_size(len(request.batch))
    """
    
    def __init__(
        self,
        inference_engine: InferenceEngine,
        max_batch_size: int = 32,
        max_graph_size: int = 1000,
    ):
        """Initialize validator.
        
        Args:
            inference_engine: Engine to validate against
            max_batch_size: Maximum graphs per batch (default: 32)
            max_graph_size: Maximum nodes per graph (default: 1000)
        """
        self.inference_engine = inference_engine
        self.max_batch_size = max_batch_size
        self.max_graph_size = max_graph_size
    
    async def validate_diagnosis_request(
        self,
        request: MinimalInferenceRequest,
    ) -> GraphTopology:
        """Validate diagnosis request and return topology.
        
        Checks:
        1. Topology exists
        2. All required sensors present
        3. Feature dimensions match model
        4. Graph size within limits
        
        Args:
            request: Diagnosis request to validate
            
        Returns:
            GraphTopology: Validated topology
            
        Raises:
            HTTPException(404): Topology not found
            HTTPException(400): Missing sensors or dimension mismatch
            HTTPException(413): Graph too large
            
        Examples:
            >>> topology = await validator.validate_diagnosis_request(request)
            >>> # Safe to proceed with inference
        """
        # 1. Get topology (raises 404 if not found)
        topology = await self._get_topology(request.topology_id)
        
        # 2. Validate sensors
        self._validate_sensors(request, topology)
        
        # 3. Validate dimensions
        self._validate_dimensions(topology)
        
        # 4. Validate graph size
        self._validate_graph_size(topology)
        
        logger.info(
            "Request validated successfully",
            extra={
                "equipment_id": request.equipment_id,
                "topology_id": request.topology_id,
                "num_components": topology.num_components,
                "num_sensors": len(request.sensor_readings),
            },
        )
        
        return topology
    
    async def validate_prediction_request(
        self,
        request: PredictionRequest,
    ) -> None:
        """Validate prediction request with batch.
        
        Checks:
        1. Batch size within limits
        2. Topology compatible with model
        3. All graphs in batch have consistent dimensions
        
        Args:
            request: Prediction request to validate
            
        Raises:
            HTTPException(413): Batch too large
            HTTPException(400): Invalid topology or dimensions
            
        Examples:
            >>> await validator.validate_prediction_request(request)
            >>> # Safe to proceed with batch inference
        """
        # 1. Validate batch size
        batch_size = len(request.batch) if hasattr(request, "batch") else 1
        self._validate_batch_size(batch_size)
        
        # 2. Validate topology if provided
        if hasattr(request, "topology") and request.topology:
            self._validate_dimensions(request.topology)
            self._validate_graph_size(request.topology)
        
        logger.info(
            "Batch request validated",
            extra={"batch_size": batch_size},
        )
    
    async def _get_topology(self, topology_id: str) -> GraphTopology:
        """Retrieve topology from service.
        
        Args:
            topology_id: Topology identifier
            
        Returns:
            GraphTopology: Retrieved topology
            
        Raises:
            HTTPException(404): Topology not found
        """
        try:
            # Assuming TopologyService integration
            # topology_service = self.inference_engine.topology_service
            # topology = await topology_service.get_topology(topology_id)
            
            # STUB: For now, raise 501 if topology service not implemented
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=(
                    "Topology service not yet integrated. "
                    "Pass topology directly in request body."
                ),
            )
        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Topology '{topology_id}' not found",
            ) from e
    
    def _validate_sensors(
        self,
        request: MinimalInferenceRequest,
        topology: GraphTopology,
    ) -> None:
        """Validate all required sensors present.
        
        Args:
            request: Request with sensor readings
            topology: Expected topology
            
        Raises:
            HTTPException(400): Missing sensors
        """
        required_sensors = {comp.component_id for comp in topology.components.values()}
        provided_sensors = set(request.sensor_readings.keys())
        missing_sensors = required_sensors - provided_sensors
        
        if missing_sensors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Missing sensor readings for components: {sorted(missing_sensors)}. "
                    f"Topology '{topology.topology_id}' requires readings for all "
                    f"{len(required_sensors)} components."
                ),
            )
        
        # Check for extra sensors (warning, not error)
        extra_sensors = provided_sensors - required_sensors
        if extra_sensors:
            logger.warning(
                "Extra sensor readings ignored",
                extra={
                    "extra_sensors": sorted(extra_sensors),
                    "topology_id": topology.topology_id,
                },
            )
    
    def _validate_dimensions(self, topology: GraphTopology) -> None:
        """Validate feature dimensions match model.
        
        Args:
            topology: Topology to validate
            
        Raises:
            HTTPException(400): Dimension mismatch
        """
        # Node feature dimension
        expected_node_dim = self.inference_engine.model.in_channels
        actual_node_dim = topology.components[next(iter(topology.components))].feature_dim
        
        if actual_node_dim != expected_node_dim:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Node feature dimension mismatch: "
                    f"topology has {actual_node_dim}, "
                    f"model expects {expected_node_dim}. "
                    f"Check FeatureConfig.node_feature_dim."
                ),
            )
        
        # Edge feature dimension
        expected_edge_dim = self.inference_engine.model.edge_in_dim
        
        # Calculate actual edge dim from topology
        # Static features (8) + dynamic features (6) = 14 default
        # But should check EdgeFeatureComputer configuration
        if hasattr(topology, "edge_feature_dim"):
            actual_edge_dim = topology.edge_feature_dim
        else:
            # Default: 8 static + 6 dynamic
            actual_edge_dim = 14
        
        if actual_edge_dim != expected_edge_dim:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Edge feature dimension mismatch: "
                    f"topology has {actual_edge_dim}, "
                    f"model expects {expected_edge_dim}. "
                    f"Check FeatureConfig.edge_in_dim and EdgeFeatureComputer."
                ),
            )
    
    def _validate_graph_size(self, topology: GraphTopology) -> None:
        """Validate graph size within limits.
        
        Args:
            topology: Topology to validate
            
        Raises:
            HTTPException(400): Graph too small
            HTTPException(413): Graph too large
        """
        num_components = topology.num_components
        
        if num_components < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Graph must have at least 2 components (nodes)",
            )
        
        if num_components > self.max_graph_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Graph too large: {num_components} components exceeds "
                    f"maximum {self.max_graph_size}. "
                    f"Consider splitting into multiple requests or increasing limits."
                ),
            )
    
    def _validate_batch_size(self, batch_size: int) -> None:
        """Validate batch size within limits.
        
        Args:
            batch_size: Number of graphs in batch
            
        Raises:
            HTTPException(413): Batch too large
        """
        if batch_size > self.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Batch size {batch_size} exceeds maximum {self.max_batch_size}. "
                    f"Reduce batch size or contact administrator to increase limits."
                ),
            )
        
        if batch_size < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size must be at least 1",
            )
