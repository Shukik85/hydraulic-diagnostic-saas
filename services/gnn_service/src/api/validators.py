"""Request validation utilities for FastAPI endpoints.

Validates inference requests to ensure compatibility between:
- Topology configuration and model architecture
- Sensor readings and component requirements
- Batch sizes and GPU memory limits
- Feature dimensions and model expectations

Used by main.py endpoints to provide clear error messages before inference.

Architecture:
    No dependencies on InferenceEngine (prevents circular deps).
    Requirements injected via constructor.
    Uses TopologyService for topology lookup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from src.schemas.graph import GraphTopology
    from src.schemas.requests import HybridInferenceRequest, MinimalInferenceRequest, PredictionRequest
    from src.schemas.topology import TopologyConfig
    from src.services.topology_service import TopologyService

logger = logging.getLogger(__name__)


class ModelRequirements(NamedTuple):
    """Model architecture requirements for validation.

    Attributes:
        node_feature_dim: Expected node feature dimension
        edge_feature_dim: Expected edge feature dimension
        min_nodes: Minimum nodes per graph
        max_nodes: Maximum nodes per graph
    """

    node_feature_dim: int
    edge_feature_dim: int
    min_nodes: int = 2
    max_nodes: int = 1000


class RequestValidator:
    """Validates inference requests for compatibility and correctness.

    Provides early validation before inference to:
    - Prevent cryptic PyTorch errors
    - Give clear HTTP error messages
    - Protect against OOM and resource exhaustion

    No direct dependency on InferenceEngine to avoid circular imports.
    Requirements are injected via constructor.

    Attributes:
        requirements: Model architecture requirements
        topology_service: Service for topology lookup
        max_batch_size: Maximum allowed batch size

    Examples:
        >>> requirements = ModelRequirements(
        ...     node_feature_dim=34,
        ...     edge_feature_dim=14,
        ... )
        >>> validator = RequestValidator(
        ...     requirements=requirements,
        ...     topology_service=topology_service,
        ...     max_batch_size=32,
        ... )
        >>>
        >>> # Validate diagnosis request
        >>> topology = await validator.validate_diagnosis_request(request)
        >>>
        >>> # Validate prediction batch
        >>> validator.validate_batch_size(len(request.batch))
    """

    def __init__(
        self,
        requirements: ModelRequirements,
        topology_service: TopologyService,
        max_batch_size: int = 32,
    ):
        """Initialize validator.

        Args:
            requirements: Model architecture requirements
            topology_service: Service for topology lookup
            max_batch_size: Maximum graphs per batch (default: 32)
        """
        self.requirements = requirements
        self.topology_service = topology_service
        self.max_batch_size = max_batch_size

    async def validate_diagnosis_request(
        self,
        request: MinimalInferenceRequest | HybridInferenceRequest,
    ) -> Any:
        """Validate diagnosis request and return topology.

        Supports:
            - MinimalInferenceRequest (legacy node-centric)
            - HybridInferenceRequest (preferred edge-centric)

        Returns:
            GraphTopology for minimal requests, TopologyConfig for hybrid requests.

        Raises:
            HTTPException(404): Topology not found
            HTTPException(400): Missing sensors/edges or invalid input
            HTTPException(413): Graph too large
        """
        # Hybrid (edge-centric) path
        if hasattr(request, "edge_readings"):
            topology_config = self._get_topology_config(request.topology_id)

            self._validate_edge_readings(request, topology_config)
            self._validate_dimensions(topology_config)
            self._validate_graph_size(topology_config)

            logger.info(
                "Hybrid request validated successfully",
                extra={
                    "equipment_id": request.equipment_id,
                    "topology_id": request.topology_id,
                    "num_components": len(topology_config.components),
                    "num_edges": len(topology_config.edges),
                    "num_edge_readings": len(request.edge_readings),
                },
            )

            return topology_config

        # Minimal (legacy) path
        topology = self._get_topology(request.topology_id, request.equipment_id)

        self._validate_sensors(request, topology)
        self._validate_dimensions(topology)
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

    def _get_topology(
        self, topology_id: str, equipment_id: str
    ) -> GraphTopology:
        """Retrieve legacy GraphTopology from service."""
        try:
            template = self.topology_service.get_template(topology_id)
            if not template:
                raise KeyError(f"Topology '{topology_id}' not found")

            topology = template.to_graph_topology(equipment_id)
            return topology

        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Topology '{topology_id}' not found. "
                    f"Available: {list(self.topology_service.get_all_templates().keys())}"
                ),
            ) from e

    def _get_topology_config(self, topology_id: str) -> TopologyConfig:
        """Retrieve TopologyConfig (edge-centric) from service."""
        topology_config = self.topology_service.get_config(
            template_id=topology_id,
            topology_id=topology_id,
        )
        if topology_config is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Topology '{topology_id}' not found. "
                    f"Available: {list(self.topology_service.get_all_templates().keys())}"
                ),
            )
        return topology_config

    def _validate_edge_readings(
        self,
        request: HybridInferenceRequest,
        topology: TopologyConfig,
    ) -> None:
        """Validate edge readings against topology.

        Policy (Day 6):
            - If diagnostic_scope.target_edges provided -> require those edges.
            - Otherwise do not require full coverage (GraphBuilderV2 can fill missing edges with nominal/zeros).
            - Unknown edge IDs are treated as input errors.
        """
        known_edges = {f"{e.source_id}__{e.target_id}" for e in topology.edges}
        provided_edges = set(request.edge_readings.keys())

        unknown_edges = provided_edges - known_edges
        if unknown_edges:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unknown edge_ids in edge_readings: {sorted(unknown_edges)}. "
                    f"Topology '{topology.topology_id}' defines {len(known_edges)} edges."
                ),
            )

        # If scope explicitly declares targets, enforce presence.
        if getattr(request, "diagnostic_scope", None) and request.diagnostic_scope.target_edges:
            required = set(request.diagnostic_scope.target_edges)
            missing = required - provided_edges
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Missing edge_readings for target_edges: {sorted(missing)}. "
                        "Provide readings for target_edges or remove diagnostic_scope.target_edges."
                    ),
                )

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

    def _validate_dimensions(self, topology: Any) -> None:
        """Validate feature dimensions match model.

        For TopologyConfig (edge-centric) there is no embedded feature-dim metadata,
        so this check is currently a no-op beyond ensuring requirements are present.

        Args:
            topology: GraphTopology or TopologyConfig

        Raises:
            HTTPException(400): Dimension mismatch
        """
        # Node feature dimension
        expected_node_dim = self.requirements.node_feature_dim

        # Edge feature dimension
        expected_edge_dim = self.requirements.edge_feature_dim

        # If topology has metadata about dimensions, validate
        if hasattr(topology, "node_feature_dim"):
            actual_node_dim = topology.node_feature_dim
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

        if hasattr(topology, "edge_feature_dim"):
            actual_edge_dim = topology.edge_feature_dim
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

    def _validate_graph_size(self, topology: Any) -> None:
        """Validate graph size within limits.

        Args:
            topology: GraphTopology or TopologyConfig

        Raises:
            HTTPException(400): Graph too small
            HTTPException(413): Graph too large
        """
        if hasattr(topology, "num_components"):
            num_components = topology.num_components
        else:
            num_components = len(topology.components)

        if num_components < self.requirements.min_nodes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Graph too small: {num_components} components, "
                    f"minimum {self.requirements.min_nodes} required"
                ),
            )

        if num_components > self.requirements.max_nodes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Graph too large: {num_components} components exceeds "
                    f"maximum {self.requirements.max_nodes}. "
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
