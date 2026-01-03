"""Tensor validation for inference.

Provides comprehensive validation of graph tensors before inference:
- Dimension checks (nodes, edges)
- NaN/Inf detection
- Device validation
- Dtype checks
"""

from __future__ import annotations

import warnings

import torch
from torch_geometric.data import Data

from .exceptions import TensorValidationError


class TensorValidator:
    """Validates graph tensors before inference.

    Attributes:
        expected_node_dim: Expected node feature dimension
        expected_edge_dim: Expected edge feature dimension
        allowed_devices: List of allowed device strings (optional)

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
        """Initialize validator.

        Args:
            expected_node_dim: Expected node feature dimension
            expected_edge_dim: Expected edge feature dimension
            allowed_devices: List of allowed device strings (e.g., ['cuda', 'cpu'])
        """
        self.expected_node_dim = expected_node_dim
        self.expected_edge_dim = expected_edge_dim
        self.allowed_devices = allowed_devices

    def validate_graph(self, graph: Data) -> None:
        """Validate graph tensors.

        Args:
            graph: PyG Data object to validate

        Raises:
            TensorValidationError: If validation fails
        """
        # Node features dimension
        if graph.x.shape[1] != self.expected_node_dim:
            raise TensorValidationError(
                f"Node dim mismatch: expected {self.expected_node_dim}, "
                f"got {graph.x.shape[1]}"
            )

        # Edge features dimension
        if graph.edge_attr is not None and graph.edge_attr.shape[1] != self.expected_edge_dim:
            raise TensorValidationError(
                f"Edge dim mismatch: expected {self.expected_edge_dim}, "
                f"got {graph.edge_attr.shape[1]}"
            )

        # NaN check - node features
        if torch.isnan(graph.x).any():
            raise TensorValidationError("NaN detected in node features")
        if torch.isinf(graph.x).any():
            raise TensorValidationError("Inf detected in node features")

        # NaN check - edge features
        if graph.edge_attr is not None:
            if torch.isnan(graph.edge_attr).any():
                raise TensorValidationError("NaN detected in edge features")
            if torch.isinf(graph.edge_attr).any():
                raise TensorValidationError("Inf detected in edge features")

        # Device validation
        if self.allowed_devices:
            device_str = str(graph.x.device)
            if not any(allowed in device_str for allowed in self.allowed_devices):
                raise TensorValidationError(
                    f"Invalid device: {device_str}. Allowed: {self.allowed_devices}"
                )

        # Dtype warning (not critical, just warn)
        if graph.x.dtype != torch.float32:
            warnings.warn(
                f"Non-float32 node features: {graph.x.dtype}",
                UserWarning,
                stacklevel=2,
            )
