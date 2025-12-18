"""Multi-task loss functions for GNN training.

Implements joint optimization for:
- Component health (node-level, 5 classes)
- Anomaly type (graph-level, 4 classes)

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskLossConfig:
    """Configuration for multi-task loss."""
    
    # Task weights
    component_health_weight: float = 1.0
    anomaly_type_weight: float = 1.0
    
    # Class weights (auto-computed if None)
    component_health_class_weights: torch.Tensor | None = None
    anomaly_type_class_weights: torch.Tensor | None = None
    
    # Label smoothing
    label_smoothing: float = 0.0
    
    # Loss functions
    component_health_loss_type: str = "cross_entropy"
    anomaly_type_loss_type: str = "cross_entropy"


class MultiTaskLoss(nn.Module):
    """Multi-task loss for simultaneous node and graph prediction.
    
    Combines:
    - Component health loss (node-level)
    - Anomaly type loss (graph-level)
    
    Examples:
        >>> config = MultiTaskLossConfig(
        ...     component_health_weight=1.0,
        ...     anomaly_type_weight=1.0
        ... )
        >>> loss_fn = MultiTaskLoss(config)
        >>> 
        >>> # Forward pass
        >>> node_logits = torch.randn(100, 5)  # [num_nodes, 5]
        >>> graph_logits = torch.randn(32, 4)  # [batch_size, 4]
        >>> node_labels = torch.randint(0, 5, (100,))
        >>> graph_labels = torch.randint(0, 4, (32,))
        >>> 
        >>> loss_dict = loss_fn(
        ...     node_logits, node_labels,
        ...     graph_logits, graph_labels
        ... )
        >>> total_loss = loss_dict['total']
    """
    
    def __init__(self, config: MultiTaskLossConfig | None = None) -> None:
        """Initialize multi-task loss.
        
        Args:
            config: Loss configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or MultiTaskLossConfig()
        
        # Component health loss
        self.component_health_loss = nn.CrossEntropyLoss(
            weight=self.config.component_health_class_weights,
            label_smoothing=self.config.label_smoothing,
            reduction='mean'
        )
        
        # Anomaly type loss
        self.anomaly_type_loss = nn.CrossEntropyLoss(
            weight=self.config.anomaly_type_class_weights,
            label_smoothing=self.config.label_smoothing,
            reduction='mean'
        )
        
        logger.info("MultiTaskLoss initialized with config: %s", self.config)
    
    def forward(
        self,
        node_logits: torch.Tensor,
        node_labels: torch.Tensor,
        graph_logits: torch.Tensor,
        graph_labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute multi-task loss.
        
        Args:
            node_logits: Node predictions [num_nodes, num_node_classes]
            node_labels: Node ground truth [num_nodes]
            graph_logits: Graph predictions [batch_size, num_graph_classes]
            graph_labels: Graph ground truth [batch_size]
            
        Returns:
            Dictionary with 'total', 'component_health', 'anomaly_type' losses
        """
        # Component health loss (node-level)
        health_loss = self.component_health_loss(node_logits, node_labels)
        
        # Anomaly type loss (graph-level)
        anomaly_loss = self.anomaly_type_loss(graph_logits, graph_labels)
        
        # Weighted sum
        total_loss = (
            self.config.component_health_weight * health_loss +
            self.config.anomaly_type_weight * anomaly_loss
        )
        
        return {
            'total': total_loss,
            'component_health': health_loss,
            'anomaly_type': anomaly_loss
        }
    
    def update_class_weights(
        self,
        component_health_weights: torch.Tensor | None = None,
        anomaly_type_weights: torch.Tensor | None = None
    ) -> None:
        """Update class weights (e.g., after computing from data).
        
        Args:
            component_health_weights: New weights for component health [5]
            anomaly_type_weights: New weights for anomaly type [4]
        """
        if component_health_weights is not None:
            self.component_health_loss.weight = component_health_weights
            logger.info("Updated component health class weights")
        
        if anomaly_type_weights is not None:
            self.anomaly_type_loss.weight = anomaly_type_weights
            logger.info("Updated anomaly type class weights")


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    normalize: bool = True
) -> torch.Tensor:
    """Compute balanced class weights from labels.
    
    Args:
        labels: Class labels [num_samples]
        num_classes: Total number of classes
        normalize: Whether to normalize weights to sum to num_classes
        
    Returns:
        Class weights [num_classes]
        
    Examples:
        >>> labels = torch.tensor([0, 0, 1, 1, 1, 2])
        >>> weights = compute_class_weights(labels, num_classes=3)
        >>> weights  # [1.5, 1.0, 3.0] (inversely proportional to frequency)
    """
    # Count occurrences
    counts = torch.bincount(labels, minlength=num_classes).float()
    
    # Avoid division by zero
    counts = torch.clamp(counts, min=1.0)
    
    # Inverse frequency
    weights = labels.numel() / (num_classes * counts)
    
    if normalize:
        weights = weights * num_classes / weights.sum()
    
    return weights
