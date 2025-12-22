"""Multi-task loss functions for GNN training (Phase 2).

Implements joint optimization for 6 tasks:
- Component health (regression, node-level)
- Component anomaly (multi-label, node-level)
- Graph health (regression, graph-level)
- Graph degradation (regression, graph-level)
- Graph anomaly (multi-label, graph-level)
- Graph RUL (regression, graph-level)

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskLossConfig:
    """Configuration for multi-task loss (Phase 2: 6 tasks)."""
    
    # Task weights (6 tasks)
    component_health_weight: float = 1.0      # Regression
    component_anomaly_weight: float = 1.0     # Multi-label
    graph_health_weight: float = 1.0          # Regression
    graph_degradation_weight: float = 1.0     # Regression
    graph_anomaly_weight: float = 1.0         # Multi-label
    graph_rul_weight: float = 1.0             # Regression
    
    # Huber loss delta for RUL (robust to outliers)
    rul_huber_delta: float = 1.0
    
    # Multi-label classification weights (optional)
    component_anomaly_class_weights: torch.Tensor | None = None  # [9]
    graph_anomaly_class_weights: torch.Tensor | None = None      # [9]
    
    # Reduction method
    reduction: str = 'mean'  # 'mean' or 'sum'


class MultiTaskLoss(nn.Module):
    """Multi-task loss for Phase 2 (6 tasks).
    
    Combines:
    - Component-level: health (MSE), anomaly (BCE)
    - Graph-level: health (MSE), degradation (MSE), anomaly (BCE), RUL (Huber)
    
    Examples:
        >>> config = MultiTaskLossConfig(
        ...     component_health_weight=1.0,
        ...     component_anomaly_weight=0.5,
        ...     graph_health_weight=1.0,
        ...     graph_degradation_weight=1.0,
        ...     graph_anomaly_weight=0.5,
        ...     graph_rul_weight=2.0,
        ... )
        >>> loss_fn = MultiTaskLoss(config)
        >>> 
        >>> # Forward pass with Phase 2 nested outputs
        >>> outputs = {
        ...     'component': {
        ...         'health': torch.randn(100, 1),     # [N, 1]
        ...         'anomaly': torch.randn(100, 9)     # [N, 9]
        ...     },
        ...     'graph': {
        ...         'health': torch.randn(4, 1),       # [B, 1]
        ...         'degradation': torch.randn(4, 1),  # [B, 1]
        ...         'anomaly': torch.randn(4, 9),      # [B, 9]
        ...         'rul': torch.randn(4, 1)           # [B, 1]
        ...     }
        ... }
        >>> targets = {
        ...     'component': {
        ...         'health': torch.rand(100, 1),
        ...         'anomaly': torch.randint(0, 2, (100, 9)).float()
        ...     },
        ...     'graph': {
        ...         'health': torch.rand(4, 1),
        ...         'degradation': torch.rand(4, 1),
        ...         'anomaly': torch.randint(0, 2, (4, 9)).float(),
        ...         'rul': torch.rand(4, 1) * 1000
        ...     }
        ... }
        >>> 
        >>> loss_dict = loss_fn(outputs, targets)
        >>> total_loss = loss_dict['total']
    """
    
    def __init__(self, config: MultiTaskLossConfig | None = None) -> None:
        """Initialize multi-task loss.
        
        Args:
            config: Loss configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or MultiTaskLossConfig()
        
        # Regression losses (MSE)
        self.mse_loss = nn.MSELoss(reduction=self.config.reduction)
        
        # Huber loss for RUL (robust to outliers)
        self.huber_loss = nn.HuberLoss(
            delta=self.config.rul_huber_delta,
            reduction=self.config.reduction
        )
        
        # Multi-label classification losses (BCE with logits)
        self.component_anomaly_loss = nn.BCEWithLogitsLoss(
            pos_weight=self.config.component_anomaly_class_weights,
            reduction=self.config.reduction
        )
        
        self.graph_anomaly_loss = nn.BCEWithLogitsLoss(
            pos_weight=self.config.graph_anomaly_class_weights,
            reduction=self.config.reduction
        )
        
        logger.info("MultiTaskLoss (Phase 2) initialized with 6 tasks")
        logger.debug("Config: %s", self.config)
    
    def forward(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        targets: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Compute multi-task loss (Phase 2).
        
        Args:
            outputs: Model outputs with nested structure:
                {
                    'component': {'health': [N, 1], 'anomaly': [N, 9]},
                    'graph': {'health': [B, 1], 'degradation': [B, 1], 
                              'anomaly': [B, 9], 'rul': [B, 1]}
                }
            targets: Ground truth with same structure
            
        Returns:
            Dictionary with individual losses and 'total'
        """
        losses = {}
        
        # Component-level losses
        if 'component' in outputs and 'component' in targets:
            # Health (regression)
            if 'health' in outputs['component'] and 'health' in targets['component']:
                losses['component_health'] = self.mse_loss(
                    outputs['component']['health'],
                    targets['component']['health']
                )
            
            # Anomaly (multi-label)
            if 'anomaly' in outputs['component'] and 'anomaly' in targets['component']:
                losses['component_anomaly'] = self.component_anomaly_loss(
                    outputs['component']['anomaly'],
                    targets['component']['anomaly']
                )
        
        # Graph-level losses
        if 'graph' in outputs and 'graph' in targets:
            # Health (regression)
            if 'health' in outputs['graph'] and 'health' in targets['graph']:
                losses['graph_health'] = self.mse_loss(
                    outputs['graph']['health'],
                    targets['graph']['health']
                )
            
            # Degradation (regression)
            if 'degradation' in outputs['graph'] and 'degradation' in targets['graph']:
                losses['graph_degradation'] = self.mse_loss(
                    outputs['graph']['degradation'],
                    targets['graph']['degradation']
                )
            
            # Anomaly (multi-label)
            if 'anomaly' in outputs['graph'] and 'anomaly' in targets['graph']:
                losses['graph_anomaly'] = self.graph_anomaly_loss(
                    outputs['graph']['anomaly'],
                    targets['graph']['anomaly']
                )
            
            # RUL (Huber loss, robust to outliers)
            if 'rul' in outputs['graph'] and 'rul' in targets['graph']:
                losses['graph_rul'] = self.huber_loss(
                    outputs['graph']['rul'],
                    targets['graph']['rul']
                )
        
        # Determine device from first available output
        device = 'cpu'
        if 'component' in outputs and outputs['component']:
            device = next(iter(outputs['component'].values())).device
        elif 'graph' in outputs and outputs['graph']:
            device = next(iter(outputs['graph'].values())).device
        
        # Compute weighted total
        total_loss = torch.tensor(0.0, device=device)
        
        if 'component_health' in losses:
            total_loss += self.config.component_health_weight * losses['component_health']
        if 'component_anomaly' in losses:
            total_loss += self.config.component_anomaly_weight * losses['component_anomaly']
        if 'graph_health' in losses:
            total_loss += self.config.graph_health_weight * losses['graph_health']
        if 'graph_degradation' in losses:
            total_loss += self.config.graph_degradation_weight * losses['graph_degradation']
        if 'graph_anomaly' in losses:
            total_loss += self.config.graph_anomaly_weight * losses['graph_anomaly']
        if 'graph_rul' in losses:
            total_loss += self.config.graph_rul_weight * losses['graph_rul']
        
        losses['total'] = total_loss
        
        return losses
    
    def update_anomaly_weights(
        self,
        component_anomaly_weights: torch.Tensor | None = None,
        graph_anomaly_weights: torch.Tensor | None = None
    ) -> None:
        """Update class weights for multi-label classification.
        
        Args:
            component_anomaly_weights: Pos weights for component anomalies [9]
            graph_anomaly_weights: Pos weights for graph anomalies [9]
        """
        if component_anomaly_weights is not None:
            self.component_anomaly_loss.pos_weight = component_anomaly_weights
            logger.info("Updated component anomaly class weights")
        
        if graph_anomaly_weights is not None:
            self.graph_anomaly_loss.pos_weight = graph_anomaly_weights
            logger.info("Updated graph anomaly class weights")


def compute_pos_weights(
    labels: torch.Tensor
) -> torch.Tensor:
    """Compute positive class weights for imbalanced multi-label data.
    
    Args:
        labels: Binary labels [num_samples, num_classes]
        
    Returns:
        Positive class weights [num_classes]
        
    Examples:
        >>> labels = torch.tensor([
        ...     [1, 0, 0],
        ...     [1, 1, 0],
        ...     [0, 0, 1]
        ... ])
        >>> weights = compute_pos_weights(labels)
        >>> weights  # [0.5, 2.0, 2.0] (inverse frequency)
    """
    # Count positive examples per class
    pos_counts = labels.sum(dim=0).float()
    
    # Count negative examples per class
    neg_counts = (1 - labels).sum(dim=0).float()
    
    # Avoid division by zero
    pos_counts = torch.clamp(pos_counts, min=1.0)
    
    # Positive weight = neg_count / pos_count
    weights = neg_counts / pos_counts
    
    return weights
