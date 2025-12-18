"""GNN models for hydraulic diagnostics.

Main components:
- UniversalTemporalGNN: GAT + LSTM with multi-task learning
- MultiTaskLoss: Joint optimization for node + graph predictions
- AttentionWeightExtractor: Interpretability tools
- Pooling layers: AttentionPooling, VirtualNodePooling

Examples:
    >>> from models import UniversalTemporalGNN, ModelConfig, MultiTaskLoss
    >>> 
    >>> # Initialize model
    >>> config = ModelConfig(node_features=34, edge_features=14)
    >>> model = UniversalTemporalGNN(config)
    >>> 
    >>> # Initialize loss
    >>> loss_fn = MultiTaskLoss()
    >>> 
    >>> # Training
    >>> outputs = model(data)
    >>> losses = loss_fn(
    ...     outputs['node_logits'], data.y_node,
    ...     outputs['graph_logits'], data.y_graph
    ... )
    >>> total_loss = losses['total']
"""

from .attention_weights import AttentionWeightExtractor
from .multi_task_loss import (
    MultiTaskLoss,
    MultiTaskLossConfig,
    compute_class_weights,
)
from .pooling import AttentionPooling, VirtualNodePooling
from .universal_temporal_gnn import ModelConfig, UniversalTemporalGNN

__all__ = [
    # Main model
    'UniversalTemporalGNN',
    'ModelConfig',
    # Loss functions
    'MultiTaskLoss',
    'MultiTaskLossConfig',
    'compute_class_weights',
    # Interpretability
    'AttentionWeightExtractor',
    # Pooling layers
    'AttentionPooling',
    'VirtualNodePooling',
]
