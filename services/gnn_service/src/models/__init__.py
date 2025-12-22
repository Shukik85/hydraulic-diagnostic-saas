"""GNN v2 models for hydraulic diagnostics.

Version 2.0.2 (Production-Hardened):
- PyTorch Geometric 2.x compatibility
- PyTorch 2.8+ torch.compile() support
- Python 3.14 native type hints
- Modern attention mechanisms
- All senior review findings addressed
- Comprehensive validation and error handling

Main components:
- UniversalTemporalGNNv2: GAT + LSTM with multi-task learning
- MultiTaskLoss: Joint optimization for node + graph predictions
- AttentionWeightExtractor: Interpretability tools
- Pooling layers: AttentionPooling, VirtualNodeAugmentation

Backward Compatibility:
- UniversalTemporalGNN (v1) → UniversalTemporalGNNv2 (alias)
- VirtualNodePooling → VirtualNodeAugmentation (alias)

Examples:
    >>> from models import UniversalTemporalGNNv2, ModelConfig, MultiTaskLoss
    >>> 
    >>> # Initialize model
    >>> config = ModelConfig(node_features=34, edge_features=14)
    >>> model = UniversalTemporalGNNv2(config)
    >>> 
    >>> # Or use v1 alias for backward compatibility
    >>> from models import UniversalTemporalGNN  # Same as v2
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
from .pooling import AttentionPooling, VirtualNodeAugmentation, VirtualNodePooling
from .universal_temporal_gnn import ModelConfig, UniversalTemporalGNNv2

# Backward compatibility alias (v1 → v2)
UniversalTemporalGNN = UniversalTemporalGNNv2

__all__ = [
    # Main model (v2.0.2)
    'UniversalTemporalGNNv2',
    'ModelConfig',
    # Backward compatibility (v1 alias)
    'UniversalTemporalGNN',
    # Loss functions
    'MultiTaskLoss',
    'MultiTaskLossConfig',
    'compute_class_weights',
    # Interpretability
    'AttentionWeightExtractor',
    # Pooling layers
    'AttentionPooling',
    'VirtualNodeAugmentation',  # New name (v2.0.2)
    'VirtualNodePooling',       # Backward compatibility alias
]

__version__ = '2.0.2'
