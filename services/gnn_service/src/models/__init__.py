"""GNN v2 models for hydraulic diagnostics.

Version 2.1.0 (Phase 2 - Multi-Level Predictions):
- 6-task architecture (4 graph + 2 component)
- Graph-level: health, degradation, anomaly (9), RUL
- Component-level: health, anomaly (9)
- Nested output structure
- PyTorch Geometric 2.x compatibility
- PyTorch 2.8+ torch.compile() support
- Python 3.14 native type hints

Main components:
- UniversalTemporalGNNv2: GAT + LSTM with 6-task predictions
- ModelConfig: Configuration for v2 architecture
- MultiTaskLoss: Joint optimization (needs update for 6 tasks)
- AttentionWeightExtractor: Interpretability tools
- Pooling layers: AttentionPooling, VirtualNodeAugmentation

Backward Compatibility:
- UniversalTemporalGNN (v1) → UniversalTemporalGNNv2 (alias)
- VirtualNodePooling → VirtualNodeAugmentation (alias)

Examples:
    >>> from models import UniversalTemporalGNNv2, ModelConfig
    >>> 
    >>> # Initialize model (Phase 2)
    >>> config = ModelConfig(
    ...     node_features=34,
    ...     edge_features=14,
    ...     graph_anomaly_classes=9,
    ...     component_anomaly_classes=9,
    ... )
    >>> model = UniversalTemporalGNNv2(config)
    >>> 
    >>> # Forward pass
    >>> outputs = model(data)  # Accepts PyG Data object
    >>> # outputs['component']['health']  # [N, 1]
    >>> # outputs['graph']['rul']  # [B, 1]
    >>> 
    >>> # Or use v1 alias for backward compatibility
    >>> from models import UniversalTemporalGNN  # Same as v2
    >>> model = UniversalTemporalGNN(config)
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
    # Main model (v2.1.0)
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
    'VirtualNodeAugmentation',  # New name (v2.0.2+)
    'VirtualNodePooling',       # Backward compatibility alias
]

__version__ = '2.1.0'
