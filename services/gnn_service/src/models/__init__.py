"""GNN models for hydraulic diagnostics.

Universal Temporal GNN с:
- GATv2 для spatial modeling
- ARMA-LSTM для temporal modeling  
- Multi-task learning (health + degradation + anomaly)
- Edge-conditioned attention
- PyTorch 2.8 torch.compile support

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from src.models.gnn_model import UniversalTemporalGNN
from src.models.layers import (
    EdgeConditionedGATv2Layer,
    ARMAAttentionLSTM,
    SpectralTemporalLayer,
)
from src.models.attention import (
    MultiHeadAttention,
    CrossTaskAttention,
    EdgeAwareAttention,
)
from src.models.utils import (
    initialize_model,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    model_summary,
)

__all__ = [
    # Main model
    "UniversalTemporalGNN",
    # Layers
    "EdgeConditionedGATv2Layer",
    "ARMAAttentionLSTM",
    "SpectralTemporalLayer",
    # Attention
    "MultiHeadAttention",
    "CrossTaskAttention",
    "EdgeAwareAttention",
    # Utils
    "initialize_model",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "model_summary",
]
