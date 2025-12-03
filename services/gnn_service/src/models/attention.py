"""Attention mechanisms для GNN.

Custom attention layers:
- MultiHeadAttention - standard multi-head attention
- CrossTaskAttention - для multi-task learning
- EdgeAwareAttention - для edge feature integration

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention.
    
    Классический Transformer attention с Scaled Dot-Product.
    
    Args:
        embed_dim: Embedding dimensionality
        num_heads: Number of attention heads
        dropout: Dropout rate
        bias: Use bias in projections
    
    Examples:
        >>> attn = MultiHeadAttention(embed_dim=256, num_heads=8)
        >>> out, attn_weights = attn(query, key, value)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            query: Query tensor [B, T_q, E]
            key: Key tensor [B, T_k, E]
            value: Value tensor [B, T_v, E]
            attn_mask: Attention mask [T_q, T_k] or None
            return_attention: Return attention weights
        
        Returns:
            output: Attention output [B, T_q, E]
            attn_weights: Attention weights [B, num_heads, T_q, T_k] (if requested)
        """
        B, T_q, E = query.shape
        T_k = key.size(1)

        # Project to Q, K, V
        Q = self.q_proj(query)  # [B, T_q, E]
        K = self.k_proj(key)  # [B, T_k, E]
        V = self.v_proj(value)  # [B, T_v, E]

        # Reshape to multi-head
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, T_q, head_dim]
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, T_k, head_dim]
        V = V.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, T_v, head_dim]

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, heads, T_q, T_k]

        # Apply mask (if provided)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, heads, T_q, T_k]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, heads, T_q, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, T_q, heads, head_dim]
        attn_output = attn_output.view(B, T_q, E)  # [B, T_q, E]

        # Output projection
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights

        return output


class CrossTaskAttention(nn.Module):
    """Cross-task attention для multi-task learning.
    
    Моделирует корреляции между задачами:
    - Низкий health -> высокая degradation
    - Высокая degradation -> вероятность anomaly
    
    Reference:
        Multi-task Graph Anomaly Detection (Microsoft, 2022)
        +11.4% F1-score improvement
    
    Args:
        hidden_dim: Hidden dimensionality
        num_tasks: Number of tasks (3: health, degradation, anomaly)
        num_heads: Number of attention heads
    
    Examples:
        >>> cross_attn = CrossTaskAttention(hidden_dim=256, num_tasks=3)
        >>> task_repr = cross_attn(shared_repr)  # [B, H] -> [3, B, H]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_tasks: int = 3,
        num_heads: int = 4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.num_heads = num_heads

        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Multi-head attention для task interaction
        self.task_interaction = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=False  # [T, B, E] format
        )

        # Task-specific projections
        self.task_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Shared representation [B, H]
        
        Returns:
            task_repr: Task-specific representations [num_tasks, B, H]
        """
        B, H = x.shape

        # Shared encoding
        shared = self.shared_encoder(x)  # [B, H]

        # Create task representations
        task_repr = torch.stack([
            proj(shared) for proj in self.task_projections
        ], dim=0)  # [num_tasks, B, H]

        # Cross-task attention
        # Each task attends to other tasks
        task_repr_attended, attn_weights = self.task_interaction(
            query=task_repr,
            key=task_repr,
            value=task_repr
        )  # [num_tasks, B, H]

        # Residual connection
        task_repr = task_repr + task_repr_attended

        return task_repr


class EdgeAwareAttention(nn.Module):
    """Edge-aware attention mechanism.
    
    Attention с учётом edge features для modulation.
    
    Args:
        node_dim: Node feature dimensionality
        edge_dim: Edge feature dimensionality
        num_heads: Number of attention heads
    
    Examples:
        >>> attn = EdgeAwareAttention(node_dim=128, edge_dim=8, num_heads=8)
        >>> out = attn(x, edge_index, edge_attr)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 8
    ):
        super().__init__()

        assert node_dim % num_heads == 0

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        # Node projections
        self.node_proj = nn.Linear(node_dim, node_dim)

        # Edge-to-attention mapping
        self.edge_to_attn = nn.Sequential(
            nn.Linear(edge_dim, num_heads),
            nn.Sigmoid()  # Gate: [0, 1]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [N, node_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
        
        Returns:
            out: Updated node features [N, node_dim]
        """
        N = x.size(0)
        E = edge_index.size(1)

        # Project nodes
        x_proj = self.node_proj(x)  # [N, node_dim]

        # Compute edge-modulated attention
        edge_gates = self.edge_to_attn(edge_attr)  # [E, num_heads]

        # Source and target nodes
        source = edge_index[0]
        target = edge_index[1]

        # Aggregate messages from neighbors
        x_source = x_proj[source]  # [E, node_dim]
        x_target = x_proj[target]  # [E, node_dim]

        # Reshape for multi-head
        x_source = x_source.view(E, self.num_heads, self.head_dim)  # [E, heads, head_dim]
        x_target = x_target.view(E, self.num_heads, self.head_dim)  # [E, heads, head_dim]

        # Edge-gated attention
        messages = x_source * edge_gates.unsqueeze(-1)  # [E, heads, head_dim]

        # Scatter to target nodes
        messages_flat = messages.view(E, -1)  # [E, node_dim]
        out = torch.zeros(N, self.node_dim, device=x.device)  # [N, node_dim]
        out.index_add_(0, target, messages_flat)

        return out
