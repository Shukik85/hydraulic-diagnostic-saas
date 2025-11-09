"""T-GAT (Temporal Graph Attention Network) implementation.

Based on:
- Xu et al. "Inductive Representation Learning on Temporal Graphs" (2020)
- Modified for hydraulic system diagnostics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class TemporalAttentionLayer(nn.Module):
    """Temporal attention layer для time-aware graph learning."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=num_heads,
            dropout=dropout,
            concat=False,  # Average heads
            add_self_loops=True,
        )
        
        # Temporal encoding
        self.temporal_proj = nn.Linear(in_dim, out_dim)
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_dim)
        
        Returns:
            Updated node features and attention weights
        """
        # Spatial attention (GAT)
        h, attention_weights = self.gat(
            x,
            edge_index,
            return_attention_weights=True,
        )
        
        # Temporal projection
        h_temporal = self.temporal_proj(x)
        
        # Combine spatial + temporal
        h = h + h_temporal
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        return h, attention_weights


class TGAT(nn.Module):
    """T-GAT classifier для hydraulic system diagnostics."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # T-GAT layers
        self.layers = nn.ModuleList([
            TemporalAttentionLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_dim)
            batch: Batch assignment (num_nodes,)
        
        Returns:
            Logits and attention weights per layer
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        
        # T-GAT layers
        attention_weights_list = []
        for layer in self.layers:
            h, attention_weights = layer(h, edge_index, edge_attr)
            attention_weights_list.append(attention_weights)
        
        # Graph-level pooling
        if batch is None:
            # Single graph (inference)
            h_graph = torch.mean(h, dim=0, keepdim=True)
        else:
            # Batch of graphs
            h_graph = global_mean_pool(h, batch)
        
        # Classification
        logits = self.classifier(h_graph)
        
        return logits, attention_weights_list
