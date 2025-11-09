"""Temporal Graph Attention Network (T-GAT) for hydraulic anomaly detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from typing import Optional

from .config import config


class TemporalGAT(nn.Module):
    """Temporal Graph Attention Network."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_channels if i == 0 else hidden_channels * num_heads
            self.gat_layers.append(
                GATv2Conv(
                    in_dim,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False,
                )
            )
        
        # Temporal encoding (LSTM over node embeddings)
        self.temporal_lstm = nn.LSTM(
            hidden_channels * (num_heads if num_layers > 1 else 1),
            hidden_channels,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
        
        Returns:
            out: Graph-level predictions [batch_size, out_channels]
            node_embeddings: Node embeddings [num_nodes, hidden_channels]
            attention_weights: List of attention weights per layer
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layers with attention weights
        attention_weights = []
        for i, gat in enumerate(self.gat_layers):
            x, (edge_index_att, alpha) = gat(x, edge_index, return_attention_weights=True)
            attention_weights.append(alpha)
            
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        node_embeddings = x
        
        # Temporal encoding (simplified: assume single timestep for now)
        # In production: reshape to [batch_size, seq_len, hidden_dim]
        x_temporal = x.unsqueeze(1)  # [num_nodes, 1, hidden_dim]
        x_temporal, _ = self.temporal_lstm(x_temporal)
        x_temporal = x_temporal.squeeze(1)
        
        # Graph-level pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x_graph = global_mean_pool(x_temporal, batch)
        
        # Output projection
        out = self.output_proj(x_graph)
        
        return out, node_embeddings, attention_weights


class GNNClassifier(nn.Module):
    """GNN-based anomaly classifier."""
    
    def __init__(
        self,
        in_channels: int = config.num_node_features,
        hidden_channels: int = config.hidden_dim,
        num_classes: int = config.num_classes,
        num_layers: int = config.num_gat_layers,
        num_heads: int = config.num_heads,
        dropout: float = config.dropout,
    ):
        super().__init__()
        
        self.encoder = TemporalGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    
    def forward(self, data):
        """Forward pass.
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch
        
        Returns:
            logits: Class logits [batch_size, num_classes]
            embeddings: Node embeddings [num_nodes, hidden_dim]
            attention_weights: Attention weights for explainability
        """
        logits, embeddings, attention_weights = self.encoder(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
            batch=data.batch if hasattr(data, "batch") else None,
        )
        
        return logits, embeddings, attention_weights
