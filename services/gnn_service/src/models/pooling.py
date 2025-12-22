"""Graph pooling layers for size-invariant representations.

Implements:
- AttentionPooling: Learnable importance weights per node
- VirtualNodeAugmentation: Virtual node for graph-level features
- Replaces naive mean/max pooling for better generalization

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter


class AttentionPooling(nn.Module):
    """Attention-based graph pooling for size-invariant representations.
    
    Learns importance weights for each node and computes weighted average.
    More expressive than mean/max pooling, especially for varying graph sizes.
    
    Examples:
        >>> pooling = AttentionPooling(hidden_dim=128)
        >>> 
        >>> # Node features from GNN
        >>> x = torch.randn(50, 128)  # [num_nodes, hidden_dim]
        >>> batch = torch.zeros(50, dtype=torch.long)  # Single graph
        >>> 
        >>> # Pool to graph-level representation
        >>> graph_embedding = pooling(x, batch)  # [1, 128]
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1
    ) -> None:
        """Initialize attention pooling.
        
        Args:
            hidden_dim: Dimension of node features
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Attention scoring network
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pool node features to graph-level representation.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]. If None, assumes single graph.
            
        Returns:
            Graph-level features [batch_size, hidden_dim]
        """
        # Handle None batch (single graph)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Handle empty graphs
        if batch.numel() == 0:
            return torch.zeros(0, self.hidden_dim, dtype=x.dtype, device=x.device)
        
        # Compute attention scores
        scores = self.attention_mlp(x)  # [num_nodes, 1]
        
        # Softmax per graph (respects batch assignment)
        attention_weights = softmax(scores, batch, dim=0)  # [num_nodes, 1]
        
        # Weighted sum
        weighted_features = x * attention_weights  # [num_nodes, hidden_dim]
        
        # Sum per graph using scatter (best practice)
        graph_features = scatter(weighted_features, batch, dim=0, reduce='sum')
        
        return graph_features
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get attention weights for interpretability.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]. If None, assumes single graph.
            
        Returns:
            Attention weights [num_nodes]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        scores = self.attention_mlp(x)
        attention_weights = softmax(scores, batch, dim=0)
        return attention_weights.squeeze(-1)


class VirtualNodeAugmentation(nn.Module):
    """Virtual node augmentation for size-invariant graph representations.
    
    Adds a learnable virtual node that aggregates information from all nodes.
    Helps models generalize across different graph sizes.
    
    Note: This is a single-pass augmentation. For iterative virtual node
    updates (as in original papers), integrate into GNN message passing.
    
    Examples:
        >>> augmentation = VirtualNodeAugmentation(node_dim=128, virtual_dim=64)
        >>> 
        >>> x = torch.randn(50, 128)  # [num_nodes, node_dim]
        >>> batch = torch.zeros(50, dtype=torch.long)
        >>> 
        >>> # Add virtual node representations
        >>> x_with_virtual = augmentation(x, batch)  # [50, 128 + 64]
    """
    
    def __init__(
        self,
        node_dim: int,
        virtual_dim: int,
        dropout: float = 0.1
    ) -> None:
        """Initialize virtual node augmentation.
        
        Args:
            node_dim: Dimension of node features
            virtual_dim: Dimension of virtual node embedding
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.virtual_dim = virtual_dim
        
        # Virtual node embedding (initialized with zeros for stable training)
        self.virtual_embedding = nn.Parameter(torch.zeros(virtual_dim))
        
        # Aggregation network
        self.aggregate_mlp = nn.Sequential(
            nn.Linear(node_dim, virtual_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Update network
        self.update_mlp = nn.Sequential(
            nn.Linear(virtual_dim * 2, virtual_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Add virtual node information to node features.
        
        Args:
            x: Node features [num_nodes, node_dim]
            batch: Batch assignment [num_nodes]. If None, assumes single graph.
            
        Returns:
            Enhanced features [num_nodes, node_dim + virtual_dim]
        """
        # Handle None batch (single graph)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Handle empty graphs
        if batch.numel() == 0:
            return torch.empty(0, self.node_dim + self.virtual_dim, 
                             dtype=x.dtype, device=x.device)
        
        # Safe batch_size calculation
        batch_size = int(batch.max()) + 1
        
        # Aggregate node features per graph
        aggregated = self.aggregate_mlp(x)  # [num_nodes, virtual_dim]
        
        # Sum per graph using scatter (best practice)
        graph_aggregated = scatter(
            aggregated, batch, dim=0, dim_size=batch_size, reduce='sum'
        )  # [batch_size, virtual_dim]
        
        # Update virtual node embedding
        virtual_batch = self.virtual_embedding.unsqueeze(0).expand(
            batch_size, -1
        )  # [batch_size, virtual_dim]
        
        virtual_updated = self.update_mlp(
            torch.cat([virtual_batch, graph_aggregated], dim=1)
        )  # [batch_size, virtual_dim]
        
        # Broadcast back to nodes
        virtual_per_node = virtual_updated[batch]  # [num_nodes, virtual_dim]
        
        # Concatenate with original features
        x_enhanced = torch.cat([x, virtual_per_node], dim=1)
        
        return x_enhanced


# Backward compatibility alias
VirtualNodePooling = VirtualNodeAugmentation
