"""Universal Temporal GNN v2 for hydraulic system diagnostics.

Combines:
- Graph Attention Networks (GAT) for spatial relationships
- LSTM for temporal patterns
- Multi-task learning for component health + anomaly detection
- Size-invariant design for different graph topologies

Version 2.0.0:
- PyTorch Geometric 2.x features (edge_dim, return_attention_weights)
- PyTorch 2.8+ compilation support
- Python 3.14 native type hints
- Modern pooling strategies (AttentionPooling, VirtualNode)

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
from torch_geometric.nn import GATConv

from .pooling import AttentionPooling, VirtualNodePooling

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for UniversalTemporalGNNv2."""
    
    # Model version
    version: str = "2.0.0"
    
    # Input dimensions
    node_features: int = 34
    edge_features: int = 14
    
    # GAT configuration
    gat_hidden_dim: int = 128
    gat_num_layers: int = 3
    gat_num_heads: int = 4
    gat_dropout: float = 0.1
    gat_concat_heads: bool = True
    
    # LSTM configuration
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    lstm_bidirectional: bool = False
    
    # Topology-aware components
    use_virtual_nodes: bool = True
    virtual_node_dim: int = 64
    use_attention_pooling: bool = True
    graph_size_as_feature: bool = True
    
    # Multi-task heads
    component_health_num_classes: int = 5  # healthy, degraded, worn, leaking, failed
    anomaly_type_num_classes: int = 4  # normal, parallel_overload, sequential_cascade, cavitation
    head_hidden_dim: int = 64
    head_dropout: float = 0.2
    
    # Attention extraction
    extract_attention_weights: bool = True
    return_attention_weights: bool = False  # Return in forward pass


class UniversalTemporalGNNv2(nn.Module):
    """Universal Temporal GNN v2 for hydraulic diagnostics.
    
    Version 2.0.0 improvements:
    - PyG 2.x GATConv with edge_dim support
    - Native attention weight extraction
    - AttentionPooling for size-invariant graphs
    - VirtualNode support for cross-topology generalization
    - PyTorch 2.8+ torch.compile() compatible
    
    Architecture:
    1. Node/edge encoding
    2. Multi-layer GAT (spatial, with edge features)
    3. Optional: LSTM (temporal)
    4. AttentionPooling → graph representation
    5. Dual prediction heads (node-level + graph-level)
    
    Examples:
        >>> config = ModelConfig(
        ...     node_features=34,
        ...     edge_features=14,
        ...     gat_hidden_dim=128,
        ...     lstm_hidden_dim=128
        ... )
        >>> model = UniversalTemporalGNNv2(config)
        >>> 
        >>> # Single graph
        >>> data = Data(x=..., edge_index=..., edge_attr=...)
        >>> outputs = model(data)
        >>> node_logits = outputs['node_logits']  # [num_nodes, 5]
        >>> graph_logits = outputs['graph_logits']  # [1, 4]
        >>> 
        >>> # Temporal sequence
        >>> sequence = [data_t0, data_t1, ..., data_t9]  # List[Data]
        >>> outputs = model(sequence, temporal=True)
        >>> graph_logits = outputs['graph_logits']  # [1, 4]
    """
    
    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize model.
        
        Args:
            config: Model configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or ModelConfig()
        
        # Effective hidden dimension after GAT
        if self.config.gat_concat_heads:
            self.gat_output_dim = self.config.gat_hidden_dim * self.config.gat_num_heads
        else:
            self.gat_output_dim = self.config.gat_hidden_dim
        
        # Node feature encoder
        self.node_encoder = nn.Linear(
            self.config.node_features,
            self.config.gat_hidden_dim
        )
        
        # Edge feature encoder (for GAT)
        self.edge_encoder = nn.Linear(
            self.config.edge_features,
            self.config.gat_hidden_dim
        )
        
        # GAT layers (PyG 2.x with edge_dim)
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(
            GATConv(
                self.config.gat_hidden_dim,
                self.config.gat_hidden_dim,
                heads=self.config.gat_num_heads,
                dropout=self.config.gat_dropout,
                concat=self.config.gat_concat_heads,
                edge_dim=self.config.gat_hidden_dim,  # PyG 2.x feature
                add_self_loops=True
            )
        )
        
        # Subsequent GAT layers
        for _ in range(self.config.gat_num_layers - 1):
            self.gat_layers.append(
                GATConv(
                    self.gat_output_dim,
                    self.config.gat_hidden_dim,
                    heads=self.config.gat_num_heads,
                    dropout=self.config.gat_dropout,
                    concat=self.config.gat_concat_heads,
                    edge_dim=self.config.gat_hidden_dim,  # PyG 2.x feature
                    add_self_loops=True
                )
            )
        
        # Virtual node (optional)
        if self.config.use_virtual_nodes:
            self.virtual_node_pooling = VirtualNodePooling(
                node_dim=self.gat_output_dim,
                virtual_dim=self.config.virtual_node_dim,
                dropout=self.config.gat_dropout
            )
            self.pooling_input_dim = self.gat_output_dim + self.config.virtual_node_dim
        else:
            self.virtual_node_pooling = None
            self.pooling_input_dim = self.gat_output_dim
        
        # Attention pooling for graph-level representation
        if self.config.use_attention_pooling:
            self.pooling = AttentionPooling(
                hidden_dim=self.pooling_input_dim,
                dropout=self.config.gat_dropout
            )
        else:
            self.pooling = None  # Will use mean pooling
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.pooling_input_dim,
            hidden_size=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            dropout=self.config.lstm_dropout if self.config.lstm_num_layers > 1 else 0.0,
            bidirectional=self.config.lstm_bidirectional,
            batch_first=True
        )
        
        lstm_output_dim = (
            self.config.lstm_hidden_dim * 2 if self.config.lstm_bidirectional
            else self.config.lstm_hidden_dim
        )
        
        # Graph size embedding (optional)
        if self.config.graph_size_as_feature:
            self.size_embedding = nn.Embedding(101, 16)  # Support up to 100 nodes
            graph_feature_dim = lstm_output_dim + 16
        else:
            self.size_embedding = None
            graph_feature_dim = lstm_output_dim
        
        # Component health head (node-level)
        self.component_health_head = nn.Sequential(
            nn.Linear(self.gat_output_dim, self.config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.head_hidden_dim, self.config.component_health_num_classes)
        )
        
        # Anomaly type head (graph-level)
        self.anomaly_type_head = nn.Sequential(
            nn.Linear(graph_feature_dim, self.config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.head_hidden_dim, self.config.anomaly_type_num_classes)
        )
        
        # Attention storage for interpretability
        self.last_attention_weights: dict[str, torch.Tensor] = {}
        
        logger.info("UniversalTemporalGNNv2 (v%s) initialized", self.config.version)
        logger.info("GAT output dim: %d, LSTM output dim: %d", self.gat_output_dim, lstm_output_dim)
    
    def forward(
        self,
        data,
        temporal: bool = False,
        return_attention: bool = False
    ) -> dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            data: PyG Data object or List[Data] if temporal=True
            temporal: Whether to use LSTM for temporal modeling
            return_attention: Return attention weights in output
            
        Returns:
            Dictionary with 'node_logits', 'graph_logits', optional 'attention_weights'
        """
        if temporal:
            return self._forward_temporal(data, return_attention)
        else:
            return self._forward_single(data, return_attention)
    
    def _forward_single(
        self,
        data,
        return_attention: bool = False
    ) -> dict[str, torch.Tensor]:
        """Forward pass for single graph.
        
        Args:
            data: PyG Data object
            return_attention: Return attention weights
            
        Returns:
            Dictionary with predictions
        """
        x = data.x  # [num_nodes, node_features]
        edge_index = data.edge_index  # [2, num_edges]
        edge_attr = data.edge_attr  # [num_edges, edge_features]
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device
        )
        
        # Encode features
        x = self.node_encoder(x)  # [num_nodes, gat_hidden_dim]
        edge_attr = self.edge_encoder(edge_attr)  # [num_edges, gat_hidden_dim]
        
        # GAT layers (PyG 2.x return_attention_weights)
        attention_weights = {}
        for i, gat_layer in enumerate(self.gat_layers):
            x_out = gat_layer(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            
            # Extract attention if tuple
            if isinstance(x_out, tuple):
                x, (edge_idx, attn) = x_out
                if self.config.extract_attention_weights:
                    attention_weights[f'gat_layer_{i}'] = attn
            else:
                x = x_out
            
            x = F.relu(x)
            x = F.dropout(x, p=self.config.gat_dropout, training=self.training)
        
        # Store attention for later extraction
        self.last_attention_weights = attention_weights
        
        # Component health prediction (node-level)
        node_logits = self.component_health_head(x)  # [num_nodes, num_classes]
        
        # Add virtual node features (optional)
        if self.virtual_node_pooling is not None:
            x = self.virtual_node_pooling(x, batch)  # [num_nodes, gat_out + virtual_dim]
        
        # Pool to graph-level representation
        if self.pooling is not None:
            graph_repr = self.pooling(x, batch)  # [batch_size, pooling_input_dim]
        else:
            # Mean pooling fallback
            batch_size = batch.max().item() + 1
            graph_repr = torch.zeros(
                batch_size, x.size(1), dtype=x.dtype, device=x.device
            )
            graph_repr.index_add_(0, batch, x)
            counts = torch.bincount(batch, minlength=batch_size).float().unsqueeze(1)
            graph_repr = graph_repr / counts.clamp(min=1)
        
        # LSTM (single timestep - just transform)
        lstm_out, _ = self.lstm(graph_repr.unsqueeze(1))  # [batch, 1, lstm_hidden]
        lstm_out = lstm_out.squeeze(1)  # [batch, lstm_hidden]
        
        # Add graph size feature (optional)
        if self.size_embedding is not None:
            # Count nodes per graph
            batch_size = batch.max().item() + 1
            node_counts = torch.bincount(batch, minlength=batch_size)
            node_counts = torch.clamp(node_counts, max=100)  # Cap at 100
            size_embed = self.size_embedding(node_counts)  # [batch, 16]
            lstm_out = torch.cat([lstm_out, size_embed], dim=1)
        
        # Anomaly type prediction (graph-level)
        graph_logits = self.anomaly_type_head(lstm_out)  # [batch, num_classes]
        
        outputs = {
            'node_logits': node_logits,
            'graph_logits': graph_logits
        }
        
        if return_attention or self.config.return_attention_weights:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def _forward_temporal(
        self,
        data_sequence: list,
        return_attention: bool = False
    ) -> dict[str, torch.Tensor]:
        """Forward pass for temporal sequence.
        
        Args:
            data_sequence: List of PyG Data objects (time sequence)
            return_attention: Return attention weights
            
        Returns:
            Dictionary with predictions
        """
        # Process each timestep through GAT
        timestep_reprs = []
        all_node_logits = []
        
        for data_t in data_sequence:
            outputs = self._forward_single(data_t, return_attention=False)
            
            # Get graph representation (before final prediction)
            # Re-extract by running partial forward
            x = data_t.x
            edge_index = data_t.edge_index
            edge_attr = data_t.edge_attr
            batch = data_t.batch if hasattr(data_t, 'batch') else torch.zeros(
                x.size(0), dtype=torch.long, device=x.device
            )
            
            # Encode and run through GAT
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)
            
            for gat_layer in self.gat_layers:
                x = gat_layer(x, edge_index, edge_attr=edge_attr)
                if isinstance(x, tuple):
                    x, _ = x
                x = F.relu(x)
                x = F.dropout(x, p=self.config.gat_dropout, training=self.training)
            
            # Virtual node
            if self.virtual_node_pooling is not None:
                x = self.virtual_node_pooling(x, batch)
            
            # Pool
            if self.pooling is not None:
                graph_repr = self.pooling(x, batch)
            else:
                batch_size = batch.max().item() + 1
                graph_repr = torch.zeros(
                    batch_size, x.size(1), dtype=x.dtype, device=x.device
                )
                graph_repr.index_add_(0, batch, x)
                counts = torch.bincount(batch, minlength=batch_size).float().unsqueeze(1)
                graph_repr = graph_repr / counts.clamp(min=1)
            
            timestep_reprs.append(graph_repr)
            all_node_logits.append(outputs['node_logits'])
        
        # Stack temporal sequence
        sequence_tensor = torch.stack(timestep_reprs, dim=1)  # [batch, seq_len, repr_dim]
        
        # LSTM
        lstm_out, _ = self.lstm(sequence_tensor)  # [batch, seq_len, lstm_hidden]
        
        # Take last timestep
        lstm_final = lstm_out[:, -1, :]  # [batch, lstm_hidden]
        
        # Add graph size
        if self.size_embedding is not None:
            batch = data_sequence[-1].batch if hasattr(data_sequence[-1], 'batch') else torch.zeros(
                data_sequence[-1].x.size(0), dtype=torch.long, device=data_sequence[-1].x.device
            )
            batch_size = batch.max().item() + 1
            node_counts = torch.bincount(batch, minlength=batch_size)
            node_counts = torch.clamp(node_counts, max=100)
            size_embed = self.size_embedding(node_counts)
            lstm_final = torch.cat([lstm_final, size_embed], dim=1)
        
        # Graph prediction
        graph_logits = self.anomaly_type_head(lstm_final)
        
        # Use node predictions from last timestep
        node_logits = all_node_logits[-1]
        
        outputs = {
            'node_logits': node_logits,
            'graph_logits': graph_logits
        }
        
        if return_attention or self.config.return_attention_weights:
            outputs['attention_weights'] = self.last_attention_weights
        
        return outputs
    
    def get_attention_weights(self) -> dict[str, torch.Tensor]:
        """Get last computed attention weights.
        
        Returns:
            Dictionary of attention weights per layer
        """
        return self.last_attention_weights
