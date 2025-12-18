"""Universal Temporal GNN v2 for hydraulic system diagnostics.

Combines:
- Graph Attention Networks v2 (GATv2) for spatial relationships
- LSTM for temporal patterns
- Multi-task learning for component health + anomaly detection
- Size-invariant design for different graph topologies

Version 2.0.2 (Production-Hardened):
- Fixed all senior review findings
- Proper configuration validation
- Scalable size embedding (supports 10k+ nodes)
- Flexible temporal processing
- Batch optimization for sequences

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)

References:
    - GATv2: "How Attentive are Graph Attention Networks?" (ICLR 2022)
      https://arxiv.org/abs/2105.14491
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

from .pooling import AttentionPooling, VirtualNodePooling

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for UniversalTemporalGNNv2.
    
    All parameters validated in __post_init__.
    
    Attributes:
        version: Model version string
        node_features: Input node feature dimension (must be > 0)
        edge_features: Input edge feature dimension (must be > 0)
        gat_hidden_dim: Hidden dimension for GATv2 layers (must be > 0)
        gat_num_layers: Number of GATv2 layers (must be >= 1)
        gat_num_heads: Number of attention heads (must be >= 1)
        gat_dropout: Dropout rate for GATv2 (must be in [0, 1])
        gat_concat_heads: Whether to concatenate attention heads
        lstm_hidden_dim: LSTM hidden dimension (must be > 0)
        lstm_num_layers: Number of LSTM layers (must be >= 1)
        lstm_dropout: LSTM dropout (must be in [0, 1])
        lstm_bidirectional: Whether LSTM is bidirectional
        use_virtual_nodes: Enable virtual node pooling
        virtual_node_dim: Virtual node dimension (must be > 0 if used)
        use_attention_pooling: Use attention-based pooling
        graph_size_as_feature: Add graph size as feature (binned)
        component_health_num_classes: Node-level classes (must be >= 2)
        anomaly_type_num_classes: Graph-level classes (must be >= 2)
        head_hidden_dim: Prediction head hidden dim (must be > 0)
        head_dropout: Prediction head dropout (must be in [0, 1])
    """
    
    # Model version
    version: str = "2.0.2"
    
    # Input dimensions
    node_features: int = 34
    edge_features: int = 14
    
    # GATv2 configuration
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
    component_health_num_classes: int = 5
    anomaly_type_num_classes: int = 4
    head_hidden_dim: int = 64
    head_dropout: float = 0.2
    
    def __post_init__(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate positive dimensions
        if self.node_features <= 0:
            raise ValueError(f"node_features must be > 0, got {self.node_features}")
        if self.edge_features <= 0:
            raise ValueError(f"edge_features must be > 0, got {self.edge_features}")
        if self.gat_hidden_dim <= 0:
            raise ValueError(f"gat_hidden_dim must be > 0, got {self.gat_hidden_dim}")
        if self.lstm_hidden_dim <= 0:
            raise ValueError(f"lstm_hidden_dim must be > 0, got {self.lstm_hidden_dim}")
        if self.head_hidden_dim <= 0:
            raise ValueError(f"head_hidden_dim must be > 0, got {self.head_hidden_dim}")
        
        # Validate layer counts
        if self.gat_num_layers < 1:
            raise ValueError(f"gat_num_layers must be >= 1, got {self.gat_num_layers}")
        if self.lstm_num_layers < 1:
            raise ValueError(f"lstm_num_layers must be >= 1, got {self.lstm_num_layers}")
        if self.gat_num_heads < 1:
            raise ValueError(f"gat_num_heads must be >= 1, got {self.gat_num_heads}")
        
        # Validate dropout rates
        if not 0.0 <= self.gat_dropout <= 1.0:
            raise ValueError(f"gat_dropout must be in [0,1], got {self.gat_dropout}")
        if not 0.0 <= self.lstm_dropout <= 1.0:
            raise ValueError(f"lstm_dropout must be in [0,1], got {self.lstm_dropout}")
        if not 0.0 <= self.head_dropout <= 1.0:
            raise ValueError(f"head_dropout must be in [0,1], got {self.head_dropout}")
        
        # Validate class counts
        if self.component_health_num_classes < 2:
            raise ValueError(
                f"component_health_num_classes must be >= 2, "
                f"got {self.component_health_num_classes}"
            )
        if self.anomaly_type_num_classes < 2:
            raise ValueError(
                f"anomaly_type_num_classes must be >= 2, "
                f"got {self.anomaly_type_num_classes}"
            )
        
        # Validate virtual node settings
        if self.use_virtual_nodes and self.virtual_node_dim <= 0:
            raise ValueError(
                f"virtual_node_dim must be > 0 when use_virtual_nodes=True, "
                f"got {self.virtual_node_dim}"
            )


class UniversalTemporalGNNv2(nn.Module):
    """Universal Temporal GNN v2 for hydraulic diagnostics.
    
    Version 2.0.2 - Production-hardened after senior review.
    
    Architecture:
    1. Node/edge encoding
    2. Multi-layer GATv2 (spatial, with edge features)
    3. Optional VirtualNode
    4. AttentionPooling → graph representation
    5. LSTM (temporal mode) or Linear projection (single mode)
    6. Dual prediction heads (node-level + graph-level)
    
    Examples:
        >>> config = ModelConfig(node_features=34, edge_features=14)
        >>> model = UniversalTemporalGNNv2(config)
        >>> 
        >>> # Single graph
        >>> outputs = model(data, temporal=False)
        >>> node_logits = outputs['node_logits']
        >>> graph_logits = outputs['graph_logits']
        >>> 
        >>> # Temporal sequence (all timesteps)
        >>> outputs = model(
        ...     sequence, 
        ...     temporal=True, 
        ...     return_all_timesteps=True
        ... )
        >>> node_logits_seq = outputs['node_logits_seq']  # List[Tensor]
    """
    
    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize model.
        
        Args:
            config: Model configuration. Uses defaults if None.
            
        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__()
        self.config = config or ModelConfig()
        
        # Node feature encoder
        self.node_encoder = nn.Linear(
            self.config.node_features,
            self.config.gat_hidden_dim
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Linear(
            self.config.edge_features,
            self.config.gat_hidden_dim
        )
        
        # Build GATv2 layers with explicit dimension tracking
        self.gat_layers = nn.ModuleList()
        in_channels = self.config.gat_hidden_dim
        
        for layer_idx in range(self.config.gat_num_layers):
            out_channels = self.config.gat_hidden_dim
            
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=self.config.gat_num_heads,
                    dropout=self.config.gat_dropout,
                    concat=self.config.gat_concat_heads,
                    edge_dim=self.config.gat_hidden_dim,
                    add_self_loops=True,
                    share_weights=False
                )
            )
            
            # Update in_channels for next layer
            if self.config.gat_concat_heads:
                in_channels = out_channels * self.config.gat_num_heads
            else:
                in_channels = out_channels
        
        # Final GATv2 output dimension
        self.gat_output_dim = in_channels
        
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
        
        # Attention pooling
        if self.config.use_attention_pooling:
            self.pooling = AttentionPooling(
                hidden_dim=self.pooling_input_dim,
                dropout=self.config.gat_dropout
            )
        else:
            self.pooling = None
        
        # LSTM for temporal mode
        lstm_output_dim = self._get_lstm_output_dim()
        self.lstm = nn.LSTM(
            input_size=self.pooling_input_dim,
            hidden_size=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            dropout=self.config.lstm_dropout if self.config.lstm_num_layers > 1 else 0.0,
            bidirectional=self.config.lstm_bidirectional,
            batch_first=True
        )
        
        # Single-graph projection (replaces LSTM for single mode)
        self.single_graph_projection = nn.Linear(
            self.pooling_input_dim,
            lstm_output_dim
        )
        
        # Graph size embedding (binned for scalability)
        if self.config.graph_size_as_feature:
            # Binned sizes: 1-10, 11-50, 51-200, 201-1000, 1001+
            self.size_bins = [10, 50, 200, 1000, 10000]
            self.size_embedding = nn.Embedding(len(self.size_bins) + 1, 16)
            graph_feature_dim = lstm_output_dim + 16
        else:
            self.size_bins = None
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
        
        logger.debug("UniversalTemporalGNNv2 (v%s) initialized", self.config.version)
        logger.debug("GATv2 output dim: %d, LSTM output dim: %d", self.gat_output_dim, lstm_output_dim)
        logger.debug("Graph feature dim: %d", graph_feature_dim)
    
    def _get_lstm_output_dim(self) -> int:
        """Calculate LSTM output dimension.
        
        Returns:
            LSTM output dimension (hidden_dim * 2 if bidirectional)
        """
        multiplier = 2 if self.config.lstm_bidirectional else 1
        return self.config.lstm_hidden_dim * multiplier
    
    def _get_size_bin(self, num_nodes: int) -> int:
        """Get binned size category for graph.
        
        Bins: [1-10], [11-50], [51-200], [201-1000], [1001-10000], [10000+]
        
        Args:
            num_nodes: Number of nodes in graph
            
        Returns:
            Bin index (0-5)
        """
        for i, bin_max in enumerate(self.size_bins):
            if num_nodes <= bin_max:
                return i
        return len(self.size_bins)  # Large graph category
    
    def _validate_batch(self, data: Data) -> torch.Tensor:
        """Validate and extract batch tensor.
        
        Args:
            data: PyG Data object
            
        Returns:
            Batch tensor [num_nodes]
            
        Raises:
            ValueError: If batch info missing for multi-graph batch
        """
        if hasattr(data, 'batch') and data.batch is not None:
            return data.batch
        
        # Single graph case
        if hasattr(data, 'num_graphs') and data.num_graphs is not None and data.num_graphs > 1:
            msg = "Batch attribute required for multi-graph batches"
            raise ValueError(msg)
        
        # Create batch tensor for single graph
        num_nodes = data.x.size(0)
        return torch.zeros(num_nodes, dtype=torch.long, device=data.x.device)
    
    def _encode_nodes(
        self,
        data,
        return_attention: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode node features through GATv2 layers.
        
        Args:
            data: PyG Data object
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (node_embeddings, attention_weights)
            Note: attention_weights is always a dict (empty if not requested)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # Encode features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # GATv2 layers
        attention_weights: dict[str, torch.Tensor] = {}
        
        for i, gat_layer in enumerate(self.gat_layers):
            x_out = gat_layer(
                x, edge_index,
                edge_attr=edge_attr,
                return_attention_weights=return_attention
            )
            
            # Extract attention if requested
            if return_attention and isinstance(x_out, tuple):
                x, (_, attn) = x_out
                attention_weights[f'gatv2_layer_{i}'] = attn.detach()
            else:
                x = x_out if not isinstance(x_out, tuple) else x_out[0]
            
            x = F.relu(x)
            x = F.dropout(x, p=self.config.gat_dropout, training=self.training)
        
        return x, attention_weights
    
    def _encode_graph(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Pool node embeddings to graph-level representation.
        
        Args:
            node_embeddings: Node features [num_nodes, dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph representation [batch_size, dim]
        """
        # Validate batch is not None
        if batch is None:
            batch = torch.zeros(
                node_embeddings.size(0),
                dtype=torch.long,
                device=node_embeddings.device
            )
        
        # Virtual node (optional)
        if self.virtual_node_pooling is not None:
            node_embeddings = self.virtual_node_pooling(node_embeddings, batch)
        
        # Pooling
        if self.pooling is not None:
            graph_repr = self.pooling(node_embeddings, batch)
        else:
            graph_repr = global_mean_pool(node_embeddings, batch)
        
        return graph_repr
    
    def forward(
        self,
        data,
        temporal: bool = False,
        return_attention: bool = False,
        return_all_timesteps: bool = False
    ) -> dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            data: PyG Data object or List[Data] if temporal=True
            temporal: Whether to use LSTM for temporal modeling
            return_attention: Return attention weights in output
            return_all_timesteps: Return predictions for all timesteps (temporal only)
            
        Returns:
            Dictionary with predictions:
            - 'node_logits': Node predictions [num_nodes, num_classes]
            - 'graph_logits': Graph predictions [batch_size, num_classes]
            - 'attention_weights': Attention (if return_attention=True)
            - 'node_logits_seq': All timestep node preds (if return_all_timesteps=True)
            - 'attention_seq': All timestep attention (if return_all_timesteps=True)
        """
        if temporal:
            return self._forward_temporal(data, return_attention, return_all_timesteps)
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
        batch = self._validate_batch(data)
        
        # Encode nodes
        node_emb, attention_weights = self._encode_nodes(data, return_attention)
        
        # Component health prediction (node-level)
        node_logits = self.component_health_head(node_emb)
        
        # Pool to graph-level
        graph_repr = self._encode_graph(node_emb, batch)
        
        # Linear projection (no LSTM for single graphs)
        graph_features = self.single_graph_projection(graph_repr)
        graph_features = F.relu(graph_features)
        
        # Add graph size feature (binned, optional)
        if self.size_embedding is not None:
            batch_size = batch.max().item() + 1
            node_counts = torch.bincount(batch, minlength=batch_size)
            
            # Convert to bin indices
            size_bins = torch.tensor(
                [self._get_size_bin(count.item()) for count in node_counts],
                dtype=torch.long,
                device=batch.device
            )
            size_embed = self.size_embedding(size_bins)
            graph_features = torch.cat([graph_features, size_embed], dim=1)
        
        # Anomaly type prediction (graph-level)
        graph_logits = self.anomaly_type_head(graph_features)
        
        outputs = {
            'node_logits': node_logits,
            'graph_logits': graph_logits
        }
        
        if return_attention and attention_weights:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def _forward_temporal(
        self,
        data_sequence: list,
        return_attention: bool = False,
        return_all_timesteps: bool = False
    ) -> dict[str, torch.Tensor]:
        """Forward pass for temporal sequence.
        
        Args:
            data_sequence: List of PyG Data objects (time sequence)
            return_attention: Return attention weights
            return_all_timesteps: Return predictions for all timesteps
            
        Returns:
            Dictionary with predictions
        """
        # Process each timestep
        timestep_reprs = []
        all_node_logits = [] if return_all_timesteps else None
        all_attention = [] if return_all_timesteps and return_attention else None
        last_node_logits = None
        last_attention: dict[str, torch.Tensor] = {}
        
        for t, data_t in enumerate(data_sequence):
            batch = self._validate_batch(data_t)
            
            # Encode nodes (single source of truth)
            node_emb, attn_weights = self._encode_nodes(data_t, return_attention)
            
            # Node predictions
            node_logits_t = self.component_health_head(node_emb)
            
            if return_all_timesteps:
                all_node_logits.append(node_logits_t.detach())
                if return_attention and attn_weights:
                    all_attention.append(attn_weights)
            
            # Keep last timestep for final output
            if t == len(data_sequence) - 1:
                last_node_logits = node_logits_t
                if return_attention:
                    last_attention = attn_weights
            
            # Pool to graph-level
            graph_repr = self._encode_graph(node_emb, batch)
            timestep_reprs.append(graph_repr)
        
        # Stack temporal sequence
        sequence_tensor = torch.stack(timestep_reprs, dim=1)  # [batch, seq_len, dim]
        
        # LSTM
        lstm_out, _ = self.lstm(sequence_tensor)
        lstm_final = lstm_out[:, -1, :]  # Take last timestep
        
        # Graph prediction (no size embedding in temporal mode)
        graph_logits = self.anomaly_type_head(lstm_final)
        
        outputs = {
            'node_logits': last_node_logits,
            'graph_logits': graph_logits
        }
        
        if return_attention and last_attention:
            outputs['attention_weights'] = last_attention
        
        if return_all_timesteps:
            outputs['node_logits_seq'] = all_node_logits
            if return_attention and all_attention:
                outputs['attention_seq'] = all_attention
        
        return outputs
