"""Universal Temporal GNN v2 for hydraulic system diagnostics.

Combines:
- Graph Attention Networks v2 (GATv2) for spatial relationships
- LSTM for temporal patterns
- Multi-task learning for component health + anomaly detection
- Size-invariant design for different graph topologies

Version 2.1.0 (Phase 2 - Multi-Level Predictions):
- 6-task architecture: 4 graph-level + 2 component-level
- Graph: health, degradation, anomaly (9 classes), RUL
- Component: health (regression), anomaly (9 classes)
- Nested output structure for better organization
- Production-ready multi-level diagnostics

Python 3.14 Features:
    - Deferred annotations (PEP 563)
    - Builtin generic types (PEP 585)

References:
    - GATv2: "How Attentive are Graph Attention Networks?" (ICLR 2022)
      https://arxiv.org/abs/2105.14491
    - Multi-level predictions: Issue #116
      https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/116
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

from .pooling import AttentionPooling, VirtualNodeAugmentation

# Backward compatibility for existing code/tests
VirtualNodePooling = VirtualNodeAugmentation

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for UniversalTemporalGNNv2.
    
    All parameters validated in __post_init__.
    
    Phase 2 Architecture (6 tasks):
    
    Graph-level predictions (4):
        - health_score: [B, 1] ∈ [0,1] - Overall system health (regression)
        - degradation_rate: [B, 1] ∈ [0,1] - System degradation rate (regression)
        - anomaly_flags: [B, 9] ∈ {0,1}^9 - 9 anomaly types (multi-label)
        - rul_hours: [B, 1] ∈ [0,∞) - Remaining Useful Life (regression)
    
    Component-level predictions (2):
        - component_health: [N, 1] ∈ [0,1] - Per-component health (regression)
        - component_anomaly: [N, 9] ∈ {0,1}^9 - Per-component anomalies (multi-label)
    
    Design decisions:
    - No size_embedding: Ensures consistency between single/temporal modes.
    - Component predictions use only GATv2 (no LSTM) for direct node attribution.
    - Graph predictions use LSTM for temporal context in temporal mode.
    
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
        use_virtual_nodes: Enable virtual node augmentation
        virtual_node_dim: Virtual node dimension (must be > 0 if used)
        use_attention_pooling: Use attention-based pooling
        graph_anomaly_classes: Number of graph-level anomaly classes (must be >= 2)
        component_anomaly_classes: Number of component-level anomaly classes (must be >= 2)
        head_hidden_dim: Prediction head hidden dim (must be > 0)
        head_dropout: Prediction head dropout (must be in [0, 1])
    """
    
    # Model version
    version: str = "2.1.0"
    
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
    
    # Multi-task heads (Phase 2)
    graph_anomaly_classes: int = 9
    component_anomaly_classes: int = 9
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
        
        # Validate anomaly class counts (Phase 2)
        if self.graph_anomaly_classes < 2:
            raise ValueError(
                f"graph_anomaly_classes must be >= 2, "
                f"got {self.graph_anomaly_classes}"
            )
        if self.component_anomaly_classes < 2:
            raise ValueError(
                f"component_anomaly_classes must be >= 2, "
                f"got {self.component_anomaly_classes}"
            )
        
        # Validate virtual node settings
        if self.use_virtual_nodes and self.virtual_node_dim <= 0:
            raise ValueError(
                f"virtual_node_dim must be > 0 when use_virtual_nodes=True, "
                f"got {self.virtual_node_dim}"
            )


class UniversalTemporalGNNv2(nn.Module):
    """Universal Temporal GNN v2 for hydraulic diagnostics.
    
    Version 2.1.0 - Phase 2: Multi-Level Predictions (6 tasks)
    
    Architecture:
    1. Node/edge encoding
    2. Multi-layer GATv2 (spatial, with edge features)
    3. Optional VirtualNodeAugmentation
    4. AttentionPooling → graph representation
    5. LSTM (temporal mode) or Linear projection (single mode)
    6. Multi-level prediction heads:
       - Component-level (from GATv2): health [N,1], anomaly [N,9]
       - Graph-level (from LSTM/projection): health [B,1], degradation [B,1], 
         anomaly [B,9], RUL [B,1]
    
    Mode differences:
    - Single: GNN → projection → graph heads (direct path)
    - Temporal: GNN → LSTM → graph heads (sequence modeling)
    - Component predictions: Same for both modes (no LSTM)
    
    Examples:
        >>> config = ModelConfig(node_features=34, edge_features=14)
        >>> model = UniversalTemporalGNNv2(config)
        >>> 
        >>> # Single graph
        >>> outputs = model(data, temporal=False)
        >>> component_health = outputs['component']['health']  # [N, 1]
        >>> graph_rul = outputs['graph']['rul']  # [B, 1]
        >>> 
        >>> # Temporal sequence
        >>> outputs = model(sequence, temporal=True)
        >>> graph_health = outputs['graph']['health']  # [B, 1]
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
        
        # Virtual node augmentation (optional)
        if self.config.use_virtual_nodes:
            self.virtual_node_pooling = VirtualNodeAugmentation(
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
        
        # ===== Component-level heads (from GATv2, no LSTM) =====
        # Component health: [N, 1] regression
        self.component_health_head = nn.Sequential(
            nn.Linear(self.gat_output_dim, self.config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.head_hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Component anomaly: [N, 9] multi-label classification
        self.component_anomaly_head = nn.Sequential(
            nn.Linear(self.gat_output_dim, self.config.head_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.head_hidden_dim * 2, self.config.component_anomaly_classes)
            # No sigmoid here - will be applied in loss function
        )
        
        # ===== Graph-level heads (from LSTM/projection) =====
        # Graph health: [B, 1] regression
        self.graph_health_head = nn.Sequential(
            nn.Linear(lstm_output_dim, self.config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.head_hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Graph degradation: [B, 1] regression
        self.graph_degradation_head = nn.Sequential(
            nn.Linear(lstm_output_dim, self.config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.head_hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Graph anomaly: [B, 9] multi-label classification
        self.graph_anomaly_head = nn.Sequential(
            nn.Linear(lstm_output_dim, self.config.head_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.head_hidden_dim * 2, self.config.graph_anomaly_classes)
            # No sigmoid here - will be applied in loss function
        )
        
        # Graph RUL: [B, 1] regression (hours until failure)
        self.graph_rul_head = nn.Sequential(
            nn.Linear(lstm_output_dim, self.config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.head_hidden_dim, 1),
            nn.Softplus()  # Ensures positive output
        )
        
        logger.debug("UniversalTemporalGNNv2 (v%s) initialized", self.config.version)
        logger.debug("GATv2 output dim: %d, LSTM output dim: %d", self.gat_output_dim, lstm_output_dim)
        logger.debug("Phase 2: 6 tasks (4 graph + 2 component)")
        logger.debug("  Component: health [N,1], anomaly [N,%d]", self.config.component_anomaly_classes)
        logger.debug("  Graph: health [B,1], degradation [B,1], anomaly [B,%d], RUL [B,1]", 
                    self.config.graph_anomaly_classes)
    
    def _get_lstm_output_dim(self) -> int:
        """Calculate LSTM output dimension.
        
        Returns:
            LSTM output dimension (hidden_dim * 2 if bidirectional)
        """
        multiplier = 2 if self.config.lstm_bidirectional else 1
        return self.config.lstm_hidden_dim * multiplier
    
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
        
        # Virtual node augmentation (optional)
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
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            data: PyG Data object or List[Data] if temporal=True
            temporal: Whether to use LSTM for temporal modeling
            return_attention: Return attention weights in output
            return_all_timesteps: Return predictions for all timesteps (temporal only)
            
        Returns:
            Dictionary with nested predictions:
            {
                'component': {
                    'health': Tensor [N, 1],
                    'anomaly': Tensor [N, 9]
                },
                'graph': {
                    'health': Tensor [B, 1],
                    'degradation': Tensor [B, 1],
                    'anomaly': Tensor [B, 9],
                    'rul': Tensor [B, 1]
                },
                'attention_weights': dict (optional),
                'component_seq': list (optional, if return_all_timesteps),
                'attention_seq': list (optional, if return_all_timesteps)
            }
        """
        if temporal:
            return self._forward_temporal(data, return_attention, return_all_timesteps)
        else:
            return self._forward_single(data, return_attention)
    
    def _forward_single(
        self,
        data,
        return_attention: bool = False
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass for single graph.
        
        Path: GNN → projection → graph heads
        Component predictions: Directly from GATv2 (no LSTM)
        
        Args:
            data: PyG Data object
            return_attention: Return attention weights
            
        Returns:
            Dictionary with nested predictions
        """
        batch = self._validate_batch(data)
        
        # Encode nodes through GATv2
        node_emb, attention_weights = self._encode_nodes(data, return_attention)
        
        # ===== Component-level predictions (from GATv2) =====
        component_health = self.component_health_head(node_emb)      # [N, 1]
        component_anomaly = self.component_anomaly_head(node_emb)    # [N, 9]
        
        # Pool to graph-level
        graph_repr = self._encode_graph(node_emb, batch)
        
        # Linear projection (no LSTM for single graphs)
        graph_features = self.single_graph_projection(graph_repr)
        graph_features = F.relu(graph_features)
        
        # ===== Graph-level predictions (from projection) =====
        graph_health = self.graph_health_head(graph_features)           # [B, 1]
        graph_degradation = self.graph_degradation_head(graph_features) # [B, 1]
        graph_anomaly = self.graph_anomaly_head(graph_features)         # [B, 9]
        graph_rul = self.graph_rul_head(graph_features)                 # [B, 1]
        
        outputs = {
            'component': {
                'health': component_health,
                'anomaly': component_anomaly
            },
            'graph': {
                'health': graph_health,
                'degradation': graph_degradation,
                'anomaly': graph_anomaly,
                'rul': graph_rul
            }
        }
        
        if return_attention and attention_weights:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def _forward_temporal(
        self,
        data_sequence: list,
        return_attention: bool = False,
        return_all_timesteps: bool = False
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass for temporal sequence.
        
        Path: GNN → LSTM → graph heads
        Component predictions: Directly from GATv2 (no LSTM)
        
        Args:
            data_sequence: List of PyG Data objects (time sequence)
            return_attention: Return attention weights
            return_all_timesteps: Return predictions for all timesteps
            
        Returns:
            Dictionary with nested predictions
        """
        # Process each timestep
        timestep_reprs = []
        all_component_preds = [] if return_all_timesteps else None
        all_attention = [] if return_all_timesteps and return_attention else None
        last_component_health = None
        last_component_anomaly = None
        last_attention: dict[str, torch.Tensor] = {}
        
        for t, data_t in enumerate(data_sequence):
            batch = self._validate_batch(data_t)
            
            # Encode nodes through GATv2
            node_emb, attn_weights = self._encode_nodes(data_t, return_attention)
            
            # Component predictions (from GATv2, no LSTM)
            component_health_t = self.component_health_head(node_emb)
            component_anomaly_t = self.component_anomaly_head(node_emb)
            
            if return_all_timesteps:
                all_component_preds.append({
                    'health': component_health_t.detach(),
                    'anomaly': component_anomaly_t.detach()
                })
                if return_attention and attn_weights:
                    all_attention.append(attn_weights)
            
            # Keep last timestep for final output
            if t == len(data_sequence) - 1:
                last_component_health = component_health_t
                last_component_anomaly = component_anomaly_t
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
        
        # ===== Graph-level predictions (from LSTM) =====
        graph_health = self.graph_health_head(lstm_final)
        graph_degradation = self.graph_degradation_head(lstm_final)
        graph_anomaly = self.graph_anomaly_head(lstm_final)
        graph_rul = self.graph_rul_head(lstm_final)
        
        outputs = {
            'component': {
                'health': last_component_health,
                'anomaly': last_component_anomaly
            },
            'graph': {
                'health': graph_health,
                'degradation': graph_degradation,
                'anomaly': graph_anomaly,
                'rul': graph_rul
            }
        }
        
        if return_attention and last_attention:
            outputs['attention_weights'] = last_attention
        
        if return_all_timesteps:
            outputs['component_seq'] = all_component_preds
            if return_attention and all_attention:
                outputs['attention_seq'] = all_attention
        
        return outputs
