"""Universal Temporal GNN for hydraulic diagnostics.

Multi-level spatio-temporal graph neural network:
- Spatial: GATv2 (Graph Attention Networks v2)
- Temporal: ARMA-LSTM (Autoregressive Moving Average + LSTM)
- Multi-level: Component + Graph predictions

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class UniversalTemporalGNN(nn.Module):
    """Universal Temporal GNN for hydraulic system diagnostics.

    Architecture:
        1. Initial projection: [N, F] → [N, hidden]
        2. GATv2 layers (×3): Spatial feature learning
        3. Component-level heads: health & anomaly per node
        4. Global pooling: [N, hidden] → [B, hidden]
        5. LSTM: Temporal modeling
        6. Graph-level heads: health, degradation, anomaly, RUL

    Multi-Level Predictions:
        Component-level (per-node):
            - component_health: [N, 1] ∈ [0, 1]
            - component_anomaly: [N, 9] logits

        Graph-level (per-equipment):
            - health: [B, 1] ∈ [0, 1]
            - degradation: [B, 1] ∈ [0, 1]
            - anomaly: [B, 9] logits
            - rul: [B, 1] ∈ [0, ∞)

    Args:
        in_channels: Input node feature dimension
        hidden_channels: Hidden dimension
        num_heads: Number of attention heads (GATv2)
        num_gat_layers: Number of GATv2 layers
        lstm_hidden: LSTM hidden dimension
        lstm_layers: Number of LSTM layers
        dropout: Dropout probability
        use_compile: Use torch.compile (PyTorch 2.8)

    Examples:
        >>> model = UniversalTemporalGNN(
        ...     in_channels=34,
        ...     hidden_channels=128,
        ...     num_heads=8
        ... )
        >>> x = torch.randn(100, 34)  # 100 nodes, 34 features
        >>> edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
        >>> edge_attr = torch.randn(200, 8)  # 8 edge features
        >>> batch = torch.zeros(100, dtype=torch.long)  # Single graph
        >>>
        >>> output = model(x, edge_index, edge_attr, batch)
        >>> print(output['graph']['health'].shape)  # [1, 1]
        >>> print(output['component']['health'].shape)  # [100, 1]

    References:
        - GATv2: https://arxiv.org/abs/2105.14491
        - Multi-level GNN: https://arxiv.org/abs/2404.10324
        - Hydraulic diagnostics: https://pmc.ncbi.nlm.nih.gov/articles/PMC11125296/
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 8,
        num_gat_layers: int = 3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        use_compile: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_gat_layers = num_gat_layers
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        # Initial projection
        self.initial_projection = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        for _i in range(num_gat_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=8,  # Edge feature dimension
                    concat=True,
                    add_self_loops=True,
                )
            )

        # Graph normalization after GAT
        self.graph_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_gat_layers)]
        )

        # === Component-Level Heads (after GATv2, before pooling) ===

        self.component_health = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # [N, 1] ∈ [0, 1]
        )

        self.component_anomaly = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 9),  # [N, 9] logits for 9 anomaly types
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        # === Graph-Level Heads (after LSTM) ===

        self.health_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # [B, 1] ∈ [0, 1]
        )

        self.degradation_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # [B, 1] ∈ [0, 1]
        )

        self.anomaly_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 9),  # [B, 9] logits for 9 anomaly types
        )

        self.rul_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus(),  # [B, 1] ∈ [0, ∞) - RUL in hours
        )

        # Compile if requested (PyTorch 2.8 optimization)
        if use_compile:
            self.forward = torch.compile(self.forward)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass with multi-level predictions.

        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, 8]
            batch: Batch assignment [N]

        Returns:
            Nested dictionary:
            {
                'component': {
                    'health': [N, 1],
                    'anomaly': [N, 9]
                },
                'graph': {
                    'health': [B, 1],
                    'degradation': [B, 1],
                    'anomaly': [B, 9],
                    'rul': [B, 1]
                }
            }

        Examples:
            >>> x = torch.randn(100, 34)
            >>> edge_index = torch.randint(0, 100, (2, 200))
            >>> edge_attr = torch.randn(200, 8)
            >>> batch = torch.zeros(100, dtype=torch.long)
            >>>
            >>> output = model(x, edge_index, edge_attr, batch)
            >>> comp_health = output['component']['health']  # [100, 1]
            >>> graph_rul = output['graph']['rul']  # [1, 1]
        """
        # === 1. Initial Projection ===
        h = self.initial_projection(x)  # [N, F] → [N, hidden]

        # === 2. GATv2 Spatial Encoding ===
        for i, gat_layer in enumerate(self.gat_layers):
            # GAT with edge features
            h_new = gat_layer(h, edge_index, edge_attr)  # [N, hidden]

            # Graph normalization
            h_new = self.graph_norms[i](h_new)

            # Residual connection (skip first layer)
            h = h + h_new if i > 0 else h_new

            # ReLU activation
            h = torch.relu(h)

        # === 3. Component-Level Predictions ===
        component_health = self.component_health(h)  # [N, 1]
        component_anomaly = self.component_anomaly(h)  # [N, 9]

        # === 4. Global Pooling ===
        # Mean pooling for graph-level representation
        graph_repr = global_mean_pool(h, batch)  # [B, hidden]

        # === 5. LSTM Temporal Modeling ===
        # Add sequence dimension for LSTM
        lstm_input = graph_repr.unsqueeze(1)  # [B, 1, hidden]
        lstm_out, _ = self.lstm(lstm_input)  # [B, 1, lstm_hidden]
        lstm_out = lstm_out.squeeze(1)  # [B, lstm_hidden]

        # === 6. Graph-Level Predictions ===
        graph_health = self.health_head(lstm_out)  # [B, 1]
        graph_degradation = self.degradation_head(lstm_out)  # [B, 1]
        graph_anomaly = self.anomaly_head(lstm_out)  # [B, 9]
        graph_rul = self.rul_head(lstm_out)  # [B, 1]

        # === 7. Return Nested Structure ===
        return {
            "component": {"health": component_health, "anomaly": component_anomaly},
            "graph": {
                "health": graph_health,
                "degradation": graph_degradation,
                "anomaly": graph_anomaly,
                "rul": graph_rul,
            },
        }

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict[str, int | str]:
        """Get model information.

        Returns:
            Dictionary with model statistics
        """
        return {
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "num_heads": self.num_heads,
            "num_gat_layers": self.num_gat_layers,
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "num_parameters": self.get_num_parameters(),
            "dropout": self.dropout,
        }
