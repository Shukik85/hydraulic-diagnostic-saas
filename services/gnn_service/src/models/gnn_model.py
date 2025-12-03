"""Universal Temporal GNN for hydraulic diagnostics.

Главная модель: GATv2 (spatial) + ARMA-LSTM (temporal) + Multi-task head.

Architecture:
    1. Node feature extraction
    2. GATv2 layers с edge-conditioned attention
    3. Temporal aggregation с ARMA-LSTM
    4. Multi-task prediction head (health, degradation, anomaly)

PyTorch 2.8 Features:
    - torch.compile для 1.5x speedup
    - torch.inference_mode для inference
    - SDPA (Scaled Dot-Product Attention)

Python 3.14 Features:
    - Deferred annotations
    - Union types с pipe operator
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool

from src.models.attention import CrossTaskAttention
from src.models.layers import ARMAAttentionLSTM, EdgeConditionedGATv2Layer


class UniversalTemporalGNN(nn.Module):
    """Universal Temporal GNN для multi-label classification.
    
    Архитектура:
        1. Input projection
        2. GATv2 layers (spatial graph modeling)
        3. ARMA-LSTM (temporal sequence modeling)
        4. Multi-task head (health + degradation + anomaly)
    
    Features:
        - GATv2 dynamic attention (vs static GAT)
        - Edge-conditioned attention (hydraulic topology)
        - ARMA-based temporal attention (ICLR 2025)
        - Cross-task attention (task correlation)
        - torch.compile optimization (PyTorch 2.8)
    
    Edge Features (14D - Phase 3.1):
        Static (8D):
            - diameter_norm
            - length_norm
            - cross_section_area_norm
            - pressure_loss_coeff_norm
            - pressure_rating_norm
            - material_onehot (3D: steel, rubber, composite)
        
        Dynamic (6D):
            - flow_rate_lpm (physics-based or measured)
            - pressure_drop_bar
            - temperature_delta_c
            - vibration_level_g
            - age_hours
            - maintenance_score
    
    Args:
        in_channels: Размерность входных node features
        hidden_channels: Размерность скрытых слоёв
        num_heads: Количество attention heads в GAT
        num_gat_layers: Количество GATv2 layers
        lstm_hidden: Размерность LSTM hidden state
        lstm_layers: Количество LSTM layers
        ar_order: Autoregressive order для ARMA attention
        ma_order: Moving average order для ARMA attention
        dropout: Dropout rate
        use_edge_features: Использовать edge features
        edge_feature_dim: Размерность edge features (14 с Phase 3.1)
        use_compile: Применить torch.compile (PyTorch 2.8)
        compile_mode: Режим компиляции
    
    Examples:
        >>> model = UniversalTemporalGNN(
        ...     in_channels=12,
        ...     hidden_channels=128,
        ...     num_heads=8,
        ...     num_gat_layers=3,
        ...     lstm_hidden=256,
        ...     lstm_layers=2,
        ...     edge_feature_dim=14,  # Phase 3.1: 14D edges
        ...     use_compile=True
        ... )
        >>> 
        >>> # Forward pass
        >>> health, degradation, anomaly = model(
        ...     x=node_features,
        ...     edge_index=edge_index,
        ...     edge_attr=edge_features,  # [E, 14]
        ...     batch=batch_assignment
        ... )
    
    Note:
        ⚠️ Breaking change in Phase 3.1:
        edge_feature_dim: 8 → 14
        Old checkpoints (v1.x) несовместимы.
        Требуется retraining с v2.0.0+
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 8,
        num_gat_layers: int = 3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        ar_order: int = 3,
        ma_order: int = 2,
        dropout: float = 0.3,
        use_edge_features: bool = True,
        edge_feature_dim: int = 14,  # Phase 3.1: 14D (was 8D)
        use_compile: bool = True,
        compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_gat_layers = num_gat_layers
        self.lstm_hidden = lstm_hidden
        self.use_edge_features = use_edge_features
        self.edge_feature_dim = edge_feature_dim  # Store for checkpoint
        self.use_compile = use_compile

        # Input projection
        self.input_projection = nn.Linear(in_channels, hidden_channels)

        # Edge feature projection (если используются)
        if use_edge_features:
            self.edge_projection = nn.Linear(edge_feature_dim, hidden_channels // num_heads)

        # GATv2 layers для spatial modeling
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            layer = EdgeConditionedGATv2Layer(
                in_channels=hidden_channels if i > 0 else hidden_channels,
                out_channels=hidden_channels // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_channels // num_heads if use_edge_features else None,
                concat=True,
                add_self_loops=True,
                bias=True
            )
            self.gat_layers.append(layer)

        # Layer normalization после каждого GAT
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_gat_layers)
        ])

        # ARMA-Attention LSTM для temporal modeling
        self.temporal_lstm = ARMAAttentionLSTM(
            input_dim=hidden_channels,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            ar_order=ar_order,
            ma_order=ma_order,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False
        )

        # Multi-task learning head с cross-task attention
        self.task_attention = CrossTaskAttention(
            hidden_dim=lstm_hidden,
            num_tasks=3,
            num_heads=4
        )

        # Task-specific heads
        self.health_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] range
        )

        self.degradation_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] range
        )

        self.anomaly_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 9)  # 9 anomaly types (from AnomalyType enum)
        )

        # Initialize weights
        self._initialize_weights()

        # Apply torch.compile (PyTorch 2.8)
        if use_compile:
            self._apply_torch_compile(compile_mode)

    def _initialize_weights(self) -> None:
        """Инициализация весов модели.
        
        Использует Xavier/Glorot для linear layers и orthogonal для LSTM.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def _apply_torch_compile(self, mode: str) -> None:
        """Применить torch.compile к forward pass (PyTorch 2.8).
        
        Args:
            mode: "default", "reduce-overhead", или "max-autotune"
        
        Note:
            Компиляция выполняется при первом forward pass.
            Expect ~30s warmup, затем 1.5x speedup.
        """
        # Compile forward method
        self.forward = torch.compile(
            self.forward,
            mode=mode,
            fullgraph=False,  # Allow graph breaks
            dynamic=True  # Support variable batch sizes
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]
    ]:
        """Forward pass через GNN.
        
        Args:
            x: Node features [N, F_in]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, 14] (Phase 3.1: 14D)
            batch: Batch assignment [N] (опционально)
            return_attention: Вернуть attention weights для визуализации
        
        Returns:
            health: Health scores [B, 1] в range [0, 1]
            degradation: Degradation rates [B, 1] в range [0, 1]
            anomaly: Anomaly logits [B, 9] (raw logits, не probabilities)
            attention_weights: List of attention weights (если requested)
        
        Shape:
            - Input: x [N, F_in], edge_index [2, E], edge_attr [E, 14]
            - Output: health [B, 1], degradation [B, 1], anomaly [B, 9]
            где N = total nodes, E = total edges, B = batch size
        """
        attention_weights = []

        # 1. Input projection
        x = self.input_projection(x)  # [N, F_in] -> [N, H]
        x = F.relu(x)

        # 2. Edge feature projection (если есть)
        if self.use_edge_features and edge_attr is not None:
            edge_attr = self.edge_projection(edge_attr)  # [E, 14] -> [E, H//num_heads]

        # 3. GATv2 spatial processing
        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            # GATv2 с edge conditioning
            if return_attention:
                x_new, attn = gat_layer(
                    x, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
                attention_weights.append(attn)
            else:
                x_new = gat_layer(x, edge_index, edge_attr=edge_attr)

            # Residual connection + normalization
            if i > 0:
                x = x + x_new  # Skip connection
            else:
                x = x_new

            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        # 4. Graph-level pooling для каждого equipment
        if batch is None:
            # Single graph - global mean pooling
            x_pooled = x.mean(dim=0, keepdim=True)  # [1, H]
        else:
            # Batch of graphs
            x_pooled = global_mean_pool(x, batch)  # [B, H]

        # 5. Temporal modeling с ARMA-LSTM
        # Reshape для LSTM: [B, T=1, H] (здесь T=1, т.к. graph уже агрегирован)
        # В реальности будет sequence of graphs
        x_temporal = x_pooled.unsqueeze(1)  # [B, 1, H]

        # ARMA-Attention LSTM
        lstm_out, (h_n, c_n) = self.temporal_lstm(x_temporal)  # lstm_out: [B, 1, lstm_hidden]

        # Use final hidden state
        final_hidden = h_n[-1]  # [B, lstm_hidden]

        # 6. Multi-task head с cross-task attention
        # Create task representations
        task_repr = self.task_attention(final_hidden)  # [3, B, lstm_hidden]

        # 7. Task-specific predictions
        health = self.health_head(task_repr[0])  # [B, 1]
        degradation = self.degradation_head(task_repr[1])  # [B, 1]
        anomaly = self.anomaly_head(task_repr[2])  # [B, 9]

        if return_attention:
            return health, degradation, anomaly, attention_weights

        return health, degradation, anomaly

    def get_model_config(self) -> dict[str, int | float | bool | str]:
        """Получить конфигурацию модели для checkpoint.
        
        Returns:
            Dictionary с параметрами модели
        """
        return {
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "num_heads": self.num_heads,
            "num_gat_layers": self.num_gat_layers,
            "lstm_hidden": self.lstm_hidden,
            "use_edge_features": self.use_edge_features,
            "edge_feature_dim": self.edge_feature_dim,  # Phase 3.1: Track edge dim
            "use_compile": self.use_compile,
            "model_version": "v2.0.0",  # Phase 3.1: Version for checkpoint compatibility
        }

    @torch.inference_mode()  # PyTorch 2.8 optimization
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Inference mode prediction.
        
        Returns:
            Dictionary с predictions и metadata
        """
        self.eval()

        health, degradation, anomaly = self.forward(
            x, edge_index, edge_attr, batch, return_attention=False
        )

        # Apply softmax к anomaly logits
        anomaly_probs = F.softmax(anomaly, dim=-1)

        return {
            "health": health,
            "degradation": degradation,
            "anomaly_logits": anomaly,
            "anomaly_probs": anomaly_probs,
        }
