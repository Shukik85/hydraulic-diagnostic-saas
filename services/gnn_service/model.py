"""
Temporal Graph Attention Network model for hydraulic diagnostics.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import model_config
from torch_geometric.nn import GATv2Conv, global_mean_pool

logger = logging.getLogger(__name__)


class TemporalGAT(nn.Module):
    """
    Temporal Graph Attention Network for hydraulic system diagnostics.

    Architecture:
    - GATv2Conv layers for spatial dependencies
    - LSTM for temporal dependencies
    - Multi-label classification for 7 components

    Args:
        num_node_features: Number of input features per node (15: 5 raw + 5 normalized + 5 deviation)
        hidden_dim: Hidden dimension size
        num_classes: Number of output classes (7 components)
        num_gat_layers: Number of GAT layers
        num_heads: Number of attention heads
        gat_dropout: Dropout rate for GAT layers
        num_lstm_layers: Number of LSTM layers
        lstm_dropout: Dropout rate for LSTM
    """

    def __init__(
        self,
        num_node_features: int = model_config.num_node_features,
        hidden_dim: int = model_config.hidden_dim,
        num_classes: int = model_config.num_classes,
        num_gat_layers: int = model_config.num_gat_layers,
        num_heads: int = model_config.num_heads,
        gat_dropout: float = model_config.gat_dropout,
        num_lstm_layers: int = model_config.num_lstm_layers,
        lstm_dropout: float = model_config.lstm_dropout,
    ):  # sourcery skip: assign-if-exp
        super().__init__()
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers

        # GAT layers for spatial processing
        self.gat_layers = nn.ModuleList()

        # ✅ ВСЕ слои одинаковые (без if/else логики!)
        for i in range(num_gat_layers):
            if i == 0:  # noqa: SIM108
                in_channels = num_node_features  # 15
            else:
                in_channels = hidden_dim * num_heads  # 384

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,  # 96
                    heads=num_heads,  # 4 (не 1!)
                    dropout=gat_dropout,
                    concat=True,  # True (не False!)
                )
            )

        # Batch normalization after each GAT layer
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(num_gat_layers)]
        )

        # LSTM for temporal processing
        lstm_input_size = hidden_dim * num_heads
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,
        )

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Attention weights storage for explainability
        self.attention_weights = None

        logger.info(
            f"TemporalGAT initialized: "
            f"{num_gat_layers} GAT layers, {num_heads} heads, "
            f"{num_lstm_layers} LSTM layers"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        sequence_length: int = model_config.sequence_length,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass of the model.

        Args:
            x: Node features [num_nodes * batch_size, num_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes * batch_size]
            sequence_length: Length of temporal sequence

        Returns:
            logits: Output logits [batch_size, num_classes]
            embeddings: Node embeddings [batch_size, hidden_dim]
            attention_weights: Dictionary of attention weights for explainability
        """
        batch_size = batch.max().item() + 1 if batch is not None else 1  # noqa: F841

        # Store attention weights for each layer
        attention_weights = {}

        # Spatial processing with GAT layers
        x_spatial = x
        for i, (gat_layer, bn) in enumerate(
            zip(self.gat_layers, self.batch_norms, strict=False)
        ):
            x_spatial, attention = gat_layer(
                x_spatial, edge_index, return_attention_weights=True
            )
            x_spatial = F.elu(bn(x_spatial))
            x_spatial = F.dropout(
                x_spatial, p=model_config.gat_dropout, training=self.training
            )

            # Store attention weights
            attention_weights[f"gat_layer_{i}"] = attention

        # Global graph embedding
        if batch is not None:
            graph_embedding = global_mean_pool(
                x_spatial, batch
            )  # [batch_size, hidden_dim]
        else:
            graph_embedding = x_spatial.mean(dim=0, keepdim=True)  # [1, hidden_dim]

        # Reshape for temporal processing (simulating sequence with current data)
        # In a real temporal setup, we'd have multiple time steps
        graph_embedding = graph_embedding.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Temporal processing with LSTM
        lstm_out, (hidden_state, cell_state) = self.lstm(graph_embedding)
        temporal_embedding = hidden_state[
            -1
        ]  # Last layer hidden state [batch_size, hidden_dim]

        # Classification
        logits = self.classifier(temporal_embedding)  # [batch_size, num_classes]

        # Store attention weights for explainability
        self.attention_weights = attention_weights

        return logits, temporal_embedding, attention_weights

    def get_attention_weights(self) -> dict:
        """Get attention weights for model explainability."""
        return self.attention_weights or {}


class HydraulicGNN(nn.Module):
    """
    Wrapper class for hydraulic diagnostics with additional utilities.
    """

    def __init__(self, model_config: model_config = model_config):
        super().__init__()
        self.gnn = TemporalGAT(
            num_node_features=model_config.num_node_features,
            hidden_dim=model_config.hidden_dim,
            num_classes=model_config.num_classes,
            num_gat_layers=model_config.num_gat_layers,
            num_heads=model_config.num_heads,
            gat_dropout=model_config.gat_dropout,
            num_lstm_layers=model_config.num_lstm_layers,
            lstm_dropout=model_config.lstm_dropout,
        )

        self.component_names = model_config.component_names

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return self.gnn(x, edge_index, batch)

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get probability predictions."""
        with torch.no_grad():
            logits, _, _ = self.forward(x, edge_index, batch)
            return torch.sigmoid(logits)

    def get_component_predictions(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get predictions for each component."""
        probabilities = self.predict_proba(x, edge_index, batch)

        return {
            component: probabilities[:, i]
            for i, component in enumerate(self.component_names)
        }


def create_model(device: str = None) -> HydraulicGNN:
    """Create and initialize model."""
    device = device or model_config.device
    model = HydraulicGNN().to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, GATv2Conv):
            nn.init.xavier_uniform_(m.lin_src.weight)
            if m.lin_src.bias is not None:
                nn.init.zeros_(m.lin_src.bias)

    model.apply(init_weights)
    logger.info(f"Model created and initialized on device: {device}")

    return model


if __name__ == "__main__":
    # Test the model
    model = create_model()

    # Create dummy data
    batch_size = 2
    num_nodes = model_config.num_nodes
    num_features = model_config.num_node_features

    x = torch.randn(batch_size * num_nodes, num_features)
    edge_index = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6, 0, 2, 3, 4, 5, 6]],
        dtype=torch.long,
    )
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    logits, embeddings, attention = model(x, edge_index, batch)
    print(f"Model output shape: {logits.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print("Model test passed!")
