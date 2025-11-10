"""
Enhanced Temporal Graph Attention Network for F1 > 90%

Improvements over v1:
- 3 GAT layers (was 2)
- LayerNorm instead of BatchNorm
- Attention Pooling (instead of mean pooling)
- Deeper classifier with residual connections
- Dropout schedule
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import model_config
from torch_geometric.nn import GATv2Conv, global_mean_pool

logger = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    """Learnable attention-based global pooling."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, features]
            batch: Batch assignment [num_nodes]

        Returns:
            Pooled features [batch_size, features]
        """
        # Compute attention scores for each node
        attention_scores = self.attention_net(x)  # [num_nodes, 1]

        # Softmax per graph
        attention_weights = torch.zeros_like(attention_scores)
        unique_batches = torch.unique(batch)

        for batch_id in unique_batches:
            mask = batch == batch_id
            batch_scores = attention_scores[mask]
            batch_weights = F.softmax(batch_scores, dim=0)
            attention_weights[mask] = batch_weights

        # Weighted sum
        weighted_features = x * attention_weights

        # Pool per graph
        pooled = global_mean_pool(weighted_features, batch)

        return pooled


class ResidualBlock(nn.Module):
    """Residual block for classifier."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.fc1(x)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.ln2(out)

        # Residual connection
        out = out + identity
        out = F.relu(out)

        return out


class EnhancedTemporalGAT(nn.Module):
    """
    Enhanced Temporal GAT for F1 > 90%.

    Architecture:
        Input [7, 15]
          ↓ GAT layer 0 (4 heads)
        [7, 384] + LayerNorm + ELU + Dropout(0.3)
          ↓ GAT layer 1 (4 heads)
        [7, 384] + LayerNorm + ELU + Dropout(0.2)
          ↓ GAT layer 2 (4 heads)
        [7, 384] + LayerNorm + ELU + Dropout(0.1)
          ↓ Attention Pooling
        [batch, 384]
          ↓ LSTM (1 layer)
        [batch, 96]
          ↓ Classifier (ResBlocks + FC)
        [batch, 7]
    """

    def __init__(
        self,
        num_node_features: int = 15,
        hidden_dim: int = 96,
        num_classes: int = 7,
        num_gat_layers: int = 3,
        num_heads: int = 4,
        gat_dropout: float = 0.2,
        num_lstm_layers: int = 1,
        lstm_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads

        # Dropout schedule (higher → lower)
        self.dropout_schedule = [0.3, 0.2, 0.1][:num_gat_layers]

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_channels = num_node_features if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=gat_dropout,
                    concat=True,
                )
            )

        # LayerNorm
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim * num_heads) for _ in range(num_gat_layers)]
        )

        # Attention Pooling
        self.attention_pool = AttentionPooling(hidden_dim * num_heads)

        # LSTM
        lstm_input_size = hidden_dim * num_heads
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        # Enhanced Classifier with residual connections
        self.classifier = nn.Sequential(
            # First projection
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Residual block
            ResidualBlock(hidden_dim, dropout=0.2),
            # Second projection
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Output
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.attention_weights = None

        logger.info(
            f"EnhancedTemporalGAT initialized: "
            f"{num_gat_layers} GAT layers, {num_heads} heads, "
            f"attention pooling, residual classifier"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass."""

        attention_weights = {}

        # GAT layers with LayerNorm
        for i, (gat_layer, ln, dropout_p) in enumerate(
            zip(self.gat_layers, self.layer_norms, self.dropout_schedule)
        ):
            x, attention = gat_layer(x, edge_index, return_attention_weights=True)
            attention_weights[f"gat_layer_{i}"] = attention

            # LayerNorm + ELU + Scheduled Dropout
            x = ln(x)
            x = F.elu(x)
            x = F.dropout(x, p=dropout_p, training=self.training)

        # Attention Pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_embedding = self.attention_pool(x, batch)  # [batch, 384]

        # LSTM
        graph_embedding = graph_embedding.unsqueeze(1)  # [batch, 1, 384]
        lstm_out, (hidden_state, _) = self.lstm(graph_embedding)
        temporal_embedding = hidden_state[-1]  # [batch, 96]

        # Classifier
        logits = self.classifier(temporal_embedding)  # [batch, 7]

        self.attention_weights = attention_weights

        return logits, temporal_embedding, attention_weights

    def get_attention_weights(self) -> dict:
        """Get attention weights for explainability."""
        return self.attention_weights or {}


def create_enhanced_model(device: str = "cuda") -> EnhancedTemporalGAT:
    """Create and initialize enhanced model."""
    model = EnhancedTemporalGAT(
        num_node_features=model_config.num_node_features,
        hidden_dim=model_config.hidden_dim,
        num_classes=model_config.num_classes,
        num_gat_layers=3,
        num_heads=model_config.num_heads,
        gat_dropout=model_config.gat_dropout,
        num_lstm_layers=model_config.num_lstm_layers,
        lstm_dropout=model_config.lstm_dropout,
    ).to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, GATv2Conv):
            if hasattr(m, "lin_l") and m.lin_l is not None:
                nn.init.xavier_uniform_(m.lin_l.weight)
                if m.lin_l.bias is not None:
                    nn.init.zeros_(m.lin_l.bias)

            if hasattr(m, "lin_r") and m.lin_r is not None:
                nn.init.xavier_uniform_(m.lin_r.weight)
                if m.lin_r.bias is not None:
                    nn.init.zeros_(m.lin_r.bias)

            if hasattr(m, "att") and m.att is not None:
                nn.init.xavier_uniform_(m.att)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)

    model.apply(init_weights)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Enhanced model created on {device}")
    logger.info(f"Total parameters: {total_params:,}")

    return model


if __name__ == "__main__":
    # Test enhanced model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_enhanced_model(device)

    # Dummy data
    batch_size = 2
    num_nodes = 7
    num_features = 15

    x = torch.randn(batch_size * num_nodes, num_features).to(device)
    edge_index = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6, 0, 2, 3, 4, 5, 6]],
        dtype=torch.long,
    ).to(device)
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]).to(device)

    logits, embeddings, attention = model(x, edge_index, batch)

    print("Model test passed!")
    print(f"  Output shape: {logits.shape}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Attention layers: {len(attention)}")
