"""
Universal Flexible Temporal GNN - Production Version
Полностью поддерживает батчинг, импорт F, корректные heads
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

logger = logging.getLogger(__name__)


def get_feature_dim(metadata):
    sensors = set()
    for comp in metadata["components"]:
        for s in comp.get("sensors", []):
            sensors.add(s)
    return len(sensors) * 5


class UniversalTemporalGNN(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_dim=96,
        num_gat_layers=3,
        num_heads=4,
        lstm_layers=2,
        dropout=0.12,
    ):
        super().__init__()
        self.n_nodes = len(metadata["components"])
        self.node_feature_dim = get_feature_dim(metadata)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.gat_layers = nn.ModuleList(
            [
                GATv2Conv(
                    self.node_feature_dim if i == 0 else hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
                for i in range(num_gat_layers)
            ]
        )
        self.gat_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim * num_heads) for _ in range(num_gat_layers)]
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim * num_heads,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.health_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.degradation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x_sequence, edge_index, batch=None):
        batch_size, T, N, Fdim = x_sequence.shape
        assert self.n_nodes == N
        if batch_size > 1:
            edge_index_list = []
            for b in range(batch_size):
                edge_index_list.append(edge_index + b * N)
            edge_index_batched = torch.cat(edge_index_list, dim=1)
        else:
            edge_index_batched = edge_index
        time_embeds = []
        for t in range(T):
            x = x_sequence[:, t].reshape(-1, Fdim)
            for gat, norm in zip(self.gat_layers, self.gat_norms):
                x = gat(x, edge_index_batched)
                x = norm(x)
                x = F.elu(x)
                x = self.dropout(x)
            x = x.view(batch_size, N, -1)
            time_embeds.append(x)
        time_embeds = torch.stack(time_embeds, dim=1)
        B, T, N, H = time_embeds.shape
        time_embeds = time_embeds.reshape(B * N, T, H)
        lstm_out, _ = self.lstm(time_embeds)
        temporal_embed = lstm_out[:, -1]
        temporal_embed = temporal_embed.view(B, N, -1)
        health_scores = self.health_head(temporal_embed).squeeze(-1)
        degradation_rate = self.degradation_head(temporal_embed).squeeze(-1)
        return health_scores, degradation_rate


def create_model(metadata, device="cuda", use_compile=True):
    model = UniversalTemporalGNN(metadata).to(device)
    if use_compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    logger.info(f"Model created on {device}")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    metadata = {
        "components": [
            {"id": "pump", "sensors": ["pressure", "flow", "temp"]},
            {"id": "valve", "sensors": ["pressure", "position"]},
        ]
    }
    model = create_model(metadata, device="cpu", use_compile=False)
    x = torch.randn(2, 12, 2, 15)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    health, deg = model(x, edge_index)
    print(f"Health shape: {health.shape}")
    print(f"Degradation shape: {deg.shape}")
    print(f"Health: {health}")
    print(f"Degradation: {deg}")
    print("✅ Model test passed!")
