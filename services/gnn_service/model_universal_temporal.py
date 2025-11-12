"""
UniversalTemporalGNN: Универсальная GNN с поддержкой RUL/Anomaly
- Поддерживает любые структуры оборудования (metadata)
- Поддержка временных окон, multi-horizon RUL, anomaly scores
- Архитектура: GAT->LSTM->Multi-head (anomaly, RUL)
- Оптимизировано под torch.compile + FP16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

def get_feature_dim(metadata):
    sensors = set()
    for comp in metadata['components']:
        for s in comp.get('sensors', []):
            sensors.add(s)
    # 5 статистик на каждый сенсор для node features: mean, std, min, max, last
    return len(sensors) * 5

class UniversalTemporalGNN(nn.Module):
    """
    Универсальная GNN для anomaly и RUL с поддержкой динамических топологий
    Вход: [batch, time, n_nodes, n_features]
    Выход: anomaly scores + multi-horizon RUL per node
    """
    def __init__(self, metadata, hidden_dim=96, num_gat_layers=3, num_heads=4,
                 lstm_layers=2, time_steps=12, horizons=[5, 15, 30], dropout=0.12):
        super().__init__()
        self.n_nodes = len(metadata['components'])
        self.node_feature_dim = get_feature_dim(metadata)
        self.horizons = horizons
        self.dropout = nn.Dropout(dropout)
        # --- GAT backbone (дл. структуры) ---
        self.gat_layers = nn.ModuleList([
            GATv2Conv(self.node_feature_dim if i==0 else hidden_dim*num_heads,
                      hidden_dim, heads=num_heads, dropout=dropout, concat=True)
            for i in range(num_gat_layers)
        ])
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim*num_heads) for _ in range(num_gat_layers)
        ])
        # --- LSTM temporal block ---
        self.lstm_input = hidden_dim*num_heads
        self.lstm = nn.LSTM(
            input_size=self.lstm_input,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers>1 else 0
        )
        # --- Heads: anomaly + multiperiod RUL ---
        self.anomaly_head = nn.Linear(hidden_dim*2, self.n_nodes)
        self.rul_heads = nn.ModuleDict({
            f"rul_{h}min": nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(), nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.n_nodes),
                nn.Sigmoid()
            ) for h in self.horizons
        })
    def forward(self, x_sequence, edge_index, batch=None):
        # x_sequence: [batch, T, n_nodes, n_features]
        batch_size, T, N, F = x_sequence.shape
        assert N == self.n_nodes
        time_embeds = []
        for t in range(T):
            x = x_sequence[:,t].reshape(-1, F)  # [batch*n_nodes, F]
            for gat, norm in zip(self.gat_layers, self.gat_norms):
                x = gat(x, edge_index)
                x = norm(x)
                x = F.elu(x)
                x = self.dropout(x)
            x = x.view(batch_size, N, -1)  # [batch, n_nodes, hid]
            time_embeds.append(x)
        # stack [batch, T, n_nodes, hid]
        time_embeds = torch.stack(time_embeds, dim=1)
        # среднее по нодам, LSTM по времени
        node_embeddings = time_embeds.mean(dim=2)  # [batch, T, hid]
        lstm_out, _ = self.lstm(node_embeddings)  # [batch, T, hid*2]
        temporal_embed = lstm_out[:,-1]           # [batch, hid*2]
        # --- Anomaly scores
        anomaly_scores = self.anomaly_head(temporal_embed)  # [batch, n_nodes]
        # --- RUL predictions
        rul_out = {h:self.rul_heads[f"rul_{h}min"](temporal_embed) for h in self.horizons}
        return anomaly_scores, rul_out
