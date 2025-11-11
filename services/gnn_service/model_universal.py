"""
Universal GNN: динамическая модель под любую структуру hydraulic/metadatasystem (экскаватор, пресс, кран, custom).
- Не требует presет/хардкода числа компонентов.
- Создает GATv2Conv/GNN skeleton, heads, classifier на основании metadata
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

def get_feature_dim(metadata):
    sensors = set()
    for comp in metadata['components']:
        for s in comp.get('sensors', []):
            sensors.add(s)
    # 5 статистик на каждый сенсор для node features: mean, std, min, max, last
    return len(sensors) * 5

class UniversalHydraulicGNN(nn.Module):
    def __init__(self, metadata, hidden_dim=96, num_gat_layers=3, num_heads=4, dropout=0.12):
        super().__init__()
        self.n_nodes = len(metadata['components'])
        self.node_feature_dim = get_feature_dim(metadata)
        self.n_classes = self.n_nodes  # 1 label per node (fault/health)
        # --- GAT backbone ---
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_ch = self.node_feature_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(GATv2Conv(in_ch, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        self.gat_norms = nn.ModuleList([nn.LayerNorm(hidden_dim*num_heads) for _ in range(num_gat_layers)])
        self.dropout = nn.Dropout(dropout)
        # --- Classifier/output ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Dropout(0.11),
            nn.Linear(hidden_dim, self.n_classes)
        )
    def forward(self, x, edge_index, batch=None):
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            x = gat(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)
        # Pooled graph-level embedding not used for node classification
        logits = self.classifier(x)
        return logits
