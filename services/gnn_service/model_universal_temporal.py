"""
Universal Flexible Temporal GNN - Production Version
Произвольная структура (metadata), временные окна, health + degradation outputs
RAG интерпретирует состояния (не модель)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import logging

logger = logging.getLogger(__name__)

def get_feature_dim(metadata):
    """Calculate feature dimension from metadata."""
    sensors = set()
    for comp in metadata['components']:
        for s in comp.get('sensors', []):
            sensors.add(s)
    return len(sensors) * 5

class UniversalTemporalGNN(nn.Module):
    """
    Universal Flexible GNN: health + degradation prediction.
    
    Input: [batch, time_steps, n_nodes, n_features]
    Output:
        - health_scores: [batch, n_nodes] (0-1)
        - degradation_rate: [batch, n_nodes] (derivative)
    """
    def __init__(self, metadata, hidden_dim=96, num_gat_layers=3, num_heads=4,
                 lstm_layers=2, dropout=0.12):
        super().__init__()
        self.n_nodes = len(metadata['components'])
        self.node_feature_dim = get_feature_dim(metadata)
        self.dropout = nn.Dropout(dropout)
        
        # GAT backbone
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                self.node_feature_dim if i == 0 else hidden_dim * num_heads,
                hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
            for i in range(num_gat_layers)
        ])
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * num_heads)
            for _ in range(num_gat_layers)
        ])
        
        # LSTM temporal
        self.lstm = nn.LSTM(
            input_size=hidden_dim * num_heads,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output heads
        self.health_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.n_nodes),
            nn.Sigmoid()  # 0-1 range
        )
        
        self.degradation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_nodes),
            nn.Tanh()  # -1 to 1 range for rate
        )
    
    def forward(self, x_sequence, edge_index, batch=None):
        """
        Args:
            x_sequence: [batch, T, n_nodes, n_features]
            edge_index: [2, n_edges]
            batch: Optional batch assignment
        
        Returns:
            health_scores: [batch, n_nodes]
            degradation_rate: [batch, n_nodes]
        """
        batch_size, T, N, F = x_sequence.shape
        assert N == self.n_nodes, f"Expected {self.n_nodes} nodes, got {N}"
        
        # Process each timestep through GAT
        time_embeds = []
        for t in range(T):
            x = x_sequence[:, t].reshape(-1, F)
            for gat, norm in zip(self.gat_layers, self.gat_norms):
                x = gat(x, edge_index)
                x = norm(x)
                x = F.elu(x)
                x = self.dropout(x)
            x = x.view(batch_size, N, -1)
            time_embeds.append(x)
        
        # Stack temporal embeddings [batch, T, n_nodes, hidden]
        time_embeds = torch.stack(time_embeds, dim=1)
        
        # Average over nodes, LSTM over time
        node_embeddings = time_embeds.mean(dim=2)  # [batch, T, hidden]
        lstm_out, _ = self.lstm(node_embeddings)    # [batch, T, hidden*2]
        temporal_embed = lstm_out[:, -1]            # [batch, hidden*2]
        
        # Predict health and degradation
        health_scores = self.health_head(temporal_embed)
        degradation_rate = self.degradation_head(temporal_embed)
        
        return health_scores, degradation_rate


def create_model(metadata, device="cuda", use_compile=True):
    """
    Create and compile universal temporal GNN.
    """
    model = UniversalTemporalGNN(metadata).to(device)
    
    if use_compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    
    logger.info(f"Model created on {device}")
    return model


if __name__ == "__main__":
    # Test model
    logging.basicConfig(level=logging.INFO)
    
    metadata = {
        "components": [
            {"id": "pump", "sensors": ["pressure", "flow", "temp"]},
            {"id": "valve", "sensors": ["pressure", "position"]},
        ]
    }
    
    model = create_model(metadata, device="cpu", use_compile=False)
    
    # Dummy input
    x = torch.randn(1, 12, 2, 15)  # batch, time, nodes, features
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    
    health, deg = model(x, edge_index)
    print(f"Health: {health}")
    print(f"Degradation: {deg}")
    print("✅ Model test passed!")
