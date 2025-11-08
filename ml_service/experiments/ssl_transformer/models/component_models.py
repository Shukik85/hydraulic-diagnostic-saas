"""
Physics-Informed Component Models (5 features)
GNN-based models treating features as graph nodes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class ComponentGNN(nn.Module):
    """Universal component model using GNN architecture"""
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 1
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        
        self.input_proj = nn.Linear(1, hidden_dim)
        
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        current_dim = hidden_dim
        for i in range(num_layers):
            if i == num_layers - 1:
                self.gat_layers.append(
                    GATConv(current_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
                )
            else:
                self.gat_layers.append(
                    GATConv(current_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout)
                )
                current_dim = hidden_dim
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index=None, batch=None, return_embedding=False):
        batch_size = x.size(0)
        num_features = self.in_features
        
        x_nodes = x.view(-1, 1)
        h = self.input_proj(x_nodes)
        h = F.relu(h)
        
        if batch is None:
            batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_features)
        
        if edge_index is not None:
            edge_list = []
            for i in range(batch_size):
                offset = i * num_features
                edge_list.append(edge_index + offset)
            edge_index_batch = torch.cat(edge_list, dim=1)
        else:
            num_nodes_total = batch_size * num_features
            edge_index_batch = torch.arange(num_nodes_total, device=x.device).unsqueeze(0).repeat(2, 1)
        
        for gat, bn in zip(self.gat_layers, self.batch_norms, strict=True):
            h = gat(h, edge_index_batch)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
        
        h_pooled = global_mean_pool(h, batch)
        
        if return_embedding:
            return h_pooled
        
        out = self.classifier(h_pooled)
        return out


class CylinderModel(ComponentGNN):
    """Cylinder model - 5 features"""
    
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__(5, hidden_dim, num_layers, dropout=dropout, num_classes=1)
        
        # 0: pressure_extend, 1: pressure_retract, 2: position, 3: velocity, 4: pressure_diff
        self.register_buffer(
            'physics_edges',
            torch.tensor([
                [0, 1], [1, 0],  # pressure_extend ↔ pressure_retract
                [0, 4], [4, 0],  # pressure_extend ↔ pressure_diff
                [1, 4], [4, 1],  # pressure_retract ↔ pressure_diff
                [2, 3], [3, 2],  # position ↔ velocity
            ], dtype=torch.long).t()
        )
    
    def forward(self, x, batch=None, return_embedding=False):
        return super().forward(x, self.physics_edges, batch, return_embedding)


class PumpModel(ComponentGNN):
    """Pump model - 5 features"""
    
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__(5, hidden_dim, num_layers, dropout=dropout, num_classes=1)
        
        self.register_buffer(
            'physics_edges',
            torch.tensor([
                [0, 1], [1, 0],  # pressure ↔ speed
                [1, 4], [4, 1],  # speed ↔ power
                [0, 4], [4, 0],  # pressure ↔ power
                [1, 3], [3, 1],  # speed ↔ vibration
                [1, 2], [2, 1],  # speed ↔ temperature
            ], dtype=torch.long).t()
        )
    
    def forward(self, x, batch=None, return_embedding=False):
        return super().forward(x, self.physics_edges, batch, return_embedding)


if __name__ == "__main__":
    print("Testing CylinderModel...")
    model = CylinderModel(hidden_dim=64)
    
    x = torch.randn(32, 5)  # 5 features
    output = model(x)
    print(f"✅ Input: {x.shape}")
    print(f"✅ Output: {output.shape}")
    
    print("\nTesting PumpModel...")
    pump = PumpModel(hidden_dim=64)
    x_pump = torch.randn(32, 5)
    
    output_pump = pump(x_pump)
    print(f"✅ Pump output: {output_pump.shape}")
    
    print("\n🎉 Models working correctly!")
