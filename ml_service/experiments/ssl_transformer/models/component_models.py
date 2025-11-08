"""
Component-Level GNN Models (LEVEL 0) - FIXED
Universal models for hydraulic components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class ComponentGNN(nn.Module):
    """
    Universal component model (pump, cylinder, valve, motor)
    Uses physics-guided graph structure
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # GAT layers (physics-aware attention!)
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        current_dim = hidden_dim
        for i in range(num_layers):
            if i < num_layers - 1:
                # Intermediate layers: concat heads
                out_channels = hidden_dim // num_heads
                self.gat_layers.append(
                    GATv2Conv(current_dim, out_channels, heads=num_heads, dropout=dropout, concat=True)
                )
                current_dim = out_channels * num_heads
            else:
                # Last layer: average heads
                self.gat_layers.append(
                    GATv2Conv(current_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=False)
                )
                current_dim = hidden_dim
            
            self.batch_norms.append(nn.BatchNorm1d(current_dim))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index=None, batch=None, return_embedding=False):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges] (optional)
            batch: Batch assignment [num_nodes]
            return_embedding: If True, return embeddings
        
        Returns:
            If return_embedding: [batch_size, hidden_dim]
            Else: [batch_size, num_classes]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        
        # If no edge_index, create complete graph
        if edge_index is None:
            num_nodes = x.size(0)
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(x.device)
        
        # GAT layers
        for gat, bn in zip(self.gat_layers, self.batch_norms):
            h = gat(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        h_pooled = global_mean_pool(h, batch)
        
        if return_embedding:
            return h_pooled
        
        # Classification
        out = self.classifier(h_pooled)
        return out


class CylinderModel(ComponentGNN):
    """
    Specialized cylinder model
    Input: 7 features
    """
    
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__(7, hidden_dim, num_layers, dropout=dropout)
        
        # Physics-guided edges (feature correlations)
        self.register_buffer(
            'physics_edges',
            torch.tensor([
                [0, 1], [1, 0],  # pressure_extend ↔ pressure_retract
                [0, 5], [5, 0],  # pressure_extend ↔ pressure_diff
                [1, 5], [5, 1],  # pressure_retract ↔ pressure_diff
                [2, 3], [3, 2],  # position ↔ velocity
                [3, 4], [4, 3],  # velocity ↔ force
                [4, 6], [6, 4],  # force ↔ load_ratio
            ], dtype=torch.long).t()
        )
    
    def forward(self, x, batch=None, return_embedding=False):
        return super().forward(x, self.physics_edges, batch, return_embedding)


class PumpModel(ComponentGNN):
    """
    Specialized pump model
    Input: 5 features
    """
    
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__(5, hidden_dim, num_layers, dropout=dropout)
        
        # Physics-guided edges
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
    
    # Dummy input: batch of 32 samples, 7 features each
    x = torch.randn(32, 7)
    
    # Forward pass
    output = model(x)
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    
    # Get embedding
    embedding = model(x, return_embedding=True)
    print(f"✅ Embedding shape: {embedding.shape}")
    
    print("\nTesting PumpModel...")
    pump_model = PumpModel(hidden_dim=64)
    x_pump = torch.randn(32, 5)
    
    output_pump = pump_model(x_pump)
    print(f"✅ Pump output shape: {output_pump.shape}")
    
    print("\n🎉 All component models working!")
