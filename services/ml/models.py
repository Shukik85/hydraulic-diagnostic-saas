"""GNN model architecture for hydraulic diagnostics."""
import torch
import torch.nn as nn

class GNNModel(nn.Module):
    """Graph Neural Network for multi-label classification."""
    
    def __init__(self, in_channels: int = 10, hidden_channels: int = 64, num_classes: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
