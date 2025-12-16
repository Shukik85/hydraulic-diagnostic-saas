"""GRAPE-based graph reconstruction for missing data.

Implements GNN-based edge embedding approach from GRAPE paper:
- Reconstructs missing edges via GNN embeddings
- Handles >50% incomplete data
- Two-stage approach: reconstruction + prediction

References:
  [3] GRAPE: Missing data via GNN edge embeddings
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv


class GraphReconstructor(nn.Module):
    """GRAPE-based graph reconstruction module.

    Reconstructs missing edges and node features via:
    1. GNN encoder to create node embeddings
    2. Edge prediction via embedding similarity
    3. Feature propagation through reconstructed edges
    """

    def __init__(
        self,
        input_dim: int = 34,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # GAT encoder for embeddings
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(
            GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        )
        for _ in range(num_layers - 1):
            self.encoder_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
            )

        # Edge prediction head
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Feature reconstruction head
        self.feature_reconstructor = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct missing edges and features.

        Args:
            x: Node features [N, F]
            edge_index: Known edges [2, E]
            mask_nodes: Node presence mask [N]

        Returns:
            x_reconstructed: Reconstructed features [N, F]
            edge_index_pred: Predicted edges [2, E_pred]
            edge_weights: Edge confidence scores [E_pred]
        """
        # Encode nodes
        h = x
        for encoder in self.encoder_layers:
            h = encoder(h, edge_index)
            h = torch.relu(h)

        # Reconstruct missing features
        x_reconstructed = self.feature_reconstructor(h)

        # Predict missing edges
        n_nodes = x.shape[0]
        edge_index_pred = []
        edge_weights = []

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # Skip if edge already exists
                if (edge_index[0] == i).any() and (edge_index[1] == j).any():
                    continue

                # Concatenate embeddings
                edge_feat = torch.cat([h[i], h[j]], dim=-1)
                weight = self.edge_predictor(edge_feat)

                if weight > 0.5:  # Threshold
                    edge_index_pred.append([i, j])
                    edge_weights.append(weight.item())

        if edge_index_pred:
            edge_index_pred = torch.tensor(edge_index_pred, dtype=torch.long).t()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            edge_index_pred = edge_index
            edge_weights = torch.ones(edge_index.shape[1])

        return x_reconstructed, edge_index_pred, edge_weights
