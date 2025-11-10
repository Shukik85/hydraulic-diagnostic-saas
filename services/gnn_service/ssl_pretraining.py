"""Self-Supervised Learning pretraining for GNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from model import TemporalGAT
from torch_geometric.data import Data


class SSLPretrainer(nn.Module):
    """SSL Pretrainer with masked node prediction + contrastive learning."""

    def __init__(
        self,
        in_channels: int = config.num_node_features,
        hidden_channels: int = config.hidden_dim,
        num_layers: int = config.num_gat_layers,
        num_heads: int = config.num_heads,
        dropout: float = config.dropout,
        mask_ratio: float = config.ssl_mask_ratio,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.in_channels = in_channels

        # Encoder
        self.encoder = TemporalGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Reconstruction head (masked node prediction)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels),
        )

        # Projection head (for contrastive learning)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 128),
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with SSL objectives.

        Returns:
            reconstructed_features: Reconstructed node features
            projections: Projected embeddings for contrastive loss
            masked_indices: Indices of masked nodes
        """
        # Mask random nodes
        num_nodes = data.x.size(0)
        num_masked = int(num_nodes * self.mask_ratio)
        masked_indices = torch.randperm(num_nodes)[:num_masked]

        # Create masked input
        x_masked = data.x.clone()
        x_masked[masked_indices] = 0  # Zero out masked nodes

        # Encode
        data_masked = Data(
            x=x_masked,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
            batch=data.batch if hasattr(data, "batch") else None,
        )

        _, node_embeddings, _ = self.encoder(
            x=data_masked.x,
            edge_index=data_masked.edge_index,
            edge_attr=data_masked.edge_attr,
            batch=data_masked.batch,
        )

        # Reconstruction
        reconstructed_features = self.reconstruction_head(node_embeddings)

        # Projection for contrastive learning
        projections = self.projection_head(node_embeddings)
        projections = F.normalize(projections, dim=-1)

        return reconstructed_features, projections, masked_indices

    def compute_ssl_loss(
        self,
        data: Data,
        temperature: float = config.ssl_contrastive_temp,
    ) -> tuple[torch.Tensor, dict]:
        """Compute SSL loss (reconstruction + contrastive).

        Returns:
            total_loss: Combined loss
            metrics: Dictionary of loss components
        """
        reconstructed, projections, masked_indices = self.forward(data)

        # 1. Masked node reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(
            reconstructed[masked_indices],
            data.x[masked_indices],
        )

        # 2. Temporal contrastive loss (InfoNCE)
        # Create positive pairs: augmented versions of same graph
        data_augmented = self._augment_graph(data)
        _, projections_aug, _ = self.forward(data_augmented)

        # InfoNCE loss
        contrastive_loss = self._infonce_loss(
            projections,
            projections_aug,
            temperature=temperature,
        )

        # Combined loss
        total_loss = reconstruction_loss + 0.5 * contrastive_loss

        metrics = {
            "ssl/reconstruction_loss": reconstruction_loss.item(),
            "ssl/contrastive_loss": contrastive_loss.item(),
            "ssl/total_loss": total_loss.item(),
        }

        return total_loss, metrics

    def _augment_graph(self, data: Data) -> Data:
        """Create augmented version of graph (edge dropout + feature noise)."""
        # Edge dropout (10%)
        edge_mask = torch.rand(data.edge_index.size(1)) > 0.1
        edge_index_aug = data.edge_index[:, edge_mask]
        edge_attr_aug = (
            data.edge_attr[edge_mask] if hasattr(data, "edge_attr") else None
        )

        # Feature noise (Gaussian)
        x_aug = data.x + torch.randn_like(data.x) * 0.1

        return Data(
            x=x_aug,
            edge_index=edge_index_aug,
            edge_attr=edge_attr_aug,
            batch=data.batch if hasattr(data, "batch") else None,
        )

    def _infonce_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss."""
        batch_size = z1.size(0)

        # Cosine similarity
        sim_matrix = torch.mm(z1, z2.t()) / temperature

        # Positive pairs on diagonal
        labels = torch.arange(batch_size, device=z1.device)

        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
