"""Tests for graph reconstruction module."""

import pytest
import torch

from src.training.graph_reconstructor import GraphReconstructor


@pytest.fixture
def reconstructor() -> GraphReconstructor:
    return GraphReconstructor(
        input_dim=34,
        hidden_dim=128,
        num_heads=8,
        num_layers=2,
    )


def test_reconstructor_forward(reconstructor: GraphReconstructor):
    x = torch.randn(5, 34)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    mask_nodes = torch.ones(5)

    x_recon, edge_pred, weights = reconstructor(x, edge_index, mask_nodes)

    assert x_recon.shape == x.shape
    assert edge_pred.shape[0] == 2
    assert weights.shape[0] == edge_pred.shape[1]


def test_reconstructor_handles_missing(reconstructor: GraphReconstructor):
    x = torch.randn(5, 34)
    x[2, :] = 0  # Mark as missing
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    mask_nodes = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])

    x_recon, edge_pred, weights = reconstructor(x, edge_index, mask_nodes)

    # Should still output valid tensors
    assert not torch.isnan(x_recon).any()
    assert not torch.isnan(weights).any()
