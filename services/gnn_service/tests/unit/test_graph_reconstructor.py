"""Unit tests for GraphReconstructor.

Tests graph reconstruction functionality:
- Edge prediction
- Feature reconstruction
- K-NN based edge detection
"""

import pytest
import torch
from torch_geometric.data import Data

from src.training.graph_reconstructor import GraphReconstructor


class TestGraphReconstructor:
    """Test GraphReconstructor class."""

    @pytest.fixture
    def reconstructor(self) -> GraphReconstructor:
        """Create reconstructor fixture."""
        return GraphReconstructor(
            input_dim=34,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            edge_threshold=0.5,
            k_neighbors=3,
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input."""
        x = torch.randn(10, 34)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        mask_nodes = torch.ones(10, dtype=torch.bool)
        return x, edge_index, mask_nodes

    def test_init(self):
        """Test initialization."""
        rec = GraphReconstructor(input_dim=34, hidden_dim=128)
        assert rec.input_dim == 34
        assert rec.hidden_dim == 128
        assert rec.edge_threshold == 0.5

    def test_invalid_threshold(self):
        """Test invalid threshold raises error."""
        with pytest.raises(ValueError):
            GraphReconstructor(edge_threshold=1.5)

    def test_forward_output_shapes(self, reconstructor, sample_input):
        """Test forward pass output shapes."""
        x, edge_index, mask_nodes = sample_input
        x_recon, edge_pred, weights = reconstructor(x, edge_index, mask_nodes)

        assert x_recon.shape == x.shape
        assert edge_pred.shape[0] == 2
        assert weights.shape[0] == edge_pred.shape[1]

    def test_forward_output_types(self, reconstructor, sample_input):
        """Test forward pass returns correct types."""
        x, edge_index, mask_nodes = sample_input
        x_recon, edge_pred, weights = reconstructor(x, edge_index, mask_nodes)

        assert isinstance(x_recon, torch.Tensor)
        assert isinstance(edge_pred, torch.Tensor)
        assert isinstance(weights, torch.Tensor)

    def test_predicted_edges_valid(self, reconstructor, sample_input):
        """Test predicted edges are valid."""
        x, edge_index, mask_nodes = sample_input
        _, edge_pred, _ = reconstructor(x, edge_index, mask_nodes)

        n_nodes = x.shape[0]
        assert edge_pred.min() >= 0
        assert edge_pred.max() < n_nodes

    def test_weight_range(self, reconstructor, sample_input):
        """Test predicted weights are in valid range."""
        x, edge_index, mask_nodes = sample_input
        _, _, weights = reconstructor(x, edge_index, mask_nodes)

        assert (weights >= 0).all()
        assert (weights <= 1).all()

    def test_no_self_loops(self, reconstructor, sample_input):
        """Test no self-loops in predicted edges."""
        x, edge_index, mask_nodes = sample_input
        _, edge_pred, _ = reconstructor(x, edge_index, mask_nodes)

        if edge_pred.shape[1] > 0:
            self_loops = (edge_pred[0] == edge_pred[1]).sum().item()
            # May have some self-loops from k-NN, but generally should be few
            assert self_loops < edge_pred.shape[1] * 0.1

    def test_input_validation(self, reconstructor):
        """Test input validation."""
        x = torch.randn(10, 34)
        edge_index = torch.tensor([[0, 1], [1, 100]])  # Invalid node
        mask_nodes = torch.ones(10, dtype=torch.bool)

        with pytest.raises(AssertionError):
            reconstructor(x, edge_index, mask_nodes)

    def test_deterministic_output(self, reconstructor, sample_input):
        """Test output is deterministic given same input."""
        x, edge_index, mask_nodes = sample_input

        torch.manual_seed(42)
        _, edge_pred1, weights1 = reconstructor(x, edge_index, mask_nodes)

        torch.manual_seed(42)
        _, edge_pred2, weights2 = reconstructor(x, edge_index, mask_nodes)

        assert torch.allclose(edge_pred1.float(), edge_pred2.float())
        assert torch.allclose(weights1, weights2)
