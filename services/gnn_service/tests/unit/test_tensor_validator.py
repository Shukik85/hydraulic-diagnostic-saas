import pytest
import torch
from torch_geometric.data import Data

from src.inference.exceptions import TensorValidationError
from src.inference.validation import TensorValidator


def test_tensor_validator_node_dim() -> None:
    validator = TensorValidator(expected_node_dim=34, expected_edge_dim=8)

    x = torch.randn(5, 30)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    graph = Data(x=x, edge_index=edge_index)

    with pytest.raises(TensorValidationError, match="Node dim mismatch"):
        validator.validate_graph(graph)

    x = torch.randn(5, 34)
    graph = Data(x=x, edge_index=edge_index)
    validator.validate_graph(graph)


def test_tensor_validator_nan_inf() -> None:
    validator = TensorValidator(expected_node_dim=34, expected_edge_dim=8)
    edge_index = torch.tensor([[0, 1], [1, 0]])

    x = torch.tensor([[1.0, 2.0], [float("nan"), 4.0]])
    graph = Data(x=x, edge_index=edge_index)
    with pytest.raises(TensorValidationError, match="NaN detected"):
        validator.validate_graph(graph)

    x = torch.randn(2, 34)
    edge_attr = torch.tensor([[1.0], [float("inf")]])
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    with pytest.raises(TensorValidationError, match="Inf detected"):
        validator.validate_graph(graph)


def test_tensor_validator_device() -> None:
    validator = TensorValidator(
        expected_node_dim=34,
        expected_edge_dim=8,
        allowed_devices=["cuda"],
    )
    x = torch.randn(2, 34)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    graph = Data(x=x, edge_index=edge_index).to("cpu")

    with pytest.raises(TensorValidationError, match="Invalid device"):
        validator.validate_graph(graph)
