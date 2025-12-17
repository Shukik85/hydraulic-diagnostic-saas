"""Pytest configuration and shared fixtures for GNN service tests.

Provides:
- Sample data fixtures
- Mock services (TimescaleDB, FeatureEngineer)
- Device fixtures
- Temporary directories
"""

import asyncio
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import MagicMock, AsyncMock

import numpy as np
import pytest
import torch
from torch_geometric.data import Data


# ============================================================================
# EVENT LOOP FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop, None]:
    """Create an event loop for pytest-asyncio (session scoped)."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


# ============================================================================
# DEVICE FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get appropriate device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """Always return CPU device."""
    return torch.device("cpu")


# ============================================================================
# DATA FIXTURES
# ============================================================================


@pytest.fixture
def sample_graph() -> Data:
    """Create a sample graph for testing.

    Returns:
        Data object with:
        - x: [10, 34] node features
        - edge_index: [2, 15] edges
        - y_health: [10, 1] health targets
        - y_anomaly: [10, 9] anomaly targets
    """
    n_nodes = 10
    n_edges = 15
    n_features = 34
    n_anomaly_classes = 9

    x = torch.randn(n_nodes, n_features)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 14)  # 14 edge features

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_graph_health=torch.rand(1),
        y_graph_degradation=torch.randn(1),
        y_graph_anomaly=torch.zeros(n_anomaly_classes),
        y_graph_rul=torch.rand(1) * 1000,
        y_component_health=torch.rand(n_nodes, 1),
        y_component_anomaly=torch.zeros(n_nodes, n_anomaly_classes),
    )


@pytest.fixture
def sample_graphs(sample_graph) -> list[Data]:
    """Create multiple sample graphs for batch testing.

    Returns:
        List of 5 sample graphs
    """
    return [sample_graph for _ in range(5)]


@pytest.fixture
def sample_batch(sample_graphs) -> Data:
    """Create a batch from sample graphs.

    Returns:
        Batched Data object
    """
    from torch_geometric.data import Batch

    return Batch.from_data_list(sample_graphs)


@pytest.fixture
def sample_temporal_sequence() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a temporal sequence of graphs.

    Returns:
        (x_sequence, edge_index, targets) where:
        - x_sequence: [T=12, N=10, F=34] temporal node features
        - edge_index: [2, E=15] edges (static)
        - targets: dict with health, anomaly, rul targets
    """
    T = 12  # 12 time steps
    N = 10  # 10 nodes
    F = 34  # 34 features
    E = 15  # 15 edges

    x_sequence = torch.randn(T, N, F)
    edge_index = torch.randint(0, N, (2, E))

    targets = {
        "health": torch.rand(T, N),
        "anomaly": torch.zeros(T, N, 9),
        "rul": torch.rand(T) * 1000,
    }

    return x_sequence, edge_index, targets


# ============================================================================
# MOCK SERVICE FIXTURES
# ============================================================================


@pytest.fixture
def mock_timescale_connector() -> AsyncMock:
    """Create a mock TimescaleDB connector.

    Returns:
        AsyncMock with fetch_sensor_data method
    """
    mock = AsyncMock()

    # Mock sensor data retrieval
    async def mock_fetch_sensor_data(equipment_id: str, start_time: str, end_time: str):
        import polars as pl

        # Create mock sensor data
        n_samples = 1000
        return pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    start="2024-01-01",
                    end="2024-01-02",
                    interval="1m",
                    eager=True,
                )[:n_samples],
                "equipment_id": [equipment_id] * n_samples,
                "sensor_id": [f"sensor_{i % 34}" for i in range(n_samples)],
                "value": np.random.randn(n_samples),
            }
        )

    mock.fetch_sensor_data = mock_fetch_sensor_data
    return mock


@pytest.fixture
def mock_feature_engineer() -> MagicMock:
    """Create a mock FeatureEngineer.

    Returns:
        MagicMock with extract_all_features method
    """
    mock = MagicMock()

    def mock_extract_features(data):
        # Return 34 features
        return np.random.randn(34)

    mock.extract_all_features = mock_extract_features
    return mock


@pytest.fixture
def mock_graph_topology() -> MagicMock:
    """Create a mock GraphTopology.

    Returns:
        MagicMock with components and connections
    """
    mock = MagicMock()
    mock.components = {
        f"component_{i}": MagicMock(component_id=f"component_{i}")
        for i in range(10)
    }
    mock.connections = [
        {"from": f"component_{i}", "to": f"component_{i+1}"}
        for i in range(9)
    ]
    return mock


# ============================================================================
# PATH FIXTURES
# ============================================================================


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create temporary directory for checkpoints.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path to checkpoint directory
    """
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """Create temporary directory for logs.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path to log directory
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


# ============================================================================
# PYTEST HOOKS
# ============================================================================


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "slow: slow tests (timeout > 5s)")
    config.addinivalue_line("markers", "gpu: tests requiring GPU")
    config.addinivalue_line("markers", "smoke: smoke tests")


@pytest.fixture(autouse=True)
def reset_seeds():
    """Reset random seeds for reproducibility."""
    import random

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
