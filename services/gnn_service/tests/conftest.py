"""Pytest configuration and shared fixtures.

Shared fixtures для всех tests.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def project_root_path():
    """Корневая директория проекта."""
    return project_root


@pytest.fixture(scope="session")
def data_dir(project_root_path):
    """Директория data/."""
    return project_root_path / "data"


@pytest.fixture(scope="session")
def models_dir(project_root_path):
    """Директория models/."""
    return project_root_path / "models"
