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
def project_root_path() -> Path:
    """Корневая директория проекта.
    
    Returns:
        Path: Root directory of the project
    """
    return project_root


@pytest.fixture(scope="session")
def data_dir(project_root_path: Path) -> Path:
    """Директория data/.
    
    Args:
        project_root_path: Root path fixture
    
    Returns:
        Path: Data directory path
    """
    return project_root_path / "data"


@pytest.fixture(scope="session")
def models_dir(project_root_path: Path) -> Path:
    """Директория models/.
    
    Args:
        project_root_path: Root path fixture
    
    Returns:
        Path: Models directory path
    """
    return project_root_path / "models"
