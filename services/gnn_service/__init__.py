"""
GNN Service package for hydraulic system diagnostics.

Temporal Graph Attention Network for multi-component fault detection
in hydraulic excavator systems.
"""

__version__ = "1.0.0"
__author__ = "Hydraulic Diagnostics Team"
__description__ = (
    "Temporal Graph Attention Network for excavator component fault detection"
)

# Package metadata
PACKAGE_METADATA = {
    "version": __version__,
    "components": [
        "prepare_bim_data.py - Data preparation and graph creation",
        "dataset.py - PyTorch Dataset class for graph data",
        "model.py - Temporal GAT model architecture",
        "train.py - Model training and evaluation",
        "inference.py - Prediction and diagnostics engine",
        "main.py - FastAPI service for real-time diagnostics",
        "config.py - Configuration and physical norms",
    ],
    "required_dependencies": [
        "torch>=2.3.1",
        "torch-geometric",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
}


def get_package_info():
    """Get package information and requirements."""
    return PACKAGE_METADATA


# Lazy imports to avoid circular dependencies and missing module errors
def __getattr__(name):
    """Lazy import of modules and classes."""

    # Config imports
    if name in ["physical_norms", "model_config", "training_config", "api_config"]:
        from . import config

        return getattr(config, name)

    # Model imports
    if name in ["TemporalGAT", "HydraulicGNN", "create_model"]:
        from . import model

        return getattr(model, name)

    # Inference imports
    if name in ["GNNInference", "get_inference_engine"]:
        from . import inference

        return getattr(inference, name)

    # Dataset imports
    if name in ["HydraulicGraphDataset", "split_dataset"]:
        from . import dataset

        return getattr(dataset, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Export list (for documentation and type checkers)
__all__ = [
    # Config
    "physical_norms",
    "model_config",
    "training_config",
    "api_config",
    # Core functionality
    "TemporalGAT",
    "HydraulicGNN",
    "create_model",
    "GNNInference",
    "get_inference_engine",
    "HydraulicGraphDataset",
    "split_dataset",
    # Utility
    "get_package_info",
]
