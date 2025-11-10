"""
Configuration file for GNN hydraulic diagnostics service.
Fixed mutable default issue with dataclasses.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class PhysicalNorms:
    """Corrected physical norms for hydraulic components."""

    # Pump specifications (corrected)
    PUMP = {
        "pressure_outlet": {
            "nominal": 250.0,
            "min": 210.0,
            "max": 270.0,
            "critical": 300.0,
        },
        "speed_rpm": {
            "nominal": 2000.0,
            "min": 1800.0,
            "max": 2200.0,
            "critical": 2500.0,
        },
        "temperature": {"nominal": 60.0, "min": 40.0, "max": 75.0, "critical": 85.0},
        "vibration": {"nominal": 2.5, "min": 1.0, "max": 4.0, "critical": 6.0},
        "power": {"nominal": 75.0, "min": 65.0, "max": 85.0, "critical": 95.0},  # kW
    }

    # Cylinder specifications (corrected and unified)
    CYLINDER_BOOM = {
        "pressure_extend": {
            "nominal": 200.0,
            "min": 180.0,
            "max": 220.0,
            "critical": 250.0,
        },
        "pressure_retract": {
            "nominal": 60.0,
            "min": 40.0,
            "max": 80.0,
            "critical": 100.0,
        },
        "position": {"nominal": 50.0, "min": 0.0, "max": 100.0, "critical": None},  # %
        "velocity": {
            "nominal": 200.0,
            "min": 100.0,
            "max": 300.0,
            "critical": 400.0,
        },  # mm/s
        "pressure_diff": {
            "nominal": 160.0,
            "min": 140.0,
            "max": 180.0,
            "critical": 200.0,
        },
    }

    CYLINDER_STICK = {
        "pressure_extend": {
            "nominal": 190.0,
            "min": 170.0,
            "max": 210.0,
            "critical": 240.0,
        },
        "pressure_retract": {
            "nominal": 55.0,
            "min": 35.0,
            "max": 75.0,
            "critical": 95.0,
        },
        "position": {"nominal": 50.0, "min": 0.0, "max": 100.0, "critical": None},
        "velocity": {"nominal": 180.0, "min": 80.0, "max": 280.0, "critical": 350.0},
        "pressure_diff": {
            "nominal": 150.0,
            "min": 130.0,
            "max": 170.0,
            "critical": 190.0,
        },
    }

    CYLINDER_BUCKET = {
        "pressure_extend": {
            "nominal": 180.0,
            "min": 160.0,
            "max": 200.0,
            "critical": 230.0,
        },
        "pressure_retract": {
            "nominal": 50.0,
            "min": 30.0,
            "max": 70.0,
            "critical": 90.0,
        },
        "position": {"nominal": 50.0, "min": 0.0, "max": 100.0, "critical": None},
        "velocity": {"nominal": 150.0, "min": 70.0, "max": 250.0, "critical": 320.0},
        "pressure_diff": {
            "nominal": 140.0,
            "min": 120.0,
            "max": 160.0,
            "critical": 180.0,
        },
    }

    # Motor specifications (corrected)
    MOTOR_SWING = {
        "speed_rpm": {
            "nominal": 1500.0,
            "min": 1200.0,
            "max": 1800.0,
            "critical": 2000.0,
        },
        "torque": {
            "nominal": 650.0,
            "min": 500.0,
            "max": 800.0,
            "critical": 900.0,
        },  # Nm
        "temperature": {"nominal": 65.0, "min": 45.0, "max": 80.0, "critical": 90.0},
        "pressure_inlet": {
            "nominal": 210.0,
            "min": 180.0,
            "max": 240.0,
            "critical": 270.0,
        },
        "vibration": {"nominal": 3.0, "min": 1.5, "max": 5.0, "critical": 7.0},
    }

    MOTOR_LEFT = {
        "speed_rpm": {
            "nominal": 1450.0,
            "min": 1150.0,
            "max": 1750.0,
            "critical": 1950.0,
        },
        "torque": {"nominal": 600.0, "min": 450.0, "max": 750.0, "critical": 850.0},
        "temperature": {"nominal": 63.0, "min": 43.0, "max": 78.0, "critical": 88.0},
        "pressure_inlet": {
            "nominal": 205.0,
            "min": 175.0,
            "max": 235.0,
            "critical": 265.0,
        },
        "vibration": {"nominal": 3.2, "min": 1.7, "max": 5.2, "critical": 7.2},
    }

    MOTOR_RIGHT = {
        "speed_rpm": {
            "nominal": 1450.0,
            "min": 1150.0,
            "max": 1750.0,
            "critical": 1950.0,
        },
        "torque": {"nominal": 600.0, "min": 450.0, "max": 750.0, "critical": 850.0},
        "temperature": {"nominal": 63.0, "min": 43.0, "max": 78.0, "critical": 88.0},
        "pressure_inlet": {
            "nominal": 205.0,
            "min": 175.0,
            "max": 235.0,
            "critical": 265.0,
        },
        "vibration": {"nominal": 3.2, "min": 1.7, "max": 5.2, "critical": 7.2},
    }


@dataclass
class GPUConfig:
    """GPU-specific configuration for optimal performance."""

    # Memory optimization
    pin_memory: bool = True
    non_blocking: bool = True
    memory_fraction: float = 0.8  # Limit GPU memory usage

    # CUDA optimization
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    torch_compile: bool = False  # Disable for 4GB GPUs

    # Mixed precision training
    mixed_precision: bool = True
    amp_dtype: str = "float16"

    # Batch size scaling for GPU
    gpu_batch_size_multiplier: int = 1  # No multiplier for low VRAM

    # Gradient checkpointing
    gradient_checkpointing: bool = True


@dataclass
class ModelConfig:
    """GNN model configuration with corrected parameters."""

    # Graph structure
    num_nodes: int = 7
    num_components: int = 7
    component_names: tuple = (
        "pump",
        "cylinder_boom",
        "cylinder_stick",
        "cylinder_bucket",
        "motor_swing",
        "motor_left",
        "motor_right",
    )

    # Feature dimensions
    num_raw_features: int = 5
    num_norm_features: int = 5
    num_deviation_features: int = 5
    num_node_features: int = 15

    # GAT architecture (optimized for 4GB)
    hidden_dim: int = 96
    num_gat_layers: int = 2
    num_heads: int = 4
    gat_dropout: float = 0.2
    num_lstm_layers: int = 1
    lstm_dropout: float = 0.1

    # Output
    num_classes: int = 7

    # Temporal dimension
    sequence_length: int = 5


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data
    batch_size: int = 12
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2

    # Memory optimization
    gradient_accumulation_steps: int = 4

    # GPU-specific - Исправлено: используем field() для mutable default
    gpu_config: GPUConfig = field(default_factory=GPUConfig)

    # Paths
    data_path: str = "data/bim_comprehensive.csv"
    model_save_path: str = "models/gnn_classifier_best.ckpt"
    graphs_save_path: str = "data/gnn_graphs_multilabel.pt"
    metadata_path: str = "data/equipment_metadata.json"


@dataclass
class APIConfig:
    """FastAPI configuration."""

    host: str = "0.0.0.0"
    port: int = 8003
    debug: bool = False
    log_level: str = "info"

    # Inference thresholds
    warning_threshold: float = 0.3
    critical_threshold: float = 0.7

    # Performance
    max_batch_size: int = 32
    inference_timeout: float = 30.0


# Global configuration instances
physical_norms = PhysicalNorms()
model_config = ModelConfig()
training_config = TrainingConfig()
api_config = APIConfig()
