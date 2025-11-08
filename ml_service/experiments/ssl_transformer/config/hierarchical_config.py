"""
Hierarchical GNN Configuration
Physics-guided multi-scale architecture
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ComponentModelConfig:
    """Configuration for component-level models (LEVEL 0)"""
    
    # Cylinder model
    cylinder_features: int = 7  # pressure_extend, pressure_retract, pos, vel, force, pressure_diff, load_ratio
    cylinder_hidden_dim: int = 64
    cylinder_num_layers: int = 2
    
    # Pump model
    pump_features: int = 5  # pressure_outlet, speed_rpm, temperature, vibration, power
    pump_hidden_dim: int = 64
    pump_num_layers: int = 2
    
    # Common
    dropout: float = 0.1
    activation: str = "relu"
    num_classes: int = 2  # normal/fault


@dataclass
class SubsystemModelConfig:
    """Configuration for subsystem models (LEVEL 1)"""
    
    # Subsystem GNN
    hidden_dim: int = 128
    num_heads: int = 4  # GAT heads
    num_layers: int = 2
    dropout: float = 0.1
    
    # Aggregation
    aggregation: str = "attention"  # "mean", "max", "attention"
    
    num_classes: int = 2


@dataclass
class MachineModelConfig:
    """Configuration for machine-level model (LEVEL 2)"""
    
    # Machine GNN
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.15
    
    # Temporal modeling
    use_temporal: bool = True
    sequence_length: int = 10
    
    num_classes: int = 2


@dataclass
class TrainingConfig:
    """Training configuration"""
    
    # Data
    data_dir: Path = Path("data")
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Training (optimized for GTX 1650)
    batch_size: int = 64  # Component level
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 50
    patience: int = 10  # Early stopping
    
    # Device
    device: str = "cuda"  # Will auto-fallback to CPU
    num_workers: int = 4
    
    # Checkpointing
    checkpoint_dir: Path = Path("results/checkpoints")
    log_dir: Path = Path("results/logs")
    save_best_only: bool = True
    
    # Monitoring
    log_interval: int = 10  # Log every N batches
    val_interval: int = 1  # Validate every N epochs


@dataclass
class HierarchicalConfig:
    """Complete hierarchical configuration"""
    
    component: ComponentModelConfig = ComponentModelConfig()
    subsystem: SubsystemModelConfig = SubsystemModelConfig()
    machine: MachineModelConfig = MachineModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    # Topology definition
    topology_file: Path = Path("config/excavator_topology.json")


# Global config
config = HierarchicalConfig()
