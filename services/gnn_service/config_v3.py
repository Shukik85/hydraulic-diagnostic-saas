"""
Optimized config for GNN V3 (F1 > 90%)
"""
from dataclasses import dataclass, field
import torch
@dataclass
class ModelConfigV3:
    num_nodes: int = 7
    num_components: int = 7
    component_names: tuple = (
        "pump","cylinder_boom","cylinder_stick","cylinder_bucket",
        "motor_swing","motor_left","motor_right",
    )
    num_node_features: int = 15
    hidden_dim: int = 96
    num_gat_layers: int = 3
    num_heads: int = 4
    gat_dropout: float = 0.18      # снизили
    num_lstm_layers: int = 1
    lstm_dropout: float = 0.08
    num_classes: int = 7
    sequence_length: int = 5
@dataclass
class TrainingConfigV3:
    batch_size: int = 16          # увеличили
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    learning_rate: float = 7e-4   # немного ниже
    weight_decay: float = 2e-4    # чуть выше
    epochs: int = 120
    patience: int = 20            # увеличили
    warmup_epochs: int = 8        # добавили warmup
    label_smoothing: float = 0.04 # снизили
    gradient_clip: float = 1.2    # увеличили
    lr_scheduler_factor: float = 0.7
    lr_scheduler_patience: int = 10
    lr_scheduler_min_lr: float = 4e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    model_save_path: str = "models/gnn_classifier_best_v3.ckpt"
    history_save_path: str = "models/training_history_v3.json"
    data_path: str = "data/bim_comprehensive.csv"
    graphs_save_path: str = "data/gnn_graphs_multilabel.pt"
@dataclass
class APIConfigV3:
    host: str = "0.0.0.0"
    port: int = 8003
    debug: bool = False
    log_level: str = "info"
    max_batch_size: int = 32
    inference_timeout: float = 30.0
model_config_v3 = ModelConfigV3()
training_config_v3 = TrainingConfigV3()
api_config_v3 = APIConfigV3()
