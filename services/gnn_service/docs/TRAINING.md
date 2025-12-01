# Training Guide - Hydraulic GNN Service

## 📖 Overview

Complete guide for training the Universal Temporal GNN for hydraulic diagnostics using PyTorch Lightning.

**Stack:**
- **Python:** 3.14.0
- **PyTorch:** 2.8.0 + CUDA 12.8
- **PyTorch Lightning:** 2.4.0
- **PyTorch Geometric:** 2.6.1

---

## 🚀 Quick Start

### Basic Training (3 Lines)

```python
from src.training import HydraulicGNNModule, create_development_trainer
from src.data import create_dataloaders

# Load data
train_loader, val_loader = create_dataloaders(batch_size=32)

# Create model
module = HydraulicGNNModule(
    in_channels=34,
    hidden_channels=128,
    num_heads=8,
    learning_rate=0.001
)

# Train
trainer = create_development_trainer()
trainer.fit(module, train_loader, val_loader)
```

**Output:**
- Checkpoints: `checkpoints/dev/`
- Logs: `logs/hydraulic_gnn/`
- View metrics: `tensorboard --logdir=logs`

---

## 🛠️ Training Configurations

### 1. Development Training

**Fast iteration, short patience:**

```python
from src.training import create_development_trainer

trainer = create_development_trainer()
trainer.fit(module, train_loader, val_loader)
```

**Features:**
- Max epochs: 50
- Mixed precision: FP16
- Early stopping: patience=10
- Checkpointing: top-2 models
- Rich progress bar

---

### 2. Production Training

**Full training, multi-GPU support:**

```python
from src.training import create_production_trainer

trainer = create_production_trainer(
    max_epochs=200,
    devices=2  # Multi-GPU
)

trainer.fit(module, train_loader, val_loader)
```

**Features:**
- Max epochs: 200
- Mixed precision: FP16
- Early stopping: patience=30
- Checkpointing: top-5 models
- DDP strategy (if devices > 1)

---

### 3. Custom Training

**Full control over all parameters:**

```python
from src.training import (
    create_trainer,
    TrainerConfig,
    CheckpointConfig,
    EarlyStoppingConfig
)

# Configure trainer
trainer_config = TrainerConfig(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    precision="16",
    gradient_clip_val=1.0,
    accumulate_grad_batches=2,  # Effective batch size *= 2
    log_every_n_steps=10
)

# Configure checkpointing
checkpoint_config = CheckpointConfig(
    dirpath="models/my_experiment",
    monitor="val/total_loss",
    save_top_k=3,
    save_last=True
)

# Configure early stopping
early_stopping_config = EarlyStoppingConfig(
    monitor="val/total_loss",
    patience=20,
    min_delta=1e-4
)

# Create trainer
trainer = create_trainer(
    trainer_config=trainer_config,
    checkpoint_config=checkpoint_config,
    early_stopping_config=early_stopping_config
)

trainer.fit(module, train_loader, val_loader)
```

---

## 🎯 Model Configuration

### LightningModule Parameters

```python
from src.training import HydraulicGNNModule

module = HydraulicGNNModule(
    # === Architecture ===
    in_channels=34,           # Input feature dimension
    hidden_channels=128,      # Hidden dimension
    num_heads=8,              # Attention heads
    num_gat_layers=3,         # GAT layers
    lstm_hidden=256,          # LSTM hidden size
    lstm_layers=2,            # LSTM layers
    
    # === Optimization ===
    learning_rate=0.001,      # Learning rate
    weight_decay=1e-5,        # L2 regularization
    scheduler_type="plateau", # plateau/cosine/none
    
    # === Loss Configuration ===
    loss_weighting="uncertainty",  # fixed/uncertainty
    use_focal_loss=True,      # FocalLoss for anomaly
    use_wing_loss=True,       # WingLoss for regression
    use_quantile_rul=True,    # QuantileRULLoss for RUL
    
    # === Fixed Loss Weights (if loss_weighting="fixed") ===
    loss_weights={
        "graph_health": 1.0,
        "graph_degradation": 1.0,
        "graph_anomaly": 1.0,
        "graph_rul": 1.0,
        "component_health": 0.5,
        "component_anomaly": 0.5,
    }
)
```

---

## 📊 Metrics

### Multi-Level Metrics Tracked

**Component-Level:**
- Health: MAE, RMSE, R², MAPE
- Anomaly: Precision, Recall, F1, AUC

**Graph-Level:**
- Health: MAE, RMSE, R², MAPE
- Degradation: MAE, RMSE, R², MAPE
- Anomaly: Precision, Recall, F1, AUC
- RUL: MAE, Asymmetric Loss, Horizon Accuracy (24h, 72h, 168h)

### Viewing Metrics

**TensorBoard:**
```bash
tensorboard --logdir=logs
# Open: http://localhost:6006
```

**CSV Logs:**
```python
import pandas as pd

metrics = pd.read_csv("logs/hydraulic_gnn_csv/version_0/metrics.csv")
print(metrics[["epoch", "train/total_loss", "val/total_loss"]])
```

---

## 🔧 Hyperparameter Tuning

### Grid Search

```python
from src.training import HydraulicGNNModule, create_trainer

hyperparameters = {
    "learning_rate": [0.0001, 0.001, 0.01],
    "hidden_channels": [64, 128, 256],
    "num_heads": [4, 8, 16],
}

for lr in hyperparameters["learning_rate"]:
    for hc in hyperparameters["hidden_channels"]:
        for nh in hyperparameters["num_heads"]:
            # Create module
            module = HydraulicGNNModule(
                in_channels=34,
                hidden_channels=hc,
                num_heads=nh,
                learning_rate=lr
            )
            
            # Train
            trainer = create_trainer()
            trainer.fit(module, train_loader, val_loader)
            
            # Log best val_loss
            best_val_loss = trainer.callback_metrics["val/total_loss"]
            print(f"LR={lr}, HC={hc}, NH={nh} -> Val Loss={best_val_loss:.4f}")
```

### Random Search with Ray Tune (Optional)

```python
import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

def train_hydraulic_gnn(config):
    module = HydraulicGNNModule(
        in_channels=34,
        hidden_channels=config["hidden_channels"],
        num_heads=config["num_heads"],
        learning_rate=config["learning_rate"]
    )
    
    trainer = create_trainer(
        additional_callbacks=[
            TuneReportCallback(
                {"val_loss": "val/total_loss"},
                on="validation_end"
            )
        ]
    )
    
    trainer.fit(module, train_loader, val_loader)

# Run tuning
analysis = tune.run(
    train_hydraulic_gnn,
    config={
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "hidden_channels": tune.choice([64, 128, 256]),
        "num_heads": tune.choice([4, 8, 16]),
    },
    num_samples=20,
    resources_per_trial={"cpu": 4, "gpu": 1}
)

best_config = analysis.best_config
print(f"Best hyperparameters: {best_config}")
```

---

## 📈 Training from Checkpoint

### Resume Training

```python
from src.training import HydraulicGNNModule, create_trainer

# Load from checkpoint
module = HydraulicGNNModule.load_from_checkpoint(
    "checkpoints/production/hydraulic-gnn-epoch=50-val_total_loss=0.1234.ckpt"
)

# Resume training
trainer = create_trainer()
trainer.fit(
    module,
    train_loader,
    val_loader,
    ckpt_path="checkpoints/production/last.ckpt"  # Resume from last
)
```

### Fine-Tuning

```python
# Load pre-trained model
module = HydraulicGNNModule.load_from_checkpoint(
    "checkpoints/pretrained.ckpt"
)

# Reduce learning rate for fine-tuning
module.learning_rate = 0.0001

# Train on new data
trainer = create_production_trainer(max_epochs=50)
trainer.fit(module, new_train_loader, new_val_loader)
```

---

## 🧪 Model Evaluation

### Test Set Evaluation

```python
# Load best model
module = HydraulicGNNModule.load_from_checkpoint(
    "checkpoints/production/best.ckpt"
)

# Evaluate on test set
trainer = create_trainer()
test_results = trainer.test(module, test_loader)

print("Test Metrics:")
for key, value in test_results[0].items():
    print(f"  {key}: {value:.4f}")
```

### Inference Example

```python
import torch
from torch_geometric.data import Data

# Load model
module = HydraulicGNNModule.load_from_checkpoint("best.ckpt")
module.eval()

# Prepare input
graph_data = Data(
    x=torch.randn(10, 34),           # 10 nodes, 34 features
    edge_index=torch.tensor([[...]]),
    edge_attr=torch.randn(15, 8),
    batch=torch.zeros(10, dtype=torch.long)
)

# Inference
with torch.no_grad():
    outputs = module(
        x=graph_data.x,
        edge_index=graph_data.edge_index,
        edge_attr=graph_data.edge_attr,
        batch=graph_data.batch
    )

# Outputs structure
print("Component Health:", outputs['component']['health'])
print("Graph RUL:", outputs['graph']['rul'])
```

---

## 🚀 Production Deployment

### Export to ONNX

```python
import torch
from src.models import UniversalTemporalGNN

# Load trained model
module = HydraulicGNNModule.load_from_checkpoint("best.ckpt")
model = module.model
model.eval()

# Dummy input
dummy_x = torch.randn(10, 34)
dummy_edge_index = torch.tensor([[0, 1], [1, 2]]).t()
dummy_edge_attr = torch.randn(2, 8)
dummy_batch = torch.zeros(10, dtype=torch.long)

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_x, dummy_edge_index, dummy_edge_attr, dummy_batch),
    "hydraulic_gnn.onnx",
    input_names=["x", "edge_index", "edge_attr", "batch"],
    output_names=["component_health", "component_anomaly", 
                  "graph_health", "graph_degradation", "graph_anomaly", "graph_rul"],
    dynamic_axes={
        "x": {0: "num_nodes"},
        "edge_index": {1: "num_edges"},
        "edge_attr": {0: "num_edges"},
        "batch": {0: "num_nodes"},
    },
    opset_version=17
)

print("✅ Model exported to hydraulic_gnn.onnx")
```

### Convert for Inference Service

```python
# Extract model weights for InferenceEngine
import torch

module = HydraulicGNNModule.load_from_checkpoint("best.ckpt")
model_state = module.model.state_dict()

# Save for inference
torch.save({
    "model_state_dict": model_state,
    "config": {
        "in_channels": 34,
        "hidden_channels": 128,
        "num_heads": 8,
    }
}, "models/hydraulic_gnn_inference.pt")
```

---

## 🔍 Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions:**
```python
# 1. Reduce batch size
train_loader = create_dataloaders(batch_size=16)  # Instead of 32

# 2. Enable gradient accumulation
trainer_config = TrainerConfig(
    accumulate_grad_batches=4  # Effective batch size = 16*4 = 64
)

# 3. Use gradient checkpointing (if supported by model)
module = HydraulicGNNModule(
    in_channels=34,
    use_checkpoint=True  # Enable if implemented
)
```

### Issue: Unstable Training (Loss explodes)

**Solutions:**
```python
# 1. Reduce learning rate
module = HydraulicGNNModule(learning_rate=0.0001)

# 2. Lower gradient clipping threshold
trainer_config = TrainerConfig(gradient_clip_val=0.5)

# 3. Use uncertainty weighting (adapts loss weights)
module = HydraulicGNNModule(loss_weighting="uncertainty")
```

### Issue: Overfitting

**Solutions:**
```python
# 1. Increase weight decay
module = HydraulicGNNModule(weight_decay=1e-4)

# 2. Add dropout (if supported)
module = HydraulicGNNModule(dropout=0.2)

# 3. Reduce early stopping patience
early_stopping_config = EarlyStoppingConfig(patience=10)
```

### Issue: Slow Training

**Solutions:**
```python
# 1. Enable mixed precision
trainer_config = TrainerConfig(precision="16")

# 2. Use multiple GPUs
trainer = create_production_trainer(devices=2)

# 3. Increase batch size (if memory allows)
train_loader = create_dataloaders(batch_size=64)

# 4. Enable PyTorch 2.8 compile (if stable)
model.use_compile = True  # Requires torch>=2.0
```

---

## 📚 Additional Resources

- **PyTorch Lightning Docs:** https://lightning.ai/docs/pytorch/stable/
- **PyTorch Geometric Docs:** https://pytorch-geometric.readthedocs.io/
- **Model Architecture:** `docs/INFERENCE.md`
- **API Reference:** `docs/API.md`
- **System Design:** `docs/ML_SYSTEM_DESIGN.md` (if exists)

---

## 🎯 Next Steps

1. ✅ **Train initial model** with development trainer
2. ✅ **Tune hyperparameters** using grid/random search
3. ✅ **Production training** with best hyperparameters
4. ✅ **Evaluate on test set** and analyze metrics
5. ✅ **Export to ONNX** for production deployment
6. ✅ **Integrate with InferenceEngine** in FastAPI service

---

**Happy Training! 🚀**
