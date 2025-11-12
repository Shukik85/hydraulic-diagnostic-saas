# Model Checkpoints

Store trained model checkpoints here.

## Structure

```
models/
├── enhanced_model_best.ckpt     # Best validation checkpoint
└── enhanced_model_latest.ckpt   # Latest training checkpoint
```

## Checkpoint Format

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "loss": loss,
    "metrics": {...},
}
```

## Usage

```python
import torch
from model_universal_temporal import create_model

model = create_model(metadata, device="cuda")
checkpoint = torch.load("models/enhanced_model_best.ckpt")
model.load_state_dict(checkpoint["model_state_dict"])
```
