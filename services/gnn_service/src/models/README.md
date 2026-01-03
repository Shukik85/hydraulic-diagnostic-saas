# UniversalTemporalGNNv2 - Model Documentation

## 🎯 Overview

**UniversalTemporalGNNv2 v2.0.2** — Production-ready GNN для диагностики гидравлических систем.

### ✅ Status: Production-Ready

- ✅ **21/21 unit tests passing** (100% success rate)
- ✅ **92% test coverage** для основного модуля
- ✅ **All senior review findings addressed**
- ✅ **Backward compatible** (v1 alias available)

---

## 🏗️ Architecture

### Core Components

```python
UniversalTemporalGNNv2
├── Node Encoder (34 → hidden_dim)
├── Edge Encoder (14 → hidden_dim)
├── GATv2 Layers (multi-layer spatial encoding)
│   ├── Multi-head attention (GAT)
│   ├── Edge features (14D dynamic + static)
│   └── Residual connections
├── VirtualNodePooling (optional)
│   └── Size-invariant graph representations
├── AttentionPooling
│   └── Learnable importance weights
├── LSTM (temporal mode)
│   └── Sequence modeling
└── Prediction Heads
    ├── Component Health (node-level, 5 classes)
    └── Anomaly Type (graph-level, 4 classes)
```

### Key Features

**1. Dual Mode:**
- **Single graph**: Direct GNN → projection → predictions
- **Temporal sequences**: GNN → LSTM → predictions

**2. Size-Invariant:**
- AttentionPooling: Learnable node importance
- VirtualNode: Global graph representation
- Handles graphs from 3 to 15,000+ nodes

**3. Multi-Task Learning:**
- Node-level: Component health (5 classes)
- Graph-level: Anomaly type (4 classes)
- Joint optimization with uncertainty weighting

**4. Interpretability:**
- Attention weight extraction
- Component importance scores
- Anomaly type predictions with confidence

---

## ⚙️ Configuration

### ModelConfig

```python
from models import ModelConfig, UniversalTemporalGNNv2

# Production configuration
config = ModelConfig(
    # Input dimensions
    node_features=34,              # Node feature dimension
    edge_features=14,              # Edge feature dimension (static + dynamic)
    
    # GATv2 architecture
    gat_hidden_dim=256,            # Hidden dimension for GAT layers
    gat_num_layers=3,              # Number of GAT layers
    gat_num_heads=4,               # Number of attention heads
    gat_dropout=0.1,               # Dropout rate
    gat_concat_heads=True,         # Concatenate vs. average heads
    
    # LSTM architecture
    lstm_hidden_dim=128,           # LSTM hidden dimension
    lstm_num_layers=2,             # Number of LSTM layers
    lstm_dropout=0.1,              # LSTM dropout
    lstm_bidirectional=False,      # Bidirectional LSTM
    
    # Topology-aware components
    use_virtual_nodes=True,        # Enable VirtualNode pooling
    virtual_node_dim=64,           # Virtual node embedding dimension
    use_attention_pooling=True,    # Enable AttentionPooling
    
    # Multi-task heads
    component_health_num_classes=5,  # Component health classes
    anomaly_type_num_classes=4,      # Anomaly type classes
    head_hidden_dim=64,              # Prediction head hidden dim
    head_dropout=0.2,                # Prediction head dropout
)

model = UniversalTemporalGNNv2(config)
```

### Configuration Validation

All parameters are validated in `__post_init__`:

```python
# ✅ Valid ranges
node_features > 0
edge_features > 0
gat_hidden_dim > 0
gat_num_layers >= 1
gat_dropout in [0, 1]
component_health_num_classes >= 2

# ❌ Invalid examples
ModelConfig(gat_dropout=1.5)  # ValueError: gat_dropout must be in [0,1]
ModelConfig(gat_num_layers=0)  # ValueError: gat_num_layers must be >= 1
```

---

## 🔄 Migration Guide: v1 → v2

### API Changes

**Old (v1):**
```python
from src.training.lightning_module import HydraulicGNNModule

model = HydraulicGNNModule(
    in_channels=34,
    hidden_channels=128,
    num_heads=4,
    num_gat_layers=3,
    lstm_hidden=128,
    lstm_layers=2,
)
```

**New (v2):**
```python
from models import UniversalTemporalGNNv2, ModelConfig

config = ModelConfig(
    node_features=34,
    edge_features=14,
    gat_hidden_dim=128,
    gat_num_heads=4,
    gat_num_layers=3,
    lstm_hidden_dim=128,
    lstm_num_layers=2,
)
model = UniversalTemporalGNNv2(config)
```

### Key Differences

| Feature | v1 | v2 | Migration |
|---------|----|----|----------|
| **Edge features** | 8D (static only) | 14D (static + dynamic) | Update edge_attr to 14D |
| **Size embedding** | Binned graph size | ❌ Removed | No action needed |
| **Batch handling** | May crash on None | ✅ Robust | Works with batch=None |
| **Anomaly head** | Separate for modes | ✅ Unified (64→4) | No action needed |
| **Gradients** | Checked all params | ✅ Excludes LSTM in single mode | Update tests |

### Backward Compatibility

```python
# v1 alias available for migration
from models import UniversalTemporalGNN  # Same as v2

model = UniversalTemporalGNN(config)  # Works!
```

---

## 📊 Usage Examples

### Single Graph Inference

```python
import torch
from torch_geometric.data import Data
from models import UniversalTemporalGNNv2, ModelConfig

# Create model
config = ModelConfig()
model = UniversalTemporalGNNv2(config)
model.eval()

# Single graph
graph = Data(
    x=torch.randn(10, 34),         # 10 nodes, 34 features
    edge_index=torch.randint(0, 10, (2, 20)),  # 20 edges
    edge_attr=torch.randn(20, 14),  # 14D edge features
)

# Inference
with torch.no_grad():
    outputs = model(graph, temporal=False)

print(outputs['node_logits'].shape)   # [10, 5] - Component health
print(outputs['graph_logits'].shape)  # [1, 4] - Anomaly type
```

### Temporal Sequence Inference

```python
# Temporal sequence (5 timesteps)
sequence = [graph for _ in range(5)]

with torch.no_grad():
    outputs = model(sequence, temporal=True)

print(outputs['node_logits'].shape)   # [10, 5] - Last timestep
print(outputs['graph_logits'].shape)  # [1, 4] - Sequence prediction

# Get all timesteps
outputs_all = model(sequence, temporal=True, return_all_timesteps=True)
print(len(outputs_all['node_logits_seq']))  # 5 timesteps
```

### Attention Weights Extraction

```python
with torch.no_grad():
    outputs = model(graph, temporal=False, return_attention=True)

attention = outputs['attention_weights']
for layer_name, attn_weights in attention.items():
    print(f"{layer_name}: {attn_weights.shape}")
    # gatv2_layer_0: [num_edges, num_heads]
    # gatv2_layer_1: [num_edges, num_heads]
```

### Training Loop

```python
from training import MultiTaskLoss

model.train()
loss_fn = MultiTaskLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for graph in dataloader:
    optimizer.zero_grad()
    
    outputs = model(graph, temporal=False)
    
    losses = loss_fn(
        outputs['node_logits'], graph.y_node,
        outputs['graph_logits'], graph.y_graph
    )
    
    total_loss = losses['total']
    total_loss.backward()
    optimizer.step()
```

---

## ✅ Production Checklist

### Before Deployment

- [ ] **Model checkpoint saved with weights_only=True**
  ```python
  torch.save({
      'model_state_dict': model.state_dict(),
      'config': config,
      'normalizer_stats': normalizer.get_stats(),
  }, 'v2.0.2.ckpt')
  ```

- [ ] **All tests passing**
  ```bash
  pytest tests/test_universal_temporal_gnn.py -v
  ```

- [ ] **Inference time < 50ms** (for single graph)
  ```python
  import time
  start = time.time()
  outputs = model(graph, temporal=False)
  print(f"Inference: {(time.time() - start)*1000:.1f}ms")
  ```

- [ ] **GPU memory usage acceptable**
  ```python
  print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
  ```

- [ ] **Batch processing works**
  ```python
  from torch_geometric.data import Batch
  batch = Batch.from_data_list([graph] * 4)
  outputs = model(batch, temporal=False)
  ```

---

## 🎓 Best Practices

### 1. Device Management

```python
# Auto device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
graph = graph.to(device)
```

### 2. Gradient Accumulation

```python
# For large graphs
accumulation_steps = 4
for i, graph in enumerate(dataloader):
    outputs = model(graph)
    loss = loss_fn(...) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for graph in dataloader:
    with autocast():
        outputs = model(graph)
        loss = loss_fn(...)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 4. Reproducibility

```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## 📚 References

- **Paper**: ["How Attentive are Graph Attention Networks?"](https://arxiv.org/abs/2105.14491) (ICLR 2022)
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **Model Tests**: `tests/test_universal_temporal_gnn.py`
- **Integration Tests**: `tests/integration/test_full_pipeline.py`

---

## 🔧 Troubleshooting

### GPU Out of Memory

```python
# Reduce batch size
config = ModelConfig(gat_hidden_dim=128)  # Smaller hidden dim

# Or use gradient checkpointing (future)
# model.enable_gradient_checkpointing()
```

### NaN Loss

```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")

# Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Slow Inference

```python
# Use torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# Or use half precision
model = model.half()
graph = graph.half()
```

---

## 📄 License

MIT License - Part of Hydraulic Diagnostics SaaS
