# GNN Service - Production-Ready Implementation

🌱 **Status:** In Active Development  
🔗 **Branch:** `feature/gnn-service-production-ready`  
📅 **Created:** 2025-11-21  
🎯 **Epic Issue:** [#92](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/92)

---

## 🚀 Overview

Production-ready Graph Neural Network service для диагностики гидравлических систем с использованием **Universal Temporal GNN** (GATv2 + ARMA-LSTM).

### Technology Stack (Updated 2025-11-21)

- 🐍 **Python 3.14.0** - Deferred annotations (PEP 649), union types
- ⚡ **PyTorch 2.8.0** - Float8 training, torch.compile, torch.inference_mode
- 🖥️ **CUDA 12.9** - Blackwell GPU support, optimizations
- 🧠 **PyTorch Lightning 2.1+** - Structured training pipeline
- 🔥 **PyTorch Geometric 2.6+** - GNN operations (GATv2Conv)
- 🚀 **FastAPI 0.109+** - Async API framework
- ✅ **Pydantic v2.6+** - Data validation with ConfigDict
- 📊 **TimescaleDB** - Time-series sensor data
- 🔄 **Redis** - Caching layer

### Key Features

- ✅ **GATv2 Architecture** - Dynamic attention (vs static GAT) [+9-10% accuracy]
- 🔥 **ARMA-LSTM** - Autoregressive moving-average temporal attention (ICLR 2025) [+9.1% forecasting]
- 🎯 **Edge-Conditioned Attention** - Hydraulic topology features (diameter, length, material)
- 🧠 **Multi-Task Learning** - Cross-task attention (health ↔ degradation ↔ anomaly) [+11.4% F1]
- ⚡ **torch.compile** - PyTorch 2.8 JIT compilation [1.5x speedup]
- 🚀 **Production Pipeline** - PyTorch Lightning, DDP, Float8 training
- 📊 **Observability** - Prometheus metrics, structured logging
- 🐳 **Containerized** - Docker with CUDA 12.9 support

---

## 📋 Current Status

### ✅ Phase 1 - Week 1 (Foundation)

**Completed (2025-11-21):**
- [x] Repository structure cleanup
- [x] Legacy files archived to `_legacy/`
- [x] New `src/` modular structure
- [x] Epic Issue #92 created
- [x] Sub-Issues #93-96 created
- [x] **[Issue #93](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/93) COMPLETE** ✅ Core Schemas (5 commits, 1550 lines, 33 tests)
  - Pydantic v2 schemas (graph, metadata, requests, responses)
  - Python 3.14 deferred annotations
  - GATv2 edge features support (EdgeSpec)
  - Multi-label classification support
  - Unit tests with 90%+ coverage

**In Progress (2025-11-21 21:00 MSK):**
- [ ] **[#94](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/94) - GNN Model Architecture** (50% done)
  - ✅ GATv2 + ARMA-LSTM implementation
  - ✅ Edge-conditioned attention layers
  - ✅ Multi-task learning head
  - ✅ Model utilities (checkpoint, summary)
  - 🔄 Documentation update (in progress)
  - [ ] Unit tests for models
  - [ ] Integration tests

**Pending:**
- [ ] [#95](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/95) - Dataset & DataLoader (14h)
- [ ] [#96](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/96) - Inference Engine (10h)

### 🔲 Phase 2 - Week 2 (Training & Integration)
- Training pipeline (PyTorch Lightning)
- Distributed training (DDP)
- Float8 training integration
- FastAPI ↔ TimescaleDB
- Model management

### 🔲 Phase 3 - Week 3 (Production Hardening)
- Observability (logging, metrics)
- Error handling & resilience
- Comprehensive testing
- API documentation
- Deployment (Docker, K8s)

---

## 🏗️ GNN Model Architecture

### Overview

**UniversalTemporalGNN** = **GATv2 (spatial)** + **ARMA-LSTM (temporal)** + **Multi-Task Head**

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Sensor Time-Series                │
│              [equipment_id, time_window, sensors]           │
└────────────────────────────┬────────────────────────────────┘
                             ↓
                    ┌────────────────┐
                    │ Graph Builder  │
                    │ - Components   │
                    │ - Edges        │
                    │ - Topology     │
                    └────────┬───────┘
                             ↓
┌────────────────────────────────────────────────────────────────┐
│                 UniversalTemporalGNN Model                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1️⃣ Input Projection                                          │
│     Linear(F_in → H)                                           │
│     ↓                                                          │
│  2️⃣ GATv2 Layers (×3) - Spatial Modeling                      │
│     ┌──────────────────────────────────────┐                  │
│     │ EdgeConditionedGATv2Layer            │                  │
│     │ - Dynamic attention (vs static GAT)  │                  │
│     │ - Edge features (diameter, length)   │                  │
│     │ - Multi-head (8 heads)               │                  │
│     │ - Skip connections                   │                  │
│     │ - Layer normalization                │                  │
│     └──────────────────────────────────────┘                  │
│     ↓                                                          │
│  3️⃣ Temporal Aggregation                                      │
│     Global Mean Pool (per graph)                              │
│     ↓                                                          │
│  4️⃣ ARMA-Attention LSTM (×2) - Temporal Modeling              │
│     ┌──────────────────────────────────────┐                  │
│     │ ARMAAttentionLSTM                    │                  │
│     │ - AR component (historical trends)   │                  │
│     │ - MA component (smoothing)           │                  │
│     │ - Multi-head attention               │                  │
│     │ - Residual connections               │                  │
│     └──────────────────────────────────────┘                  │
│     ↓                                                          │
│  5️⃣ Multi-Task Head - Cross-Task Attention                    │
│     ┌──────────────────────────────────────┐                  │
│     │ CrossTaskAttention (4 heads)         │                  │
│     │ - Shared encoder                     │                  │
│     │ - Task interaction (health ↔ anom)   │                  │
│     │ - Task-specific projections          │                  │
│     └──────────────────────────────────────┘                  │
│     ↓                                                          │
│  6️⃣ Task-Specific Heads                                       │
│     ├─ Health Head: Linear(H → 64 → 1) + Sigmoid            │
│     ├─ Degradation Head: Linear(H → 64 → 1) + Sigmoid       │
│     └─ Anomaly Head: Linear(H → 64 → 9) (multi-label)       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────────┐
│                    Outputs (3 tasks)                           │
├────────────────────────────────────────────────────────────────┤
│  • Health Score: [0, 1] (1 = healthy)                          │
│  • Degradation Rate: [0, 1] (0 = stable, 1 = rapid)           │
│  • Anomaly Logits: [9] (pressure_drop, cavitation, etc.)      │
└────────────────────────────────────────────────────────────────┘
```

### Why GATv2 (not GAT)?

**GAT (2018):** Static attention - node ranking independent of query node  
```python
# GAT attention
alpha = LeakyReLU(a^T [Wh_i || Wh_j])  # Static!
```

**GATv2 (2021, improved 2024-2025):** Dynamic attention - query-dependent ranking  
```python
# GATv2 attention
alpha = a^T LeakyReLU(W [h_i || h_j])  # Dynamic!
```

**Результаты:**
- **+9-10% accuracy** на fraud detection tasks
- **Лучше работает** на heterophilic graphs (разные соседи)
- **Production-proven** (используется в Microsoft, Google)

### Edge-Conditioned Attention

**Идея:** Модулировать attention weights характеристиками соединений.

```python
# Edge features для hydraulic systems:
edge_features = {
    "diameter_mm": 16.0,           # Диаметр трубы
    "length_m": 2.5,                # Длина
    "pressure_rating_bar": 350,     # Номинальное давление
    "material": "steel",            # Материал
    "flow_direction": "unidirectional"  # Направление потока
}

# Computed:
cross_section_area = π * (diameter/2)^2
pressure_loss_coeff = length / diameter^4

# Attention modulation:
attn_weight = base_attention * edge_gate(edge_features)
```

**Почему важно для гидравлики:**
- Длинная тонкая труба → **больше потери давления** → проблемы распространяются медленнее
- Короткая широкая труба → **быстрое распространение** → проблемы видны сразу
- Материал влияет на вибрации и износ

### ARMA-Attention LSTM

**Reference:** *Autoregressive Moving-average Attention Mechanism for Time Series Forecasting* (ICLR 2025 submission)  
**Results:** +9.1% improvement в forecasting accuracy

**Компоненты:**

**1. AR (Autoregressive)** - учёт исторических трендов:
```python
AR_component = Σ(i=1 to p) φ_i * X_{t-i}
# φ_i - learnable AR coefficients
# Captures: degradation trends, seasonal patterns
```

**2. MA (Moving Average)** - сглаживание и инерционные процессы:
```python
MA_component = Σ(i=1 to q) θ_i * ε_{t-i}
# θ_i - learnable MA coefficients  
# Captures: smoothing, inertial hydraulic processes
```

**3. Combined Attention:**
```python
attn_modulation = exp(AR_component + MA_component)
attn_final = softmax(base_attention * attn_modulation)
```

**Применение к hydraulics:**
- **AR:** Долгосрочная деградация (износ уплотнений, накопление загрязнений)
- **MA:** Краткосрочные флуктуации (pressure spikes, temperature changes)
- **Result:** Более точное prediction времени до отказа

### Multi-Task Learning Head

**Reference:** *Multi-task Graph Anomaly Detection Network* (Microsoft, 2022)  
**Results:** +11.4% F1-score improvement

**Идея:** Моделировать корреляции между задачами:

```python
# Task correlations:
Low health → High degradation (obvious)
High degradation → Anomaly likely (predictive)
Anomaly detected → Re-assess health (feedback)

# Cross-task attention:
task_repr = [health_repr, degradation_repr, anomaly_repr]  # [3, B, H]
attended_repr = MultiheadAttention(task_repr, task_repr, task_repr)

# Each task "sees" other tasks during prediction
```

**Результаты:**
- **Улучшение consistency** predictions между задачами
- **Robustness** к шумным данным (один таск помогает другим)
- **Early warning** - degradation предсказывает anomaly

---

## 🧠 Detailed Model Architecture

### Model Configuration

```python
from src.models import UniversalTemporalGNN

model = UniversalTemporalGNN(
    in_channels=12,           # Sensor features per component
    hidden_channels=128,      # GNN hidden dimension
    num_heads=8,              # Attention heads
    num_gat_layers=3,         # GAT depth
    lstm_hidden=256,          # LSTM hidden dimension
    lstm_layers=2,            # LSTM depth
    ar_order=3,               # Autoregressive order
    ma_order=2,               # Moving average order
    dropout=0.3,              # Dropout rate
    use_edge_features=True,   # Enable edge conditioning
    edge_feature_dim=8,       # Edge feature dimension
    use_compile=True,         # Enable torch.compile (PyTorch 2.8)
    compile_mode="reduce-overhead"  # Compilation mode
)
```

### Forward Pass Example

```python
import torch
from torch_geometric.data import Data, Batch

# Подготовка данных
graph = Data(
    x=node_features,        # [N, 12] - sensor features per component
    edge_index=edge_index,  # [2, E] - connectivity
    edge_attr=edge_attr,    # [E, 8] - edge features (diameter, length, etc.)
)

# Batch of graphs
batch = Batch.from_data_list([graph1, graph2, graph3])

# Inference
model.eval()
with torch.inference_mode():  # PyTorch 2.8 optimization
    health, degradation, anomaly = model(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        batch=batch.batch
    )

# Outputs:
# health: [3, 1] - health scores для 3 equipment
# degradation: [3, 1] - degradation rates
# anomaly: [3, 9] - anomaly logits (9 types)
```

### Attention Visualization

```python
# Debug mode - return attention weights
health, degradation, anomaly, attention_weights = model(
    x=batch.x,
    edge_index=batch.edge_index,
    edge_attr=batch.edge_attr,
    batch=batch.batch,
    return_attention=True
)

# attention_weights: List[Tensor]
# - attention_weights[0]: Layer 1 attention [E, num_heads]
# - attention_weights[1]: Layer 2 attention [E, num_heads]
# - attention_weights[2]: Layer 3 attention [E, num_heads]

# Visualize which components are most important
import matplotlib.pyplot as plt
from src.visualization import plot_attention_graph

plot_attention_graph(
    edge_index=batch.edge_index,
    attention_weights=attention_weights[0],  # First layer
    component_names=["pump", "valve", "cylinder"],
    save_path="attention_layer1.png"
)
```

### PyTorch 2.8 torch.compile

**Automatic optimization при инициализации:**

```python
model = UniversalTemporalGNN(
    ...,
    use_compile=True,
    compile_mode="reduce-overhead"  # Options: default, reduce-overhead, max-autotune
)

# Compilation происходит при первом forward pass
# Expect: ~30s warmup, затем 1.5x speedup

# First call (compilation happens)
output = model(x, edge_index)  # Takes ~30s

# Subsequent calls (compiled)
output = model(x, edge_index)  # 1.5x faster!
```

**Compilation modes:**
- `"default"` - Balanced speed/memory
- `"reduce-overhead"` - Minimize overhead (recommended)
- `"max-autotune"` - Maximum performance (longer compile time)

---

## 📚 Layer-by-Layer Explanation

### Layer 1: Input Projection

```python
self.input_projection = nn.Linear(in_channels, hidden_channels)
x = self.input_projection(x)  # [N, F_in] -> [N, H]
x = F.relu(x)
```

**Purpose:** Проектировать raw sensor features в latent space.

### Layer 2: EdgeConditionedGATv2

```python
class EdgeConditionedGATv2Layer:
    def __init__(self, in_channels, out_channels, heads, edge_dim):
        self.gatv2 = GATv2Conv(
            in_channels, out_channels, heads, edge_dim=edge_dim
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, heads),
            nn.Sigmoid()  # Gate: [0, 1]
        )
```

**Attention Computation:**
```python
# 1. GATv2 base attention
alpha_base = GATv2(x_i, x_j, edge_attr)  # [E, heads]

# 2. Edge gating
edge_gates = edge_gate(edge_attr)  # [E, heads]

# 3. Modulated attention
alpha_final = alpha_base * edge_gates
alpha_final = softmax(alpha_final)  # Normalize
```

**Why важно:**
- Короткая wide труба: high gate → strong attention → быстрое распространение проблем
- Длинная thin труба: low gate → weak attention → медленное распространение

### Layer 3: ARMAAttentionLSTM

```python
class ARMAAttentionLSTM:
    def __init__(self, input_dim, hidden_dim, ar_order=3, ma_order=2):
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.ar_weights = nn.Parameter(torch.randn(ar_order))
        self.ma_weights = nn.Parameter(torch.randn(ma_order))
```

**ARMA Modulation Computation:**
```python
# Time distance matrix
time_dists = |i - j|  # [T, T]

# AR component (учёт прошлого)
AR = Σ φ_i * (time_dists == i+1)  # i = 1..3

# MA component (сглаживание)
MA = Σ θ_i * (time_dists <= i+1)  # i = 1..2

# ARMA modulation
modulation = exp(AR + MA)  # [T, T]

# Apply к attention
attn = softmax(Q @ K^T / √d_k * modulation)
```

**Captures:**
- **AR:** Degradation trends (постепенный износ)
- **MA:** Инерционные процессы (тепловая инерция, fluid momentum)

### Layer 4: CrossTaskAttention

```python
class CrossTaskAttention:
    def forward(self, shared_repr):  # [B, H]
        # Create task representations
        task_repr = stack([
            health_proj(shared_repr),
            degradation_proj(shared_repr),
            anomaly_proj(shared_repr)
        ])  # [3, B, H]
        
        # Cross-task attention
        attended = MultiheadAttention(
            query=task_repr,
            key=task_repr,
            value=task_repr
        )  # [3, B, H]
        
        # Residual
        task_repr = task_repr + attended
        
        return task_repr
```

**Example correlation:**
```
Health task "sees":
  - Own prediction: 0.5 (warning)
  - Degradation task: 0.8 (high degradation)
  - Anomaly task: 0.9 (anomaly detected)
  → Adjusts health down to 0.4 (critical)
```

### Layer 5: Task-Specific Heads

```python
# Health Head
health = Sequential(
    Linear(lstm_hidden, 64),
    ReLU(),
    Dropout(0.3),
    Linear(64, 1),
    Sigmoid()  # [0, 1]
)

# Degradation Head (similar)
degradation = Sequential(...)

# Anomaly Head (multi-label)
anomaly = Sequential(
    Linear(lstm_hidden, 64),
    ReLU(),
    Dropout(0.3),
    Linear(64, 9)  # 9 anomaly types
)
# Note: No sigmoid here - logits для multi-label loss
```

**9 Anomaly Types:**
1. `pressure_drop` - Падение давления
2. `overheating` - Перегрев
3. `cavitation` - Кавитация
4. `leakage` - Утечка
5. `vibration_anomaly` - Аномальная вибрация
6. `flow_restriction` - Ограничение потока
7. `contamination` - Загрязнение жидкости
8. `seal_degradation` - Износ уплотнений
9. `valve_stiction` - Залипание клапана

---

## 🎯 Model Parameters

### Default Configuration

```python
model = UniversalTemporalGNN(
    in_channels=12,           # 3-4 sensors per component (pressure, temp, vibration)
    hidden_channels=128,      # GNN latent dimension
    num_heads=8,              # Attention heads (128 / 8 = 16 per head)
    num_gat_layers=3,         # 3-layer GAT
    lstm_hidden=256,          # LSTM hidden state
    lstm_layers=2,            # 2-layer LSTM
    ar_order=3,               # AR(3) - 3 historical timesteps
    ma_order=2,               # MA(2) - 2-step smoothing
    dropout=0.3,              # 30% dropout
    use_edge_features=True,   # Edge conditioning enabled
    edge_feature_dim=8,       # 8D edge features
    use_compile=True          # torch.compile enabled
)
```

### Model Size

```python
from src.models.utils import print_model_summary

print_model_summary(model)

# Output:
# Model: UniversalTemporalGNN
# ==================================================
# Total Parameters: ~2.5M
# Trainable Parameters: ~2.5M
# Memory Footprint: ~9.5 MB (float32)
# ==================================================
# 
# Top Layers:
# - temporal_lstm.lstm.weight_ih_l0    | 131,072 params
# - temporal_lstm.lstm.weight_hh_l0    | 262,144 params
# - gat_layers.0.gatv2.lin_src.weight  | 16,384 params
# ...
```

**Comparison:**
- Original (stub): ~500K params
- **New (production):** ~2.5M params (+5x capacity)
- Memory: 9.5 MB (CPU) / 12-15 MB (GPU with buffers)

---

## 📊 Training

### Basic Training

```python
from src.training import GNNTrainer
from src.data import HydraulicGraphDataset
import lightning as L

# Load dataset
train_dataset = HydraulicGraphDataset(
    data_path="data/processed/train",
    sequence_length=10,
    transform=None
)

val_dataset = HydraulicGraphDataset(
    data_path="data/processed/val",
    sequence_length=10,
    transform=None
)

# Initialize trainer
trainer = GNNTrainer(
    model=model,
    learning_rate=0.001,
    weight_decay=0.0001,
    scheduler="cosine",
    loss_weights={"health": 1.0, "degradation": 1.0, "anomaly": 2.0}
)

# Lightning trainer
trainer_pl = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    precision="16-mixed",  # AMP
    log_every_n_steps=10,
    val_check_interval=0.25
)

# Train
trainer_pl.fit(trainer, train_dataset, val_dataset)
```

### Distributed Training (Multi-GPU)

```python
trainer_pl = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=4,              # 4 GPUs
    strategy="ddp",         # Distributed Data Parallel
    precision="16-mixed",
    sync_batchnorm=True
)

trainer_pl.fit(trainer, train_dataset, val_dataset)
```

### Float8 Training (PyTorch 2.8)

**Requirements:** A100/H100 GPU

```python
from torchao.float8 import convert_to_float8_training

# Convert model to float8
model = convert_to_float8_training(model)

# Train as usual - 1.5x faster!
trainer_pl.fit(trainer, train_dataset, val_dataset)

# Results:
# - 1.5x training speedup
# - Same accuracy (no degradation)
# - Lower memory footprint
```

---

## 💡 Advanced Features

### Spectral-Temporal Layer

**Optional:** Frequency domain processing для periodic patterns.

```python
from src.models.layers import SpectralTemporalLayer

# Add после LSTM
model.spectral_layer = SpectralTemporalLayer(
    hidden_dim=256,
    num_frequencies=32
)

# Использование
out, hidden = model.temporal_lstm(x)
out = model.spectral_layer(out)  # FFT processing
```

**Captures:**
- Periodic pressure oscillations
- Resonance frequencies (cavitation, vibration)
- Harmonics в sensor signals

### Dynamic Batching

**Production optimization** для throughput:

```python
from src.inference import DynamicBatchProcessor

processor = DynamicBatchProcessor(
    model=model,
    max_batch_size=32,
    max_wait_ms=50  # Max latency tolerance
)

# Accumulate requests
await processor.add_request(request1)
await processor.add_request(request2)
# ...

# Automatic batching & processing
# Result: 3-5x throughput improvement
```

---

## 🔗 Integration with Other Services

### TimescaleDB Integration

```python
from src.data import TimescaleConnector

# Fetch sensor data
connector = TimescaleConnector(db_url=DATABASE_URL)

sensor_data = await connector.fetch_sensor_data(
    equipment_id="excavator_001",
    start_time=datetime(2025, 11, 1),
    end_time=datetime(2025, 11, 21),
    sensors=["pressure_pump_out", "temperature_fluid", "vibration"]
)

# Returns: pandas DataFrame with time-series data
```

### Redis Caching

```python
from src.inference import CachedInferenceEngine

engine = CachedInferenceEngine(
    model=model,
    redis_url=REDIS_URL,
    ttl_seconds=300  # 5 minutes cache
)

# First call: cache miss, runs inference
result = await engine.predict(equipment_id="exc_001", ...)

# Second call (within 5 min): cache hit, instant response
result = await engine.predict(equipment_id="exc_001", ...)  # From cache!
```

---

## 📖 Documentation Structure

### Main Docs
- **[README.md](README.md)** (this file) - Overview & quick start
- **[STRUCTURE.md](STRUCTURE.md)** - Detailed architecture
- **[MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)** - Migration from legacy
- **[Epic Issue #92](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/92)** - Full roadmap

### API Documentation
- **Swagger UI:** http://localhost:8002/docs
- **ReDoc:** http://localhost:8002/redoc
- **OpenAPI JSON:** http://localhost:8002/openapi.json

### Code Documentation
- **Schemas:** `src/schemas/` - Pydantic models с docstrings
- **Models:** `src/models/` - GNN architecture
- **Data:** `src/data/` - Dataset & preprocessing
- **Training:** `src/training/` - Training pipeline
- **Inference:** `src/inference/` - Inference engine

---

## 🧪 Testing

### Run Tests

```bash
# All tests with coverage
pytest --cov=src --cov-report=term-missing --cov-report=html

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_schemas.py -v

# GPU tests (requires CUDA)
pytest -m gpu

# Slow tests
pytest -m slow

# Parallel testing
pytest -n auto
```

### Code Quality

```bash
# Format
ruff format src/ tests/

# Lint
ruff check src/ tests/

# Auto-fix
ruff check --fix src/ tests/

# Type check (strict mode)
mypy src/ tests/

# All checks
./scripts/quality_checks.sh
```

---

## 🐳 Docker

### Development

```dockerfile
# Dockerfile.dev
FROM nvidia/cuda:12.9.0-cudnn9-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.14

WORKDIR /app
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
```

### Production

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.9.0-cudnn9-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.14

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8002

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "4"]
```

---

## 📈 Performance Benchmarks

### Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **Inference Latency** | < 500ms | Single equipment, p95 |
| **Batch Throughput** | > 100 eq/s | Batch size 32 |
| **Health MAE** | < 0.05 | Validation set |
| **Degradation MAE** | < 0.05 | Validation set |
| **Anomaly F1** | > 0.85 | Multi-label avg |
| **GPU Memory** | < 4 GB | Inference mode |
| **Training Time** | < 12 hours | 100 epochs, 10K samples, 1x A100 |

### Optimization Gains

| Technique | Speedup | Source |
|-----------|---------|--------|
| **torch.compile** | 1.5x | PyTorch 2.8 |
| **Float8 training** | 1.5x | PyTorch 2.8 (A100/H100) |
| **Dynamic batching** | 3-5x | Uber production |
| **GATv2 (vs GAT)** | +9% accuracy | Papers 2024-2025 |
| **ARMA attention** | +9.1% forecast | ICLR 2025 |
| **Multi-task head** | +11.4% F1 | Microsoft 2022 |

---

## 🔧 Configuration

### Environment Variables

```bash
# Service
SERVICE_NAME=gnn-service
SERVICE_VERSION=2.0.0
LOG_LEVEL=INFO

# PyTorch
CUDA_VISIBLE_DEVICES=0
TORCH_COMPILE=true
FLOAT8_TRAINING=false  # Requires A100/H100

# Model
MODEL_PATH=models/checkpoints/best.ckpt
BATCH_SIZE=32
MAX_SEQUENCE_LENGTH=10

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/hydraulic_db
REDIS_URL=redis://localhost:6379/0

# Monitoring
PROMETHEUS_PORT=9090
ENABLE_METRICS=true
```

### SystemConfig (Pydantic)

See [src/schemas/metadata.py](src/schemas/metadata.py) для полной конфигурации.

---

## 📝 Development Notes

### Python 3.14 Features Used

✅ **Deferred Annotations (PEP 649)**
```python
from __future__ import annotations

class GraphTopology(BaseModel):
    components: Dict[str, ComponentSpec]  # Forward reference!
```

✅ **Union Types с Pipe Operator**
```python
def forward(x: torch.Tensor, edge_attr: torch.Tensor | None = None):
    # Instead of Optional[torch.Tensor]
    ...
```

### PyTorch 2.8 Features Used

✅ **torch.compile**
```python
model.forward = torch.compile(model.forward, mode="reduce-overhead")
```

✅ **torch.inference_mode**
```python
@torch.inference_mode()  # Faster than torch.no_grad()
def predict(self, x):
    return self(x)
```

✅ **Float8 Training (optional)**
```python
from torchao.float8 import convert_to_float8_training
model = convert_to_float8_training(model)  # 1.5x faster on A100/H100
```

---

## 🔗 Related Links

### Issues
- [Epic #92 - GNN Service Production Ready](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/92)
- [#93 - Core Schemas](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/93) ✅ COMPLETE
- [#94 - GNN Model Architecture](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/94) 🔄 IN PROGRESS
- [#95 - Dataset & DataLoader](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/95)
- [#96 - Inference Engine](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/96)

### Documentation
- [3-Week Roadmap](../../docs/GNN_SERVICE_ROADMAP.md)
- [Architecture Details](STRUCTURE.md)
- [API Documentation](http://localhost:8002/docs)

### References
- [GATv2 Paper](https://arxiv.org/abs/2105.14491) - "How Attentive are Graph Attention Networks?"
- [ARMA Attention (ICLR 2025)](https://openreview.net/forum?id=Z9N3J7j50k)
- [Multi-task Anomaly Detection (Microsoft)](https://arxiv.org/abs/2211.12141)
- [PyTorch 2.8 Release](https://pytorch.org/blog/pytorch-2-8/)
- [CUDA 12.9 Features](https://docs.nvidia.com/cuda/archive/12.9.0/)

---

## 🤝 Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

---

## 📧 Support

- **GitHub Issues:** [Create Issue](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/new)
- **Email:** shukik85@ya.ru
- **Documentation:** [docs/](../../docs/)

---

**Last Updated:** 2025-11-21 22:00 MSK  
**Status:** 🚧 Active Development (Phase 1: 25% → 50% complete)  
**Next Milestone:** Issue #94 Complete → Dataset Implementation (#95)