# Hydraulic Diagnostics GNN - ML System Design

**Version:** 2.0  
**Last Updated:** 2025-11-29  
**Status:** Production-Ready Architecture  

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Data Schema](#2-data-schema)
3. [Model Architecture](#3-model-architecture)
4. [Training Strategy](#4-training-strategy)
5. [Production Pipeline](#5-production-pipeline)
6. [Roadmap](#6-roadmap)
7. [References](#7-references)

---

## 1. Problem Statement

### 1.1 Overview

Hydraulic systems in mobile and industrial equipment require continuous monitoring and predictive maintenance to prevent failures and optimize operations. Our ML system addresses:

1. **Multi-level fault detection** - Detect anomalies at system, component, and connection levels
2. **RUL prediction** - Estimate Remaining Useful Life for maintenance planning
3. **Component diagnostics** - Identify which specific components are degrading
4. **Connection health** - Monitor wear, leakage, and blockage in hydraulic lines

### 1.2 Business Impact

- **Reduce downtime** - Predict failures before they occur
- **Optimize maintenance** - Schedule service based on actual component condition
- **Extend equipment life** - Identify and address issues early
- **Cost savings** - Prevent catastrophic failures and unnecessary servicing

### 1.3 Technical Challenges

- **Graph-structured data** - Hydraulic systems are networks of interconnected components
- **Temporal dynamics** - System behavior evolves over time
- **Multi-task learning** - Predict multiple correlated outputs simultaneously
- **Class imbalance** - Normal operation dominates, anomalies are rare
- **Real-time inference** - Production requirements demand low latency (<500ms)

---

## 2. Data Schema

### 2.1 Graph Representation

**Nodes (Components):**
- One node = one hydraulic component (pump, valve, cylinder, filter, etc.)
- Total nodes per graph: Variable (N), typically 10-50 components

**Edges (Connections):**
- One edge = one physical connection (hose, pipe, tube)
- Directed graph representing fluid flow
- Total edges per graph: Variable (E), typically 15-80 connections

**Batch Structure:**
- Multiple equipment graphs batched together
- Batch size: B (typically 16-32)

### 2.2 Feature Engineering

#### Node Features [N, 34]

Extracted from raw sensor time-series data:

**Statistical Features (11):**
```
- mean, std, min, max, median
- percentiles: 5th, 25th, 50th, 75th, 95th
- skewness, kurtosis
```

**Frequency Features (12):**
```
- top 10 FFT magnitudes
- dominant frequency
- spectral entropy
```

**Temporal Features (11):**
```
- rolling mean/std (windows: 5, 10, 30)
- exponential moving average
- autocorrelation (lags: 1, 5, 10)
- linear trend slope
```

**Total:** 34 features per component

#### Edge Features [E, 14]

**Static Features (8):**
```
1. diameter_norm         # Normalized hose diameter
2. length_norm           # Normalized connection length
3. cross_section_area    # Flow area
4. pressure_loss_coeff   # Pressure drop coefficient
5. pressure_rating_norm  # Rated pressure (normalized)
6. material_steel        # One-hot: steel
7. material_rubber       # One-hot: rubber
8. material_composite    # One-hot: composite
```

**Dynamic Features (6) - Phase 2:**
```
9. flow_rate            # Current flow through connection
10. pressure_diff       # Pressure drop across connection
11. temperature_diff    # Temperature delta
12. vibration_level     # Connection vibration
13. age_hours           # Operating hours
14. maintenance_counter # Service count
```

### 2.3 Targets (Multi-Task)

#### Graph-Level Targets

Predictions for entire equipment:

```python
# 1. Health Score (Regression)
health_score: Tensor[B, 1]  # ∈ [0, 1]
  # 1.0 = perfect health
  # 0.0 = critical condition
  # Loss: WingLoss (robust regression)
  # Metric: MAE, RMSE, R²

# 2. Degradation Rate (Regression)
degradation_rate: Tensor[B, 1]  # ∈ [0, 1]
  # 0.0 = no degradation
  # 1.0 = rapid degradation
  # Loss: WingLoss
  # Metric: MAE, RMSE

# 3. RUL - Remaining Useful Life (Regression) - NEW
rul_hours: Tensor[B, 1]  # ∈ [0, ∞)
  # Hours until predicted failure
  # Loss: QuantileLoss (asymmetric penalties)
  # Metric: MAE, RMSE, prediction horizon accuracy

# 4. Anomaly Detection (Multi-Label Classification)
anomaly_flags: Tensor[B, 9]  # ∈ {0, 1}^9
  # 9 anomaly types (see below)
  # Loss: FocalLoss (handles class imbalance)
  # Metric: F1, Precision, Recall, AUC-ROC per class
```

**9 Anomaly Types:**
1. `pressure_drop` - Unexpected pressure loss
2. `overheating` - Temperature above normal
3. `cavitation` - Bubble formation in fluid
4. `leakage` - Fluid escaping system
5. `vibration_anomaly` - Abnormal vibration patterns
6. `flow_restriction` - Reduced flow rate
7. `contamination` - Fluid contamination detected
8. `seal_degradation` - Seal wear indicators
9. `valve_stiction` - Valve sticking/friction

#### Component-Level Targets - NEW

Predictions for each component:

```python
# 1. Component Health (Regression)
component_health: Tensor[N, 1]  # ∈ [0, 1]
  # Health score per component
  # Enables fault localization
  # Loss: WingLoss
  # Metric: MAE, RMSE

# 2. Component Anomaly (Multi-Label Classification)
component_anomaly: Tensor[N, 9]  # ∈ {0, 1}^9
  # Same 9 anomaly types, per component
  # Identifies which component has which anomaly
  # Loss: FocalLoss
  # Metric: F1, Precision, Recall per class
```

#### Edge-Level Targets - Phase 2

Predictions for connections:

```python
# 1. Edge Wear (Regression)
edge_wear: Tensor[E, 1]  # ∈ [0, 1]
  # Hose/pipe wear prediction

# 2. Edge Leakage (Regression)
edge_leakage: Tensor[E, 1]  # ∈ [0, 1]
  # Leakage probability

# 3. Edge Blockage (Regression)
edge_blockage: Tensor[E, 1]  # ∈ [0, 1]
  # Flow restriction indicator
```

---

## 3. Model Architecture

### 3.1 Overview

**UniversalTemporalGNN** combines spatial and temporal modeling:

```
Input → GATv2 (Spatial) → Multi-Level Heads
           ↓
        Pooling
           ↓
      LSTM (Temporal)
           ↓
     Graph-Level Heads
```

### 3.2 Spatial Module: GATv2

**Graph Attention Networks v2** for spatial feature learning:

```python
GATv2 Architecture:
  - Num layers: 3
  - Attention heads: 8
  - Hidden channels: 128
  - Edge-conditioned attention
  - Residual connections
  - Graph normalization

Features:
  - Dynamic attention weights
  - Edge feature integration
  - Captures component interactions
  - Learns spatial patterns
```

**Formula:**
```
h_i^(l+1) = σ(Σ_j∈N(i) α_ij W h_j^(l))

α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j || W e_ij]))
```

Where:
- `h_i` = node features
- `e_ij` = edge features
- `α_ij` = attention weights
- `W` = learnable weights

### 3.3 Temporal Module: ARMA-LSTM

**Autoregressive Moving Average + LSTM** for temporal dynamics:

```python
ARMA-LSTM Architecture:
  - LSTM hidden: 256
  - LSTM layers: 2
  - Dropout: 0.1
  - Bidirectional: No (causal)

Features:
  - Captures temporal evolution
  - Long-term dependencies
  - Sequence modeling
  - Autoregressive patterns
```

### 3.4 Multi-Level Prediction Heads

#### Component-Level Heads

Applied **after GATv2, before pooling**:

```python
# Component Health Head
component_health = nn.Sequential(
    nn.Linear(hidden_channels, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 1),
    nn.Sigmoid()  # [N, 1] ∈ [0, 1]
)

# Component Anomaly Head
component_anomaly = nn.Sequential(
    nn.Linear(hidden_channels, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 9)  # [N, 9] logits
)
```

#### Graph-Level Heads

Applied **after LSTM**:

```python
# Health Head
health_head = nn.Sequential(
    nn.Linear(lstm_hidden, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 1),
    nn.Sigmoid()  # [B, 1] ∈ [0, 1]
)

# Degradation Head
degradation_head = nn.Sequential(
    nn.Linear(lstm_hidden, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 1),
    nn.Sigmoid()  # [B, 1] ∈ [0, 1]
)

# RUL Head - NEW
rul_head = nn.Sequential(
    nn.Linear(lstm_hidden, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 1),
    nn.Softplus()  # [B, 1] ∈ [0, ∞)
)

# Anomaly Head
anomaly_head = nn.Sequential(
    nn.Linear(lstm_hidden, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 9)  # [B, 9] logits
)
```

### 3.5 Forward Pass

```python
def forward(self, x, edge_index, edge_attr, batch):
    # === Spatial Encoding ===
    h = self.initial_projection(x)  # [N, F] → [N, hidden]
    
    for gat_layer in self.gat_layers:
        h = gat_layer(h, edge_index, edge_attr)  # [N, hidden]
    
    # === Component-Level Predictions ===
    component_health = self.component_health(h)      # [N, 1]
    component_anomaly = self.component_anomaly(h)    # [N, 9]
    
    # === Global Pooling ===
    graph_repr = global_mean_pool(h, batch)          # [B, hidden]
    
    # === Temporal Modeling ===
    lstm_out = self.lstm(graph_repr.unsqueeze(1))    # [B, 1, lstm_hidden]
    lstm_out = lstm_out.squeeze(1)                   # [B, lstm_hidden]
    
    # === Graph-Level Predictions ===
    graph_health = self.health_head(lstm_out)        # [B, 1]
    graph_degradation = self.degradation_head(lstm_out)  # [B, 1]
    graph_anomaly = self.anomaly_head(lstm_out)      # [B, 9]
    graph_rul = self.rul_head(lstm_out)              # [B, 1]
    
    return {
        'component': {
            'health': component_health,
            'anomaly': component_anomaly
        },
        'graph': {
            'health': graph_health,
            'degradation': graph_degradation,
            'anomaly': graph_anomaly,
            'rul': graph_rul
        }
    }
```

### 3.6 Model Statistics

```
Total Parameters: ~2.5M
Input Shape: Variable (graph-dependent)
Output Shape: Nested dict (multi-level)
Memory: ~200MB (GPU)
Inference Time: ~50ms (single graph, GPU)
```

---

## 4. Training Strategy

### 4.1 Loss Functions

#### WingLoss (Robust Regression)

For health and degradation predictions:

```python
class WingLoss(nn.Module):
    """Combines L1 and L2 losses with smooth transition.
    
    More robust to outliers than MSE.
    Less biased than MAE.
    
    Args:
        omega: Transition threshold (default: 10.0)
        epsilon: Smoothness parameter (default: 2.0)
    """
```

**Benefits:**
- Robust to sensor noise
- Smooth gradients
- Better convergence

#### QuantileLoss (Asymmetric RUL)

For RUL prediction:

```python
class QuantileRULLoss(nn.Module):
    """Quantile loss with asymmetric penalties.
    
    Penalizes underestimation (predict too early) more than
    overestimation (predict too late).
    
    Args:
        quantiles: List of quantiles (default: [0.1, 0.5, 0.9])
    
    References:
        - https://openreview.net/forum?id=tzFjcVqmxw
    """
```

**Benefits:**
- Conservative predictions (safer)
- Uncertainty quantification
- Better for maintenance planning

#### FocalLoss (Class Imbalance)

For anomaly detection:

```python
class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.
    
    Down-weights easy examples, focuses on hard cases.
    
    Args:
        alpha: Class weight (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
    """
```

**Benefits:**
- Handles rare anomalies
- Focuses on hard examples
- Better than BCE for imbalanced data

#### UncertaintyWeighting (Multi-Task)

For combining losses:

```python
class UncertaintyWeighting(nn.Module):
    """Learn task weights based on homoscedastic uncertainty.
    
    Automatically balances tasks without manual tuning.
    
    References:
        - Multi-Task Learning Using Uncertainty
          https://arxiv.org/abs/1705.07115
    """
```

**Benefits:**
- Automatic balancing
- No manual weight tuning
- Adapts during training

### 4.2 Metrics

#### Regression Metrics

```python
Health & Degradation:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)

RUL:
  - MAE
  - RMSE
  - Prediction Horizon Accuracy
  - Quantile Coverage
```

#### Classification Metrics

```python
Anomaly Detection:
  - F1 Score (per class + macro/micro)
  - Precision (per class)
  - Recall (per class)
  - AUC-ROC (per class + macro)
```

### 4.3 Optimization

```python
Optimizer: Adam
  - Learning rate: 0.001
  - Weight decay: 1e-5
  - Betas: (0.9, 0.999)

Scheduler: ReduceLROnPlateau
  - Mode: min (validation loss)
  - Factor: 0.5
  - Patience: 10 epochs
  - Verbose: True

Alternative: CosineAnnealingLR
  - T_max: 100 epochs
  - eta_min: 1e-6
```

### 4.4 Training Configuration

```python
Batch size: 32
Epochs: 100 (with early stopping)
Gradient clipping: 1.0
Mixed precision: float16 (AMP)
Device: CUDA (auto-detect)

Callbacks:
  - ModelCheckpoint (save best)
  - EarlyStopping (patience: 15)
  - LearningRateMonitor
```

---

## 5. Production Pipeline

### 5.1 FastAPI Inference Service

```python
Endpoints:
  - POST /predict         # Single prediction
  - POST /predict/batch   # Batch predictions
  - GET /health           # Basic health check
  - GET /health/detailed  # Detailed metrics
  - GET /stats            # Service statistics
  - GET /models/versions  # Available models
  - GET /models/current   # Active model info
```

### 5.2 Model Management

```python
Versioning:
  - Checkpoint-based versioning
  - Semantic versioning (v2.0.0)
  - Metadata tracking
  - Rollback support

Deployment:
  - Docker containers
  - Kubernetes orchestration
  - Auto-scaling
  - Load balancing
```

### 5.3 Performance Requirements

```
Latency:
  - p50: <200ms
  - p95: <500ms
  - p99: <1000ms

Throughput:
  - >100 req/s per GPU
  - >50 req/s per CPU

Availability:
  - 99.9% uptime
  - Graceful degradation
```

---

## 6. Roadmap

### Phase 1: Core Extensions (Current)

**Issue #116: Multi-Level Predictions** (3-4h)
- [x] QuantileRULLoss implementation
- [x] Component-level prediction heads
- [x] Multi-level output structure
- [x] Training integration
- [x] Tests

**Issue #115: Training Pipeline** (2-3h remaining)
- [x] LightningModule ✅
- [x] Advanced losses ✅
- [x] Multi-task integration ✅
- [ ] Metrics implementation
- [ ] Trainer factory
- [ ] Documentation

### Phase 2: Edge Intelligence (Week 2)

**Issue #118: Dynamic Edge Features**
- Add flow_rate, pressure_diff, temperature_diff
- Add vibration_level, age_hours, maintenance_counter
- Update FeatureEngineer
- Update GraphBuilder

**Issue #119: Edge-Level Predictions**
- EdgePredictor module
- Edge wear/leakage/blockage heads
- Edge-level loss functions
- Integration & tests

### Phase 3: Advanced Features (Week 3-4)

**Issue #120: Hierarchical Graph Structure**
- Component-level graph (current)
- System-level graph (new)
- Hierarchical pooling
- Multi-scale predictions

---

## 7. References

### Research Papers

1. **Spatio-Temporal GNN for Hydraulic Systems**
   - https://www.sciencedirect.com/science/article/abs/pii/S095219762300982X
   - Multi-level diagnostics in complex systems

2. **Multi-Task ST-GNN for Predictive Maintenance**
   - https://arxiv.org/pdf/2401.15964.pdf
   - Joint RUL and health prediction

3. **GNN for Water Distribution Networks**
   - https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR036741
   - Node-level predictions in hydraulic networks

4. **Quantile Loss for RUL Prediction**
   - https://openreview.net/forum?id=tzFjcVqmxw
   - Asymmetric loss functions for maintenance

5. **Component-Level Fault Localization**
   - https://arxiv.org/abs/2404.10324
   - Per-node predictions in GNN

6. **Industrial PdM with GNN**
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC11125296/
   - Production deployment strategies

7. **Edge-Level Predictions in Hydraulic Systems**
   - https://ijarsct.co.in/Paper19379.pdf
   - Leakage and wear detection

### Frameworks & Tools

- **PyTorch 2.8** - https://pytorch.org/
- **PyTorch Geometric** - https://pytorch-geometric.readthedocs.io/
- **PyTorch Lightning** - https://lightning.ai/docs/pytorch/
- **FastAPI** - https://fastapi.tiangolo.com/

---

**Document Version:** 2.0  
**Last Updated:** 2025-11-29  
**Next Review:** 2025-12-15
