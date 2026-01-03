# V2 Clean Architecture: Node-Centric GNN + Semisynthetic Features

**Date:** 2026-01-03  
**Status:** 🚨 Pre-Alpha  
**Objective:** Replace fragmented V1 with clean, production-ready pipeline  

---

## 퉪 Problem Statement

V1 architecture failed to deliver:

| Problem | Impact | V2 Solution |
|---------|--------|-------------|
| **Fragmented data pipeline** | Mapping, Parquet, Dataset loading disconnected | Single RawDataLoader → Cycles → Features |
| **Homogeneous graph** | All sensors treated equally, ignored physics | Node-centric: 17 sensors + 22 diagnostic edges |
| **Black-box features** | Raw timeseries only, no domain knowledge | 48 semisynthetic features per sensor |
| **Train-serving skew** | Inference used edge-centric graphs, training didn't | Both use node-centric topology |
| **Undocumented formats** | Multiple CSV/Parquet/PT transformations | Single canonical format: Cycles |
| **No feature importance** | Hard to debug model decisions | Physical meaning encoded in features |
| **Incorrect assumptions** | Docs claimed 9.1/10 quality, but code had fallbacks | Architecture matches reality |

---

## 🏛️ Architecture Overview

### Data Pipeline

```
╯────────────────────────────────────────────────╮
│ RAW UCI DATA (100 Hz, 17 sensors)                         │
│ services/gnn_service/data/raw_real_dataset/               │
│ │─ PS1.txt (46.9 MB)                                      │
│ │─ PS2.txt (82.4 MB)                                      │
│ │─ ... 15 more sensor files ...                           │
│ │─ documentation.txt (labels)                             │
╰────────────────────────────────────────────────╯
           ⬇️ RawDataLoader
╯────────────────────────────────────────────────╮
│ CYCLES (resampled to 10 Hz)                              │
│ 600 samples @ 10 Hz × 17 sensors                          │
│ 2600 total cycles from ~13 hours of data                  │
│                                                            │
│ Cycle(┋                                                   │
│   cycle_id=0,                                             │
│   data=(600, 17),                    # 60 sec @ 10 Hz    │
│   label=CycleLabel(cooler=0, ...),                        │
│   metadata={...}┋)                                        │
╰────────────────────────────────────────────────╯
           ⬇️ SemisyntheticFeatureEngineer
╯────────────────────────────────────────────────╮
│ NODE FEATURES (48 per sensor)                             │
│                                                            │
│ Per sensor:                                               │
│   Raw (3):         value, min10, max10                    │
│   Statistical (8): mean, std, skew, kurtosis, q1-q3, rng │
│   Temporal (4):    trend, ROC, accel, cyclicity           │
│   Frequency (8):   FFT @ 0.5-40 Hz                        │
│   Relational (6):  correlations (filled in dataset)       │
│   Operational (4): normalized states                      │
│   Diagnostic (12): outliers, drift, saturation, entropy   │
│   Energy (3):      power, work, efficiency                │
│                                                            │
│   Per 17 sensors: 17 × 48 = 816-dim feature space         │
╰────────────────────────────────────────────────╯
           ⬇️ NodeCentricTopology
╯────────────────────────────────────────────────╮
│ PyG DATA GRAPHS                                           │
│                                                            │
│ Nodes (17):                                               │
│   Pressure (6): PS1-PS6                                   │
│   Temperature (4): TS1-TS4                                │
│   Flow (2): FS1-FS2                                       │
│   Other (5): VS1, SE, CE, CP, EPS1                        │
│                                                            │
│ Edges (22): Diagnostic connections                        │
│   Pump → Motor: pressure/flow                             │
│   Motor → Cooler: heat                                     │
│   Accumulator ↔ Motor: energy                            │
│   Pressure + Temp: wear                                   │
│   Solenoid: control signals                               │
│   Energy + Power: efficiency                              │
│   Vibration + Pressure/Temp/Flow: bearing health          │
│                                                            │
│ Node features: (17, 48)                                   │
│ Edge index: (2, 22)                                       │
│ Labels: (4,) multi-label                                  │
╰────────────────────────────────────────────────╯
           ⬇️ NodeCentricGraphDataset
╯────────────────────────────────────────────────╮
│ PyTorch Dataset                                           │
│   Split: train (70%), val (15%), test (15%)              │
│   Feature normalization: per-split statistics             │
│   Relational feature filling: inter-sensor correlations   │
╰────────────────────────────────────────────────╯
           ⬇️ DataLoader
╯────────────────────────────────────────────────╮
│ Batched Training (GPU-ready)                             │
│   batch.x: (B×17, 48)                                     │
│   batch.edge_index: (2, B×22)                             │
│   batch.y: (B, 4)                                         │
│   batch.batch: (B×17,) graph indices                       │
╰────────────────────────────────────────────────╯
           ⬇️ GNN Model (next phase)
╯────────────────────────────────────────────────╮
│ GAT + LSTM + MLP                                        │
│   Input: Node features (17, 48)                           │
│   GNN layer: Graph Attention + LSTM                       │
│   Output: Multi-label classification (4,)                 │
╰────────────────────────────────────────────────╯
```

### Key Design Decisions

#### 1. **Node-Centric Topology (NOT Edge-Centric)**

**Why?** Sensors are physical entities with properties; edges are relationships.

```python
Nodes (17): Sensors = physical entities on hydraulic system
Edges (22): Diagnostic connections = relationships for fault propagation
```

Example edges:
- `PS1 → PS2`: Pump outlet → Motor inlet (pressure drop detects leaks)
- `PS1 → TS1`: Pump pressure → Pump temperature (wear correlation)
- `SE ↔ FS1`: Solenoid state ↔ Pump flow (control signal)

#### 2. **Semisynthetic Features (NOT Raw Time-Series)**

**Why?** Domain knowledge reduces overfitting and improves generalization.

```python
Raw     + Engineered = Semisynthetic
[600]   + [48 features] = [17 × 48] node features
```

Example:
- Raw: `PS1_value = 250 bar`
- Engineered: `PS1_mean = 220 bar`, `PS1_drift = +0.3 bar/min`, `PS1_outliers = 2%`
- Semisynthetic: All 48 features concatenated

#### 3. **Direct Pipeline (NOT Multiple Formats)**

**Why?** Eliminates intermediate format confusion and ensures consistency.

```
V1: raw .txt → CSV → Parquet → .pt → Dataset
     (mapping lost between formats)

V2: raw .txt → Cycles → Features → PyG Data
    (single source of truth)
```

---

## 💪 Why This Works Better

### ✅ Explainability
Every feature has physical meaning:
- `diag_below_min`: Sensor reading out of spec → Filter clogging?
- `diag_drift`: Long-term trend → Pump wear?
- `diag_entropy`: High variability → Cavitation?
- `relat_correlation`: Pressure-temperature link → Thermal coupling?

### ✅ Robustness
Semisynthetic features reduce noise:
- Raw pressure: 200, 202, 201, 250 (spike?) → outlier detection catches it
- Time-series: Hard to interpret jumps → Diagnostic features show abnormality

### ✅ Generalization
Domain knowledge improves transfer learning:
- Frequency features at pump/motor/bearing Hz catch system-specific patterns
- Pressure-temperature relationships encode hydraulic physics
- Can retrain on new equipment with same topology

### ✅ Debuggability
Node features are interpretable:
```python
# What's wrong with PS1 (pump pressure)?
graph.x[0, :48]  # All 48 features for PS1
# ✓ mean=250 (healthy)
# ✓ std=15 (normal variability)
# ❌ outliers=0.05 (5% spikes!)
# ❌ drift=+2.0 bar/min (increasing!)
# → Conclusion: Pump wearing out, producing spikes
```

---

## 💿 Engineering Decisions

### Memory Efficiency

```python
Cycle data: 600 samples × 17 sensors × 4 bytes = 40.8 KB per cycle
2600 cycles × 40.8 KB = ~106 MB in memory (compressed)

Node features: 17 sensors × 48 features × 4 bytes = 3.26 KB per cycle
X 2600 cycles = ~8.5 MB (vs 106 MB for raw data)

Compression ratio: ~12.5x
```

### Feature Computation Speed

```python
Semisynthetic features per cycle:
  - Load cycle: ~0.001 sec
  - Feature engineering (17 sensors): ~0.05 sec
  - Total: ~0.051 sec per cycle

2600 cycles: 0.051 × 2600 = ~133 seconds (~2.2 min)
```

### Scalability

- **Single machine:** 2600 cycles, batch_size=32 → ~156 batches/epoch
- **Multi-GPU:** Use DataLoader with num_workers=8, pin_memory=True
- **Distributed:** Can shard dataset by cycle_id

---

## 퉰b Validation Checklist

- [x] Raw data loads correctly (17 sensors, 100 Hz)
- [x] Resampling works (100 Hz → 10 Hz)
- [x] Cycles created (2600 × (600, 17) shape)
- [x] Feature engineering extracts 48 features
- [x] Node-centric topology has 17 nodes + 22 edges
- [x] PyG Data graphs created with correct shapes
- [x] DataLoader batches correctly
- [ ] Labels parsed from documentation.txt
- [ ] GNN model architecture defined
- [ ] Training loop implemented
- [ ] Inference endpoint created

---

## 🔨 Next Steps

### Phase 1: Label Parsing (1-2 days)
```python
# Read documentation.txt
# Map rows to cycles (timestamp-based)
# Validate label distribution
```

### Phase 2: GNN Model (3-5 days)
```python
# GAT + LSTM architecture
# Multi-label focal loss
# Hyperparameter tuning
```

### Phase 3: Production (2-3 days)
```python
# FastAPI inference server
# Redis graph caching
# Monitoring + observability
```

---

## 📚 References

- **Module docs:** `src/v2_clean/README.md`
- **Data source:** UCI Hydraulic System Condition Monitoring
- **Graph framework:** PyTorch Geometric
- **Model:** GAT (Veličković et al., 2017)

---

**Status:** Ready for testing. Run `python -m src.v2_clean.test_quick_start`
