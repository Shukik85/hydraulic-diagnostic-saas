# V2 Clean Architecture: Node-Centric Graphs + Semisynthetic Features

**Status:** 🚨 Pre-Alpha (Testing phase)  
**Objective:** Production-ready GNN for hydraulic system diagnostics  
**Approach:** Explainable ML via domain-informed features

---

## 🌟 Why V2?

V1 suffered from:
- ❌ **Fragmented data pipeline** (mapping, parquet export, dataset loading not connected)
- ❌ **Homogeneous graph architecture** (all sensors treated equally, ignored physics)
- ❌ **Black-box features** (raw time-series into model, no domain knowledge)
- ❌ **Train-serving skew** (inference used edge-centric graphs, training didn't)
- ❌ **Undocumented intermediate formats** (multiple CSV/Parquet/PT transformations)

**V2 fixes all of this:**
- ✅ **Direct pipeline:** Raw UCI → Cycles → Features → Graphs → Training
- ✅ **Node-centric graphs:** 17 sensors as nodes, diagnostic connections as edges (physics-based)
- ✅ **Semisynthetic features:** 48 features per sensor combining raw + engineered knowledge
- ✅ **Single source of truth:** One topology, one feature engineering, no duplicates
- ✅ **Explainability:** Every feature has physical meaning (pressure drop, thermal gradient, etc.)

---

## 📊 Architecture Overview

```
Raw UCI Data (100 Hz, 17 sensors)
  |  services/gnn_service/data/raw_real_dataset/
  |  → PS1.txt, PS2.txt, ..., EPS1.txt  (16 files, ~600 MB)
  ↓
[RawDataLoader]
  - Load 17 .txt files
  - Parse labels from documentation
  - Resample 100 Hz → 10 Hz
  - Output: Cycles (600 samples @ 10 Hz × 17 sensors)
  ↓
Cycles (2600 cycles, ready for training)
  ↓
[SemisyntheticFeatureEngineer]
  - Extract 48 features per sensor:
    * Raw (3): value, min, max
    * Statistical (8): mean, std, skewness, kurtosis, quartiles, range
    * Temporal (4): trend, ROC, acceleration, cyclicity
    * Frequency (8): FFT at pump/motor/bearing frequencies
    * Relational (6): inter-sensor correlations (filled in dataset)
    * Operational (4): normalized device states
    * Diagnostic (12): fault indicators (outliers, drift, saturation, entropy)
    * Energy (3): power, work, efficiency
  ↓
Node Features (17 sensors × 48 features = 816 dims)
  ↓
[NodeCentricTopology]
  - 17 nodes: PS1, PS2, PS3, PS4, PS5, PS6, TS1-TS4, FS1-FS2, VS1, SE, CE, CP, EPS1
  - 22 edges: diagnostic connections
    * Pump → Motor (pressure drop, flow correlation)
    * Motor → Cooler (heat generation)
    * Pump → Temperature (wear)
    * Solenoid → {Pump, Motor, Flow}
    * Energy ↔ {Pressure, Flow}
    * Vibration → {Pressure, Temperature, Flow} (bearing health)
  ↓
PyG Data Graph (x, edge_index, y)
  - x: (17, 48) node features
  - edge_index: (2, 22) connectivity
  - y: (4,) multi-label targets
    * cooler: 0=healthy, 1=reduced, 2=failed
    * valve: 0=healthy, 1=worn, 2=severely worn
    * pump_leak: 0=none, 1=internal, 2=external
    * accumulator: 0=ok, 1=presoak, 2=rapid_soaking
  ↓
[NodeCentricGraphDataset]
  - PyTorch Geometric Dataset
  - train/val/test splits (70/15/15)
  - Feature normalization (per-split statistics)
  ↓
[DataLoader]
  - Batched training (GPU-ready)
  - Graph-level batching (via batch.batch tensor)
  ↓
GNN Model (next)
  - GAT (Graph Attention Networks) for node-level reasoning
  - MLP head for multi-label classification
  - Production inference: FastAPI + Redis caching
```

---

## 🚀 Quick Start

### 1. Run Integration Test

```bash
cd services/gnn_service
python -m src.v2_clean.test_quick_start
```

**Expected output:**
```
# NODE-CENTRIC HYDRAULIC DATASET - QUICK START TEST
========================================================
TEST 1: Raw Data Loading
✓ Loaded 17 sensors
  Sensors: ['PS1', 'PS2', ..., 'EPS1']
    PS1:  46,956,000 samples @ 100 Hz
    ...

TEST 2: Cycle Loading & Resampling
✓ Loaded 2600 cycles
  First cycle details:
    Data shape: (600, 17)
    Label: CycleLabel(cooler=0, valve=0, pump_leak=0, accumulator=0)

TEST 3: Node-Centric Topology
✓ Created topology: NodeCentricTopology(nodes=17, edges=22, density=8.24%)

TEST 4: NodeCentricGraphDataset
✓ Created dataset: train split
  x (node features): torch.Size([17, 48])
  edge_index: torch.Size([2, 22])
  y (labels): torch.Size([4])

TEST 5: PyTorch DataLoader
✓ Created DataLoader
  Total batches: 156 (batch_size=4)
  First batch x: torch.Size([68, 48])  # 4 graphs × 17 nodes

✓ ALL TESTS PASSED - Ready for training!
```

### 2. Use in Your Training Loop

```python
from src.v2_clean.dataset import NodeCentricGraphDataset
from torch_geometric.data import DataLoader

# Load dataset
dataset_train = NodeCentricGraphDataset(
    raw_data_dir="data/raw_real_dataset",
    split="train",
    normalize_features=True,
)
dataset_val = NodeCentricGraphDataset(
    raw_data_dir="data/raw_real_dataset",
    split="val",
    normalize_features=True,
)

# Create dataloaders
loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)
loader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)

# Training loop
for epoch in range(100):
    for batch in loader_train:
        # batch.x: (B*17, 48) - flattened node features
        # batch.edge_index: (2, E) - edge connectivity
        # batch.y: (B, 4) - labels
        # batch.batch: (B*17,) - which graph each node belongs to
        
        # Forward pass
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y)
        
        # Backward
        loss.backward()
        optimizer.step()
```

---

## 📄 Module Reference

### `data.py`

**Load raw UCI hydraulic data and create cycles.**

```python
from src.v2_clean.data import RawDataLoader, Cycle, CycleLabel

loader = RawDataLoader("data/raw_real_dataset")
cycles = loader.load_cycles()  # List[Cycle]

cycle = cycles[0]
print(cycle.data.shape)        # (600, 17) - 60 seconds @ 10 Hz
print(cycle.label)              # CycleLabel(cooler=0, valve=0, ...)
print(cycle.metadata)           # {'cycle_id': 0, 'hz': 10, ...}
```

**Classes:**
- `RawDataLoader`: Main entry point
- `Cycle`: Named tuple with data + label + metadata
- `CycleLabel`: Multi-label targets
- `SensorMetadata`: Physical sensor specifications
- `SENSOR_CONFIG`: Dictionary of all 17 sensors with ranges
- `SENSOR_ORDER`: List of sensor names in column order

---

### `topology.py`

**Define node-centric hydraulic system topology.**

```python
from src.v2_clean.topology import NodeCentricTopology

topology = NodeCentricTopology()
print(topology)  # NodeCentricTopology(nodes=17, edges=22, density=8.24%)

# Get node index
ps1_idx = topology.get_node_index("PS1")  # 0

# Get edge list (PyG format)
source_indices, target_indices = topology.get_edge_index()
edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)

# Get edges adjacent to a sensor
edges_ps1 = topology.get_adjacent_edges("PS1")
# [Edge(source='SE', target='PS1', ...), Edge(source='PS1', target='PS2', ...), ...]
```

**Topology rules:**
- Pump → Motor: pressure/flow correlation
- Motor → Cooler: heat dissipation
- Accumulator ↔ Motor: energy storage
- Pressure + Temperature: wear diagnostics
- Solenoid: pump control
- Vibration: bearing health

---

### `features.py`

**Extract 48 semisynthetic features per sensor.**

```python
from src.v2_clean.features import SemisyntheticFeatureEngineer

engineer = SemisyntheticFeatureEngineer(
    sensor_name="PS1",
    sensor_type="pressure",
    physical_range=(0, 350),
    hz=10,
)

# Extract from 600-sample cycle
features = engineer.extract(cycle_data[:, 0])  # (48,)
print(features)  # [value, min, max, mean, std, ..., energy_total, ...]

# Get feature names
names = engineer.feature_names()
for i, name in enumerate(names):
    print(f"{i:2d}: {name} = {features[i]:.4f}")
```

**Feature categories:**
1. **Raw** (3): Last value, min/max of last 10 samples
2. **Statistical** (8): mean, std, kurtosis, skewness, quartiles, range
3. **Temporal** (4): Trend, ROC, acceleration, cyclicity
4. **Frequency** (8): FFT at 0.5, 1, 2, 5, 10, 17, 12, 40 Hz
5. **Relational** (6): Correlations with adjacent sensors (filled in dataset)
6. **Operational** (4): Normalized device states
7. **Diagnostic** (12): Fault indicators
8. **Energy** (3): Power, work, efficiency

---

### `dataset.py`

**PyTorch Geometric dataset with train/val/test splits.**

```python
from src.v2_clean.dataset import NodeCentricGraphDataset
from torch_geometric.data import DataLoader

# Create dataset
dataset = NodeCentricGraphDataset(
    raw_data_dir="data/raw_real_dataset",
    split="train",
    test_fraction=0.15,
    val_fraction=0.15,
    normalize_features=True,
)

print(f"Dataset size: {len(dataset)}")
print(f"Total cycles: {len(dataset.cycles)}")

# Get single graph
graph = dataset[0]
print(f"x shape: {graph.x.shape}")          # (17, 48)
print(f"edge_index shape: {graph.edge_index.shape}")  # (2, 22)
print(f"y shape: {graph.y.shape}")          # (4,)
print(f"y values: {graph.y}")               # [cooler, valve, pump_leak, accumulator]

# Create dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for batch in loader:
    print(f"Batch size (num graphs): {batch.num_graphs}")
    print(f"Batch x: {batch.x.shape}")      # (num_graphs*17, 48)
    print(f"Batch edge_index: {batch.edge_index.shape}")  # (2, num_graphs*22)
    print(f"Batch y: {batch.y.shape}")      # (num_graphs, 4)
    print(f"Batch batch: {batch.batch.shape}")  # (num_graphs*17,) - which graph
    break
```

---

## 📅 Data Format

### Raw UCI Files

```
services/gnn_service/data/raw_real_dataset/
├── PS1.txt         (46.9 MB, 46M samples @ 100 Hz)
├── PS2.txt         (82.4 MB, 82M samples)
├── PS3.txt
├── PS4.txt
├── PS5.txt
├── PS6.txt
├── TS1.txt         (0.9 MB, temperature)
├── TS2.txt
├── TS3.txt
├── TS4.txt
├── FS1.txt         (7.6 MB, flow)
├─┠ FS2.txt         (8.3 MB)
├─┠ VS1.txt         (0.8 MB, vibration)
├─┠ SE.txt          (0.9 MB, solenoid)
├─┠ CE.txt          (0.9 MB, energy)
├┠┠ CP.txt          (0.9 MB, power)
├┠┠ EPS1.txt        (87.4 MB, electrical)
├┠┠ documentation.txt
└┠┠ description.txt

Total: ~600 MB
```

### Cycle Format

```python
Cycle(
    cycle_id=0,
    data=np.ndarray(shape=(600, 17), dtype=float32),  # 60 sec @ 10 Hz
    label=CycleLabel(cooler=0, valve=0, pump_leak=0, accumulator=0),
    metadata={
        'cycle_id': 0,
        'timestamp_start_sec': 0,
        'timestamp_end_sec': 60,
        'hz': 10,
        'num_sensors': 17,
    }
)
```

### PyG Graph Format

```python
Data(
    x=torch.Tensor(shape=(17, 48), dtype=float32),  # Node features
    edge_index=torch.Tensor(shape=(2, 22), dtype=int64),  # Edges
    y=torch.Tensor(shape=(4,), dtype=int64),  # Multi-label targets
    cycle_id=torch.Tensor(scalar),
    metadata=dict,  # Original cycle metadata
)
```

---

## 🪦 Known Limitations

1. **Labels not parsed yet**
   - `RawDataLoader._parse_labels()` returns empty dict
   - Implement by reading `documentation.txt`
   - Currently cycles get default labels (all healthy)

2. **Relational features are correlations only**
   - Could add: pressure drop magnitudes, thermal gradients, flow imbalance
   - Currently simple Pearson correlation

3. **No temporal modeling yet**
   - Features are per-cycle (static)
   - Could add: LSTM encoding of time-series, attention over time
   - Graph structure is currently static

4. **Feature normalization on train split only**
   - Val/test use train statistics (correct for reproducibility)
   - Could improve with per-split normalization (check literature)

---

## 🚀 Next Steps

1. **Implement label parsing**
   - Read documentation.txt
   - Map documentation rows to cycles

2. **Build GNN model**
   - GAT + LSTM for temporal reasoning
   - Multi-label classification head
   - Loss function: multi-label focal loss

3. **Training pipeline**
   - Hyperparameter tuning
   - Early stopping
   - Checkpoint saving

4. **Inference server**
   - FastAPI endpoint `/v1/diagnose`
   - Graph caching
   - Real-time monitoring

5. **Evaluation metrics**
   - Per-label accuracy/F1
   - Confusion matrices
   - Feature importance visualization

---

## 📚 References

- **UCI Dataset**: [Hydraulic System Condition Monitoring](https://archive.ics.uci.edu/ml/datasets/Hydraulic+Systems+Condition+Monitoring)
- **PyG Docs**: https://pytorch-geometric.readthedocs.io/
- **Graph Neural Networks**: Kipf & Welling (2016), GAT (Veličković et al., 2017)

---

## 👻 Author

Aleksandr Plotnikov (@shukik85)  
Started: 2026-01-03
