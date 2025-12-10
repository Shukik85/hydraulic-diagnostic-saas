# Universal GNN - Examples

Complete working examples for the Universal Temporal GNN inference pipeline.

## 🚀 Quick Start

### Prerequisites

```bash
# Navigate to GNN service
cd services/gnn_service

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### Run Example

```bash
# Run complete inference example
python -m examples.example_inference

# Or directly
python examples/example_inference.py
```

### Expected Output

```
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀
Universal GNN - Example Inference Pipeline
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀

Configuration:
  - Python: 3.14+
  - PyTorch: 2.4+
  - Time: 2025-12-10T22:00:00

================================================================================
TEST 1: Pump Equipment (5 sensors)
================================================================================
Building graph for pump_001...
Graph built successfully!
  - Nodes: 5 (expected: 5)
  - Node features: 34
  - Edges: 10
  - Edge features: 14 (edge_in_dim=14)
  - Validation: ✅ PASS

...

================================================================================
✅ ALL TESTS PASSED!
================================================================================
```

---

## 📋 What's Demonstrated

### Test 1: Pump Equipment (5 sensors)

**What it shows:**
- Building a graph for equipment with 5 sensors
- DynamicGraphBuilder reading from mock TimescaleDB
- Creating nodes and edges from topology definition
- Graph validation

**Key features:**
- Pump topology: 5 sensors in series with feedback loop
- Real sensor names (pump_1, pump_2, etc.)
- Edge features computed from sensor correlation

**Output:**
```
Nodes: 5 (one per sensor)
Edges: 10 (bidirectional connections)
Node features: 34D (standard feature vector)
Edge features: 14D (static + dynamic)
```

### Test 2: Compressor Equipment (7 sensors)

**What it shows:**
- Different equipment type with different sensor count
- Variable topology support
- DynamicGraphBuilder adapts to different topologies

**Key features:**
- Compressor topology: 7 sensors (more complex than pump)
- Demonstrates Universal GNN flexibility
- Same inference pipeline works for both

**Output:**
```
Nodes: 7 (different from pump!)
Edges: 14 (more connections)
Both use same 14D edge features
```

### Test 3: Variable Edge Dimensions

**What it shows:**
- Edge feature dimension flexibility
- 8D edges (static features only)
- 14D edges (static + dynamic)
- 20D edges (extended features)

**Key features:**
- All work with same DynamicGraphBuilder
- Graph validation confirms correct dimensions
- Production-ready for different edge feature sets

### Test 4: Batch Inference

**What it shows:**
- Building multiple graphs in sequence
- Handling mixed equipment types
- Variable-sized graphs in single batch

**Key features:**
- 3 pump graphs + 1 compressor graph
- Different node counts per graph
- All successfully created and validated

---

## 🔧 Code Structure

### Mock Components

**MockTimescaleConnector**
```python
connector = MockTimescaleConnector()
df = await connector.read_sensor_data(
    equipment_id="pump_001",
    lookback_minutes=10
)
```

Simulates reading from TimescaleDB:
- Different equipment types → different sensor counts
- Returns pandas DataFrame with sensor time series
- No real DB connection needed

**MockSensorData**
```python
generator = MockSensorData(
    equipment_id="pump_001",
    num_sensors=5,
    num_samples=100
)
df = generator.generate()
```

Generates realistic mock sensor data:
- Base sine wave (periodic signals)
- Random noise (realistic variations)
- Trends (degradation simulation)

### DynamicGraphBuilder

```python
builder = DynamicGraphBuilder(
    timescale_connector=connector,
    feature_engineer=engineer,
    feature_config=config
)

graph = await builder.build_from_timescale(
    equipment_id="pump_001",
    topology=topology,
    lookback_minutes=10
)
```

Key operations:
1. Reads sensor data from database
2. Creates node features from time series
3. Creates edges from topology
4. Computes edge features
5. Returns PyG Data object

### Topology Definition

```python
topology = GraphTopology(
    topology_id="pump_standard_v1",
    equipment_type="pump",
    sensor_ids=["pump_1", "pump_2", "pump_3", "pump_4", "pump_5"],
    connections=[
        {"from": "pump_1", "to": "pump_2", "type": "flow"},
        {"from": "pump_2", "to": "pump_3", "type": "flow"},
        # ... more connections
    ]
)
```

Defines:
- Equipment type
- Sensor list
- How sensors are connected
- Connection types (flow, feedback, etc.)

---

## 🎯 Adapting for Real Data

### Option 1: Replace MockTimescaleConnector

```python
from src.data import TimescaleConnector  # Real connector

connector = TimescaleConnector(
    db_url="postgresql://user:password@localhost/hydraulics"
)
await connector.connect()
```

The rest of the code stays the same!

### Option 2: Load from CSV

```python
df = pd.read_csv("sensor_data.csv", index_col="timestamp")

graph = builder._create_node_features(
    sensor_data=df,
    topology=topology,
    equipment_id="pump_001"
)
```

### Option 3: Use TemporalGraphDataset

```python
from src.data import TemporalGraphDataset

dataset = TemporalGraphDataset(
    data_path="data/gnn_graphs_multilabel.pt",
    feature_config=config
)

graph = dataset[0]  # Already preprocessed
```

---

## 🧪 Testing

### Run Tests

```bash
# Unit tests
python -m pytest tests/unit/test_edge_in_dim.py -v

# Integration tests (Phase 2)
python -m pytest tests/integration/test_dataloader_universal_gnn.py -v

# Integration tests (Phase 3)
python -m pytest tests/integration/test_inference_phase3.py -v
```

### Expected Results

```
✅ test_edge_in_dim.py::TestFeatureConfigEdgeDim - 6 tests
✅ test_edge_in_dim.py::TestGraphBuilderEdgeDim - 6 tests
✅ test_edge_in_dim.py::TestTemporalGraphDataset - 3 tests
✅ test_dataloader_universal_gnn.py::TestDataLoaderUniversalGNN - 6 tests
✅ test_inference_phase3.py::TestDynamicGraphBuilder - 5 tests

✅ Total: 26 tests PASSED
```

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
cd services/gnn_service
python -m examples.example_inference  # Use module path
```

### Issue: `ImportError: cannot import name 'DynamicGraphBuilder'`

**Solution:**
```bash
# Ensure you're on feature/gnn-service-production-ready branch
git checkout feature/gnn-service-production-ready

# Reinstall in development mode
pip install -e .
```

### Issue: Tests fail with `RuntimeError: CUDA out of memory`

**Solution:**
```python
# In example_inference.py, set device to CPU
config = FeatureConfig(edge_in_dim=14)
# (CPU is already used in mock tests)
```

---

## 📊 Performance Notes

**Mock Example (CPU):**
- Graph building: ~50ms per equipment
- Batch of 4: ~200ms
- Memory: ~100MB

**Real Data (GPU):**
- Graph building: ~10-20ms per equipment
- Batch of 32: ~500-1000ms
- Inference: ~5-10ms per graph

---

## 🎓 Key Learnings

This example demonstrates:

1. **Universal GNN is truly universal:**
   - ✅ Works with 5 sensors (pump)
   - ✅ Works with 7 sensors (compressor)
   - ✅ Works with arbitrary sensor counts

2. **Flexible edge dimensions:**
   - ✅ 8D edges (static only)
   - ✅ 14D edges (standard)
   - ✅ 20D edges (extended)

3. **Production-ready:**
   - ✅ Mock data for testing
   - ✅ Easy to adapt for real data
   - ✅ Proper error handling
   - ✅ Comprehensive logging

4. **Batch processing:**
   - ✅ Multiple equipment types
   - ✅ Variable-sized graphs
   - ✅ Efficient GPU utilization

---

## 📚 Next Steps

1. **Run the example locally** ✅
2. **Adapt for your data** (see "Adapting for Real Data")
3. **Run unit tests** to verify
4. **Run integration tests** to check full pipeline
5. **Deploy to production** with real TimescaleDB

---

**Questions?** Check the main [README.md](../README.md) or [UNIVERSAL_GNN_PROGRESS.md](../UNIVERSAL_GNN_PROGRESS.md)!
