# GNN Service - Production-Ready Implementation

🎉 **Status:** Phase 3 COMPLETE (Production Ready)  
🔗 **Branch:** `feature/gnn-service-production-ready`  
📅 **Updated:** 2025-12-04  
🎯 **Epic Issue:** [#92](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/92)

---

## 🚀 Overview

Production-ready Graph Neural Network service для диагностики гидравлических систем с использованием **Universal Temporal GNN** (GATv2 + ARMA-LSTM).

### Version 2.0.1 - NEW! 🎆

**Universal GNN Update (04.12.2025):**
- ✅ **Edge Feature Flexibility** - поддержка произвольной размерности edge-фич (edge_in_dim)
- ✅ **Variable Graph Topology** - работа с графами разного размера (N, E)
- ✅ **Edge Projection Layer** - нелинейное преобразование edge-фич
- ✅ **MODEL_CONTRACT.md** - полная спецификация входов/выходов
- ✅ **Backward Compatible** - старые чекпоинты работают (edge_in_dim=8)

🔗 **Tracking Issue:** [#124 Universal GNN Variable Topology](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/124)

**Phase 3 Completed (03.12.2025):**
- ✅ **Dynamic Edge Features** (14D) - Physics-based flow estimation
- ✅ **API v2** - Simplified inference endpoints
- ✅ **Topology Management** - Pre-configured templates
- ✅ **Backward Compatible** - v1 API still works
- ✅ **Production Ready** - <200ms inference, 85%+ test coverage

### Technology Stack (Updated 2025-12-04)

- 🐍 **Python 3.14.0** - Deferred annotations (PEP 649), union types
- ⚡ **PyTorch 2.8.0** - Float8 training, torch.compile
- 🖥️ **CUDA 12.9** - GPU optimization
- 🧠 **PyTorch Geometric 2.6+** - GNN operations
- 🚀 **FastAPI 0.109+** - Async API (NEW: v2 endpoints)
- ✅ **Pydantic v2.6+** - Data validation
- 📊 **TimescaleDB** - Time-series data
- 🔄 **Redis** - Caching

### Key Features

#### Core GNN
- ✅ **GATv2 Architecture** - Dynamic attention [+9-10% accuracy]
- 🔥 **ARMA-LSTM** - Temporal attention (ICLR 2025) [+9.1% forecasting]
- 🎯 **14D Edge Features** - 8 static + 6 dynamic (Phase 3)
- 🔄 **Universal Topology** - Variable N, E, B (v2.0.1)
- 🧠 **Multi-Task Learning** - Health, degradation, 9 anomalies
- ⚡ **torch.compile** - 1.5x speedup

#### Phase 3 (v2.0.0)
- 🔬 **Physics-Based Flow** - Darcy-Weisbach equation
- 📊 **Mixed Normalization** - Per-feature strategy
- 🏭 **Topology Templates** - Pre-configured systems
- 🌐 **API v2** - Minimal inference (4 fields)
- ⏱️ **Sub-200ms Inference** - Production SLA
- 🧪 **165+ Tests** - 85%+ coverage

#### Universal GNN (v2.0.1)
- 🔄 **Edge Feature Flexibility** - Arbitrary edge_in_dim
- 🏭 **Variable Graph Size** - N, E invariant architecture
- 🧱 **Edge Projection** - Learnable edge transformation
- 📝 **MODEL_CONTRACT.md** - Complete I/O specification
- ✅ **Backward Compatible** - Works with v2.0.0 checkpoints

---

## 📊 Architecture

### High-Level Overview

```
Client Request (v2 API)
         ↓
   FastAPI main.py
         ↓
TopologyService → Load template
         ↓
 InferenceEngine
         ├→ EdgeFeatureComputer (compute dynamic features)
         ├→ EdgeFeatureNormalizer (normalize)
         └→ GraphBuilder (build PyG graph)
         ↓
UniversalTemporalGNN (14D edges, variable N/E)
         ├→ Edge Projection (edge_in_dim → edge_hidden) [v2.0.1]
         ├→ GATv2 layers (spatial)
         ├→ ARMA-LSTM (temporal)
         └→ Multi-task heads
         ↓
PredictionResponse (health, degradation, anomaly)
```

### Universal Temporal GNN Architecture (v2.0.1)

```python
UniversalTemporalGNN(
    in_channels=34,      # Node feature dimension (fixed per config)
    hidden_channels=128,
    edge_in_dim=14,      # NEW: Edge feature dimension (configurable!)
    num_heads=8,
    num_gat_layers=3,
    lstm_hidden=256,
    lstm_layers=2
)
```

**Universal Properties:**
1. **Node Count (N):** Работает с графами от 1 до N узлов
2. **Edge Count (E):** Произвольное число рёбер
3. **Batch Size (B):** Любое количество графов в батче
4. **Edge Features (edge_in_dim):** Произвольная размерность (8D, 14D, 20D, etc.)

**Example: Variable Graph Sizes**

```python
import torch
from torch_geometric.data import Data, Batch

# Small system: 20 nodes, 30 edges
small_graph = Data(
    x=torch.randn(20, 34),
    edge_index=torch.randint(0, 20, (2, 30)),
    edge_attr=torch.randn(30, 14)
)

# Large system: 150 nodes, 300 edges
large_graph = Data(
    x=torch.randn(150, 34),
    edge_index=torch.randint(0, 150, (2, 300)),
    edge_attr=torch.randn(300, 14)
)

# Batch different sizes together
batch = Batch.from_data_list([small_graph, large_graph])

# Model processes both seamlessly
output = model(
    x=batch.x,              # [170, 34] (20+150)
    edge_index=batch.edge_index,  # [2, 330] (30+300)
    edge_attr=batch.edge_attr,    # [330, 14]
    batch=batch.batch       # [170] (indices 0,1)
)

print(output['graph']['health'].shape)  # [2, 1] - 2 graphs
print(output['component']['health'].shape)  # [170, 1] - 170 nodes
```

🔗 **Full Specification:** [docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md)

### Phase 3 Components (v2.0.0)

#### 1. EdgeFeatureComputer

**Physics-based dynamic feature computation:**

```python
from src.data.edge_features import EdgeFeatureComputer

computer = EdgeFeatureComputer()

features = computer.compute_edge_features(
    edge=edge_spec,
    sensor_readings={
        "pump_1": ComponentSensorReading(pressure_bar=150.0, ...),
        "valve_1": ComponentSensorReading(pressure_bar=148.0, ...)
    },
    current_time=datetime.now()
)

# Returns:
# {
#     "flow_rate_lpm": 115.3,       # Darcy-Weisbach
#     "pressure_drop_bar": 2.0,
#     "temperature_delta_c": 1.0,
#     "vibration_level_g": 0.8,
#     "age_hours": 12000.0,
#     "maintenance_score": 0.85
# }
```

#### 2. EdgeFeatureNormalizer

**Mixed normalization strategy:**

```python
from src.data.normalization import EdgeFeatureNormalizer

normalizer = EdgeFeatureNormalizer()

# Fit on training data
normalizer.fit(training_features)

# Normalize
normalized = normalizer.normalize(features)

# Save/load stats
stats = normalizer.get_stats()
normalizer.load_stats(stats)
```

**Strategies:**
- Flow: log + z-score (right-skewed)
- Pressure/Temp: z-score (negative OK)
- Vibration/Age: min-max [0, 1]
- Maintenance: pass-through

#### 3. TopologyService

**Template management:**

```python
from src.services.topology_service import TopologyService

service = TopologyService.get_instance()

# List templates
templates = service.list_templates()

# Get template
template = service.get_template("standard_pump_system")
topology = template.to_graph_topology("equipment_001")

# Validate custom
is_valid, errors = service.validate_topology(custom_topology)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas
cd hydraulic-diagnostic-saas/services/gnn_service

# Create virtual environment
python3.14 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Service

```bash
# Set environment variables
export MODEL_PATH="models/v2.0.1.ckpt"
export DEVICE="cuda"  # or "cpu"
export BATCH_SIZE="32"

# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# API docs available at:
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

### API Usage (v2 - Simplified)

```python
import requests
from datetime import datetime

# Minimal inference (simplest API)
response = requests.post(
    "http://localhost:8000/api/v2/inference/minimal",
    json={
        "equipment_id": "pump_001",
        "timestamp": datetime.now().isoformat(),
        "sensor_readings": {
            "pump_main": {
                "pressure_bar": 150.0,
                "temperature_c": 65.0,
                "vibration_g": 0.8,
                "rpm": 1450
            },
            "filter_main": {
                "pressure_bar": 148.0,
                "temperature_c": 66.0
            },
            "valve_control": {
                "pressure_bar": 145.0,
                "temperature_c": 67.0
            },
            "cylinder_1": {
                "pressure_bar": 140.0,
                "temperature_c": 68.0
            }
        },
        "topology_id": "standard_pump_system"
    }
)

result = response.json()
print(f"Health: {result['health']['score']:.2f}")
print(f"Degradation: {result['degradation']['rate']:.2f}")
print(f"Inference time: {result['inference_time_ms']:.1f}ms")
```

### List Available Topologies

```python
response = requests.get("http://localhost:8000/api/v2/topologies")
templates = response.json()["templates"]

for t in templates:
    print(f"{t['template_id']}: {t['name']} ({t['num_components']} components)")
# Output:
# standard_pump_system: Standard Pump System (4 components)
# dual_pump_system: Dual Pump System (7 components)
# hydraulic_circuit_type_a: Hydraulic Circuit Type A (5 components)
```

---

## 🧪 Testing

### Run All Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_api_v2.py -v

# Integration tests only
pytest tests/test_dynamic_edges_integration.py -v
```

### Test Coverage

- **Unit Tests:** 110+ (Phase 3.1)
- **Integration Tests:** 55+ (Phase 3.2)
- **Total Coverage:** 85%+
- **Critical Paths:** 95%+

---

## 📚 Documentation

### Available Docs

- **[MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md)** - Universal GNN I/O specification (v2.0.1)
- **[API_DOCS.md](docs/API_DOCS.md)** - Complete API reference
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **README.md** - This file
- **OpenAPI Docs** - http://localhost:8000/docs

### Code Documentation

- **100% docstring coverage** - All functions documented
- **100% type hints** - Full typing support
- **JSON schema examples** - Request/response samples
- **Inline comments** - Complex logic explained

---

## 🚀 Performance

### Inference Metrics (v2.0.1)

- **Latency:** <200ms (validated)
  - Graph construction: ~50ms
  - Edge projection: ~5ms (NEW)
  - Edge features: ~5ms per edge
  - Model forward: ~100ms
  - **Total:** ~160ms (4 components, 3 edges)

- **Throughput:** >50 predictions/second
- **Memory:** Minimal overhead (+600 bytes per system with edge_projection)
- **GPU Utilization:** 80-90% (batch inference)

### Optimization Tips

```python
# 1. Use batch inference
responses = await engine.predict_batch(requests)

# 2. Enable torch.compile
model = UniversalTemporalGNN(use_compile=True)

# 3. GPU memory pinning
config = InferenceConfig(pin_memory=True)

# 4. Persistent workers
config = DataLoaderConfig(persistent_workers=True)

# 5. Configure edge_in_dim appropriately
model = UniversalTemporalGNN(
    in_channels=34,
    hidden_channels=128,
    edge_in_dim=14,  # Match your data!
    num_heads=8
)
```

---

## 🛠️ Development

### Project Structure

```
services/gnn_service/
├── main.py                    # FastAPI application
├── src/
│   ├── data/                  # Data processing
│   │   ├── edge_features.py   # EdgeFeatureComputer (v2.0.0)
│   │   ├── normalization.py   # EdgeFeatureNormalizer (v2.0.0)
│   │   ├── graph_builder.py   # GraphBuilder (14D)
│   │   └── ...
│   ├── models/                # GNN models
│   │   ├── universal_temporal_gnn.py  # UniversalTemporalGNN (v2.0.1)
│   │   └── ...
│   ├── services/              # Business logic
│   │   └── topology_service.py  # TopologyService (v2.0.0)
│   ├── inference/             # Inference engine
│   │   └── inference_engine.py  # InferenceEngine
│   └── schemas/               # Pydantic models
│       ├── topology.py        # TopologyTemplate (v2.0.0)
│       ├── requests.py        # MinimalInferenceRequest (v2.0.0)
│       └── ...
├── configs/                   # Configuration
│   └── topology_templates.json  # Built-in templates (v2.0.0)
├── tests/                     # Test suite
│   ├── test_api_v2.py         # API tests (v2.0.0)
│   ├── test_edge_features.py  # Edge feature tests (v2.0.0)
│   └── ...
└── docs/                      # Documentation
    ├── MODEL_CONTRACT.md      # Model I/O spec (v2.0.1)
    ├── API_DOCS.md            # API reference (v2.0.0)
    └── ...
```

### Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes with tests
3. Run tests: `pytest tests/ -v`
4. Commit: `git commit -m "feat: add feature"`
5. Push: `git push origin feature/my-feature`
6. Create Pull Request

---

## 🗺️ Roadmap

### v2.0.1 (Current) ✅
- [x] Edge feature dimension flexibility (edge_in_dim)
- [x] Edge projection layer
- [x] MODEL_CONTRACT.md documentation
- [x] Backward compatibility preserved
- [x] Variable graph topology support (#124 Phase 1)

### v2.1.0 (Q1 2026)
- [ ] PyTorch Geometric DataLoader (#124 Phase 2)
- [ ] Dynamic graph construction from TimescaleDB (#124 Phase 3)
- [ ] Real flow meter integration
- [ ] Online learning for normalizer
- [ ] Edge feature importance analysis
- [ ] Advanced topology templates
- [ ] Authentication (API keys)

### v2.2.0 (Q2 2026)
- [ ] Rate limiting
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] Multi-region deployment

### v3.0.0 (Q3 2026)
- [ ] Multi-equipment batch inference
- [ ] Temporal predictions
- [ ] Attention visualization
- [ ] Explainability features

---

## 📝 License

Proprietary - All rights reserved

---

## 📞 Support

- **Issues:** https://github.com/Shukik85/hydraulic-diagnostic-saas/issues
- **Universal GNN Tracking:** [#124](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/124)
- **Email:** support@example.com
- **Slack:** #gnn-service

---

**Last Updated:** 2025-12-04 23:50 MSK  
**Version:** 2.0.1 (Universal GNN Phase 1 ✅)  
**Status:** 🚀 **PRODUCTION READY**