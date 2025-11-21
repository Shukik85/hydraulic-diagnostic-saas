# Issue #95: Dataset & DataLoader - Quick Start Checklist

☕ **Post-Coffee Implementation Guide**

---

## ✅ Step-by-Step Checklist

### 🎯 Phase 1: Foundation (30 min)

- [ ] Create `src/data/__init__.py` with exports
- [ ] Create `src/data/feature_config.py` (FeatureConfig, DataLoaderConfig)
- [ ] Update `requirements.txt` with:
  - `asyncpg>=0.29.0`
  - `pandas>=2.2.0`
  - `scikit-learn>=1.4.0`
  - `scipy>=1.12.0`

---

### 📊 Phase 2: TimescaleDB Connector (60 min)

- [ ] Create `src/data/timescale_connector.py`
- [ ] Implement `TimescaleConnector` class:
  - [ ] `__init__()` - connection pool setup
  - [ ] `async fetch_sensor_data()` - single equipment query
  - [ ] `async fetch_batch_sensor_data()` - batch query
  - [ ] `async get_equipment_metadata()` - metadata query
  - [ ] `async close()` - cleanup
- [ ] Add retry logic with exponential backoff
- [ ] Add connection health check
- [ ] Create `tests/unit/test_timescale_connector.py`
- [ ] Run: `pytest tests/unit/test_timescale_connector.py -v`

**Quick snippet:**
```python
import asyncpg
from src.schemas import TimeWindow, EquipmentMetadata

class TimescaleConnector:
    def __init__(self, db_url: str, pool_size: int = 10):
        self.db_url = db_url
        self.pool: asyncpg.Pool | None = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
    
    async def fetch_sensor_data(...) -> pd.DataFrame:
        # Query TimescaleDB
        # Return DataFrame
```

---

### 🧮 Phase 3: Feature Engineering (90 min)

- [ ] Create `src/data/feature_engineer.py`
- [ ] Implement `FeatureEngineer` class:
  - [ ] `extract_statistical_features()` - mean, std, percentiles, etc.
  - [ ] `extract_frequency_features()` - FFT, PSD, dominant freq
  - [ ] `extract_temporal_features()` - rolling windows, autocorr
  - [ ] `extract_hydraulic_features()` - pressure ratio, efficiency
  - [ ] `normalize_features()` - standardization/minmax
  - [ ] `extract_all_features()` - combine all
- [ ] Create `tests/unit/test_feature_engineer.py`
- [ ] Run: `pytest tests/unit/test_feature_engineer.py -v`

**Quick snippet:**
```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq

class FeatureEngineer:
    def extract_statistical_features(self, data: pd.DataFrame) -> np.ndarray:
        features = [
            data.mean(),
            data.std(),
            data.min(),
            data.max(),
            data.quantile([0.25, 0.5, 0.75]),
            stats.skew(data),
            stats.kurtosis(data)
        ]
        return np.concatenate(features)
```

---

### 🌐 Phase 4: Graph Builder (60 min)

- [ ] Create `src/data/graph_builder.py`
- [ ] Implement `GraphBuilder` class:
  - [ ] `build_graph()` - main construction method
  - [ ] `build_component_features()` - node features
  - [ ] `build_edge_features()` - edge attributes
  - [ ] `validate_graph()` - topology check
  - [ ] `_construct_edge_index()` - connectivity matrix
- [ ] Create `tests/unit/test_graph_builder.py`
- [ ] Run: `pytest tests/unit/test_graph_builder.py -v`

**Quick snippet:**
```python
import torch
from torch_geometric.data import Data
from src.schemas import GraphTopology, EquipmentMetadata

class GraphBuilder:
    def build_graph(
        self,
        sensor_data: pd.DataFrame,
        topology: GraphTopology,
        metadata: EquipmentMetadata,
        features: np.ndarray
    ) -> Data:
        # Build nodes
        x = torch.from_numpy(features).float()
        
        # Build edges
        edge_index = self._construct_edge_index(topology)
        edge_attr = self._build_edge_features(topology)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

---

### 📚 Phase 5: Dataset Implementation (60 min)

- [ ] Create `src/data/dataset.py`
- [ ] Implement `HydraulicGraphDataset` class:
  - [ ] `__init__()` - setup
  - [ ] `__len__()` - dataset size
  - [ ] `__getitem__()` - fetch item with caching
  - [ ] `_load_equipment_list()` - load metadata
  - [ ] `_get_cache_path()` - cache file path
  - [ ] `_load_from_cache()` - load cached graph
  - [ ] `_save_to_cache()` - save graph
  - [ ] `get_equipment_ids()` - list equipment
  - [ ] `get_statistics()` - dataset stats
- [ ] Create `tests/unit/test_dataset.py`
- [ ] Run: `pytest tests/unit/test_dataset.py -v`

**Quick snippet:**
```python
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class HydraulicGraphDataset(Dataset):
    def __init__(self, data_path, connector, engineer, builder, cache_dir=None):
        self.equipment_list = self._load_equipment_list(data_path)
        self.connector = connector
        self.engineer = engineer
        self.builder = builder
        self.cache_dir = Path(cache_dir) if cache_dir else None
    
    def __len__(self):
        return len(self.equipment_list)
    
    def __getitem__(self, idx) -> Data:
        equipment_id = self.equipment_list[idx]
        
        # Try cache first
        if self.cache_dir:
            cached = self._load_from_cache(equipment_id)
            if cached is not None:
                return cached
        
        # Build graph
        graph = self._build_graph(equipment_id)
        
        # Cache it
        if self.cache_dir:
            self._save_to_cache(equipment_id, graph)
        
        return graph
```

---

### 🚀 Phase 6: DataLoader Factory (30 min)

- [ ] Create `src/data/loader.py`
- [ ] Implement:
  - [ ] `hydraulic_collate_fn()` - custom collate
  - [ ] `create_dataloader()` - factory function
  - [ ] `create_train_val_loaders()` - train/val split
- [ ] Create `tests/unit/test_loader.py`
- [ ] Run: `pytest tests/unit/test_loader.py -v`

**Quick snippet:**
```python
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

def hydraulic_collate_fn(batch: list[Data]) -> Batch:
    return Batch.from_data_list(batch)

def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=hydraulic_collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
```

---

### 🧪 Phase 7: Integration Testing (45 min)

- [ ] Create `tests/integration/test_data_pipeline.py`
- [ ] Test end-to-end pipeline:
  - [ ] TimescaleDB → Features → Graph → Batch
  - [ ] Multiple equipment
  - [ ] Real schema instances
  - [ ] DataLoader iteration
- [ ] Create `tests/integration/test_model_data_integration.py`
- [ ] Test model forward pass with real data:
  - [ ] Load batch from DataLoader
  - [ ] Forward through UniversalTemporalGNN
  - [ ] Verify output shapes
- [ ] Run: `pytest tests/integration/ -v`

---

### 📚 Phase 8: Documentation (45 min)

- [ ] Update `README.md`:
  - [ ] Add "Data Pipeline" section
  - [ ] Document feature engineering
  - [ ] Add DataLoader examples
  - [ ] Performance tuning guide
- [ ] Verify all docstrings complete
- [ ] Add `examples/data_pipeline_usage.py`
- [ ] Update `STRUCTURE.md` if needed

---

## 🛠️ Quick Commands (Copy-Paste Ready)

### Create Files
```bash
cd services/gnn_service

# Create data module files
touch src/data/__init__.py
touch src/data/feature_config.py
touch src/data/timescale_connector.py
touch src/data/feature_engineer.py
touch src/data/graph_builder.py
touch src/data/dataset.py
touch src/data/loader.py

# Create test files
touch tests/unit/test_timescale_connector.py
touch tests/unit/test_feature_engineer.py
touch tests/unit/test_graph_builder.py
touch tests/unit/test_dataset.py
touch tests/unit/test_loader.py
touch tests/integration/test_data_pipeline.py
touch tests/integration/test_model_data_integration.py
```

### Run Tests
```bash
# Run all data tests
pytest tests/unit/test_data*.py tests/unit/test_timescale*.py tests/unit/test_feature*.py tests/unit/test_graph*.py tests/unit/test_loader.py -v

# With coverage
pytest tests/unit/test_data*.py tests/unit/test_timescale*.py tests/unit/test_feature*.py tests/unit/test_graph*.py tests/unit/test_loader.py --cov=src/data --cov-report=term-missing

# Integration tests
pytest tests/integration/ -v
```

### Type Check
```bash
mypy src/data/ --strict
```

### Lint
```bash
ruff check src/data/
ruff format src/data/
```

---

## 📊 Success Metrics

**Code:**
- [ ] ~1500 lines of production code
- [ ] ~800 lines of test code
- [ ] 7 new files (5 src + 2 test)
- [ ] 85%+ test coverage
- [ ] 100% type hints
- [ ] 100% docstrings

**Functionality:**
- [ ] TimescaleDB queries working
- [ ] Feature extraction < 200ms per equipment
- [ ] Graph construction < 50ms
- [ ] DataLoader throughput > 50 graphs/s
- [ ] Cache hit ratio > 90%

**Quality:**
- [ ] All tests passing
- [ ] mypy strict mode passing
- [ ] ruff checks passing
- [ ] Documentation complete

---

## 📦 Commit Strategy

**Planned commits:**

1. `feat(data): implement TimescaleDB async connector (#95)`
2. `feat(data): add feature engineering pipeline (#95)`
3. `feat(data): implement graph builder with schema integration (#95)`
4. `feat(data): add HydraulicGraphDataset with caching (#95)`
5. `feat(data): implement DataLoader factory (#95)`
6. `test(data): add comprehensive unit tests (#95)`
7. `test(data): add integration tests for data pipeline (#95)`
8. `docs(readme): update with data pipeline documentation (#95)`

**Target: 8 commits**

---

## ⚡ Quick Start Commands (After Coffee)

```bash
# Navigate to directory
cd services/gnn_service

# Start with Step 1: Foundation
# 1. Create __init__.py
# 2. Create feature_config.py
# 3. Update requirements.txt

# Then proceed through phases 2-8
# Test after each phase
# Commit when phase complete

# Final check:
pytest tests/ --cov=src --cov-report=term-missing
mypy src/ --strict
ruff check src/
```

---

## 🎯 Ready State

✅ Plan documented  
✅ Steps outlined  
✅ Code snippets prepared  
✅ Tests planned  
✅ Documentation checklist ready  

**🚀 Ready to implement after coffee break!**

---

**Estimated time:** 4-5 hours  
**Start when ready:** ☕ → 💻 → ✅