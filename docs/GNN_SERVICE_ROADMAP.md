# GNN Service Production Roadmap

**Branch:** `feature/gnn-service-production-ready`  
**Created:** 2025-11-21  
**Status:** 🚧 In Progress

## Executive Summary

Переход от prototype stage с критическими пробелами к production-ready GNN service.

### Key Objectives

1. ✅ **Clean Architecture** - устранение stub files, модульная структура
2. ⏳ **Core Implementation** - полная реализация GNN model, dataset, inference
3. ⏳ **Training Pipeline** - PyTorch Lightning, checkpointing, distributed training
4. ⏳ **Production Integration** - FastAPI ↔ Inference ↔ TimescaleDB
5. ⏳ **Modern Stack** - Python 3.13.5, PyTorch 2.8, CUDA 12.9
6. ⏳ **Observability** - structured logging, metrics, health checks

---

## Phase 1: Foundation (Week 1) - 🔵 CURRENT

### ✅ Completed

#### Repository Cleanup
- [x] Created branch `feature/gnn-service-production-ready`
- [x] Moved stub files to `_legacy/`
- [x] Created new `src/` structure
- [x] Documented new architecture in `STRUCTURE.md`

### 🚧 In Progress

#### Day 1-2: Core Models Implementation

**File:** `src/schemas/graph.py`
```python
from pydantic import BaseModel, Field
from typing import Dict, List

class ComponentSpec(BaseModel):
    component_id: str
    component_type: str
    sensors: List[str]
    feature_dim: int

class GraphTopology(BaseModel):
    components: Dict[str, ComponentSpec]
    edges: List[tuple[str, str]]
    edge_types: Dict[tuple[str, str], str]
```

**File:** `src/models/gnn_model.py`
```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class UniversalTemporalGNN(nn.Module):
    """GAT + LSTM for hydraulic diagnostics."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 8,
        num_gat_layers: int = 3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATConv(in_channels if i == 0 else hidden_channels,
                    hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout)
            for i in range(num_gat_layers)
        ])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_channels,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output heads
        self.health_head = nn.Linear(lstm_hidden, 1)  # Health score
        self.degradation_head = nn.Linear(lstm_hidden, 1)  # Degradation rate
    
    def forward(self, x, edge_index, batch):
        # GAT processing
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = F.relu(x)
        
        # Temporal aggregation
        x, (h_n, c_n) = self.lstm(x.unsqueeze(1))
        
        # Predictions
        health = torch.sigmoid(self.health_head(h_n[-1]))
        degradation = self.degradation_head(h_n[-1])
        
        return health, degradation
```

**Tasks:**
- [ ] Implement `src/schemas/graph.py` - GraphTopology, ComponentSpec
- [ ] Implement `src/schemas/metadata.py` - EquipmentMetadata, SensorConfig
- [ ] Implement `src/schemas/requests.py` - InferenceRequest, TrainingRequest
- [ ] Implement `src/schemas/responses.py` - InferenceResponse, ComponentHealth
- [ ] Implement `src/models/gnn_model.py` - UniversalTemporalGNN
- [ ] Implement `src/models/layers.py` - Custom GAT/LSTM wrappers
- [ ] Implement `src/models/attention.py` - Attention mechanisms
- [ ] Unit tests for models

**Acceptance Criteria:**
- ✅ All schemas have Pydantic validation
- ✅ Model forward pass works
- ✅ Model can be saved/loaded
- ✅ Test coverage ≥ 80%

---

#### Day 3-4: Dataset & DataLoader

**File:** `src/data/dataset.py`
```python
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch_geometric.data import Data

class HydraulicGraphDataset(Dataset):
    """PyTorch Dataset for hydraulic system graphs."""
    
    def __init__(
        self,
        data_path: str,
        metadata: EquipmentMetadata,
        sequence_length: int = 10,
        window_minutes: int = 60,
        split: str = "train"
    ):
        self.data = pd.read_csv(data_path)
        self.metadata = metadata
        self.sequence_length = sequence_length
        self.window_minutes = window_minutes
        self.split = split
        self._prepare_data()
    
    def _prepare_data(self):
        # Time-series windowing
        # Component grouping
        # Feature engineering
        pass
    
    def __getitem__(self, idx):
        # Build graph
        # Return Data object
        pass
    
    def __len__(self):
        return len(self.windows)
```

**Tasks:**
- [ ] Implement `src/data/dataset.py` - HydraulicGraphDataset
- [ ] Implement `src/data/loader.py` - create_dataloaders factory
- [ ] Implement `src/data/preprocessing.py` - feature engineering
- [ ] Implement `src/data/graph_builder.py` - dynamic graph construction
- [ ] Add data augmentation for hydraulic systems
- [ ] Memory-efficient data loading
- [ ] Tests for data pipeline

**Acceptance Criteria:**
- ✅ Dataset loads CSV correctly
- ✅ Graph construction works
- ✅ DataLoader batching works
- ✅ Memory usage optimized

---

#### Day 5: Inference Engine

**File:** `src/inference/engine.py`
```python
class InferenceEngine:
    """Production inference engine."""
    
    def __init__(
        self,
        model_path: str,
        metadata_path: str,
        device: str = "cuda",
        batch_size: int = 32
    ):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.metadata = self._load_metadata(metadata_path)
        self.batch_size = batch_size
    
    async def predict(
        self,
        equipment_id: str,
        sensor_data: Dict[str, np.ndarray]
    ) -> InferenceResponse:
        # Build graph
        # Run inference
        # Post-process
        pass
    
    async def batch_predict(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        # Batch optimization
        pass
```

**Tasks:**
- [ ] Implement `src/inference/engine.py` - InferenceEngine
- [ ] Implement `src/inference/post_processing.py` - result processing
- [ ] Implement `src/inference/batch_processor.py` - batch optimization
- [ ] Add GPU memory management
- [ ] Add model caching
- [ ] Error handling and fallbacks
- [ ] Tests

**Acceptance Criteria:**
- ✅ Inference latency < 500ms (p95)
- ✅ Batch processing works
- ✅ GPU memory managed correctly
- ✅ Graceful error handling

---

## Phase 2: Training Pipeline (Week 2)

### Day 1-2: PyTorch Lightning Trainer

**File:** `src/training/trainer.py`
```python
import pytorch_lightning as pl

class GNNTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for GNN."""
    
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        # Loss calculation
        # Logging
        pass
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        # AdamW optimizer
        # Learning rate scheduling
        pass
```

**Tasks:**
- [ ] Implement `src/training/trainer.py` - GNNTrainer with Lightning
- [ ] Implement `src/training/callbacks.py` - training callbacks
- [ ] Implement `src/training/metrics.py` - custom metrics
- [ ] Add distributed training support (DDP)
- [ ] Add automatic mixed precision (AMP)
- [ ] Add gradient accumulation
- [ ] Add learning rate scheduling
- [ ] Add early stopping
- [ ] Tests

**Acceptance Criteria:**
- ✅ Training runs end-to-end
- ✅ Distributed training works
- ✅ Checkpointing automatic
- ✅ Metrics logged correctly

---

### Day 3-4: FastAPI ↔ TimescaleDB Integration

**File:** `config/database.py`
```python
import asyncpg

class DatabaseManager:
    """Async TimescaleDB connection manager."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            min_size=2,
            max_size=self.config.pool_size,
            timeout=self.config.timeout
        )
    
    async def query_sensor_data(
        self,
        equipment_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, pd.DataFrame]:
        # Time-series query with hyperfunction
        pass
```

**Tasks:**
- [ ] Improve `config/database.py` - async connection pool
- [ ] Refactor `api/main.py` - full async/await
- [ ] Implement `api/routes/inference.py` - inference endpoints
- [ ] Implement `api/routes/admin.py` - admin endpoints
- [ ] Implement `api/routes/monitoring.py` - monitoring
- [ ] Implement `api/routes/health.py` - health checks
- [ ] Add request validation
- [ ] Add response streaming
- [ ] Add rate limiting
- [ ] Add circuit breaker
- [ ] Integration tests

**Acceptance Criteria:**
- ✅ All endpoints async
- ✅ DB connection pooling works
- ✅ Request validation strict
- ✅ Error handling comprehensive

---

### Day 5: Model Management

**File:** `api/routes/admin.py`
```python
@router.post("/model/deploy")
async def deploy_model(
    model_file: UploadFile,
    version: str,
    metadata: ModelMetadata
):
    # Validate model
    # Save checkpoint
    # Update production pointer
    # Rollback support
    pass
```

**Tasks:**
- [ ] Model versioning system
- [ ] Deployment API endpoints
- [ ] Rollback mechanism
- [ ] A/B testing support
- [ ] Model registry
- [ ] Tests

**Acceptance Criteria:**
- ✅ Model deployment works
- ✅ Versioning tracked
- ✅ Rollback tested
- ✅ A/B testing functional

---

## Phase 3: Production Readiness (Week 3)

### Day 1-2: Observability

**Tasks:**
- [ ] Structured logging with python-json-logger
- [ ] Prometheus metrics
- [ ] Health check endpoints (liveness/readiness)
- [ ] Custom metrics for domain KPIs
- [ ] Log aggregation ready (ELK/Datadog format)
- [ ] Trace ID propagation

**Metrics to track:**
- Request latency histogram
- Inference time per model version
- GPU utilization
- Error rates
- Queue depth
- Model performance metrics

---

### Day 3-4: Error Handling & Resilience

**Tasks:**
- [ ] Graceful degradation
- [ ] Retry mechanisms with exponential backoff
- [ ] Circuit breaker for external services
- [ ] Detailed error responses
- [ ] Error categorization
- [ ] Tests for error scenarios

---

### Day 5: Testing & Documentation

**Tasks:**
- [ ] Unit tests (≥ 80% coverage)
- [ ] Integration tests
- [ ] End-to-end tests
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Architecture diagrams
- [ ] Deployment guide
- [ ] Model development guide

---

## Technology Stack Updates

### Python 3.13.5 Advantages

1. **Free-threaded mode** - снятие GIL для параллельной обработки inference
2. **Improved JIT** - 10-15% ускорение Python кода
3. **Better memory** - уменьшенный memory footprint
4. **Enhanced typing** - TypeIs для type safety

### PyTorch 2.8 New Features

1. **Stable libtorch ABI** - упрощение C++/CUDA extensions
2. **Quantized inference** - оптимизация CPU inference
3. **Control flow operators** - лучшая компиляция
4. **Wheel Variants** - оптимизированные сборки

### CUDA 12.9 Improvements

1. Family-specific features
2. A100/H100 optimizations
3. Better memory management

---

## Metrics & Success Criteria

### Production Readiness Checklist

- [ ] All core components implemented (no stubs)
- [ ] Inference latency < 500ms (p95)
- [ ] Training pipeline automated
- [ ] Test coverage ≥ 80%
- [ ] Structured logging + metrics
- [ ] Health checks working
- [ ] Error rate < 1%
- [ ] Documentation complete
- [ ] Docker images built
- [ ] Kubernetes manifests tested

### Performance Targets

- **Inference latency**: < 500ms (p95), < 200ms (p50)
- **Throughput**: > 100 req/s per GPU
- **Training speed**: < 4 hours for full dataset
- **Model accuracy**: > 90% health prediction
- **Uptime**: 99.9% availability

---

## Risk Mitigation

### Technical Risks

1. **GPU memory** - batch size tuning, gradient checkpointing
2. **Training stability** - gradient clipping, learning rate tuning
3. **Data quality** - validation pipeline, outlier detection
4. **Integration complexity** - modular design, comprehensive tests

### Timeline Risks

- **Buffer**: 20% time buffer for unexpected issues
- **Parallel work**: Core models and data pipeline can be developed in parallel
- **Incremental delivery**: Each phase produces working software

---

## Next Steps

**Immediate (Today):**
1. ✅ Repository structure created
2. 🚧 Start implementing `src/schemas/`
3. 🚧 Start implementing `src/models/gnn_model.py`

**This Week:**
- Complete Phase 1 (Foundation)
- Begin Phase 2 (Training Pipeline)

**Next Week:**
- Complete Phase 2
- Begin Phase 3 (Production Readiness)

**Week 3:**
- Complete Phase 3
- Production deployment ready

---

## Resources

- **Code**: `services/gnn_service/`
- **Documentation**: `docs/`
- **Issues**: GitHub Issues #44, #86
- **Branch**: `feature/gnn-service-production-ready`

## Contact

- **ML Engineer**: Responsible for model implementation
- **DevOps**: Docker, Kubernetes deployment
- **QA**: Testing strategy and execution