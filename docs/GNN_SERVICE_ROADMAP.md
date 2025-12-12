# GNN Service Production Roadmap

**Branch:** `feature/gnn-service-production-ready`  
**Created:** 2025-11-21  
**Epic Issue:** [#92](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/92)  
**Status:** 🚧 In Progress

---

## 📋 Executive Summary

Переход от prototype stage с критическими пробелами к production-ready GNN service для диагностики гидравлических систем.

### Key Objectives

1. ✅ **Clean Architecture** - устранение stub files, модульная структура
2. 🔲 **Core Implementation** - полная реализация GNN model, dataset, inference
3. 🔲 **Training Pipeline** - PyTorch Lightning, checkpointing, distributed training
4. 🔲 **Production Integration** - FastAPI ↔ Inference ↔ TimescaleDB
5. ✅ **Modern Stack** - Python 3.14, PyTorch 2.8, CUDA 12.9
6. 🔲 **Observability** - structured logging, metrics, health checks

---

## 🏗️ Phase 1: Foundation (Week 1) - 🔵 CURRENT

### ✅ Completed (2025-11-21)

#### Repository Cleanup & Structure
- [x] Created branch `feature/gnn-service-production-ready`
- [x] Moved stub files to `_legacy/`
- [x] Created new `src/` modular structure
- [x] Documented architecture in `STRUCTURE.md`
- [x] Updated dependencies to Python 3.14, PyTorch 2.8, CUDA 12.9
- [x] Created comprehensive README
- [x] Created migration documentation
- [x] Created Epic Issue #92
- [x] Created Sub-Issues #93, #94, #95, #96

---

### 🚧 In Progress - Days 1-2: Core Components

#### Issue #93: Core Schemas Implementation
**Time:** 8 hours  
**Link:** [#93](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/93)

**Files to create:**
```python
src/schemas/
├── __init__.py          # Exports all schemas
├── graph.py            # GraphTopology, ComponentSpec, EdgeSpec
├── metadata.py         # EquipmentMetadata, SensorConfig, SystemConfig
├── requests.py         # InferenceRequest, TrainingRequest, TimeWindow
└── responses.py        # InferenceResponse, ComponentHealth, Anomaly
```

**Python 3.14 features:**
- ✅ Deferred annotations (PEP 649) для динамических типов
- ✅ Type hints с `from __future__ import annotations`
- ✅ Pydantic v2.6+ с ConfigDict

**Example:**
```python
from __future__ import annotations  # PEP 649
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List

class ComponentSpec(BaseModel):
    """Спецификация компонента гидравлической системы."""
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        json_schema_extra={
            "example": {
                "component_id": "pump_main_001",
                "component_type": "hydraulic_pump",
                "sensors": ["pressure_in", "pressure_out", "temperature"],
                "feature_dim": 12
            }
        }
    )
    
    component_id: str = Field(..., min_length=1, max_length=50)
    component_type: str = Field(..., pattern=r"^[a-z_]+$")
    sensors: List[str] = Field(..., min_length=1)
    feature_dim: int = Field(..., gt=0, le=256)
    metadata: Dict[str, str | int | float] = Field(default_factory=dict)

class GraphTopology(BaseModel):
    """Топология гидравлического графа."""
    components: Dict[str, ComponentSpec]
    edges: List[tuple[str, str]]
    edge_types: Dict[str, str]
    
    def validate_connectivity(self) -> bool:
        """Проверка связности графа."""
        # Implementation
        pass
```

**Acceptance Criteria:**
- [ ] All schemas implemented with full type hints
- [ ] Pydantic v2 validation working
- [ ] Deferred annotations used
- [ ] JSON schema export working
- [ ] Unit tests coverage ≥ 90%

---

#### Issue #94: GNN Model Architecture
**Time:** 12 hours  
**Link:** [#94](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/94)

**Files to create:**
```python
src/models/
├── __init__.py          # Exports
├── gnn_model.py        # UniversalTemporalGNN (main)
├── layers.py           # TemporalGATLayer, TemporalLSTMLayer
└── attention.py        # SpatialAttention, TemporalAttention
```

**PyTorch 2.8 features:**
- ✅ `torch.compile` для оптимизации (mode="reduce-overhead")
- ✅ `@torch.inference_mode()` вместо `torch.no_grad()`
- ✅ Stable API patterns
- ✅ SDPA (Scaled Dot-Product Attention)

**Architecture:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class UniversalTemporalGNN(nn.Module):
    """Universal Temporal GNN для гидравлических систем.
    
    Architecture:
        Input → GAT (×3) → LSTM (×2) → Output Heads
        
    Features:
        - Multi-head GAT для spatial attention
        - Bidirectional LSTM для temporal modeling
        - Multiple output heads (health, degradation, anomaly)
        - torch.compile optimization (PyTorch 2.8)
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension (default: 128)
        num_heads: Number of attention heads (default: 8)
        num_gat_layers: Number of GAT layers (default: 3)
        lstm_hidden: LSTM hidden dimension (default: 256)
        lstm_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.3)
        use_compile: Enable torch.compile (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 8,
        num_gat_layers: int = 3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        use_compile: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.use_compile = use_compile
        
        # GAT layers for spatial attention
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels if i == 0 else hidden_channels,
                hidden_channels // num_heads,
                heads=num_heads,
                dropout=dropout,
                add_self_loops=True,
                concat=True
            )
            for i in range(num_gat_layers)
        ])
        
        # Layer normalization after each GAT
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels)
            for _ in range(num_gat_layers)
        ])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_channels,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output heads
        self.health_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Health score [0, 1]
        )
        
        self.degradation_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Degradation rate (continuous)
        )
        
        self.anomaly_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # 3 anomaly types
            nn.Softmax(dim=-1)
        )
        
        # Compile model if enabled (PyTorch 2.8)
        if use_compile and torch.__version__ >= "2.8":
            self._compiled_forward = torch.compile(
                self._forward_impl,
                mode="reduce-overhead",
                fullgraph=True
            )
        else:
            self._compiled_forward = self._forward_impl
    
    def _forward_impl(self, x, edge_index, batch):
        """Internal forward implementation."""
        attention_weights = []
        
        # GAT processing with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            x_new, (edge_idx, attn) = gat(
                x, edge_index, return_attention_weights=True
            )
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            
            # Residual connection (after first layer)
            if i > 0:
                x = x + x_new
            else:
                x = x_new
            
            attention_weights.append(attn)
        
        # Reshape for LSTM: aggregate nodes by batch
        # [N, F] → [B, T, F] where T = num_nodes_per_graph
        x_temporal = self._aggregate_temporal(x, batch)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x_temporal)
        
        # Use final hidden state from last layer
        final_hidden = h_n[-1]  # [B, H]
        
        # Predictions from multiple heads
        health = self.health_head(final_hidden)  # [B, 1]
        degradation = self.degradation_head(final_hidden)  # [B, 1]
        anomaly = self.anomaly_head(final_hidden)  # [B, 3]
        
        return health, degradation, anomaly, attention_weights
    
    def forward(self, x, edge_index, batch, return_attention=False):
        """Forward pass with optional attention weights.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment [N]
            return_attention: Return attention weights
        
        Returns:
            health: Health scores [B, 1]
            degradation: Degradation rates [B, 1]
            anomaly: Anomaly probabilities [B, 3]
            attention: Attention weights (if return_attention=True)
        """
        health, degradation, anomaly, attention = self._compiled_forward(
            x, edge_index, batch
        )
        
        if return_attention:
            return health, degradation, anomaly, attention
        return health, degradation, anomaly
    
    def _aggregate_temporal(self, x, batch):
        """Aggregate node features by batch for temporal processing."""
        # Group nodes by batch index
        batch_size = batch.max().item() + 1
        max_nodes = (batch == 0).sum().item()
        
        # Create tensor [B, max_nodes, F]
        x_batched = torch.zeros(
            batch_size, max_nodes, x.size(1),
            device=x.device, dtype=x.dtype
        )
        
        for b in range(batch_size):
            mask = (batch == b)
            nodes = x[mask]
            x_batched[b, :nodes.size(0)] = nodes
        
        return x_batched
```

**Acceptance Criteria:**
- [ ] Model forward/backward pass works
- [ ] torch.compile works without errors
- [ ] GPU/CPU compatibility
- [ ] Model save/load works
- [ ] Test coverage ≥ 85%

---

### 🚧 In Progress - Days 3-4: Data Pipeline

#### Issue #95: Dataset & DataLoader
**Time:** 14 hours  
**Link:** [#95](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/95)

**Files to create:**
```python
src/data/
├── __init__.py
├── dataset.py          # HydraulicGraphDataset
├── loader.py           # create_dataloaders factory
├── preprocessing.py    # Feature engineering
└── graph_builder.py    # Dynamic graph construction
```

**Implementation:**
```python
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class HydraulicGraphDataset(Dataset):
    """PyTorch Dataset для гидравлических систем.
    
    Features:
    - Time-series windowing
    - Dynamic graph construction
    - Feature engineering
    - Data augmentation
    - Memory-efficient loading
    - Caching mechanism
    
    Args:
        data_path: Path to sensor data (CSV/Parquet)
        metadata_path: Path to equipment metadata (JSON)
        sequence_length: Number of timesteps per sample
        window_minutes: Window size in minutes
        timestep_minutes: Step size for sliding window
        split: 'train', 'val', or 'test'
        augment: Enable data augmentation
        cache_graphs: Cache built graphs in memory
    """
    
    def __init__(
        self,
        data_path: str | Path,
        metadata_path: str | Path,
        sequence_length: int = 10,
        window_minutes: int = 60,
        timestep_minutes: int = 5,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        augment: bool = True,
        cache_graphs: bool = True,
        num_workers: int = 0  # For parallel preprocessing
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.metadata_path = Path(metadata_path)
        self.sequence_length = sequence_length
        self.window_minutes = window_minutes
        self.timestep_minutes = timestep_minutes
        self.split = split
        self.augment = augment and split == "train"
        self.cache_graphs = cache_graphs
        
        # Load data
        self.data = self._load_data()
        self.metadata = self._load_metadata()
        
        # Create windows
        self.windows = self._create_windows()
        
        # Split data
        self._split_data(train_ratio, val_ratio)
        
        # Cache
        self._graph_cache: Dict[int, Data] = {} if cache_graphs else None
    
    def _load_data(self) -> pd.DataFrame:
        """Load sensor data from CSV/Parquet."""
        if self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path, parse_dates=['timestamp'])
        elif self.data_path.suffix == '.parquet':
            df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported format: {self.data_path.suffix}")
        
        # Sort by equipment_id and timestamp
        df = df.sort_values(['equipment_id', 'timestamp'])
        return df
    
    def __getitem__(self, idx: int) -> Data:
        """Get graph sample at index."""
        # Check cache
        if self._graph_cache is not None and idx in self._graph_cache:
            graph = self._graph_cache[idx]
            if self.augment:
                graph = self._augment_graph(graph)
            return graph
        
        # Get window info
        window_info = self.windows[idx]
        
        # Extract sensor data for window
        window_data = self._extract_window_data(window_info)
        
        # Build graph
        from src.data.graph_builder import build_dynamic_graph
        graph = build_dynamic_graph(
            window_data,
            self.metadata,
            window_info['equipment_id']
        )
        
        # Cache
        if self._graph_cache is not None:
            self._graph_cache[idx] = graph
        
        # Augment if training
        if self.augment:
            graph = self._augment_graph(graph)
        
        return graph
    
    def _augment_graph(self, graph: Data) -> Data:
        """Apply data augmentation.
        
        Augmentations:
        - Gaussian noise on node features
        - Random edge dropout
        - Temporal jittering
        """
        # Clone to avoid modifying cache
        graph = graph.clone()
        
        # Add Gaussian noise to node features (5% std)
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(graph.x) * 0.05
            graph.x = graph.x + noise
        
        # Random edge dropout (10%)
        if torch.rand(1).item() < 0.3:
            mask = torch.rand(graph.edge_index.size(1)) > 0.1
            graph.edge_index = graph.edge_index[:, mask]
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                graph.edge_attr = graph.edge_attr[mask]
        
        return graph
    
    def __len__(self) -> int:
        return len(self.windows)
```

**DataLoader Factory:**
```python
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

def create_dataloaders(
    data_path: str,
    metadata_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_ds = HydraulicGraphDataset(
        data_path, metadata_path, split="train", **dataset_kwargs
    )
    val_ds = HydraulicGraphDataset(
        data_path, metadata_path, split="val", augment=False, **dataset_kwargs
    )
    test_ds = HydraulicGraphDataset(
        data_path, metadata_path, split="test", augment=False, **dataset_kwargs
    )
    
    # Create dataloaders (PyG DataLoader handles batching automatically)
    train_loader = PyGDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    val_loader = PyGDataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = PyGDataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader
```

**Acceptance Criteria:**
- [ ] Dataset loads data correctly
- [ ] Graph construction works
- [ ] DataLoader batching works
- [ ] Multi-worker loading works
- [ ] Data augmentation works
- [ ] Test coverage ≥ 80%

---

### 🚧 In Progress - Day 5: Inference Engine

#### Issue #96: Inference Engine Implementation
**Time:** 10 hours  
**Link:** [#96](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/96)

**Files to create:**
```python
src/inference/
├── __init__.py
├── engine.py              # InferenceEngine (main)
├── post_processing.py     # Result processing
└── batch_processor.py     # Batch optimization
```

**Python 3.14 + PyTorch 2.8 features:**
- ✅ Free-threaded mode для parallel inference
- ✅ `@torch.inference_mode()` decorator
- ✅ Async/await with asyncio
- ✅ GPU memory management

**Implementation:**
```python
from __future__ import annotations
import torch
import asyncio
from pathlib import Path
from typing import Dict, List
from contextlib import asynccontextmanager

from src.models.gnn_model import UniversalTemporalGNN
from src.schemas.requests import InferenceRequest
from src.schemas.responses import InferenceResponse, ComponentHealth, Anomaly

class InferenceEngine:
    """Production inference engine.
    
    Features:
    - Model loading/caching
    - GPU memory management
    - Batch optimization
    - Async inference (Python 3.14 free-threading)
    - Error handling with fallbacks
    
    Python 3.14:
    - Free-threaded mode for parallel requests
    - Multiple interpreters for isolation
    
    PyTorch 2.8:
    - @torch.inference_mode() for optimization
    - torch.compile for speed
    - AMP (Automatic Mixed Precision)
    """
    
    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        device: str = "cuda",
        batch_size: int = 32,
        use_amp: bool = True,
        enable_free_threading: bool = True
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.use_amp = use_amp and self.device.type == "cuda"
        
        # Load model and metadata
        self.model = self._load_model()
        self.metadata = self._load_metadata()
        
        # Setup GPU monitoring
        if self.device.type == "cuda":
            self._setup_gpu_monitoring()
        
        # Stats
        self._inference_count = 0
        self._total_time_ms = 0.0
    
    @torch.inference_mode()  # PyTorch 2.8 optimization
    async def predict(
        self,
        request: InferenceRequest
    ) -> InferenceResponse:
        """Single inference request.
        
        Uses Python 3.14 async/await with free-threading.
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Query sensor data from TimescaleDB
            sensor_data = await self._query_sensor_data(request)
            
            # Build graph
            from src.data.graph_builder import build_dynamic_graph
            graph = build_dynamic_graph(
                sensor_data,
                self.metadata,
                request.equipment_id
            )
            
            # Move to device
            graph = graph.to(self.device)
            
            # Run inference with AMP
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp
            ):
                health, degradation, anomaly = self.model(
                    graph.x,
                    graph.edge_index,
                    graph.batch
                )
            
            # Post-process
            response = await self._post_process(
                request,
                health,
                degradation,
                anomaly,
                graph
            )
            
            # Update stats
            inference_time = (time.perf_counter() - start_time) * 1000
            self._inference_count += 1
            self._total_time_ms += inference_time
            
            response.inference_time_ms = inference_time
            response.model_version = self._get_model_version()
            
            return response
            
        except Exception as e:
            return self._handle_error(request, e, start_time)
    
    @torch.inference_mode()
    async def batch_predict(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """Batch inference with optimization.
        
        Processes multiple requests in parallel batches.
        Uses Python 3.14 free-threading for efficiency.
        """
        # Group into batches
        batches = [
            requests[i:i + self.batch_size]
            for i in range(0, len(requests), self.batch_size)
        ]
        
        all_responses = []
        
        for batch_requests in batches:
            # Build batch graph
            graphs = []
            for req in batch_requests:
                sensor_data = await self._query_sensor_data(req)
                from src.data.graph_builder import build_dynamic_graph
                graph = build_dynamic_graph(
                    sensor_data,
                    self.metadata,
                    req.equipment_id
                )
                graphs.append(graph)
            
            # Batch graphs
            batch_graph = Batch.from_data_list(graphs)
            batch_graph = batch_graph.to(self.device)
            
            # Batch inference
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp
            ):
                health, degradation, anomaly = self.model(
                    batch_graph.x,
                    batch_graph.edge_index,
                    batch_graph.batch
                )
            
            # Post-process each result
            for i, req in enumerate(batch_requests):
                response = await self._post_process(
                    req,
                    health[i:i+1],
                    degradation[i:i+1],
                    anomaly[i:i+1],
                    graphs[i]
                )
                all_responses.append(response)
        
        return all_responses
    
    def _load_model(self) -> UniversalTemporalGNN:
        """Load model from checkpoint."""
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True  # PyTorch 2.8 security
        )
        
        model = UniversalTemporalGNN(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def get_stats(self) -> Dict:
        """Get inference statistics."""
        avg_time = (
            self._total_time_ms / self._inference_count
            if self._inference_count > 0
            else 0
        )
        
        stats = {
            "total_inferences": self._inference_count,
            "avg_inference_time_ms": avg_time,
            "device": str(self.device)
        }
        
        if self.device.type == "cuda":
            stats.update(self._get_gpu_stats())
        
        return stats
```

**Acceptance Criteria:**
- [ ] Single inference works
- [ ] Batch inference works
- [ ] GPU memory managed
- [ ] Error handling robust
- [ ] Inference latency < 500ms (p95)
- [ ] Test coverage ≥ 85%

---

## 🏗️ Phase 2: Training Pipeline (Week 2)

### Issue #97: PyTorch Lightning Trainer (TBD)
**Time:** 14 hours  
**Priority:** HIGH

**Tasks:**
- [ ] Implement `src/training/trainer.py` with PyTorch Lightning
- [ ] Add distributed training (DDP)
- [ ] Add float8 training (PyTorch 2.8)
- [ ] Add AMP support
- [ ] Add gradient accumulation
- [ ] Add learning rate scheduling
- [ ] Add early stopping
- [ ] Add custom metrics
- [ ] Add TensorBoard logging

---

### Issue #98: FastAPI Integration (TBD)
**Time:** 12 hours  
**Priority:** HIGH

**Tasks:**
- [ ] Refactor `api/main.py` to full async
- [ ] Implement `api/routes/inference.py`
- [ ] Implement `api/routes/admin.py`
- [ ] Add request validation
- [ ] Add rate limiting
- [ ] Add circuit breaker
- [ ] Integration tests

---

### Issue #99: Model Management System (TBD)
**Time:** 10 hours  
**Priority:** HIGH

**Tasks:**
- [ ] Model versioning
- [ ] Deployment API
- [ ] Rollback mechanism
- [ ] A/B testing support
- [ ] Model registry

---

## 🏗️ Phase 3: Production Hardening (Week 3)

### Issue #100: Observability (TBD)
**Time:** 10 hours  
**Priority:** MEDIUM

**Tasks:**
- [ ] Structured logging (python-json-logger)
- [ ] Prometheus metrics
- [ ] Health checks (liveness/readiness)
- [ ] Trace ID propagation
- [ ] Custom domain metrics

---

### Issue #101: Testing & Documentation (TBD)
**Time:** 12 hours  
**Priority:** MEDIUM

**Tasks:**
- [ ] Unit tests (≥ 80% coverage)
- [ ] Integration tests
- [ ] E2E tests
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] Deployment guide

---

## 📊 Technology Stack (Updated 2025-11-21)

### Python 3.14.0 (Released: 07.10.2025)

**Major Features:**
- ✅ **PEP 779:** Free-threaded Python (no GIL) - параллельная inference
- ✅ **PEP 649:** Deferred annotations - динамическая типизация
- ✅ **PEP 750:** t-string literals - безопасные шаблонные строки
- ✅ **PEP 734:** Multiple interpreters - изоляция workloads
- ✅ **PEP 784:** compression.zstd - быстрая компрессия
- ✅ **New REPL:** Цветной синтаксис, автокомплит, улучшенные ошибки
- ✅ **Improved JIT:** 10-15% ускорение
- ✅ **Better memory:** Уменьшенный footprint

**Documentation:**
- [What's new in Python 3.14](https://docs.python.org/3/whatsnew/3.14.html)
- [Python 3.14 Release](https://www.python.org/downloads/release/python-3140/)

---

### PyTorch 2.8 (Released: 06.08.2025)

**Major Features:**
- ✅ **Stable/Unstable API system** - чёткая классификация
- ✅ **Float8 training** - 1.5x ускорение обучения LLM/NN
- ✅ **Quantized inference (Intel CPU)** - Int8 quantization с AMX
- ✅ **torch.compile для Apple Silicon** - M1/M2/M3 optimization
- ✅ **Expanded platform support** - SYCL, XPU, ROCm 6.4+
- ✅ **torchao registry** - новая система quantization
- ✅ **Int4 quantization (CUDA-only)** - сжатие моделей
- ✅ **weights_only=True** - безопасная загрузка моделей
- ✅ **SDPA improvements** - Scaled Dot-Product Attention

**Documentation:**
- [PyTorch 2.8 Release Notes](https://dev-discuss.pytorch.org/t/pytorch-release-2-8-key-information/3039)

---

### CUDA 12.9 (Released: Summer 2025)

**Major Features:**
- ✅ **Family-specific features** - SM 10.3/12.1, Blackwell optimization
- ✅ **PTX compatibility** - универсальные бинарники
- ✅ **Cubin forward/backward compatibility** - между архитектурами
- ✅ **Blackwell architecture support** - новейшие GPU
- ✅ **Improved memory management** - для больших моделей
- ✅ **Multi-GPU optimization** - лучшая поддержка NVLink
- ⚠️ **Maxwell/Pascal/Volta deprecated** - только поддержка, без новых фич

**Documentation:**
- [CUDA 12.9 Blackwell Features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)

---

## 🎯 Practical Usage in GNN Service

### Python 3.14 Applications

**1. Free-threaded parallel inference:**
```python
# Multiple concurrent inference requests without GIL
import sys
sys.set_gil_mode(0)  # Disable GIL (Python 3.14)

async def handle_parallel_requests(requests: List[InferenceRequest]):
    # Process in parallel without GIL blocking
    tasks = [engine.predict(req) for req in requests]
    return await asyncio.gather(*tasks)
```

**2. Deferred annotations for dynamic schemas:**
```python
from __future__ import annotations

class DynamicConfig(BaseModel):
    # Type evaluation deferred until needed
    components: Dict[str, ComponentSpec]
    # Reduces import overhead
```

**3. t-strings for safe templating:**
```python
# Safe SQL template (example)
from template import Template

query_template = Template(t"""
    SELECT * FROM sensor_data 
    WHERE equipment_id = {equipment_id}
    AND timestamp BETWEEN {start_time} AND {end_time}
""")
```

**4. Multiple interpreters for isolation:**
```python
import interpreters

# Create isolated interpreter for training
training_interp = interpreters.create()
training_interp.exec("""
    # Training runs in isolated interpreter
    trainer.fit(model, train_loader)
""")
```

---

### PyTorch 2.8 Applications

**1. Float8 training:**
```python
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.experimental import float8_training

# Enable float8 training (1.5x speedup)
with float8_training():
    for batch in train_loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

**2. Quantized inference:**
```python
import torchao

# Int8 quantization for CPU deployment
quantized_model = torchao.quantize(
    model,
    quantization_config=torchao.Int8DynActInt8WeightConfig()
)

# 2-4x speedup on Intel CPUs with AMX
```

**3. torch.compile optimization:**
```python
# Compile model for optimization
compiled_model = torch.compile(
    model,
    mode="reduce-overhead",  # or "max-autotune"
    fullgraph=True
)

# 1.5-2x speedup in inference
```

---

### CUDA 12.9 Applications

**1. Family-specific compilation:**
```bash
# Compile for Blackwell GPUs (SM 10.3)
nvcc -arch=sm_103 -o optimized.cubin kernel.cu

# Compile for Hopper (SM 9.0)
nvcc -arch=sm_90 -o optimized.cubin kernel.cu
```

**2. PTX universal binaries:**
```python
# PyTorch will use PTX for runtime compilation
# Automatically targets available GPU architecture
torch.cuda.set_device(0)  # Auto-selects best SM version
```

---

## 📊 Performance Targets

### With New Stack (Python 3.14 + PyTorch 2.8 + CUDA 12.9)

| Metric | Target | Improvement |
|--------|--------|-------------|
| **Inference Latency (p95)** | < 500ms | 1.5-2x faster |
| **Inference Latency (p50)** | < 200ms | 1.5-2x faster |
| **Training Speed** | < 4 hours | 1.5x faster (float8) |
| **Throughput** | > 100 req/s | 2-3x higher |
| **GPU Utilization** | > 70% | +20% |
| **CPU Inference** | < 2s | 2-4x faster (quantized) |
| **Memory Usage** | -20% | Better allocation |
| **Parallel Requests** | 10x+ | Free-threading |

---

## ✅ Success Criteria

### Production Readiness Checklist

**Functionality:**
- [ ] All core components implemented (no stubs)
- [ ] Inference latency < 500ms (p95)
- [ ] Training pipeline automated
- [ ] Model versioning working
- [ ] FastAPI ↔ TimescaleDB integration

**Quality:**
- [ ] Test coverage ≥ 80%
- [ ] All code quality checks pass
- [ ] No critical security issues
- [ ] Documentation complete

**Performance:**
- [ ] Targets met (see table above)
- [ ] No memory leaks
- [ ] GPU utilization optimized

**Production:**
- [ ] Structured logging
- [ ] Prometheus metrics
- [ ] Health checks
- [ ] Error rate < 1%
- [ ] Docker images
- [ ] K8s manifests

---

## 🔗 Resources

### Documentation
- **Epic Issue:** [#92](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/92)
- **Roadmap:** This file
- **Structure:** [STRUCTURE.md](../services/gnn_service/STRUCTURE.md)
- **README:** [README.md](../services/gnn_service/README.md)
- **Migration:** [MIGRATION_SUMMARY.md](../services/gnn_service/MIGRATION_SUMMARY.md)

### Issues
- **Phase 1:** [#93](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/93), [#94](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/94), [#95](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/95), [#96](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/96)
- **Phase 2:** TBD
- **Phase 3:** TBD

### External Links
- [Python 3.14 Docs](https://docs.python.org/3.14/)
- [PyTorch 2.8 Docs](https://pytorch.org/docs/2.8/)
- [CUDA 12.9 Docs](https://docs.nvidia.com/cuda/)

---

## 📅 Timeline

**Week 1 (Nov 21-27):** Foundation
- ✅ Day 0 (Nov 21): Structure & planning
- 🔲 Days 1-2: Core schemas & GNN model
- 🔲 Days 3-4: Dataset & DataLoader
- 🔲 Day 5: Inference engine

**Week 2 (Nov 28 - Dec 4):** Training & Integration
- 🔲 Days 1-2: PyTorch Lightning trainer
- 🔲 Days 3-4: FastAPI integration
- 🔲 Day 5: Model management

**Week 3 (Dec 5-11):** Production Readiness
- 🔲 Days 1-2: Observability
- 🔲 Days 3-4: Testing
- 🔲 Day 5: Documentation & deployment

---

**Last Updated:** 2025-11-21 04:00 MSK  
**Next Review:** After Phase 1 completion