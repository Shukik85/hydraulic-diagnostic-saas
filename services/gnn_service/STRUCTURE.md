# GNN Service - Production Structure

## Overview

Clean, modular architecture for production-ready GNN service.

## Directory Structure

```
services/gnn_service/
├── src/                          # Source code (new clean implementation)
│   ├── models/                   # GNN model implementations
│   │   ├── __init__.py
│   │   ├── gnn_model.py          # UniversalTemporalGNN (GAT + LSTM)
│   │   ├── layers.py             # Custom layers (GAT, LSTM wrappers)
│   │   └── attention.py          # Attention mechanisms
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py            # HydraulicGraphDataset
│   │   ├── loader.py             # DataLoader factory
│   │   ├── preprocessing.py      # Feature engineering
│   │   └── graph_builder.py      # Dynamic graph construction
│   ├── inference/               # Inference engine
│   │   ├── __init__.py
│   │   ├── engine.py             # InferenceEngine class
│   │   ├── post_processing.py    # Result processing
│   │   └── batch_processor.py    # Batch optimization
│   ├── training/                # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py            # GNNTrainer (PyTorch Lightning)
│   │   ├── callbacks.py          # Training callbacks
│   │   └── metrics.py            # Custom metrics
│   ├── schemas/                 # Pydantic models
│   │   ├── __init__.py
│   │   ├── graph.py              # GraphTopology, ComponentSpec
│   │   ├── metadata.py           # EquipmentMetadata, SensorConfig
│   │   ├── requests.py           # API request models
│   │   └── responses.py          # API response models
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── device.py             # CUDA/CPU device management
│       ├── checkpointing.py      # Model checkpointing
│       └── logging_config.py     # Structured logging setup
├── api/                         # FastAPI application
│   ├── __init__.py
│   ├── main.py                   # FastAPI app (refactored)
│   ├── routes/
│   │   ├── inference.py          # Inference endpoints
│   │   ├── admin.py              # Admin endpoints
│   │   ├── monitoring.py         # Monitoring endpoints
│   │   └── health.py             # Health checks
│   ├── middleware/
│   │   └── error_handling.py     # Error handlers
│   └── dependencies.py           # FastAPI dependencies
├── config/                      # Configuration
│   ├── __init__.py
│   ├── settings.py               # Pydantic settings
│   └── database.py               # DB configuration
├── _legacy/                     # Old/deprecated code (archived)
│   ├── README_LEGACY.md
│   ├── model_dynamic_gnn_stub.py
│   ├── dataset_dynamic_stub.py
│   └── ... (other legacy files)
├── tests/                       # Tests
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_dataset.py
│   │   └── test_inference.py
│   ├── integration/
│   │   ├── test_api.py
│   │   └── test_training.py
│   └── conftest.py
├── data/                        # Data directory
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── models/                      # Saved models
│   ├── checkpoints/
│   └── production/
├── logs/                        # Logs
├── kubernetes/                  # K8s manifests
├── protos/                      # gRPC protos (if needed)
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Dev dependencies
├── Dockerfile                   # Production Docker image
├── Dockerfile.dev               # Development Docker image
├── docker-compose.yml           # Docker compose
├── pyproject.toml               # Python project config
├── .env.example                 # Environment variables template
└── README.md                    # Service documentation
```

## Key Changes from Old Structure

### Before (Problematic)
```
services/gnn_service/
├── model_dynamic_gnn.py      # STUB - only comments
├── dataset_dynamic.py       # STUB - only comments
├── schemas.py               # STUB - only comments
├── train_dynamic.py         # Uses non-existent imports
├── inference_dynamic.py     # Uses non-existent imports
├── data_loader_dynamic.py   # Imports missing dataset
└── ... (mixed files)
```

### After (Clean)
```
services/gnn_service/
├── src/                     # All source code organized
│   ├── models/              # Clear module boundaries
│   ├── data/
│   ├── inference/
│   ├── training/
│   ├── schemas/
│   └── utils/
├── api/                    # FastAPI separated
├── config/                 # Configuration isolated
├── tests/                  # Test organization
└── _legacy/                # Old code archived
```

## Benefits

1. **Clear module boundaries** - каждый модуль имеет единственную ответственность
2. **No stub files** - все файлы содержат полную реализацию
3. **Testable** - модульная структура упрощает тестирование
4. **Production-ready** - следует best practices Python packaging
5. **Type-safe** - использование Pydantic v2 для всех схем
6. **Legacy isolated** - старый код изолирован, не мешает разработке

## Migration Status

- ✅ Created new src/ structure
- ✅ Moved stub files to _legacy/
- ⏳ Implementing core models (next step)
- ⏳ Implementing data pipeline
- ⏳ Implementing inference engine
- ⏳ Implementing training pipeline
- ⏳ Refactoring FastAPI application

## Next Steps

See [../../docs/GNN_SERVICE_ROADMAP.md](../../docs/GNN_SERVICE_ROADMAP.md) for detailed implementation plan.