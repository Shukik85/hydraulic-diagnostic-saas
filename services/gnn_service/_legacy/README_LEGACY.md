# Legacy Files Directory

**Created:** 2025-11-21
**Branch:** feature/gnn-service-production-ready

## Purpose

This directory contains old, stub, and deprecated files from the GNN service that are not part of the production-ready implementation.

## Files Moved to Legacy

### Stub Files (Comment Placeholders Only)
- `model_dynamic_gnn_stub.py` - Universal Dynamic GNN stub
- `dataset_dynamic_stub.py` - DynamicTemporalGraphDataset stub
- `schemas_stub.py` - Dynamic data schemas stub

### Deprecated Implementation Files
See existing files in this directory:
- `model_universal_temporal.py` - Old model implementation
- `train_universal.py` - Old training script
- `inference_service.py` - Old inference engine
- `benchmark_optimizations.py` - Benchmark code

## Migration Path

Old structure:
```
services/gnn_service/
├── model_dynamic_gnn.py (stub)
├── dataset_dynamic.py (stub)
├── schemas.py (stub)
└── ...
```

New production structure:
```
services/gnn_service/
├── src/
│   ├── models/
│   │   └── gnn_model.py (full implementation)
│   ├── data/
│   │   ├── dataset.py (full implementation)
│   │   └── loader.py
│   ├── schemas/
│   │   └── models.py (pydantic schemas)
│   └── ...
├── _legacy/ (this directory)
└── main.py
```

## Note

These files are kept for reference only. Do not use in production code.