"""FastAPI application for GNN inference service.

Endpoints:
- POST /predict - Single prediction
- POST /predict/batch - Batch prediction
- GET /health - Health check
- GET /stats - Service statistics

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations
