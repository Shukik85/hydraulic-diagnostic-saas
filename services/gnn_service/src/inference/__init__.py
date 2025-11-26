"""Inference module for production deployment.

Components:
- ModelManager - Model loading, caching, versioning
- InferenceEngine - Batch inference, GPU optimization
- FastAPI integration - REST API endpoints

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from src.inference.model_manager import ModelManager

__all__ = [
    "ModelManager",
]
