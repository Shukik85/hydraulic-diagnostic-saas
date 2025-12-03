"""Model versioning schemas.

Pydantic models for model management and versioning.

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    """Model information.
    
    Attributes:
        path: Model file path
        version: Model version (extracted from filename)
        device: Device model is loaded on
        num_parameters: Number of model parameters
        size_mb: Model file size in MB
        loaded: Whether model is currently loaded
        loaded_at: When model was loaded
    
    Examples:
        >>> info = ModelInfo(
        ...     path="models/checkpoints/v2.0.0.ckpt",
        ...     version="2.0.0",
        ...     device="cuda:0",
        ...     num_parameters=2500000,
        ...     size_mb=45.3,
        ...     loaded=True
        ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "path": "models/checkpoints/v2.0.0.ckpt",
                "version": "2.0.0",
                "device": "cuda:0",
                "num_parameters": 2500000,
                "size_mb": 45.3,
                "loaded": True,
                "loaded_at": "2025-11-26T20:00:00Z",
                "compiled": True
            }
        }
    )

    path: str = Field(..., description="Model file path")
    version: str = Field(..., description="Model version")
    device: str = Field(..., description="Device (cpu/cuda)")
    num_parameters: int = Field(..., ge=0, description="Number of parameters")
    size_mb: float = Field(..., ge=0, description="File size (MB)")
    loaded: bool = Field(..., description="Currently loaded")
    loaded_at: datetime | None = Field(default=None, description="Load timestamp")
    compiled: bool = Field(default=False, description="torch.compile enabled")


class ModelVersion(BaseModel):
    """Model version metadata.
    
    Attributes:
        version: Semantic version
        path: Model checkpoint path
        created_at: Creation timestamp
        size_mb: File size
        num_parameters: Parameter count
        architecture: Architecture name
        is_current: Currently active model
    
    Examples:
        >>> version = ModelVersion(
        ...     version="2.0.0",
        ...     path="models/v2.0.0.ckpt",
        ...     size_mb=45.3,
        ...     num_parameters=2500000,
        ...     architecture="GATv2-ARMA-LSTM"
        ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "version": "2.0.0",
                "path": "models/checkpoints/v2.0.0.ckpt",
                "created_at": "2025-11-21T20:00:00Z",
                "size_mb": 45.3,
                "num_parameters": 2500000,
                "architecture": "GATv2-ARMA-LSTM",
                "is_current": True
            }
        }
    )

    version: str = Field(..., description="Semantic version (e.g., 2.0.0)")
    path: str = Field(..., description="Checkpoint path")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    size_mb: float = Field(..., ge=0, description="File size (MB)")
    num_parameters: int = Field(..., ge=0, description="Parameter count")
    architecture: str = Field(
        default="GATv2-ARMA-LSTM",
        description="Architecture name"
    )
    is_current: bool = Field(
        default=False,
        description="Currently active model"
    )
