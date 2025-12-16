"""Multi-model registry with A/B testing support.

Provides:
- ModelConfig: Configuration for individual model versions
- ModelRegistry: Registry with traffic-based routing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model version configuration.

    Attributes:
        path: Model checkpoint path
        version: Version identifier
        traffic: Traffic split ratio [0.0, 1.0]
        device: Device override (optional)
        enabled: Whether model is active

    Examples:
        >>> config = ModelConfig(
        ...     path="models/v2.0.0.ckpt",
        ...     version="v2",
        ...     traffic=0.2
        ... )
    """

    path: str
    version: str
    traffic: float = 1.0
    device: str | None = None
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration.

        Note: Path.exists() check here is acceptable for early validation.
        Model loading happens lazily during engine initialization.
        """
        if not 0.0 <= self.traffic <= 1.0:
            msg = f"traffic must be [0,1], got {self.traffic}"
            raise ValueError(msg)
        if not Path(self.path).exists():
            msg = f"Model not found: {self.path}"
            raise FileNotFoundError(msg)


class ModelRegistry:
    """Multi-model registry with A/B testing.

    Features:
    - Traffic-based routing (consistent hashing)
    - Model versioning
    - Runtime model swapping

    Examples:
        >>> registry = ModelRegistry()
        >>> registry.register("v1", ModelConfig(path="v1.ckpt", traffic=0.8))
        >>> registry.register("v2", ModelConfig(path="v2.ckpt", traffic=0.2))
        >>> version = registry.select_model("req_123")  # Consistent routing
    """

    def __init__(self):
        """Initialize empty registry."""
        self._models: dict[str, ModelConfig] = {}
        self._loaded_models: dict[str, Any] = {}

    def register(self, version: str, config: ModelConfig) -> None:
        """Register model version.

        Args:
            version: Version identifier
            config: Model configuration
        """
        self._models[version] = config
        logger.info(
            f"Registered model '{version}'",
            extra={"traffic": config.traffic},
        )

    def select_model(self, request_id: str) -> str:
        """Select model version based on traffic split.

        Uses consistent hashing to ensure same request_id always routes
        to the same model version (unless traffic weights change).

        Args:
            request_id: Request identifier for hashing

        Returns:
            Selected model version

        Raises:
            ValueError: If no enabled models
        """
        enabled = {v: c for v, c in self._models.items() if c.enabled}
        if not enabled:
            msg = "No enabled models"
            raise ValueError(msg)
        if len(enabled) == 1:
            return next(iter(enabled))

        # Consistent hashing
        hash_val = hash(request_id) % 100
        cumulative = 0.0
        for version, config in enabled.items():
            cumulative += config.traffic * 100
            if hash_val < cumulative:
                return version
        return next(iter(enabled))

    def get_model(self, version: str) -> Any:
        """Get loaded model instance.

        Args:
            version: Model version

        Returns:
            Loaded model

        Raises:
            KeyError: If model not loaded
        """
        if version not in self._loaded_models:
            msg = f"Model '{version}' not loaded"
            raise KeyError(msg)
        return self._loaded_models[version]

    def set_loaded_model(self, version: str, model: Any) -> None:
        """Store loaded model instance.

        Args:
            version: Model version
            model: Loaded model instance
        """
        self._loaded_models[version] = model

    def get_all_versions(self) -> list[str]:
        """Get all registered versions.

        Returns:
            List of version identifiers
        """
        return list(self._models.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_versions": len(self._models),
            "enabled_versions": sum(1 for c in self._models.values() if c.enabled),
            "loaded_versions": len(self._loaded_models),
            "traffic_split": {v: c.traffic for v, c in self._models.items() if c.enabled},
        }
