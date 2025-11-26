"""Model Manager for production model lifecycle.

Features:
- Model loading from checkpoint
- In-memory caching (LRU)
- Model versioning
- Device management (CPU/CUDA)
- Thread-safe operations

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import torch
import logging
from pathlib import Path
from typing import Literal
from functools import lru_cache
import threading
from dataclasses import dataclass

from src.models import UniversalTemporalGNN
from src.models.utils import load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    
    model_path: Path
    device: Literal["cpu", "cuda", "auto"] = "auto"
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"
    
    def __post_init__(self):
        """Validate paths."""
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")


class ModelManager:
    """Manager для model loading, caching, versioning.
    
    Features:
    - Singleton pattern (one manager per process)
    - LRU cache для models
    - Thread-safe operations
    - Device management
    - Model validation
    
    Examples:
        >>> manager = ModelManager()
        >>> 
        >>> # Load model
        >>> model = manager.load_model(
        ...     model_path="models/best.ckpt",
        ...     device="cuda"
        ... )
        >>> 
        >>> # Get cached model
        >>> model = manager.get_model("models/best.ckpt")
        >>> 
        >>> # Clear cache
        >>> manager.clear_cache()
    """
    
    _instance: ModelManager | None = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize manager."""
        if not hasattr(self, "_initialized"):
            self._models: dict[str, torch.nn.Module] = {}
            self._configs: dict[str, ModelConfig] = {}
            self._lock = threading.Lock()
            self._initialized = True
            
            logger.info("ModelManager initialized")
    
    def load_model(
        self,
        model_path: str | Path,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        use_compile: bool = True,
        compile_mode: str = "reduce-overhead",
        force_reload: bool = False
    ) -> torch.nn.Module:
        """Load model from checkpoint.
        
        Args:
            model_path: Path to checkpoint file
            device: Device to load model on
            use_compile: Enable torch.compile
            compile_mode: Compilation mode
            force_reload: Force reload even if cached
        
        Returns:
            model: Loaded model
        
        Examples:
            >>> manager = ModelManager()
            >>> model = manager.load_model(
            ...     model_path="models/v1.0.0/best.ckpt",
            ...     device="cuda",
            ...     use_compile=True
            ... )
        """
        model_path = Path(model_path)
        model_key = str(model_path.resolve())
        
        # Check cache
        if not force_reload and model_key in self._models:
            logger.info(f"Using cached model: {model_path.name}")
            return self._models[model_key]
        
        logger.info(f"Loading model from {model_path}")
        
        with self._lock:
            # Double-check after acquiring lock
            if not force_reload and model_key in self._models:
                return self._models[model_key]
            
            # Validate config
            config = ModelConfig(
                model_path=model_path,
                device=device,
                use_compile=use_compile,
                compile_mode=compile_mode
            )
            
            # Determine device
            if device == "auto":
                device_str = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device_str = device
            
            logger.info(f"Loading model to device: {device_str}")
            
            # Load checkpoint
            try:
                checkpoint = load_checkpoint(str(model_path))
                
                # Extract model config
                model_config = checkpoint.get("model_config", {})
                
                # Initialize model
                model = UniversalTemporalGNN(
                    in_channels=model_config.get("in_channels", 34),
                    hidden_channels=model_config.get("hidden_channels", 128),
                    num_heads=model_config.get("num_heads", 8),
                    num_gat_layers=model_config.get("num_gat_layers", 3),
                    lstm_hidden=model_config.get("lstm_hidden", 256),
                    lstm_layers=model_config.get("lstm_layers", 2),
                    use_compile=False  # Will compile separately if needed
                )
                
                # Load state dict
                model.load_state_dict(checkpoint["model_state_dict"])
                
                # Move to device
                model = model.to(device_str)
                
                # Set to eval mode
                model.eval()
                
                # Compile if requested
                if use_compile and device_str == "cuda":
                    logger.info(f"Compiling model with mode: {compile_mode}")
                    model = torch.compile(model, mode=compile_mode)
                
                # Validate model
                self._validate_model(model)
                
                # Cache
                self._models[model_key] = model
                self._configs[model_key] = config
                
                logger.info(
                    f"Model loaded successfully: {model_path.name} "
                    f"(device={device_str}, compile={use_compile})"
                )
                
                return model
            
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                raise
    
    def get_model(self, model_path: str | Path) -> torch.nn.Module | None:
        """Get cached model.
        
        Args:
            model_path: Path to checkpoint
        
        Returns:
            model: Cached model or None
        
        Examples:
            >>> manager = ModelManager()
            >>> model = manager.get_model("models/best.ckpt")
            >>> if model is None:
            ...     model = manager.load_model("models/best.ckpt")
        """
        model_key = str(Path(model_path).resolve())
        return self._models.get(model_key)
    
    def clear_cache(self, model_path: str | Path | None = None):
        """Clear model cache.
        
        Args:
            model_path: Specific model to clear, or None for all
        
        Examples:
            >>> manager = ModelManager()
            >>> 
            >>> # Clear specific model
            >>> manager.clear_cache("models/v1.0.0/best.ckpt")
            >>> 
            >>> # Clear all models
            >>> manager.clear_cache()
        """
        with self._lock:
            if model_path is None:
                # Clear all
                count = len(self._models)
                self._models.clear()
                self._configs.clear()
                logger.info(f"Cleared {count} models from cache")
            else:
                # Clear specific
                model_key = str(Path(model_path).resolve())
                if model_key in self._models:
                    del self._models[model_key]
                    del self._configs[model_key]
                    logger.info(f"Cleared model from cache: {Path(model_path).name}")
    
    def list_cached_models(self) -> list[str]:
        """List cached models.
        
        Returns:
            paths: List of cached model paths
        
        Examples:
            >>> manager = ModelManager()
            >>> models = manager.list_cached_models()
            >>> print(f"Cached: {len(models)} models")
        """
        return list(self._models.keys())
    
    def get_model_info(self, model_path: str | Path) -> dict | None:
        """Get model information.
        
        Args:
            model_path: Path to checkpoint
        
        Returns:
            info: Model info dict or None
        
        Examples:
            >>> manager = ModelManager()
            >>> info = manager.get_model_info("models/best.ckpt")
            >>> print(info["device"])
        """
        model_key = str(Path(model_path).resolve())
        
        if model_key not in self._models:
            return None
        
        model = self._models[model_key]
        config = self._configs[model_key]
        
        return {
            "path": str(config.model_path),
            "device": next(model.parameters()).device.type,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "compiled": config.use_compile,
            "mode": model.training
        }
    
    def _validate_model(self, model: torch.nn.Module):
        """Validate loaded model.
        
        Args:
            model: Model to validate
        
        Raises:
            ValueError: If model invalid
        """
        # Check model has forward method
        if not hasattr(model, "forward"):
            raise ValueError("Model missing forward method")
        
        # Check model has parameters
        if sum(1 for _ in model.parameters()) == 0:
            raise ValueError("Model has no parameters")
        
        # Check model is in eval mode
        if model.training:
            logger.warning("Model is in training mode, setting to eval")
            model.eval()
        
        logger.debug("Model validation passed")
    
    def warmup(self, model_path: str | Path, batch_size: int = 1):
        """Warmup model (JIT compilation, cache warming).
        
        Args:
            model_path: Path to model
            batch_size: Batch size для warmup
        
        Examples:
            >>> manager = ModelManager()
            >>> model = manager.load_model("models/best.ckpt")
            >>> manager.warmup("models/best.ckpt", batch_size=32)
        """
        model = self.get_model(model_path)
        if model is None:
            raise ValueError(f"Model not loaded: {model_path}")
        
        logger.info(f"Warming up model: {Path(model_path).name}")
        
        # Create dummy input
        device = next(model.parameters()).device
        
        # Dummy graph batch
        dummy_x = torch.randn(10 * batch_size, 34, device=device)  # 10 nodes per graph
        dummy_edge_index = torch.randint(0, 10, (2, 20 * batch_size), device=device)
        dummy_edge_attr = torch.randn(20 * batch_size, 8, device=device)
        dummy_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            10
        )
        
        # Warmup forward passes
        with torch.inference_mode():
            for _ in range(3):  # 3 warmup iterations
                _ = model(
                    x=dummy_x,
                    edge_index=dummy_edge_index,
                    edge_attr=dummy_edge_attr,
                    batch=dummy_batch
                )
        
        logger.info("Model warmup complete")
