"""Model utilities.

Утилиты для:
- Инициализация моделей
- Checkpoint management (save/load)
- Model summary и статистика
- Parameter counting

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def initialize_model(model: nn.Module, method: str = "xavier_uniform") -> nn.Module:
    """Inicialize weights of the model.

    Args:
        model: PyTorch model
        method: Initialization method (xavier_uniform, kaiming_normal, orthogonal)

    Returns:
        model: Model with initialized weights

    Examples:
        >>> model = UniversalTemporalGNN(...)
        >>> model = initialize_model(model, method="xavier_uniform")
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if method == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif method == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif method == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif method == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    logger.info(f"Model initialized with {method}")
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    loss: float,
    metrics: dict[str, float],
    save_path: str | Path,
    model_config: dict[str, Any] | None = None,
) -> None:
    """Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer (or None)
        epoch: Current epoch
        loss: Loss value
        metrics: Dictionary with metrics
        save_path: Path for saving
        model_config: Model configuration

    Examples:
        >>> save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=100,
        ...     loss=0.0234,
        ...     metrics={"health_mae": 0.045},
        ...     save_path="checkpoints/model_epoch100.ckpt"
        ... )
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "metrics": metrics,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if model_config is not None:
        checkpoint["model_config"] = model_config

    # Add PyTorch and CUDA versions
    checkpoint["pytorch_version"] = torch.__version__
    checkpoint["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: PyTorch model for loading weights
        optimizer: Optimizer for loading state (optional)
        device: Device for loading

    Returns:
        checkpoint: Dictionary with checkpoint information

    Examples:
        >>> model = UniversalTemporalGNN(...)
        >>> checkpoint = load_checkpoint("checkpoints/best.ckpt", model)
        >>> print(f"Loaded epoch {checkpoint['epoch']}")
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # type: ignore[no-any-return]

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model weights loaded from {checkpoint_path}")

    # Load optimizer state (if provided)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state loaded")

    # Log metadata
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")
    logger.info(f"PyTorch version: {checkpoint.get('pytorch_version', 'unknown')}")

    return checkpoint


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count number of model parameters.

    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters

    Returns:
        count: Number of parameters

    Examples:
        >>> model = UniversalTemporalGNN(in_channels=12, hidden_channels=128)
        >>> total = count_parameters(model)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total:,} | Trainable: {trainable:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(
    model: nn.Module, _input_size: tuple[int, ...] | None = None, _device: str = "cpu"
) -> dict[str, Any]:
    """Generate model summary.

    Args:
        model: PyTorch model
        _input_size: Input tensor size (reserved for future FLOPs estimation)
        _device: Device for computation (reserved for future use)

    Returns:
        summary: Dictionary with model metrics

    Examples:
        >>> model = UniversalTemporalGNN(in_channels=12, hidden_channels=128)
        >>> summary = model_summary(model)
        >>> print(f"Parameters: {summary['total_params']:,}")
        >>> print(f"Memory: {summary['memory_mb']:.2f} MB")
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    # Calculate memory footprint (approximate)
    param_memory_mb = (total_params * 4) / (1024**2)  # 4 bytes per float32

    summary: dict[str, Any] = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "memory_mb": param_memory_mb,
    }

    # Layer breakdown
    layer_summary: list[dict[str, str | int]] = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                layer_summary.append(
                    {"name": name, "type": module.__class__.__name__, "params": num_params}
                )

    summary["layers"] = layer_summary

    # Model type
    summary["model_type"] = model.__class__.__name__

    return summary


def print_model_summary(model: nn.Module) -> None:
    """Print human-readable model summary.

    Args:
        model: PyTorch model

    Examples:
        >>> model = UniversalTemporalGNN(in_channels=12, hidden_channels=128)
        >>> print_model_summary(model)

        Model: UniversalTemporalGNN
        ================================
        Total Parameters: 1,234,567
        Trainable Parameters: 1,234,567
        Memory Footprint: 4.71 MB
        ================================
    """
    summary = model_summary(model)

    # Top 10 layers by parameter count
    if "layers" in summary:
        sorted_layers = sorted(summary["layers"], key=lambda x: x["params"], reverse=True)[:10]

        for _layer in sorted_layers:
            pass


def get_device(model: nn.Module) -> torch.device:
    """Get device of model.

    Args:
        model: PyTorch model

    Returns:
        device: torch.device ('cuda:0', 'cpu', etc.)

    Examples:
        >>> model = UniversalTemporalGNN(...).to('cuda')
        >>> device = get_device(model)
        >>> print(device)  # cuda:0
    """
    return next(model.parameters()).device


def model_to_device(model: nn.Module, device: str | torch.device) -> nn.Module:
    """Move model to device with logging.

    Args:
        model: PyTorch model
        device: Target device

    Returns:
        model: Model on specified device

    Examples:
        >>> model = UniversalTemporalGNN(...)
        >>> model = model_to_device(model, 'cuda:0')
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)

    logger.info(f"Model moved to {device_obj}")

    # Log GPU memory if CUDA
    if device_obj.type == "cuda":
        memory_allocated = torch.cuda.memory_allocated(device_obj) / (1024**2)
        logger.info(f"GPU memory allocated: {memory_allocated:.2f} MB")

    return model


def freeze_layers(model: nn.Module, layer_names: list[str]) -> nn.Module:
    """Freeze specific layers (for fine-tuning).

    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze

    Returns:
        model: Model with frozen layers

    Examples:
        >>> model = UniversalTemporalGNN(...)
        >>> model = freeze_layers(model, ["gat_layers.0", "gat_layers.1"])
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if name.startswith(layer_name):
                param.requires_grad = False
                logger.info(f"Frozen parameter: {name}")

    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    logger.info(f"Trainable parameters: {trainable:,} / {total:,}")

    return model


def unfreeze_all_layers(model: nn.Module) -> nn.Module:
    """Unfreeze all layers.

    Args:
        model: PyTorch model

    Returns:
        model: Model with unfrozen parameters
    """
    for param in model.parameters():
        param.requires_grad = True

    logger.info("All layers unfrozen")
    return model
