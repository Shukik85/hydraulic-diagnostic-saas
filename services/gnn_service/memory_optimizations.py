"""
Additional memory optimizations for low VRAM GPUs.
"""

import torch


def apply_memory_optimizations(model, config):
    """Apply memory saving techniques."""

    # Gradient checkpointing for GNN layers
    if config.gpu_config.gradient_checkpointing and (
        hasattr(model, "gnn") and hasattr(model.gnn, "gat_layers")
    ):
        for layer in model.gnn.gat_layers:
            layer.checkpoint = True


def clear_gpu_cache():
    """Clear GPU cache and reset memory stats."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()


def get_memory_usage():
    """Get current memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


class MemoryOptimizer:
    """Context manager for memory optimization."""

    def __enter__(self):
        clear_gpu_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_gpu_cache()

    def checkpoint(self, name=""):
        """Checkpoint memory usage."""
        if torch.cuda.is_available():
            memory_gb = get_memory_usage()
            print(f"Memory checkpoint '{name}': {memory_gb:.2f} GB")
