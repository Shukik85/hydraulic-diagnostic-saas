"""
Test script to verify memory optimizations work correctly.
"""

import torch
from memory_optimizations import (
    apply_memory_optimizations,
    MemoryOptimizer,
    clear_gpu_cache,
)
from model import create_model
from config import training_config, model_config


def test_memory_optimizations():
    """Test that memory optimizations reduce memory usage."""

    print("🧪 Testing Memory Optimizations")
    print("=" * 50)

    # Create a test model
    model = create_model()

    # Test memory usage without optimizations
    clear_gpu_cache()
    baseline_memory = get_memory_usage()

    # Create some test data
    batch_size = 4
    x = torch.randn(batch_size * 7, model_config.num_node_features).cuda()
    edge_index = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6, 0, 2, 3, 4, 5, 6]],
        dtype=torch.long,
    ).cuda()
    batch = torch.tensor([0] * 7 + [1] * 7 + [2] * 7 + [3] * 7).cuda()

    # Test forward pass without optimizations
    with torch.no_grad():
        model.eval()
        model(x, edge_index, batch)

    memory_without_opt = get_memory_usage() - baseline_memory
    print(f"Memory usage without optimizations: {memory_without_opt:.2f} GB")

    # Apply optimizations
    clear_gpu_cache()
    baseline_memory_opt = get_memory_usage()

    apply_memory_optimizations(model, training_config)

    # Test forward pass with optimizations
    with torch.no_grad():
        model.eval()
        model(x, edge_index, batch)

    memory_with_opt = get_memory_usage() - baseline_memory_opt
    print(f"Memory usage with optimizations: {memory_with_opt:.2f} GB")

    # Calculate savings
    savings = ((memory_without_opt - memory_with_opt) / memory_without_opt) * 100
    print(f"Memory savings: {savings:.1f}%")

    # Test MemoryOptimizer context manager
    print("\nTesting MemoryOptimizer context manager:")
    with MemoryOptimizer() as mem_opt:
        mem_opt.checkpoint("start")

        # Some memory-intensive operation
        large_tensor = torch.randn(1000, 1000).cuda()
        result = large_tensor @ large_tensor.T

        mem_opt.checkpoint("after_computation")

        del large_tensor, result

    print("✅ Memory optimizations test completed!")


def get_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def test_gradient_checkpointing():
    """Test gradient checkpointing functionality."""
    print("\n🧪 Testing Gradient Checkpointing")

    model = create_model()

    # Enable gradient checkpointing in config
    training_config.gpu_config.gradient_checkpointing = True

    apply_memory_optimizations(model, training_config)

    # Check if checkpointing was applied
    if hasattr(model, "gnn") and hasattr(model.gnn, "gat_layers"):
        for i, layer in enumerate(model.gnn.gat_layers):
            if hasattr(layer, "checkpoint"):
                print(f"GAT layer {i}: checkpointing = {layer.checkpoint}")
            else:
                print(f"GAT layer {i}: checkpointing not available")

    print("✅ Gradient checkpointing test completed!")


if __name__ == "__main__":
    test_memory_optimizations()
    test_gradient_checkpointing()
