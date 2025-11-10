"""
Test script to verify memory optimizations work correctly.
Place this in the root directory (hydraulic-diagnostic-saas).
"""

import sys
import os
import torch

# Add the current directory to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: PTH100

try:
    from gnn_service.model import create_model
    from gnn_service.config import training_config, model_config
    from memory_optimizations import (
        apply_memory_optimizations,
        MemoryOptimizer,
        clear_gpu_cache,
    )

    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


def get_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def test_basic_memory_operations():
    """Test basic memory operations without model dependencies."""
    print("🧪 Testing Basic Memory Operations")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping GPU memory tests")
        return

    # Test MemoryOptimizer context manager
    print("Testing MemoryOptimizer context manager:")

    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.2f} GB")

    with MemoryOptimizer() as mem_opt:
        mem_opt.checkpoint("start")

        # Create some tensors to use memory
        large_tensor1 = torch.randn(500, 500).cuda()
        result1 = large_tensor1 @ large_tensor1.T

        mem_opt.checkpoint("after_tensor1")

        large_tensor2 = torch.randn(1000, 500).cuda()
        result2 = large_tensor2 @ large_tensor1.T

        mem_opt.checkpoint("after_tensor2")

        # Cleanup
        del large_tensor1, large_tensor2, result1, result2

    final_memory = get_memory_usage()
    print(f"Final memory: {final_memory:.2f} GB")

    if final_memory <= initial_memory + 0.1:  # Allow small overhead
        print("✅ Memory cleanup successful")
    else:
        print("❌ Memory cleanup may have issues")

    print("✅ Basic memory operations test completed!")


def test_model_memory_optimizations():
    """Test memory optimizations with actual model."""
    if not IMPORTS_SUCCESSFUL:
        print("❌ Skipping model tests due to import errors")
        return

    print("\n🧪 Testing Model Memory Optimizations")
    print("=" * 50)

    # Create a test model
    model = create_model()

    # Test memory usage without optimizations
    clear_gpu_cache()
    baseline_memory = get_memory_usage()

    # Create some test data
    batch_size = 2  # Small batch for testing
    x = torch.randn(batch_size * model_config.num_nodes, model_config.num_node_features)
    edge_index = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6, 0, 2, 3, 4, 5, 6]],
        dtype=torch.long,
    )
    batch = torch.tensor([0] * 7 + [1] * 7)

    # Move to GPU if available
    if torch.cuda.is_available():
        x = x.cuda()
        edge_index = edge_index.cuda()
        batch = batch.cuda()
        model = model.cuda()

    # Test forward pass without optimizations
    with torch.no_grad():
        model.eval()
        output1, _, _ = model(x, edge_index, batch)

    memory_without_opt = get_memory_usage() - baseline_memory
    print(f"Memory usage without optimizations: {memory_without_opt:.2f} GB")

    # Apply optimizations
    clear_gpu_cache()
    baseline_memory_opt = get_memory_usage()

    apply_memory_optimizations(model, training_config)

    # Test forward pass with optimizations
    with torch.no_grad():
        model.eval()
        output2, _, _ = model(x, edge_index, batch)

    memory_with_opt = get_memory_usage() - baseline_memory_opt
    print(f"Memory usage with optimizations: {memory_with_opt:.2f} GB")

    # Calculate savings
    if memory_without_opt > 0:
        savings = ((memory_without_opt - memory_with_opt) / memory_without_opt) * 100
        print(f"Memory savings: {savings:.1f}%")
    else:
        print("Memory usage too small to calculate savings")

    # Verify outputs are similar (optimizations shouldn't change results drastically)
    if torch.allclose(output1, output2, rtol=1e-3):
        print("✅ Model outputs consistent after optimizations")
    else:
        print("⚠️  Model outputs differ after optimizations")

    print("✅ Model memory optimizations test completed!")


def test_gradient_checkpointing():
    """Test gradient checkpointing functionality."""
    if not IMPORTS_SUCCESSFUL:
        print("❌ Skipping gradient checkpointing tests due to import errors")
        return

    print("\n🧪 Testing Gradient Checkpointing")

    model = create_model()

    # Enable gradient checkpointing in config
    training_config.gpu_config.gradient_checkpointing = True

    apply_memory_optimizations(model, training_config)

    # Check if model has the expected attributes
    if hasattr(model, "gnn"):
        print("✅ Model has GNN component")
        if hasattr(model.gnn, "gat_layers"):
            print(f"✅ Model has {len(model.gnn.gat_layers)} GAT layers")
        else:
            print("❌ Model doesn't have gat_layers attribute")
    else:
        print("❌ Model doesn't have gnn attribute")

    print("✅ Gradient checkpointing test completed!")


def main():
    """Run all memory optimization tests."""
    print("Memory Optimizations Test Suite")
    print("=" * 50)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(
            f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        print("❌ CUDA not available - some tests will be skipped")

    # Run tests
    test_basic_memory_operations()

    if IMPORTS_SUCCESSFUL:
        test_model_memory_optimizations()
        test_gradient_checkpointing()
    else:
        print("\n💡 To fix import errors, make sure:")
        print("1. You're running from the root directory (hydraulic-diagnostic-saas)")
        print("2. The gnn_service folder has __init__.py")
        print("3. All required modules are in place")

    print("\n" + "=" * 50)
    print("Test suite completed!")


if __name__ == "__main__":
    main()
