#!/usr/bin/env python3
"""GPU Verification Script for GTX 1650 SUPER"""
import sys

def check_gpu():
    print("=" * 70)
    print("🔍 GPU VERIFICATION FOR GTX 1650 SUPER")
    print("=" * 70)
    print()

    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed!")
        return False

    cuda_available = torch.cuda.is_available()
    print(f"{'✅' if cuda_available else '❌'} CUDA available: {cuda_available}")

    if not cuda_available:
        print("\n⚠️  CUDA not available!")
        return False

    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ cuDNN version: {torch.backends.cudnn.version()}")
    print()

    props = torch.cuda.get_device_properties(0)
    print(f"GPU 0: {props.name}")
    print(f"  - Compute capability: {props.major}.{props.minor}")
    print(f"  - Total memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"  - Multi-processors: {props.multi_processor_count}")
    print()

    # Test tensor operation
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print("✅ GPU tensor operations working!")
        print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        return False

    print()
    print("=" * 70)
    print("✅ GPU VERIFICATION PASSED!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    sys.exit(0 if check_gpu() else 1)
