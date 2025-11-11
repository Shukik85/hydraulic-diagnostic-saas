"""
Script to check GPU availability and configure training accordingly.
"""

import torch


def check_gpu_setup():
    """Check GPU setup and provide recommendations."""

    print("🔍 GPU Configuration Check")
    print("=" * 50)

    # Basic CUDA check
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅' if cuda_available else '❌'}")

    if cuda_available:
        _extracted_from_check_gpu_setup_13()
    else:
        print("\n💡 CPU-only training recommendations:")
        print("✅ Use smaller batch sizes (8-16)")
        print("✅ Set num_workers=0 for DataLoader")
        print("✅ Consider cloud GPU for faster training")

    # Check for mixed precision support
    if cuda_available:
        compute_capability = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

        if compute_capability[0] >= 7:
            print("✅ GPU supports mixed precision training (AMP)")
        else:
            print("⚠️  GPU may have limited mixed precision support")

    print("=" * 50)


# TODO Rename this here and in `check_gpu_setup`
def _extracted_from_check_gpu_setup_13():
    # GPU information
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")

    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"GPU {i}: {gpu_name} ({memory:.1f} GB)")

    # CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA Version: {cuda_version}")

    # Memory usage
    current_memory = torch.cuda.memory_allocated() / 1e9
    reserved_memory = torch.cuda.memory_reserved() / 1e9
    print(f"Current GPU memory: {current_memory:.2f} GB")
    print(f"Reserved GPU memory: {reserved_memory:.2f} GB")

    # Performance recommendations
    print("\n💡 Performance Recommendations:")

    if device_count > 1:
        print("✅ Multiple GPUs detected - consider using DistributedDataParallel")
    else:
        print("✅ Single GPU - optimized settings applied")

    if memory >= 16:
        print("✅ High memory GPU - can use larger batch sizes")
    elif memory >= 8:
        print("⚠️  Medium memory GPU - moderate batch sizes recommended")
    else:
        print("❌ Low memory GPU - consider gradient accumulation")


def get_optimized_config():
    """Get optimized configuration based on GPU availability."""

    config = {
        "batch_size": 32 if torch.cuda.is_available() else 8,
        "num_workers": 4 if torch.cuda.is_available() else 0,
        "mixed_precision": torch.cuda.is_available(),
        "gradient_accumulation": (1 if torch.cuda.is_available() else 2),
    }

    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        if memory_gb >= 16:
            config["batch_size"] = 64
            config["gpu_batch_multiplier"] = 4
        elif memory_gb >= 8:
            config["batch_size"] = 32
            config["gpu_batch_multiplier"] = 2
        else:
            config["batch_size"] = 16
            config["gpu_batch_multiplier"] = 1
            config["gradient_accumulation"] = 2  # Use accumulation for low memory

    return config


if __name__ == "__main__":
    check_gpu_setup()

    optimized_config = get_optimized_config()
    print("\n🎯 Recommended Configuration:")
    for key, value in optimized_config.items():
        print(f"  {key}: {value}")

    print("\n💡 To apply these settings, update training_config in config.py")
