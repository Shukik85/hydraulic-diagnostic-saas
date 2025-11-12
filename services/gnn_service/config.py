"""
Configuration for GTX 1650 SUPER (4GB VRAM)
Optimized for production use
"""
import torch

# Model Configuration (reduced for 4GB VRAM)
MODEL_CONFIG = {
    "in_channels": 10,
    "hidden_channels": 32,  # Reduced from 64
    "out_channels": 1,
    "num_layers": 2,  # Reduced from 3
    "heads": 4,  # Reduced from 8
    "dropout": 0.1,
}

# Inference Configuration
INFERENCE_CONFIG = {
    "use_mixed_precision": True,  # FP16 saves 50% memory
    "compile_model": True,  # torch.compile optimization
    "max_batch_size": 5,  # Limited by 4GB VRAM
}

# GPU Configuration
GPU_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cuda_memory_fraction": 0.85,  # Use max 85% (3.4GB)
    "enable_cudnn_benchmark": True,
    "torch_compile_mode": "reduce-overhead",
}

# Memory Management
MEMORY_CONFIG = {
    "clear_cache_after_inference": True,
    "max_cached_graphs": 10,
}

def apply_gpu_config():
    """Apply GPU optimization settings"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(
            GPU_CONFIG["cuda_memory_fraction"],
            device=0
        )
        torch.backends.cudnn.benchmark = GPU_CONFIG["enable_cudnn_benchmark"]
        print(f"✅ GPU configured for GTX 1650 SUPER")
        print(f"   Max VRAM: {GPU_CONFIG['cuda_memory_fraction'] * 4:.1f} GB")
