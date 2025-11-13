# services/rag_service/model_loader.py
"""
Загрузка и управление DeepSeek-R1-Distill-32B с vLLM.
"""
import os
import logging
from typing import Optional
from vllm import LLM, SamplingParams
import torch

logger = logging.getLogger(__name__)


class DeepSeekModel:
    """
    Self-hosted DeepSeek-R1-Distill-32B с vLLM для 2x A100.
    
    vLLM features:
    - PagedAttention для эффективной памяти
    - Continuous batching
    - Tensor parallelism для multi-GPU
    - Оптимизированные CUDA kernels
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        tensor_parallel_size: int = 2,  # 2x A100
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        
        logger.info(f"Loading {model_name} with {tensor_parallel_size} GPUs...")
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        gpu_count = torch.cuda.device_count()
        if gpu_count < tensor_parallel_size:
            raise RuntimeError(
                f"Requested {tensor_parallel_size} GPUs, "
                f"but only {gpu_count} available"
            )
        
        logger.info(f"Found {gpu_count} GPUs")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"GPU {i}: {props.name} - "
                f"{props.total_memory / 1024**3:.1f} GB"
            )
        
        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",  # Эффективнее float16 на A100
            trust_remote_code=True,
            download_dir="/app/models"
        )
        
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None
    ) -> str:
        """
        Generate response with reasoning.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            
        Returns:
            str: Generated text with reasoning steps
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or ["</думает>", "<|end|>"]
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> list[str]:
        """
        Batch generation для efficiency.
        
        Args:
            prompts: List of prompts
            max_tokens: Max tokens per response
            temperature: Sampling temperature
            
        Returns:
            list[str]: Generated responses
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


# Global model instance
_model: Optional[DeepSeekModel] = None


def get_model() -> DeepSeekModel:
    """
    Get global model instance (singleton).
    
    Returns:
        DeepSeekModel: Loaded model
    """
    global _model
    if _model is None:
        _model = DeepSeekModel(
            tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "2")),
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTIL", "0.90")),
            max_model_len=int(os.getenv("MAX_MODEL_LEN", "8192"))
        )
    return _model
