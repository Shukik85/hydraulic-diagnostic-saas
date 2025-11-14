# services/rag_service/model_loader.py
"""
Загрузка и управление DeepSeek-R1-Distill-32B с vLLM.

UPDATED: Config-based, no hardcoded values.
"""
import logging
from typing import Optional
from datetime import datetime, timezone

from vllm import LLM, SamplingParams
from fastapi import HTTPException
import torch
import structlog

from config import config

logger = structlog.get_logger()


class DeepSeekModel:
    """
    Self-hosted DeepSeek-R1-Distill-32B с vLLM.
    
    Features:
    - PagedAttention for memory efficiency
    - Continuous batching
    - Tensor parallelism for multi-GPU
    - Optimized CUDA kernels
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None
    ):
        # Use config defaults if not specified
        self.model_name = model_name or config.MODEL_NAME
        self.tensor_parallel_size = tensor_parallel_size or config.TENSOR_PARALLEL_SIZE
        self.gpu_memory_utilization = gpu_memory_utilization or config.GPU_MEMORY_UTIL
        self.max_model_len = max_model_len or config.MAX_MODEL_LEN
        
        logger.info(
            "model_initialization_started",
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_util=self.gpu_memory_utilization,
            max_model_len=self.max_model_len
        )
        
        # GPU checks
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This service requires GPU.")
        
        gpu_count = torch.cuda.device_count()
        if gpu_count < self.tensor_parallel_size:
            raise RuntimeError(
                f"Requested {self.tensor_parallel_size} GPUs, "
                f"but only {gpu_count} available"
            )
        
        logger.info("gpu_detected", gpu_count=gpu_count)
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                "gpu_info",
                gpu_id=i,
                name=props.name,
                memory_gb=f"{props.total_memory / 1024**3:.1f}"
            )
        
        # Initialize vLLM
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
            download_dir=config.MODEL_PATH  # ✅ From config!
        )
        
        logger.info("model_loaded_successfully")
    
    def _validate_params(
        self,
        temperature: float,
        top_p: float,
        max_tokens: int
    ):
        """Validate generation parameters."""
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {temperature}")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0.0, 1.0], got {top_p}")
        if not 1 <= max_tokens <= config.MAX_MODEL_LEN:
            raise ValueError(
                f"max_tokens must be in [1, {config.MAX_MODEL_LEN}], got {max_tokens}"
            )
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list] = None
    ) -> str:
        """
        Generate response with reasoning.
        
        Raises:
            HTTPException: If generation fails
        """
        # Use config defaults
        max_tokens = max_tokens or config.DEFAULT_MAX_TOKENS
        temperature = temperature or config.DEFAULT_TEMPERATURE
        top_p = top_p or config.DEFAULT_TOP_P
        
        try:
            # Validate
            if not prompt or not prompt.strip():
                raise ValueError("prompt cannot be empty")
            
            self._validate_params(temperature, top_p, max_tokens)
            
            logger.debug(
                "generation_params",
                prompt_length=len(prompt),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["</думает>", "<|end|>"]
            )
            
            outputs = self.llm.generate([prompt], sampling_params)
            
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("vLLM returned empty outputs")
            
            return outputs[0].outputs[0].text
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error("gpu_oom", error=str(e))
            raise HTTPException(
                status_code=503,
                detail="GPU out of memory. Try reducing max_tokens or batch size."
            )
        
        except ValueError as e:
            logger.error("invalid_params", error=str(e))
            raise HTTPException(
                status_code=400,
                detail=f"Invalid parameters: {str(e)}"
            )
        
        except Exception as e:
            logger.error("generation_error", error=str(e), error_type=type(e).__name__)
            raise HTTPException(
                status_code=503,
                detail=f"Model inference failed: {str(e)}"
            )
    
    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> list[str]:
        """
        Batch generation for efficiency.
        
        Raises:
            HTTPException: If generation fails
        """
        max_tokens = max_tokens or config.DEFAULT_MAX_TOKENS
        temperature = temperature or config.DEFAULT_TEMPERATURE
        
        try:
            if not prompts:
                raise ValueError("prompts list cannot be empty")
            
            if len(prompts) > 10:
                logger.warning(
                    "large_batch_size",
                    batch_size=len(prompts),
                    message="Consider splitting into smaller batches"
                )
            
            self._validate_params(temperature, 0.9, max_tokens)
            
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            outputs = self.llm.generate(prompts, sampling_params)
            
            if len(outputs) != len(prompts):
                raise RuntimeError(
                    f"Expected {len(prompts)} outputs, got {len(outputs)}"
                )
            
            return [output.outputs[0].text for output in outputs]
            
        except Exception as e:
            logger.error("batch_generation_error", error=str(e))
            raise HTTPException(
                status_code=503,
                detail=f"Batch generation failed: {str(e)}"
            )


# Global model instance
_model: Optional[DeepSeekModel] = None


def get_model() -> DeepSeekModel:
    """
    Get global model instance (singleton).
    
    Returns:
        DeepSeekModel: Loaded model
        
    Raises:
        RuntimeError: If model initialization fails
    """
    global _model
    if _model is None:
        _model = DeepSeekModel(
            model_name=config.MODEL_NAME,
            tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=config.GPU_MEMORY_UTIL,
            max_model_len=config.MAX_MODEL_LEN
        )
    return _model
