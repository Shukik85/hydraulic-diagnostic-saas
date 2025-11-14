# services/rag_service/model_loader.py
"""
Загрузка и управление DeepSeek-R1-Distill-32B с vLLM.

FIXED:
- Added comprehensive error handling for all methods
- Added parameter validation
- Added detailed logging
- Replaced datetime.utcnow() with datetime.now(timezone.utc)
- Better exception messages with context
"""
import os
import logging
from typing import Optional
from datetime import datetime, timezone

from vllm import LLM, SamplingParams
import torch
from fastapi import HTTPException

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
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        
        logger.info(f"Initializing DeepSeekModel...")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"  GPU memory utilization: {gpu_memory_utilization}")
        logger.info(f"  Max model length: {max_model_len}")
        logger.info(f"  dtype: bfloat16")
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This service requires GPU.")
        
        gpu_count = torch.cuda.device_count()
        if gpu_count < tensor_parallel_size:
            raise RuntimeError(
                f"Requested {tensor_parallel_size} GPUs, "
                f"but only {gpu_count} available"
            )
        
        logger.info(f"Found {gpu_count} GPUs:")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU {i}: {props.name} - "
                f"{props.total_memory / 1024**3:.1f} GB"
            )
        
        # Initialize vLLM
        try:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype="bfloat16",  # Эффективнее float16 на A100
                trust_remote_code=True,
                download_dir="/app/models"
            )
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vLLM: {e}", exc_info=True)
            raise RuntimeError(f"vLLM initialization failed: {str(e)}")
    
    def _validate_params(
        self,
        temperature: float,
        top_p: float,
        max_tokens: int
    ):
        """
        Валидация параметров генерации.
        
        Raises:
            ValueError: If parameters are invalid
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                f"temperature must be in [0.0, 2.0], got {temperature}"
            )
        
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(
                f"top_p must be in [0.0, 1.0], got {top_p}"
            )
        
        if not 1 <= max_tokens <= self.max_model_len:
            raise ValueError(
                f"max_tokens must be in [1, {self.max_model_len}], "
                f"got {max_tokens}"
            )
    
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
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            stop: Stop sequences
            
        Returns:
            str: Generated text with reasoning steps
            
        Raises:
            ValueError: If parameters are invalid
            HTTPException: If generation fails
        """
        try:
            # Валидация параметров
            self._validate_params(temperature, top_p, max_tokens)
            
            if not prompt or not prompt.strip():
                raise ValueError("prompt cannot be empty")
            
            # Логирование запроса
            logger.debug(
                f"Generating: temp={temperature}, top_p={top_p}, "
                f"max_tokens={max_tokens}, prompt_len={len(prompt)}"
            )
            
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["</думает>", "<|end|>"]
            )
            
            # Генерация
            outputs = self.llm.generate([prompt], sampling_params)
            
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("vLLM returned empty outputs")
            
            result = outputs[0].outputs[0].text
            
            logger.debug(f"Generated {len(result)} characters")
            return result
            
        except ValueError as e:
            logger.error(f"Invalid parameters: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid parameters: {str(e)}"
            )
        
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM during generation: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="GPU out of memory. Try reducing max_tokens or wait for current requests to complete."
            )
        
        except RuntimeError as e:
            logger.error(f"vLLM runtime error: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail=f"Model inference failed: {str(e)}"
            )
        
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {str(e)}"
            )
    
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
            
        Raises:
            ValueError: If prompts list is invalid
            HTTPException: If generation fails
        """
        try:
            # Валидация
            if not prompts:
                raise ValueError("prompts list cannot be empty")
            
            if not all(isinstance(p, str) and p.strip() for p in prompts):
                raise ValueError("All prompts must be non-empty strings")
            
            if len(prompts) > 10:
                logger.warning(
                    f"Large batch size: {len(prompts)} prompts. "
                    f"Consider splitting into smaller batches."
                )
            
            self._validate_params(temperature, 0.9, max_tokens)
            
            logger.info(f"Batch generating {len(prompts)} prompts...")
            
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            outputs = self.llm.generate(prompts, sampling_params)
            
            # Проверка результатов
            if len(outputs) != len(prompts):
                raise RuntimeError(
                    f"Expected {len(prompts)} outputs, got {len(outputs)}"
                )
            
            results = [output.outputs[0].text for output in outputs]
            
            logger.info(f"✅ Batch generated {len(results)} responses")
            return results
            
        except ValueError as e:
            logger.error(f"Invalid batch parameters: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid parameters: {str(e)}"
            )
        
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM during batch generation: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail=f"GPU out of memory with batch size {len(prompts)}. Try smaller batches."
            )
        
        except Exception as e:
            logger.error(f"Batch generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
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
        try:
            _model = DeepSeekModel(
                tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "2")),
                gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTIL", "0.90")),
                max_model_len=int(os.getenv("MAX_MODEL_LEN", "8192"))
            )
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}", exc_info=True)
            raise
    
    return _model
