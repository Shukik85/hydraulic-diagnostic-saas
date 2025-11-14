# services/rag_service/config.py
"""
Centralized configuration для RAG Service.
All environment variables и constants в одном месте.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class RAGServiceConfig(BaseSettings):
    """
    RAG Service configuration с type checking и validation.
    
    All values can be overridden via environment variables.
    Priority: ENV > .env file > defaults
    """
    
    # === Service Info ===
    SERVICE_NAME: str = "Hydraulic Diagnostic RAG"
    SERVICE_VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", description="dev|staging|production")
    
    # === Model Settings ===
    MODEL_NAME: str = Field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        description="Hugging Face model identifier"
    )
    MODEL_PATH: str = Field(
        default="/app/models",
        description="Local path for model storage"
    )
    MAX_MODEL_LEN: int = Field(
        default=8192,
        ge=512,
        le=32768,
        description="Maximum context length"
    )
    TENSOR_PARALLEL_SIZE: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of GPUs for tensor parallelism"
    )
    GPU_MEMORY_UTIL: float = Field(
        default=0.90,
        ge=0.1,
        le=0.99,
        description="GPU memory utilization fraction"
    )
    
    # === Generation Defaults ===
    DEFAULT_MAX_TOKENS: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Default max tokens for generation"
    )
    DEFAULT_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature"
    )
    DEFAULT_TOP_P: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Default nucleus sampling parameter"
    )
    
    # === Vector Store Settings ===
    VECTOR_STORE_PATH: str = Field(
        default="/app/vector_store",
        description="Path for FAISS indices and documents"
    )
    VECTOR_STORE_NAME: str = Field(
        default="hydraulics",
        description="Name of the knowledge base index"
    )
    EMBEDDING_MODEL: str = Field(
        default="intfloat/multilingual-e5-large",
        description="Embeddings model"
    )
    EMBEDDING_DIM: int = Field(
        default=1024,
        description="Embedding vector dimensions"
    )
    
    # === S3 Settings (Optional) ===
    S3_ENABLED: bool = Field(
        default=False,
        description="Enable S3 backup/restore"
    )
    S3_BUCKET: Optional[str] = Field(
        default=None,
        description="S3 bucket name for KB backup"
    )
    S3_PREFIX: str = Field(
        default="knowledge_base/",
        description="S3 key prefix"
    )
    S3_REGION: str = Field(
        default="eu-central-1",
        description="AWS region"
    )
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # === Rate Limiting ===
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting middleware"
    )
    # Per-endpoint limits
    RATE_LIMIT_DIAGNOSIS: str = Field(
        default="10/minute",
        description="Rate limit for /interpret/diagnosis"
    )
    RATE_LIMIT_GENERATE: str = Field(
        default="5/minute",
        description="Rate limit for /generate (stricter)"
    )
    RATE_LIMIT_EXPLAIN: str = Field(
        default="15/minute",
        description="Rate limit for /explain/anomaly"
    )
    RATE_LIMIT_KB_UPLOAD: str = Field(
        default="20/hour",
        description="Rate limit for KB document uploads"
    )
    
    # === Logging ===
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG|INFO|WARNING|ERROR"
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format: json|text"
    )
    LOG_FILE: Optional[str] = Field(
        default=None,
        description="Optional log file path (None = stdout only)"
    )
    
    # === API Settings ===
    API_HOST: str = Field(default="0.0.0.0", description="API bind host")
    API_PORT: int = Field(default=8004, ge=1, le=65535, description="API port")
    API_WORKERS: int = Field(default=1, ge=1, le=8, description="Uvicorn workers")
    CORS_ORIGINS: list[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    
    # === Admin Settings ===
    ADMIN_SECRET_KEY: str = Field(
        default="change-me-in-production",
        description="JWT secret key for admin auth"
    )
    ADMIN_TOKEN_EXPIRE_HOURS: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Admin token expiration in hours"
    )
    
    # === Monitoring ===
    METRICS_ENABLED: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    METRICS_PORT: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Prometheus metrics port"
    )
    
    # === Search Defaults ===
    SEARCH_TOP_K: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Default top-k for KB search"
    )
    SEARCH_THRESHOLD: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Default similarity threshold (0.65=balanced)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Игнорировать неизвестные env vars


# Global config instance
config = RAGServiceConfig()


def get_config() -> RAGServiceConfig:
    """Get global config instance."""
    return config


# Config validation on import
def validate_config():
    """
    Validate critical config values.
    
    Raises:
        ValueError: If production config is invalid
    """
    if config.ENVIRONMENT == "production":
        # Production checks
        if config.ADMIN_SECRET_KEY == "change-me-in-production":
            raise ValueError(
                "ADMIN_SECRET_KEY must be changed in production! "
                "Set via environment variable."
            )
        
        if not config.RATE_LIMIT_ENABLED:
            raise ValueError(
                "Rate limiting must be enabled in production for GPU protection!"
            )
        
        if config.LOG_LEVEL == "DEBUG":
            import warnings
            warnings.warn(
                "DEBUG logging in production! Consider using INFO or WARNING."
            )
    
    # S3 validation
    if config.S3_ENABLED and not config.S3_BUCKET:
        raise ValueError("S3_BUCKET must be set when S3_ENABLED=true")
    
    return True


# Auto-validate on import
validate_config()
