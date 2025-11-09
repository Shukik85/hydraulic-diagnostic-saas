"""Configuration management for RAG service using pydantic-settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""

    # Service configuration
    service_name: str = Field(default="rag-service", description="Service name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Security
    internal_api_key: str = Field(
        ..., description="Shared secret for internal service auth"
    )

    # RAG Configuration
    llm_model: str = Field(default="llama3.2:latest", description="LLM model name")
    ollama_base_url: str = Field(
        default="http://ollama:11434", description="Ollama API URL"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model",
    )
    vector_store_path: str = Field(
        default="/data/faiss_index", description="FAISS index storage path"
    )

    # Performance
    max_context_length: int = Field(default=4096, description="Max context tokens")
    chunk_size: int = Field(default=512, description="Document chunk size")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size")

    class Config:
        env_file = ".env"
        env_prefix = "RAG_"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
