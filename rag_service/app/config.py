"""Конфигурация RAG Service через Pydantic Settings."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")
    rag_internal_api_key: str = "changeme-rag-key"
    rag_ollama_base_url: str = "http://ollama:11434"
    embedding_model: str = "intfloat/multilingual-e5-large"
    faiss_index_path: str = "/data/faiss_index.bin"
    qdrant_enabled: bool = False
    qdrant_url: str = "http://qdrant:6333"
    max_context_docs: int = 3
    log_level: str = "INFO"
    debug: bool = False
@lru_cache
def get_settings() -> Settings:
    return Settings()
