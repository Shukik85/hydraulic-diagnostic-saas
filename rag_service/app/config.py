"""RAG Service Configuration with DeepSeek-R1."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """RAG Service Settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # Application
    rag_debug: bool = Field(default=False, env="RAG_DEBUG")
    rag_host: str = Field(default="0.0.0.0", env="RAG_HOST")
    rag_port: int = Field(default=8002, env="RAG_PORT")

    # Security
    internal_api_key: str = Field(..., env="RAG_INTERNAL_API_KEY")

    # LLM Configuration (DeepSeek-R1)
    llm_model: str = Field(default="deepseek-r1:7b", env="RAG_LLM_MODEL")
    ollama_base_url: str = Field(default="http://ollama:11434", env="RAG_OLLAMA_BASE_URL")

    # Embeddings
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-large",
        env="RAG_EMBEDDING_MODEL",
    )

    # Vector Store
    vector_store_path: str = Field(default="/data/faiss_index", env="RAG_VECTOR_STORE_PATH")

    # Performance
    max_context_length: int = Field(default=4096, env="RAG_MAX_CONTEXT_LENGTH")

    # CORS
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")

    # Logging
    log_level: str = Field(default="INFO", env="RAG_LOG_LEVEL")


settings = Settings()
