"""
Configuration management using Pydantic Settings
Environment-based configuration for development, staging, production
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application
    VERSION: str = "1.0.0"
    ENV: str = "development"  # development, staging, production
    DEBUG: bool = True

    # Database (PostgreSQL + TimescaleDB)
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/hydraulic_db"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # Redis (caching, rate limiting)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    API_KEY_HEADER: str = "X-API-Key"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    # GNN Service
    GNN_SERVICE_URL: str = "http://localhost:8001"

    # RAG Service (future)
    RAG_SERVICE_URL: str = "http://localhost:8002"

    # TimescaleDB
    TIMESCALE_RETENTION_DAYS: int = 1825  # 5 years
    TIMESCALE_COMPRESSION_INTERVAL: str = "7 days"

    # Rate Limiting (per subscription tier)
    RATE_LIMIT_FREE: int = 100  # requests per day
    RATE_LIMIT_BASIC: int = 10000
    RATE_LIMIT_PRO: int = 100000

    # Monitoring
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 9090

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or console

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
