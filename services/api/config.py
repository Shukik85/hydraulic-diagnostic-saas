"""Configuration management."""
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings."""
    
    APP_NAME: str = "Hydraulic Diagnostic Service"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    DATABASE_URL: str = "postgresql://user:pass@db:5432/hydraulic"
    DATABASE_POOL_SIZE: int = 20
    
    LOG_LEVEL: str = "INFO"
    
    LOGGING_CONFIG = {
        "version": 1,
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {"level": "INFO", "handlers": ["default"]},
    }
    
    class Config:
        env_file = ".env"

settings = Settings()
