"""
ML Inference Service Configuration
Enterprise конфигурация для гидравлической диагностики
"""

from pathlib import Path

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """ML Service Settings with enterprise defaults."""

    # Application
    app_name: str = "Hydraulic ML Inference Service"
    version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    # Server
    host: str = Field(default="0.0.0.0", env="ML_HOST")
    port: int = Field(default=8001, env="ML_PORT")
    workers: int = Field(default=4, env="ML_WORKERS")

    # ML Models - UPDATED FOR CATBOOST ENSEMBLE
    model_path: Path = Field(default=Path("./models"), env="MODEL_PATH")
    ensemble_weights: list[float] = [0.5, 0.3, 0.15, 0.05]  # CatBoost, XGBoost, RandomForest, Adaptive
    prediction_threshold: float = Field(default=0.6, env="PREDICTION_THRESHOLD")

    # Performance - OPTIMIZED FOR CATBOOST
    max_inference_time_ms: int = Field(default=20, env="MAX_INFERENCE_TIME_MS")  # Reduced from 100ms
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    cache_predictions: bool = Field(default=True, env="CACHE_PREDICTIONS")
    cache_ttl_seconds: int = Field(default=300, env="CACHE_TTL_SECONDS")  # 5 min

    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")

    # Database (for model metadata)
    database_url: str = Field(default="postgresql://user:pass@localhost:5432/hydraulic", env="DATABASE_URL")

    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Health Checks
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    model_warmup_timeout: int = Field(default=60, env="MODEL_WARMUP_TIMEOUT")

    # Security
    api_key: str | None = Field(default=None, env="ML_API_KEY")
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8000"], env="CORS_ORIGINS")

    # Feature Engineering
    feature_window_minutes: int = Field(default=10, env="FEATURE_WINDOW_MINUTES")
    sampling_frequency_hz: float = Field(default=100.0, env="SAMPLING_FREQUENCY_HZ")

    # Anomaly Detection
    anomaly_sensitivity: float = Field(default=0.8, env="ANOMALY_SENSITIVITY")
    min_data_points: int = Field(default=100, env="MIN_DATA_POINTS")

    # External Services
    backend_api_url: str = Field(default="http://localhost:8000/api", env="BACKEND_API_URL")
    notification_service_url: str | None = Field(default=None, env="NOTIFICATION_SERVICE_URL")

    @validator("ensemble_weights")
    def validate_ensemble_weights(cls, v):
        if len(v) != 4:  # Updated for 4 models
            raise ValueError("Ensemble weights must have exactly 4 values")
        if abs(sum(v) - 1.0) > 0.01:
            raise ValueError("Ensemble weights must sum to 1.0")
        return v

    @validator("prediction_threshold")
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Prediction threshold must be between 0.0 and 1.0")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from .env
        json_encoders = {
            Path: str  # Converts Path to string when serializing
        }


# Global settings instance
settings = Settings()

# Model Configuration - UPDATED FOR CATBOOST ENSEMBLE
MODEL_CONFIG = {
    "catboost": {
        "name": "CatBoost Anomaly Detection",
        "file": "catboost_model.joblib",
        "weight": settings.ensemble_weights[0],  # 50% - Primary model
        "accuracy_target": 0.999,  # Higher than HELM
        "latency_target_ms": 5,  # Much faster than HELM
        "description": "Enterprise gradient boosting for hydraulic anomaly detection (HELM replacement)",
        "license": "Apache 2.0",
        "commercial_safe": True,
        "russian_registry_compliant": True,
    },
    "xgboost": {
        "name": "XGBoost Classifier",
        "file": "xgboost_model.joblib",
        "weight": settings.ensemble_weights[1],  # 30%
        "accuracy_target": 0.998,
        "latency_target_ms": 15,
        "description": "Gradient boosting for valve/accumulator component specialization",
        "license": "Apache 2.0",
        "commercial_safe": True,
    },
    "random_forest": {
        "name": "Random Forest",
        "file": "random_forest_model.joblib",
        "weight": settings.ensemble_weights[2],  # 15%
        "accuracy_target": 0.996,
        "latency_target_ms": 25,
        "description": "Ensemble stabilizer for robust predictions",
        "license": "BSD-3-Clause",
        "commercial_safe": True,
    },
    "adaptive": {
        "name": "Adaptive Threshold",
        "file": "adaptive_model.joblib",
        "weight": settings.ensemble_weights[3],  # 5%
        "accuracy_target": 0.992,
        "latency_target_ms": 3,
        "description": "Dynamic threshold adjustment based on system state",
        "license": "Own Implementation",
        "commercial_safe": True,
    },
}

# ENSEMBLE PERFORMANCE TARGETS
ENSEMBLE_TARGETS = {
    "accuracy": 0.996,  # Combined: 99.6%+
    "latency_p90_ms": 20,  # Down from 100ms
    "latency_p99_ms": 35,  # Conservative estimate
    "memory_mb": 500,  # Total memory budget
    "throughput_rps": 100,  # Requests per second
}

# Feature Configuration
FEATURE_CONFIG = {
    "sensor_features": [
        "pressure_mean",
        "pressure_std",
        "pressure_max",
        "pressure_min",
        "temperature_mean",
        "temperature_std",
        "temperature_max",
        "temperature_min",
        "flow_mean",
        "flow_std",
        "flow_max",
        "flow_min",
        "vibration_mean",
        "vibration_std",
        "vibration_max",
        "vibration_min",
    ],
    "derived_features": [
        "pressure_gradient",
        "temperature_gradient",
        "flow_gradient",
        "vibration_rms",
        "pressure_flow_ratio",
        "temp_pressure_correlation",
        "system_efficiency",
        "anomaly_score_rolling",
    ],
    "window_features": [
        "trend_slope",
        "seasonality_score",
        "stationarity_test",
        "autocorrelation_lag1",
        "cross_correlation_max",
    ],
}

# Anomaly Thresholds
ANOMALY_THRESHOLDS = {
    "normal": {"min": 0.0, "max": 0.3, "color": "green", "priority": "low"},
    "warning": {"min": 0.3, "max": 0.6, "color": "yellow", "priority": "medium"},
    "critical": {"min": 0.6, "max": 1.0, "color": "red", "priority": "high"},
}

# System Health Metrics
HEALTH_METRICS = {
    "cpu_threshold": 80.0,  # %
    "memory_threshold": 85.0,  # %
    "disk_threshold": 90.0,  # %
    "response_time_threshold": settings.max_inference_time_ms,  # ms
    "error_rate_threshold": 0.05,  # 5%
    "model_accuracy_threshold": 0.95,  # 95%
}