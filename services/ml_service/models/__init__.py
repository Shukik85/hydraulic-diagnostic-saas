"""
ML Models Package for Hydraulic Diagnostic Platform
Production-ready ensemble of 4 anomaly detection models
"""

from .adaptive_model import AdaptiveModel
from .base_model import BaseMLModel
from .catboost_model import CatBoostModel
from .ensemble import EnsembleModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel

# Model registry for dynamic loading
MODEL_REGISTRY = {
    "catboost": CatBoostModel,
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel,
    "adaptive": AdaptiveModel,
    "ensemble": EnsembleModel,
}

# Available models list
AVAILABLE_MODELS = list(MODEL_REGISTRY.keys())

# Model information
MODEL_INFO = {
    "catboost": {
        "name": "CatBoost Gradient Boosting",
        "type": "gradient_boosting",
        "description": "High-performance gradient boosting optimized for categorical features",
        "strengths": ["High accuracy", "Handles categorical features", "Robust to overfitting"],
        "use_case": "Primary model for most predictions",
    },
    "xgboost": {
        "name": "XGBoost Gradient Boosting",
        "type": "gradient_boosting",
        "description": "Extreme gradient boosting with advanced regularization",
        "strengths": ["Fast training", "Feature importance", "Cross-validation support"],
        "use_case": "Alternative gradient boosting with different optimization",
    },
    "random_forest": {
        "name": "Random Forest Ensemble",
        "type": "ensemble",
        "description": "Ensemble of decision trees with bootstrap aggregating",
        "strengths": ["Robust to overfitting", "Feature importance", "Handles missing values"],
        "use_case": "Stable predictions with uncertainty estimation",
    },
    "adaptive": {
        "name": "Adaptive Online Learning",
        "type": "online_learning",
        "description": "Adaptive model that learns from streaming data and detects concept drift",
        "strengths": ["Adapts to changes", "Online learning", "Drift detection"],
        "use_case": "Dynamic systems with changing patterns",
    },
    "ensemble": {
        "name": "Intelligent 4-Model Ensemble",
        "type": "meta_ensemble",
        "description": "Combines all models with dynamic weight adjustment",
        "strengths": ["Highest accuracy", "Robust predictions", "Automatic optimization"],
        "use_case": "Production system requiring maximum reliability",
    },
}

# Version information
__version__ = "1.0.0"
__models_version__ = {
    "catboost": "1.0.0",
    "xgboost": "1.0.0",
    "random_forest": "1.0.0",
    "adaptive": "1.0.0",
    "ensemble": "1.0.0",
}


def get_model_class(model_name: str) -> type[BaseMLModel] | type[EnsembleModel]:
    """Get model class by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {AVAILABLE_MODELS}")
    return MODEL_REGISTRY[model_name]


def create_model(model_name: str) -> BaseMLModel | EnsembleModel:
    """Create model instance by name."""
    model_class = get_model_class(model_name)
    return model_class()


def get_model_info(model_name: str | None = None) -> dict:
    """Get information about models."""
    if model_name:
        if model_name not in MODEL_INFO:
            raise ValueError(f"Unknown model: {model_name}")
        return MODEL_INFO[model_name]
    return MODEL_INFO


def list_available_models() -> list[str]:
    """List all available model names."""
    return AVAILABLE_MODELS.copy()


def check_model_availability() -> dict[str, bool]:
    """Check which models can be instantiated."""
    availability = {}

    for model_name, model_class in MODEL_REGISTRY.items():
        try:
            # Try to create instance (don't load)
            model = model_class()
            availability[model_name] = True
        except Exception:
            availability[model_name] = False

    return availability


# Export all classes and functions
__all__ = [
    # Base class
    "BaseMLModel",
    # Individual models
    "CatBoostModel",
    "XGBoostModel",
    "RandomForestModel",
    "AdaptiveModel",
    # Ensemble
    "EnsembleModel",
    # Registry and utilities
    "MODEL_REGISTRY",
    "AVAILABLE_MODELS",
    "MODEL_INFO",
    "get_model_class",
    "create_model",
    "get_model_info",
    "list_available_models",
    "check_model_availability",
    # Version info
    "__version__",
    "__models_version__",
]
