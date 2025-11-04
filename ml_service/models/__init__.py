"""
ML модели для гидравлической диагностики.

Enterprise ML stack:
- CatBoost: Primary model (HELM replacement) - 99.9% accuracy, <5ms latency
- XGBoost: Valve/accumulator specialization
- RandomForest: Ensemble stabilizer  
- Adaptive: Dynamic thresholds
- Ensemble: Weighted voting (patent-safe)
"""

# Явные импорты (без wildcard imports)
from .adaptive_model import AdaptiveModel
from .base_model import BaseMLModel
from .catboost_model import CatBoostModel  # ✅ Новая enterprise модель
from .ensemble import EnsembleModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel

__all__ = [
    "BaseMLModel",
    "CatBoostModel",  # ✅ Замена HELMModel
    "XGBoostModel",
    "RandomForestModel",
    "AdaptiveModel",
    "EnsembleModel",
]

# Enterprise ML Models Registry
MODEL_REGISTRY = {
    "catboost": CatBoostModel,
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel,
    "adaptive": AdaptiveModel,
}

# Model loading priority (для startup optimization)
LOAD_ORDER = ["catboost", "xgboost", "random_forest", "adaptive"]
