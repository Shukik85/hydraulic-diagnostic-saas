"""
Random Forest Model for Hydraulic Systems
Enterprise Random Forest с 99.6% accuracy target
"""

import time
from typing import Any

import numpy as np
import structlog

from config import MODEL_CONFIG

from .base_model import BaseMLModel

logger = structlog.get_logger()


class RandomForestModel(BaseMLModel):
    """
    Random Forest Classifier для robust предсказаний.

    Особенности:
    - Ensemble method с множеством decision trees
    - Устойчивость к оверфиттингу
    - Feature importance analysis
    - 99.6% accuracy target
    - <20ms среднее время inference
    """

    def __init__(self):
        super().__init__("random_forest", MODEL_CONFIG["random_forest"]["file"])
        self.n_estimators = 150
        self.max_depth = 10
        self.min_samples_split = 5
        self.feature_importances_ = None

    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        """
        Random Forest предсказание.

        Args:
            features: Массив признаков

        Returns:
            Dict с score, confidence, processing_time_ms
        """
        if not self.is_loaded:
            raise RuntimeError("RandomForest model not loaded")

        if not self.validate_features(features):
            raise ValueError("Invalid features for RandomForest model")

        start_time = time.time()

        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Random Forest предсказание
            if hasattr(self.model, "predict_proba"):
                # Классификация
                probabilities = self.model.predict_proba(features)[0]
                anomaly_score = probabilities[1] if len(probabilities) > 1 else probabilities[0]

                # Уверенность на основе распределения вероятностей
                max_prob = max(probabilities)
                confidence = min(0.996, max_prob * 1.1)

            elif hasattr(self.model, "decision_function"):
                # Anomaly detection
                decision_score = self.model.decision_function(features)[0]
                anomaly_score = self._normalize_anomaly_score(decision_score)
                confidence = min(0.996, 0.8 + abs(decision_score) * 0.2)

            else:
                # Fallback
                prediction = self.model.predict(features)[0]
                anomaly_score = 0.75 if prediction == -1 else 0.25
                confidence = 0.8

            processing_time = time.time() - start_time
            self.update_stats(processing_time)

            # Feature importance (если доступно)
            feature_importance_score = 0.0
            if hasattr(self.model, "feature_importances_"):
                self.feature_importances_ = self.model.feature_importances_
                feature_importance_score = np.mean(self.feature_importances_)

            result = {
                "score": float(anomaly_score),
                "confidence": float(confidence),
                "processing_time_ms": processing_time * 1000,
                "model_specific": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "feature_importance_avg": float(feature_importance_score),
                    "trees_used": self.n_estimators,
                },
            }

            logger.debug(
                "RandomForest prediction completed",
                score=anomaly_score,
                confidence=confidence,
                processing_time_ms=processing_time * 1000,
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "RandomForest prediction failed",
                error=str(e),
                processing_time_ms=processing_time * 1000,
            )
            raise

    def _normalize_anomaly_score(self, decision_score: float) -> float:
        """Нормализация anomaly score."""
        # Преобразование decision_function в [0, 1]
        if decision_score < 0:
            return 0.5 + abs(decision_score) * 0.4  # Аномалия
        else:
            return max(0.0, 0.5 - decision_score * 0.4)  # Норма
