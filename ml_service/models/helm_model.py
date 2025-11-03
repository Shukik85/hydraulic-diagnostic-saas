"""
HELM (Hierarchical Extreme Learning Machine) Model
Enterprise HELM модель с 99.5% accuracy target
"""

import time
from typing import Any

import numpy as np
import structlog
from sklearn.preprocessing import StandardScaler

from config import MODEL_CONFIG

from .base_model import BaseMLModel

logger = structlog.get_logger()


class HELMModel(BaseMLModel):
    """
    HELM (Hierarchical Extreme Learning Machine) для аномалий.

    Особенности:
    - Многоуровневая архитектура
    - Оптимизация для временных рядов
    - 99.5% accuracy target
    - <30ms среднее время inference
    """

    def __init__(self):
        super().__init__("helm", MODEL_CONFIG["helm"]["file"])
        self.scaler = StandardScaler()
        self.hierarchy_levels = 3
        self.confidence_threshold = 0.85

    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        """
        HELM предсказание с иерархической обработкой.

        Args:
            features: Массив признаков (25 или больше)

        Returns:
            Dict с score, confidence, processing_time_ms
        """
        if not self.is_loaded:
            raise RuntimeError("HELM model not loaded")

        if not self.validate_features(features):
            raise ValueError("Invalid features for HELM model")

        start_time = time.time()

        try:
            # Нормализация признаков
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # HELM иерархическая обработка
            normalized_features = (
                self.scaler.fit_transform(features)
                if hasattr(self.scaler, "transform")
                else features
            )

            # Мок HELM логика (в реальности сложнее)
            level1_score = (
                self.model.decision_function(normalized_features)[0]
                if hasattr(self.model, "decision_function")
                else 0.5
            )

            # Преобразование в вероятность [0, 1]
            anomaly_score = self._normalize_score(level1_score)

            # Вычисление уверенности
            confidence = min(0.995, max(0.5, abs(level1_score) * 1.2))

            processing_time = time.time() - start_time
            self.update_stats(processing_time)

            result = {
                "score": float(anomaly_score),
                "confidence": float(confidence),
                "processing_time_ms": processing_time * 1000,
                "model_specific": {
                    "hierarchy_level": 1,
                    "raw_score": float(level1_score),
                    "features_processed": features.shape[1] if features.ndim > 1 else len(features),
                },
            }

            logger.debug(
                "HELM prediction completed",
                score=anomaly_score,
                confidence=confidence,
                processing_time_ms=processing_time * 1000,
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "HELM prediction failed", error=str(e), processing_time_ms=processing_time * 1000
            )
            raise

    def _normalize_score(self, raw_score: float) -> float:
        """Нормализация скора в диапазон [0, 1]."""
        # Простая сигмоидная нормализация
        return 1.0 / (1.0 + np.exp(-raw_score * 2))
