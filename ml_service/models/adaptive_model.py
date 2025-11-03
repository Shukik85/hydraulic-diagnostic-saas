"""
Adaptive Threshold Model for Dynamic Anomaly Detection
Enterprise adaptive модель с динамическими порогами
"""

import time
from typing import Any

import numpy as np
import structlog

from config import MODEL_CONFIG, settings

from .base_model import BaseMLModel

logger = structlog.get_logger()


class AdaptiveModel(BaseMLModel):
    """
    Adaptive Threshold Model с динамической настройкой.

    Особенности:
    - Адаптация к состоянию системы
    - Динамические пороги
    - Онлайн обучение
    - 99.2% accuracy target
    """

    def __init__(self):
        super().__init__("adaptive", MODEL_CONFIG["adaptive"]["file"])
        self.base_threshold = settings.prediction_threshold
        self.adaptation_rate = 0.01
        self.history_window = 100
        self.recent_scores = []
        self.system_state_cache = {}

    async def predict(self, features: np.ndarray, system_id: str = None) -> dict[str, Any]:
        """
        Adaptive предсказание с динамическими порогами.

        Args:
            features: Массив признаков
            system_id: ID системы для контекстной адаптации

        Returns:
            Dict с score, confidence, dynamic_threshold
        """
        if not self.is_loaded:
            raise RuntimeError("Adaptive model not loaded")

        if not self.validate_features(features):
            raise ValueError("Invalid features for Adaptive model")

        start_time = time.time()

        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Базовое предсказание
            if hasattr(self.model, "decision_function"):
                base_score = self.model.decision_function(features)[0]
                normalized_score = self._normalize_score(base_score)
            else:
                prediction = self.model.predict(features)[0]
                normalized_score = 0.7 if prediction == -1 else 0.3

            # Динамическая адаптация порога
            adaptive_threshold = await self._calculate_adaptive_threshold(features, system_id)

            # Коррекция скора на основе динамического порога
            adapted_score = self._apply_adaptive_correction(normalized_score, adaptive_threshold)

            # Обновление истории
            self._update_history(adapted_score)

            # Уверенность на основе стабильности
            confidence = self._calculate_confidence(adapted_score)

            processing_time = time.time() - start_time
            self.update_stats(processing_time)

            result = {
                "score": float(adapted_score),
                "confidence": float(confidence),
                "processing_time_ms": processing_time * 1000,
                "model_specific": {
                    "base_score": float(normalized_score),
                    "adaptive_threshold": float(adaptive_threshold),
                    "threshold_adjustment": float(adaptive_threshold - self.base_threshold),
                    "history_size": len(self.recent_scores),
                    "system_adaptation": system_id is not None,
                },
            }

            logger.debug(
                "Adaptive prediction completed",
                score=adapted_score,
                threshold=adaptive_threshold,
                processing_time_ms=processing_time * 1000,
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Adaptive prediction failed",
                error=str(e),
                processing_time_ms=processing_time * 1000,
            )
            raise

    async def _calculate_adaptive_threshold(
        self,
        _features: np.ndarray,
        system_id: str = None,  # ✅ Исправил ARG002
    ) -> float:
        """Вычисление динамического порога."""
        base_threshold = self.base_threshold

        # Адаптация на основе истории
        if len(self.recent_scores) >= 10:
            recent_mean = np.mean(self.recent_scores[-10:])
            recent_std = np.std(self.recent_scores[-10:])

            # Коррекция на основе волатильности
            volatility_factor = min(0.2, recent_std * 2)

            if recent_mean > base_threshold:
                # Повышаем порог при частых аномалиях
                base_threshold += volatility_factor * self.adaptation_rate
            else:
                # Понижаем порог при стабильности
                base_threshold -= volatility_factor * self.adaptation_rate * 0.5

        # Адаптация по системе
        if system_id and system_id in self.system_state_cache:
            system_factor = self.system_state_cache[system_id].get("threshold_adjustment", 0.0)
            base_threshold += system_factor

        # Ограничение диапазона
        return max(0.1, min(0.9, base_threshold))

    def _apply_adaptive_correction(self, base_score: float, threshold: float) -> float:
        """Применение адаптивной коррекции."""
        # Коррекция на основе расстояния от порога
        distance_from_threshold = abs(base_score - threshold)

        if base_score > threshold:
            # Усиление сигнала аномалии
            correction = min(0.2, distance_from_threshold * 0.5)
            return min(1.0, base_score + correction)
        else:
            # Смягчение ложных срабатываний
            correction = min(0.1, distance_from_threshold * 0.2)
            return max(0.0, base_score - correction)

    def _update_history(self, score: float) -> None:
        """Обновление истории скоров."""
        self.recent_scores.append(score)

        # Ограничиваем размер истории
        if len(self.recent_scores) > self.history_window:
            self.recent_scores = self.recent_scores[-self.history_window :]

    def _calculate_confidence(self, score: float) -> float:
        """Расчет уверенности на основе стабильности."""
        if len(self.recent_scores) < 5:
            return 0.5  # Минимальная уверенность

        # Стабильность на основе стандартного отклонения
        recent_std = np.std(self.recent_scores[-10:])
        stability_factor = max(0.0, 1.0 - recent_std * 2)

        # Базовая уверенность + фактор стабильности
        base_confidence = 0.7 + abs(score - 0.5) * 0.4

        return min(0.992, base_confidence * stability_factor)

    def _normalize_score(self, raw_score: float) -> float:
        """Нормализация скора."""
        return 1.0 / (1.0 + np.exp(-raw_score * 1.2))

    async def update_system_state(self, system_id: str, state_info: dict[str, Any]) -> None:
        """Обновление состояния системы для адаптации."""
        self.system_state_cache[system_id] = {**state_info, "updated_at": time.time()}

        # Ограничиваем размер кеша
        if len(self.system_state_cache) > 1000:
            # Удаляем старые записи
            oldest_systems = sorted(self.system_state_cache.items(), key=lambda x: x[1].get("updated_at", 0))[:100]

            for system_id, _ in oldest_systems:
                del self.system_state_cache[system_id]
