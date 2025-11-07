"""
Adaptive Threshold Service for Dynamic Anomaly Detection
Enterprise-grade seasonal baselines + EMA threshold adaptation
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog

from config import settings

logger = structlog.get_logger()


class AdaptiveThresholdService:
    """
    Адаптивные пороги с сезонными baseline и EMA-обновлением.

    Возможности:
    - Seasonal context: день недели + час суток
    - EMA threshold adaptation по скользящему окну
    - Coverage-aware adjustment (quality penalty)
    - Target FPR calibration
    """

    def __init__(self, cache_service=None):
        self.cache = cache_service
        self.enabled = getattr(settings, "adaptive_thresholds_enabled", True)
        self.adaptation_rate = getattr(settings, "threshold_adaptation_rate", 0.05)
        self.target_fpr = getattr(settings, "target_fpr", 0.10)
        self.coverage_penalty = getattr(settings, "threshold_coverage_penalty", 0.05)
        self.window_size = getattr(settings, "threshold_window_size", 100)  # последние N окон

        # In-memory fallback если нет cache
        self._memory_storage: dict[str, dict[str, Any]] = defaultdict(dict)
        self._score_buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window_size))

        logger.info(
            "AdaptiveThresholdService initialized",
            enabled=self.enabled,
            adaptation_rate=self.adaptation_rate,
            target_fpr=self.target_fpr,
            window_size=self.window_size,
        )

    def _get_context_key(self, timestamp: float | None = None) -> str:
        """Генерирует контекстный ключ для сезонного baseline"""
        if timestamp is None:
            timestamp = time.time()

        dt = datetime.fromtimestamp(timestamp, tz=UTC)
        dow = dt.weekday()  # 0=Monday, 6=Sunday
        hour = dt.hour

        # Группируем часы для стабильности baseline
        hour_group = hour // 4  # 0-5: 6 групп по 4 часа

        return f"dow{dow}_h{hour_group}"

    async def get_threshold(
        self, system_id: str, context: dict[str, Any] | None = None, coverage: float = 1.0
    ) -> dict[str, Any]:
        """
        Получить адаптивный порог для системы с учётом контекста и покрытия данных.

        Returns:
            {
                "threshold": float,
                "source": str,
                "context_key": str,
                "confidence_multiplier": float
            }
        """
        if not self.enabled:
            return {
                "threshold": settings.prediction_threshold,
                "source": "fixed",
                "context_key": "disabled",
                "confidence_multiplier": 1.0,
            }

        context_key = self._get_context_key()
        storage_key = f"adaptive_threshold:{system_id}:{context_key}"

        # Попытка загрузить из cache/storage
        try:
            if self.cache:
                stored = await self.cache.get(storage_key)
                if stored:
                    data = json.loads(stored) if isinstance(stored, str) else stored
                    threshold = float(data.get("threshold", settings.prediction_threshold))
                    source = "seasonal_ema"
                else:
                    threshold = settings.prediction_threshold
                    source = "fallback"
            else:
                # In-memory fallback
                data = self._memory_storage[system_id].get(context_key, {})
                threshold = float(data.get("threshold", settings.prediction_threshold))
                source = "memory_ema" if data else "fallback"
        except Exception as e:
            logger.warning("Failed to load adaptive threshold", error=str(e), system_id=system_id)
            threshold = settings.prediction_threshold
            source = "error_fallback"

        # Coverage-aware adjustment
        confidence_multiplier = 1.0
        if coverage < 0.8:
            # При низком покрытии данных повышаем порог (консервативнее)
            penalty = self.coverage_penalty * (0.8 - coverage)
            threshold += penalty
            confidence_multiplier = coverage  # понижаем confidence
            source += "_coverage_adjusted"

        return {
            "threshold": float(min(0.95, max(0.05, threshold))),  # clamp [0.05, 0.95]
            "source": source,
            "context_key": context_key,
            "confidence_multiplier": float(confidence_multiplier),
        }

    async def update_baseline(
        self,
        system_id: str,
        score: float,
        is_normal: bool | None = None,
        coverage: float = 1.0,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """
        Обновить baseline и адаптировать порог по EMA.

        Args:
            score: текущий ensemble_score
            is_normal: если известно — True/False/None (для supervised adjustment)
            coverage: качество данных для взвешивания
        """
        if not self.enabled:
            return {"updated": False, "reason": "disabled"}

        context_key = self._get_context_key(timestamp)
        storage_key = f"adaptive_threshold:{system_id}:{context_key}"

        try:
            # Добавляем score в скользящий буфер
            buffer_key = f"{system_id}:{context_key}"
            self._score_buffers[buffer_key].append(score)

            # EMA threshold adaptation каждые 10+ окон
            buffer = list(self._score_buffers[buffer_key])
            if len(buffer) >= 10:
                # Текущая оценка p95 по скользящему окну
                current_p95 = float(np.percentile(buffer, 95))

                # Загружаем предыдущий threshold
                old_threshold = settings.prediction_threshold
                try:
                    if self.cache:
                        stored = await self.cache.get(storage_key)
                        if stored:
                            data = json.loads(stored) if isinstance(stored, str) else stored
                            old_threshold = float(data.get("threshold", settings.prediction_threshold))
                    else:
                        data = self._memory_storage[system_id].get(context_key, {})
                        old_threshold = float(data.get("threshold", settings.prediction_threshold))
                except Exception:
                    old_threshold = settings.prediction_threshold

                # EMA update
                new_threshold = (1 - self.adaptation_rate) * old_threshold + self.adaptation_rate * current_p95

                # Клэмп и сохранение
                new_threshold = float(min(0.95, max(0.05, new_threshold)))

                threshold_data = {
                    "threshold": new_threshold,
                    "last_update": time.time(),
                    "samples_count": len(buffer),
                    "current_p95": current_p95,
                    "coverage_weight": coverage,
                    "context": context_key,
                }

                # Сохранение
                try:
                    if self.cache:
                        await self.cache.set(storage_key, json.dumps(threshold_data), ttl=7 * 24 * 3600)  # 1 week
                    else:
                        self._memory_storage[system_id][context_key] = threshold_data
                except Exception as e:
                    logger.warning("Failed to save adaptive threshold", error=str(e))

                logger.info(
                    "Adaptive threshold updated",
                    system_id=system_id,
                    context=context_key,
                    old_threshold=old_threshold,
                    new_threshold=new_threshold,
                    current_p95=current_p95,
                    samples=len(buffer),
                )

                return {
                    "updated": True,
                    "old_threshold": old_threshold,
                    "new_threshold": new_threshold,
                    "samples_used": len(buffer),
                }
            else:
                return {"updated": False, "reason": f"insufficient_samples_{len(buffer)}"}

        except Exception as e:
            logger.error("Baseline update failed", error=str(e), system_id=system_id)
            return {"updated": False, "reason": f"error_{str(e)}"}

    def get_system_summary(self, system_id: str) -> dict[str, Any]:
        """Сводка по адаптивным порогам для системы (для debug/admin)"""
        summary = {"system_id": system_id, "enabled": self.enabled, "contexts": {}}

        # Проверяем все контексты в памяти
        for buffer_key, buffer in self._score_buffers.items():
            if buffer_key.startswith(f"{system_id}:"):
                context = buffer_key.replace(f"{system_id}:", "")
                summary["contexts"][context] = {
                    "buffer_size": len(buffer),
                    "recent_scores": list(buffer)[-5:] if buffer else [],
                    "p95_estimate": float(np.percentile(list(buffer), 95)) if len(buffer) >= 5 else None,
                }

        return summary
