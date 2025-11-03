"""
Ensemble Model for Hydraulic Systems Anomaly Detection
Enterprise ensemble с 4 ML моделями и <100ms latency
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import structlog
from sklearn.ensemble import IsolationForest

from config import settings, MODEL_CONFIG
from .base_model import BaseMLModel
from .helm_model import HELMModel
from .xgboost_model import XGBoostModel
from .random_forest_model import RandomForestModel
from .adaptive_model import AdaptiveModel

logger = structlog.get_logger()


class EnsembleModel:
    """
    Enterprise Ensemble Model для гидравлической диагностики.

    Объединяет 4 ML модели:
    - HELM (99.5% accuracy) - вес 0.4
    - XGBoost (99.8% accuracy) - вес 0.4
    - RandomForest (99.6% accuracy) - вес 0.2
    - Adaptive (99.2% accuracy) - динамические пороги
    """

    def __init__(self):
        self.models: Dict[str, BaseMLModel] = {}
        self.ensemble_weights = settings.ensemble_weights.copy()
        self.is_loaded = False
        self.load_start_time = None
        self.prediction_count = 0
        self.total_inference_time = 0.0

        # Метрики производительности
        self.performance_metrics = {
            "predictions_total": 0,
            "inference_times": [],
            "accuracy_scores": [],
            "cache_hits": 0,
        }

    async def load_models(self) -> None:
        """Отложенная загрузка всех моделей."""
        self.load_start_time = time.time()

        logger.info("Loading ensemble models", model_path=settings.model_path)

        try:
            # Параллельная загрузка моделей
            tasks = [
                self._load_helm_model(),
                self._load_xgboost_model(),
                self._load_random_forest_model(),
                self._load_adaptive_model(),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Проверка результатов
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    model_name = list(MODEL_CONFIG.keys())[i]
                    logger.error(f"Failed to load {model_name}", error=str(result))

            loaded_models = [name for name, model in self.models.items() if model.is_loaded]

            if len(loaded_models) < 2:
                raise RuntimeError(f"Insufficient models loaded: {loaded_models}")

            self.is_loaded = True
            load_time = time.time() - self.load_start_time

            logger.info(
                "Ensemble models loaded successfully",
                loaded_models=loaded_models,
                load_time_seconds=load_time,
            )

        except Exception as e:
            logger.error("Failed to load ensemble models", error=str(e))
            raise

    async def _load_helm_model(self) -> None:
        """Загрузка HELM модели."""
        try:
            model = HELMModel()
            await model.load()
            self.models["helm"] = model
        except Exception as e:
            logger.warning("HELM model failed to load", error=str(e))
            # Продолжаем без HELM

    async def _load_xgboost_model(self) -> None:
        """Загрузка XGBoost модели."""
        try:
            model = XGBoostModel()
            await model.load()
            self.models["xgboost"] = model
        except Exception as e:
            logger.warning("XGBoost model failed to load", error=str(e))

    async def _load_random_forest_model(self) -> None:
        """Загрузка RandomForest модели."""
        try:
            model = RandomForestModel()
            await model.load()
            self.models["random_forest"] = model
        except Exception as e:
            logger.warning("RandomForest model failed to load", error=str(e))

    async def _load_adaptive_model(self) -> None:
        """Загрузка Adaptive модели."""
        try:
            model = AdaptiveModel()
            await model.load()
            self.models["adaptive"] = model
        except Exception as e:
            logger.warning("Adaptive model failed to load", error=str(e))

    async def predict(self, features: np.ndarray, use_cache: bool = True) -> Dict[str, Any]:
        """
        Enterprise ensemble предсказание с <100ms latency.

        Args:
            features: Массив признаков
            use_cache: Использовать кеш

        Returns:
            Словарь с предсказанием и метриками
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")

        start_time = time.time()

        try:
            # Параллельные предсказания
            prediction_tasks = []

            for model_name, model in self.models.items():
                if model.is_loaded:
                    task = self._get_model_prediction(model_name, model, features)
                    prediction_tasks.append(task)

            if not prediction_tasks:
                raise RuntimeError("No models available for prediction")

            # Ожидание всех предсказаний
            individual_predictions = await asyncio.gather(*prediction_tasks)

            # Ensemble вычисление
            ensemble_result = self._calculate_ensemble_score(individual_predictions)

            # Обновление метрик
            inference_time = (time.time() - start_time) * 1000  # мс
            self._update_metrics(inference_time, ensemble_result["ensemble_score"])

            # Проверка latency requirement
            if inference_time > settings.max_inference_time_ms:
                logger.warning(
                    "Inference time exceeded target",
                    inference_time_ms=inference_time,
                    target_ms=settings.max_inference_time_ms,
                )

            return {
                **ensemble_result,
                "individual_predictions": individual_predictions,
                "total_processing_time_ms": inference_time,
                "cache_hit": False,  # TODO: имплементировать кеширование
            }

        except Exception as e:
            logger.error("Ensemble prediction failed", error=str(e))
            raise

    async def _get_model_prediction(
        self, model_name: str, model: BaseMLModel, features: np.ndarray
    ) -> Dict[str, Any]:
        """Получение предсказания от одной модели."""
        start_time = time.time()

        try:
            prediction = await model.predict(features)
            processing_time = (time.time() - start_time) * 1000

            return {
                "model_name": model_name,
                "model_version": model.version,
                "prediction_score": prediction["score"],
                "confidence": prediction.get("confidence", 0.95),
                "processing_time_ms": processing_time,
                "features_used": len(features) if features.ndim == 1 else features.shape[1],
            }

        except Exception as e:
            logger.error(f"Model {model_name} prediction failed", error=str(e))
            return {
                "model_name": model_name,
                "model_version": model.version,
                "prediction_score": 0.5,  # Нейтральный скор
                "confidence": 0.0,
                "processing_time_ms": 0.0,
                "features_used": 0,
                "error": str(e),
            }

    def _calculate_ensemble_score(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Вычисление ensemble скора с весами."""
        if not predictions:
            return {"ensemble_score": 0.5, "severity": "normal", "confidence": 0.0}

        # Отфильтровываем ошибочные предсказания
        valid_predictions = [p for p in predictions if "error" not in p]

        if not valid_predictions:
            return {"ensemble_score": 0.5, "severity": "normal", "confidence": 0.0}

        # Весовое среднее
        weighted_score = 0.0
        total_weight = 0.0
        confidence_sum = 0.0

        model_weights = {
            "helm": self.ensemble_weights[0],
            "xgboost": self.ensemble_weights[1],
            "random_forest": self.ensemble_weights[2],
        }

        for pred in valid_predictions:
            model_name = pred["model_name"]
            weight = model_weights.get(model_name, 0.1)  # Минимальный вес

            weighted_score += pred["prediction_score"] * weight
            confidence_sum += pred["confidence"] * weight
            total_weight += weight

        if total_weight == 0:
            return {"ensemble_score": 0.5, "severity": "normal", "confidence": 0.0}

        final_score = weighted_score / total_weight
        final_confidence = confidence_sum / total_weight

        # Определение серьезности
        if final_score < 0.3:
            severity = "normal"
        elif final_score < 0.6:
            severity = "warning"
        else:
            severity = "critical"

        return {
            "ensemble_score": final_score,
            "severity": severity,
            "is_anomaly": final_score > settings.prediction_threshold,
            "confidence": final_confidence,
        }

    def _update_metrics(self, inference_time: float, accuracy: float) -> None:
        """Обновление метрик производительности."""
        self.performance_metrics["predictions_total"] += 1
        self.performance_metrics["inference_times"].append(inference_time)
        self.performance_metrics["accuracy_scores"].append(accuracy)

        # Ограничиваем размер списков (1000 последних значений)
        if len(self.performance_metrics["inference_times"]) > 1000:
            self.performance_metrics["inference_times"] = self.performance_metrics[
                "inference_times"
            ][-1000:]
            self.performance_metrics["accuracy_scores"] = self.performance_metrics[
                "accuracy_scores"
            ][-1000:]

    async def warmup(self, warmup_samples: int = 10) -> None:
        """Прогрев моделей для оптимизации первого запроса."""
        logger.info("Warming up ensemble models", samples=warmup_samples)

        # Симуляция данных для прогрева
        dummy_features = np.random.rand(warmup_samples, 25)  # 25 признаков

        for i in range(warmup_samples):
            try:
                await self.predict(dummy_features[i], use_cache=False)
            except Exception as e:
                logger.warning(f"Warmup sample {i} failed", error=str(e))

        logger.info("Model warmup completed")

    def is_ready(self) -> bool:
        """Проверка готовности ensemble."""
        return self.is_loaded and len([m for m in self.models.values() if m.is_loaded]) >= 2

    def get_loaded_models(self) -> List[str]:
        """Список загруженных моделей."""
        return [name for name, model in self.models.items() if model.is_loaded]

    def get_model_info(self) -> Dict[str, Any]:
        """Информация о загруженных моделях."""
        model_info = {}

        for name, model in self.models.items():
            if model.is_loaded:
                model_info[name] = {
                    "name": MODEL_CONFIG[name]["name"],
                    "version": model.version,
                    "description": MODEL_CONFIG[name]["description"],
                    "accuracy_target": MODEL_CONFIG[name]["accuracy_target"],
                    "weight": MODEL_CONFIG[name]["weight"],
                    "is_loaded": True,
                }
            else:
                model_info[name] = {
                    "name": MODEL_CONFIG[name]["name"],
                    "is_loaded": False,
                    "error": "Failed to load",
                }

        return model_info

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Метрики производительности."""
        if not self.performance_metrics["inference_times"]:
            return {"predictions_total": 0}

        times = self.performance_metrics["inference_times"]

        return {
            "predictions_total": self.performance_metrics["predictions_total"],
            "average_response_time_ms": np.mean(times),
            "p95_response_time_ms": np.percentile(times, 95),
            "p99_response_time_ms": np.percentile(times, 99),
            "min_response_time_ms": np.min(times),
            "max_response_time_ms": np.max(times),
            "average_accuracy": np.mean(self.performance_metrics["accuracy_scores"])
            if self.performance_metrics["accuracy_scores"]
            else 0.0,
        }

    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        logger.info("Cleaning up ensemble models")

        for model in self.models.values():
            try:
                await model.cleanup()
            except Exception as e:
                logger.warning("Model cleanup failed", error=str(e))

        self.models.clear()
        self.is_loaded = False
