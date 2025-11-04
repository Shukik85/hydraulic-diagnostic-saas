"""
Ensemble Model for Hydraulic Systems Anomaly Detection
Dynamic features + fail-soft model errors
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import structlog

from config import MODEL_CONFIG, settings

from .adaptive_model import AdaptiveModel
from .base_model import BaseMLModel
from .catboost_model import CatBoostModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel

logger = structlog.get_logger()


class EnsembleModel:
    def __init__(self):
        self.models: dict[str, BaseMLModel] = {}
        self.ensemble_weights = settings.ensemble_weights.copy()
        self.is_loaded = False
        self.load_start_time = None
        self.prediction_count = 0
        self.total_inference_time = 0.0

        self.fallback_strategies = {
            "primary": ["catboost"],
            "secondary": ["catboost", "xgboost"],
            "tertiary": ["xgboost", "random_forest"],
            "emergency": ["adaptive"],
        }

        self.performance_metrics = {
            "predictions_total": 0,
            "inference_times": [],
            "accuracy_scores": [],
            "cache_hits": 0,
            "fallback_usage": {"primary": 0, "secondary": 0, "tertiary": 0, "emergency": 0},
        }

    async def load_models(self) -> None:
        self.load_start_time = time.time()
        logger.info("Loading ensemble models", model_path=str(settings.model_path))
        tasks = [self._load_catboost_model(), self._load_xgboost_model(), self._load_random_forest_model(), self._load_adaptive_model()]
        await asyncio.gather(*tasks, return_exceptions=True)
        loaded_models = [name for name, model in self.models.items() if model.is_loaded]
        if len(loaded_models) < 1:
            raise RuntimeError(f"No models loaded: {loaded_models}")
        self.is_loaded = True
        load_time = time.time() - self.load_start_time
        logger.info("Ensemble models loaded successfully", loaded_models=loaded_models, load_time_seconds=load_time)

    async def _load_catboost_model(self) -> None:
        try:
            model = CatBoostModel()
            await model.load()
            self.models["catboost"] = model
        except Exception as e:
            logger.warning("CatBoost model failed to load", error=str(e))

    async def _load_xgboost_model(self) -> None:
        try:
            model = XGBoostModel()
            await model.load()
            self.models["xgboost"] = model
        except Exception as e:
            logger.warning("XGBoost model failed to load", error=str(e))

    async def _load_random_forest_model(self) -> None:
        try:
            model = RandomForestModel()
            await model.load()
            self.models["random_forest"] = model
        except Exception as e:
            logger.warning("RandomForest model failed to load", error=str(e))

    async def _load_adaptive_model(self) -> None:
        try:
            model = AdaptiveModel()
            await model.load()
            self.models["adaptive"] = model
        except Exception as e:
            logger.warning("Adaptive model failed to load", error=str(e))

    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")

        # Ensure 1D and soft-adjust length to first model's expectation
        if not isinstance(features, np.ndarray):
            features = np.asarray(features, dtype=float)
        if features.ndim != 1:
            features = features.ravel()

        expected = None
        for m in self.models.values():
            if m.is_loaded:
                expected = int(m.metadata.get("features_count", features.size))
                break
        expected = expected or features.size

        if features.size != expected:
            logger.warning("Adjusting feature vector length", got=int(features.size), expected=expected)
            if features.size > expected:
                features = features[:expected]
            else:
                features = np.pad(features, (0, expected - features.size), constant_values=0.0)

        logger.info("Ensemble received vector", len=int(features.size), nonzero=int(np.count_nonzero(features)))

        start_time = time.time()
        for strategy_name, model_names in self.fallback_strategies.items():
            try:
                available = [(name, self.models[name]) for name in model_names if name in self.models and self.models[name].is_loaded]
                if not available:
                    continue
                individual_predictions = []
                for name, model in available:
                    try:
                        pred = await model.predict(features)
                        individual_predictions.append({
                            "ml_model": name,
                            "version": model.version,
                            "prediction_score": float(pred.get("score", 0.5)),
                            "confidence": float(pred.get("confidence", 0.0)),
                            "processing_time_ms": float(pred.get("processing_time_ms", 0.0)),
                            "features_used": int(features.size),
                        })
                    except Exception as e:
                        logger.warning("Model prediction failed", model=name, error=str(e))
                        # Fail-soft: добавляем error-запись, чтобы стратегия могла продолжиться
                        individual_predictions.append({
                            "ml_model": name,
                            "version": getattr(model, "version", "v1"),
                            "prediction_score": 0.5,
                            "confidence": 0.0,
                            "processing_time_ms": 0.0,
                            "features_used": int(features.size),
                            "error": str(e),
                        })
                # Оставляем только валидные предсказания без error
                valid = [p for p in individual_predictions if "error" not in p]
                if not valid:
                    continue
                result = self._calculate_ensemble_score(valid)
                inference_time = (time.time() - start_time) * 1000
                self.performance_metrics["predictions_total"] += 1
                self.performance_metrics["inference_times"].append(inference_time)
                self.performance_metrics["fallback_usage"][strategy_name] += 1
                logger.info(
                    "Ensemble prediction",
                    strategy_used=strategy_name,
                    features_extracted=int(features.size),
                    inference_time_ms=inference_time,
                    models_used=len(valid),
                )
                return {
                    **result,
                    "individual_predictions": valid,
                    "total_processing_time_ms": inference_time,
                    "cache_hit": False,
                    "strategy_used": strategy_name,
                }
            except Exception as e:
                logger.warning("Fallback strategy failed", strategy=strategy_name, error=str(e))
                continue
        total_time = (time.time() - start_time) * 1000
        logger.error("All fallback strategies failed", processing_time_ms=total_time)
        raise RuntimeError("All fallback strategies failed")

    def _calculate_ensemble_score(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        if not predictions:
            return {"ensemble_score": 0.5, "severity": "normal", "confidence": 0.0, "is_anomaly": False}
        weighted_score, total_weight, confidence_sum = 0.0, 0.0, 0.0
        model_weights = {
            "catboost": self.ensemble_weights[0],
            "xgboost": self.ensemble_weights[1],
            "random_forest": self.ensemble_weights[2],
            "adaptive": self.ensemble_weights[3],
        }
        for p in predictions:
            w = model_weights.get(p["ml_model"], 0.05)
            weighted_score += p["prediction_score"] * w
            confidence_sum += p["confidence"] * w
            total_weight += w
        if total_weight == 0:
            return {"ensemble_score": 0.5, "severity": "normal", "confidence": 0.0, "is_anomaly": False}
        final_score = weighted_score / total_weight
        final_confidence = confidence_sum / total_weight
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

    async def warmup(self, warmup_samples: int = 10) -> None:
        logger.info("Warming up ensemble models", samples=warmup_samples)
        dummy_features = np.random.rand(warmup_samples, 25)
        for i in range(warmup_samples):
            try:
                result = await self.predict(dummy_features[i])
                logger.debug(
                    f"Warmup sample {i} completed",
                    strategy=result.get("strategy_used", "unknown"),
                    time_ms=result.get("total_processing_time_ms", 0),
                )
            except Exception as e:
                logger.warning(f"Warmup sample {i} failed", error=str(e))
        logger.info("Model warmup completed", fallback_usage=self.performance_metrics["fallback_usage"])

    def is_ready(self) -> bool:
        return self.is_loaded and len([m for m in self.models.values() if m.is_loaded]) >= 1

    async def cleanup(self) -> None:
        logger.info("Cleaning up ensemble models")
        for model in self.models.values():
            try:
                await model.cleanup()
            except Exception as e:
                logger.warning("Model cleanup failed", error=str(e))
        self.models.clear()
        self.is_loaded = False
