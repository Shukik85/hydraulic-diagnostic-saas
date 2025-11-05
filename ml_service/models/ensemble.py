"""
Ensemble Model for Hydraulic Systems Anomaly Detection
CatBoost-based implementation with fallback strategies
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import structlog

from config import MODEL_CONFIG, settings

from .base_model import BaseMLModel
from .catboost_model import CatBoostModel

logger = structlog.get_logger()


class EnsembleModel:
    """
    Production-ready ensemble focused on CatBoost as primary model.
    
    NOTE: This is NOT a true ensemble - only CatBoost is implemented.
    Other models (XGBoost, RandomForest, Adaptive) are placeholders.
    """
    
    def __init__(self):
        self.models: dict[str, BaseMLModel] = {}
        # Only CatBoost weight matters in practice
        self.ensemble_weights = [1.0, 0.0, 0.0, 0.0]  # CatBoost only
        self.is_loaded = False
        self.load_start_time = None
        self.prediction_count = 0
        self.total_inference_time = 0.0

        # Simplified fallback - only CatBoost available
        self.fallback_strategies = {
            "primary": ["catboost"],
            "emergency": ["catboost"],  # Same model, but with error handling
        }

        self.performance_metrics = {
            "predictions_total": 0,
            "inference_times": [],
            "accuracy_scores": [],
            "cache_hits": 0,
            "fallback_usage": {"primary": 0, "emergency": 0},
        }

    async def load_models(self) -> None:
        """Load only CatBoost model - other models are not implemented."""
        self.load_start_time = time.time()
        logger.info("Loading CatBoost model", model_path=str(settings.model_path))
        
        try:
            model = CatBoostModel()
            await model.load()
            self.models["catboost"] = model
            logger.info("CatBoost model loaded successfully")
        except Exception as e:
            logger.error("CatBoost model failed to load", error=str(e))
            raise RuntimeError(f"Failed to load primary CatBoost model: {e}")
        
        self.is_loaded = True
        load_time = time.time() - self.load_start_time
        logger.info("Model loading completed", 
                   loaded_models=["catboost"], 
                   load_time_seconds=load_time)

    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Make prediction using CatBoost model with fallback handling."""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")

        # Ensure proper feature format
        if not isinstance(features, np.ndarray):
            features = np.asarray(features, dtype=float)
        if features.ndim != 1:
            features = features.ravel()

        # Get expected feature count from CatBoost model
        catboost_model = self.models.get("catboost")
        if not catboost_model or not catboost_model.is_loaded:
            raise RuntimeError("CatBoost model not available")
        
        expected = int(catboost_model.metadata.get("features_count", features.size))
        
        # Adjust feature vector if needed
        if features.size != expected:
            logger.warning("Adjusting feature vector length", 
                          got=int(features.size), 
                          expected=expected)
            if features.size > expected:
                features = features[:expected]
            else:
                features = np.pad(features, (0, expected - features.size), constant_values=0.0)

        logger.info("Processing prediction request", 
                   features_count=int(features.size), 
                   nonzero_features=int(np.count_nonzero(features)))

        start_time = time.time()
        
        # Try primary strategy (CatBoost)
        try:
            pred = await catboost_model.predict(features)
            
            individual_prediction = {
                "ml_model": "catboost",
                "version": catboost_model.version,
                "prediction_score": float(pred.get("score", 0.5)),
                "confidence": float(pred.get("confidence", 0.0)),
                "processing_time_ms": float(pred.get("processing_time_ms", 0.0)),
                "features_used": int(features.size),
            }
            
            # Calculate final result (no real ensemble, just CatBoost score)
            result = self._calculate_result([individual_prediction])
            
            inference_time = (time.time() - start_time) * 1000
            self._update_metrics("primary", inference_time)
            
            logger.info("Prediction completed", 
                       strategy_used="primary",
                       features_extracted=int(features.size),
                       inference_time_ms=inference_time,
                       prediction_score=result["ensemble_score"])
            
            return {
                **result,
                "individual_predictions": [individual_prediction],
                "total_processing_time_ms": inference_time,
                "cache_hit": False,
                "strategy_used": "primary",
            }
            
        except Exception as e:
            # Emergency fallback - return safe default
            logger.error("Primary prediction failed, using emergency fallback", error=str(e))
            
            inference_time = (time.time() - start_time) * 1000
            self._update_metrics("emergency", inference_time)
            
            return {
                "ensemble_score": 0.5,  # Safe default
                "severity": "normal",
                "is_anomaly": False,
                "confidence": 0.0,
                "individual_predictions": [{
                    "ml_model": "emergency_fallback",
                    "version": "1.0.0",
                    "prediction_score": 0.5,
                    "confidence": 0.0,
                    "processing_time_ms": inference_time,
                    "features_used": int(features.size),
                    "error": str(e)
                }],
                "total_processing_time_ms": inference_time,
                "cache_hit": False,
                "strategy_used": "emergency",
                "error": f"Primary model failed: {str(e)}"
            }

    def _calculate_result(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate final result from predictions (currently just CatBoost)."""
        if not predictions:
            return {
                "ensemble_score": 0.5, 
                "severity": "normal", 
                "confidence": 0.0, 
                "is_anomaly": False
            }
        
        # Since we only have CatBoost, just use its score directly
        pred = predictions[0]
        final_score = pred["prediction_score"]
        final_confidence = pred["confidence"]
        
        # Determine severity based on score
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

    def _update_metrics(self, strategy: str, inference_time: float) -> None:
        """Update performance metrics."""
        self.performance_metrics["predictions_total"] += 1
        self.performance_metrics["inference_times"].append(inference_time)
        self.performance_metrics["fallback_usage"][strategy] += 1

    async def warmup(self, warmup_samples: int = 5) -> None:
        """Warm up the CatBoost model with dummy predictions."""
        logger.info("Warming up CatBoost model", samples=warmup_samples)
        
        dummy_features = np.random.rand(warmup_samples, 25)  # Standard feature count
        
        for i in range(warmup_samples):
            try:
                result = await self.predict(dummy_features[i])
                logger.debug(
                    f"Warmup sample {i+1}/{warmup_samples} completed",
                    strategy=result.get("strategy_used", "unknown"),
                    time_ms=round(result.get("total_processing_time_ms", 0), 1)
                )
            except Exception as e:
                logger.warning(f"Warmup sample {i+1} failed", error=str(e))
        
        logger.info("Model warmup completed", 
                   fallback_usage=self.performance_metrics["fallback_usage"])

    def is_ready(self) -> bool:
        """Check if the ensemble is ready (CatBoost model loaded)."""
        catboost_model = self.models.get("catboost")
        return self.is_loaded and catboost_model and catboost_model.is_loaded

    def get_loaded_models(self) -> list[str]:
        """Return list of actually loaded models."""
        return [name for name, model in self.models.items() if model.is_loaded]

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded models."""
        model_info = {}
        
        # CatBoost info
        catboost_model = self.models.get("catboost")
        if catboost_model and catboost_model.is_loaded:
            model_info["catboost"] = {
                "name": "CatBoost Anomaly Detection",
                "version": catboost_model.version,
                "description": "Primary gradient boosting model for hydraulic anomaly detection",
                "accuracy_target": "Real UCI performance",
                "weight": 1.0,  # 100% since it's the only model
                "is_loaded": True,
            }
        else:
            model_info["catboost"] = {
                "name": "CatBoost Anomaly Detection",
                "is_loaded": False,
                "error": "Failed to load or not initialized",
            }
        
        # Note about missing models
        model_info["_note"] = "XGBoost, RandomForest, and Adaptive models are not implemented"
        
        return model_info

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        if not self.performance_metrics["inference_times"]:
            return {"predictions_total": 0, "note": "No predictions made yet"}
        
        times = self.performance_metrics["inference_times"]
        return {
            "predictions_total": self.performance_metrics["predictions_total"],
            "average_response_time_ms": np.mean(times),
            "p95_response_time_ms": np.percentile(times, 95),
            "p99_response_time_ms": np.percentile(times, 99),
            "min_response_time_ms": np.min(times),
            "max_response_time_ms": np.max(times),
            "fallback_usage": self.performance_metrics["fallback_usage"],
            "cache_hits": self.performance_metrics["cache_hits"],
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up ensemble models")
        
        for model in self.models.values():
            try:
                await model.cleanup()
            except Exception as e:
                logger.warning("Model cleanup failed", error=str(e))
        
        self.models.clear()
        self.is_loaded = False
        logger.info("Cleanup completed")
