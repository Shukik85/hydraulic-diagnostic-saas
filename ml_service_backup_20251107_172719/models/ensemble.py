"""
True Ensemble Model for Hydraulic Systems Anomaly Detection
Intelligent 4-model ensemble with dynamic weight balancing
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
from .xgboost_model import XGBoostModel
from .random_forest_model import RandomForestModel
from .adaptive_model import AdaptiveModel

logger = structlog.get_logger()


class EnsembleModel:
    """
    Production-ready 4-model ensemble with intelligent weight balancing.
    
    Models:
    - CatBoost: High-accuracy gradient boosting (primary)
    - XGBoost: Alternative gradient boosting (backup)
    - RandomForest: Ensemble of decision trees (robust)
    - Adaptive: Online learning with drift detection (dynamic)
    
    Features:
    - Dynamic weight adjustment based on performance
    - Intelligent fallback strategies
    - Real-time model health monitoring
    - Consensus-based confidence calculation
    """
    
    def __init__(self):
        self.models: dict[str, BaseMLModel] = {}
        
        # Initial ensemble weights (will be dynamically adjusted)
        self.ensemble_weights = {
            "catboost": 0.4,      # Primary model - highest weight
            "xgboost": 0.3,       # Strong alternative
            "random_forest": 0.2, # Robust ensemble
            "adaptive": 0.1       # Dynamic adaptation
        }
        
        self.is_loaded = False
        self.load_start_time = None
        self.prediction_count = 0
        self.total_inference_time = 0.0

        # Intelligent fallback strategies
        self.fallback_strategies = {
            "primary": ["catboost", "xgboost", "random_forest", "adaptive"],
            "gradient_boost": ["catboost", "xgboost"],
            "ensemble_only": ["random_forest", "adaptive"],
            "emergency": ["adaptive"],  # Most resilient model
        }

        # Performance tracking for dynamic weight adjustment
        self.performance_metrics = {
            "predictions_total": 0,
            "inference_times": [],
            "model_performance": {
                model_name: {
                    "success_count": 0,
                    "error_count": 0,
                    "avg_confidence": 0.0,
                    "avg_processing_time": 0.0,
                    "last_error_time": None
                } for model_name in self.ensemble_weights.keys()
            },
            "ensemble_consensus": [],
            "weight_adjustments": 0,
            "fallback_usage": {strategy: 0 for strategy in self.fallback_strategies.keys()},
        }

    async def load_models(self) -> None:
        """Load all models in the ensemble."""
        self.load_start_time = time.time()
        logger.info("Loading 4-model ensemble", models=list(self.ensemble_weights.keys()))
        
        # Create model instances
        model_classes = {
            "catboost": CatBoostModel,
            "xgboost": XGBoostModel,
            "random_forest": RandomForestModel,
            "adaptive": AdaptiveModel,
        }
        
        # Load models concurrently for faster startup
        load_tasks = []
        for model_name, model_class in model_classes.items():
            try:
                model = model_class()
                self.models[model_name] = model
                load_tasks.append(self._load_single_model(model_name, model))
            except Exception as e:
                logger.error(f"Failed to create {model_name} model", error=str(e))
        
        # Wait for all models to load
        load_results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        # Check which models loaded successfully
        loaded_models = []
        for i, (model_name, result) in enumerate(zip(model_classes.keys(), load_results)):
            if isinstance(result, Exception):
                logger.error(f"Failed to load {model_name}", error=str(result))
                # Remove failed model from ensemble
                if model_name in self.models:
                    del self.models[model_name]
                self.ensemble_weights[model_name] = 0.0
            else:
                loaded_models.append(model_name)
                logger.info(f"{model_name} model loaded successfully")
        
        # Normalize weights for successfully loaded models
        self._normalize_weights()
        
        self.is_loaded = len(loaded_models) > 0
        load_time = time.time() - self.load_start_time
        
        if self.is_loaded:
            logger.info("Ensemble loading completed", 
                       loaded_models=loaded_models,
                       load_time_seconds=load_time,
                       weights=self.ensemble_weights)
        else:
            raise RuntimeError("Failed to load any models in the ensemble")

    async def _load_single_model(self, model_name: str, model: BaseMLModel) -> None:
        """Load a single model with error handling."""
        try:
            await model.load()
            logger.debug(f"{model_name} loaded", version=model.version)
        except Exception as e:
            logger.error(f"Failed to load {model_name}", error=str(e))
            raise

    def _normalize_weights(self) -> None:
        """Normalize ensemble weights to sum to 1.0."""
        total_weight = sum(w for w in self.ensemble_weights.values() if w > 0)
        if total_weight > 0:
            for model_name in self.ensemble_weights:
                if self.ensemble_weights[model_name] > 0:
                    self.ensemble_weights[model_name] /= total_weight
        
        logger.debug("Weights normalized", weights=self.ensemble_weights)

    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Make intelligent ensemble prediction."""
        if not self.is_loaded:
            raise RuntimeError("Ensemble not loaded")

        # Ensure proper feature format
        if not isinstance(features, np.ndarray):
            features = np.asarray(features, dtype=float)
        if features.ndim != 1:
            features = features.ravel()

        logger.info("Processing ensemble prediction request", 
                   features_count=int(features.size), 
                   nonzero_features=int(np.count_nonzero(features)))

        start_time = time.time()
        
        # Try primary strategy first
        for strategy_name, model_list in self.fallback_strategies.items():
            try:
                predictions = await self._get_predictions_from_models(features, model_list)
                
                if predictions:
                    # Calculate ensemble result
                    result = self._calculate_ensemble_result(predictions, strategy_name)
                    
                    inference_time = (time.time() - start_time) * 1000
                    self._update_metrics(strategy_name, inference_time, predictions)
                    
                    # Dynamic weight adjustment based on performance
                    await self._adjust_weights_based_on_performance(predictions)
                    
                    logger.info("Ensemble prediction completed", 
                               strategy_used=strategy_name,
                               models_used=len(predictions),
                               inference_time_ms=inference_time,
                               ensemble_score=result["ensemble_score"],
                               consensus_strength=result.get("consensus_strength", 0.0))
                    
                    return {
                        **result,
                        "individual_predictions": predictions,
                        "total_processing_time_ms": inference_time,
                        "strategy_used": strategy_name,
                        "models_used": [p["ml_model"] for p in predictions],
                        "ensemble_weights_used": {p["ml_model"]: self.ensemble_weights.get(p["ml_model"], 0.0) for p in predictions}
                    }
                    
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed", error=str(e))
                continue
        
        # If all strategies fail, return emergency fallback
        inference_time = (time.time() - start_time) * 1000
        self._update_metrics("emergency_fallback", inference_time, [])
        
        return {
            "ensemble_score": 0.5,  # Safe default
            "severity": "normal",
            "is_anomaly": False,
            "confidence": 0.0,
            "individual_predictions": [],
            "total_processing_time_ms": inference_time,
            "strategy_used": "emergency_fallback",
            "models_used": [],
            "error": "All ensemble strategies failed"
        }

    async def _get_predictions_from_models(self, features: np.ndarray, model_list: list[str]) -> list[dict[str, Any]]:
        """Get predictions from specified models."""
        predictions = []
        
        # Create prediction tasks
        tasks = []
        available_models = []
        
        for model_name in model_list:
            if model_name in self.models and self.models[model_name].is_loaded:
                model = self.models[model_name]
                
                # Adjust features for model if needed
                model_features = self._adjust_features_for_model(features, model)
                
                tasks.append(self._predict_with_timeout(model, model_features, model_name))
                available_models.append(model_name)
        
        if not tasks:
            return []
        
        # Execute predictions concurrently with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for model_name, result in zip(available_models, results):
            if isinstance(result, Exception):
                logger.warning(f"Model {model_name} prediction failed", error=str(result))
                self.performance_metrics["model_performance"][model_name]["error_count"] += 1
                self.performance_metrics["model_performance"][model_name]["last_error_time"] = time.time()
            else:
                # Add model name and weight to prediction
                result["ml_model"] = model_name
                result["model_weight"] = self.ensemble_weights.get(model_name, 0.0)
                predictions.append(result)
                
                # Update performance metrics
                perf = self.performance_metrics["model_performance"][model_name]
                perf["success_count"] += 1
                perf["avg_confidence"] = (perf["avg_confidence"] * (perf["success_count"] - 1) + result.get("confidence", 0.0)) / perf["success_count"]
                perf["avg_processing_time"] = (perf["avg_processing_time"] * (perf["success_count"] - 1) + result.get("processing_time_ms", 0.0)) / perf["success_count"]
        
        return predictions

    async def _predict_with_timeout(self, model: BaseMLModel, features: np.ndarray, model_name: str, timeout: float = 5.0) -> dict[str, Any]:
        """Make prediction with timeout to prevent hanging."""
        try:
            return await asyncio.wait_for(model.predict(features), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Model {model_name} prediction timed out after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"Model {model_name} prediction failed: {str(e)}")

    def _adjust_features_for_model(self, features: np.ndarray, model: BaseMLModel) -> np.ndarray:
        """Adjust feature vector for specific model requirements."""
        expected_features = model.metadata.get("features_count", features.size)
        
        if features.size != expected_features:
            if features.size > expected_features:
                return features[:expected_features]
            else:
                return np.pad(features, (0, expected_features - features.size), constant_values=0.0)
        
        return features

    def _calculate_ensemble_result(self, predictions: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
        """Calculate weighted ensemble result from individual predictions."""
        if not predictions:
            return {
                "ensemble_score": 0.5, 
                "severity": "normal", 
                "confidence": 0.0, 
                "is_anomaly": False,
                "consensus_strength": 0.0
            }
        
        # Calculate weighted ensemble score
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        
        for pred in predictions:
            model_name = pred["ml_model"]
            weight = self.ensemble_weights.get(model_name, 0.0)
            
            if weight > 0:
                weighted_score += pred["score"] * weight
                weighted_confidence += pred["confidence"] * weight
                total_weight += weight
        
        # Normalize if we have weights
        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            # Fallback to simple average
            final_score = np.mean([p["score"] for p in predictions])
            final_confidence = np.mean([p["confidence"] for p in predictions])
        
        # Calculate consensus strength
        scores = [p["score"] for p in predictions]
        consensus_strength = 1.0 - (np.std(scores) / max(np.mean(scores), 0.1))  # Lower std = higher consensus
        consensus_strength = max(0.0, min(1.0, consensus_strength))
        
        # Adjust confidence based on consensus
        final_confidence = final_confidence * (0.7 + 0.3 * consensus_strength)
        
        # Determine severity based on score
        if final_score < 0.3:
            severity = "normal"
        elif final_score < 0.6:
            severity = "warning"
        else:
            severity = "critical"
        
        return {
            "ensemble_score": float(final_score),
            "severity": severity,
            "is_anomaly": final_score > settings.prediction_threshold,
            "confidence": float(final_confidence),
            "consensus_strength": float(consensus_strength),
            "contributing_models": len(predictions),
        }

    async def _adjust_weights_based_on_performance(self, predictions: list[dict[str, Any]]) -> None:
        """Dynamically adjust ensemble weights based on model performance."""
        if len(predictions) < 2 or self.prediction_count % 50 != 0:  # Adjust every 50 predictions
            return
        
        try:
            # Calculate performance scores for each model
            performance_scores = {}
            
            for model_name in self.ensemble_weights.keys():
                if model_name not in self.models:
                    continue
                    
                perf = self.performance_metrics["model_performance"][model_name]
                
                if perf["success_count"] > 0:
                    # Performance score based on success rate, confidence, and speed
                    success_rate = perf["success_count"] / (perf["success_count"] + perf["error_count"])
                    avg_confidence = perf["avg_confidence"]
                    speed_score = max(0.1, 1.0 - (perf["avg_processing_time"] / 1000.0))  # Normalize to 0-1
                    
                    # Combined performance score
                    performance_scores[model_name] = (success_rate * 0.5 + 
                                                     avg_confidence * 0.3 + 
                                                     speed_score * 0.2)
                else:
                    performance_scores[model_name] = 0.0
            
            # Adjust weights based on performance
            if performance_scores:
                total_performance = sum(performance_scores.values())
                
                if total_performance > 0:
                    # Update weights with momentum (gradual adjustment)
                    momentum = 0.1  # Adjust slowly to avoid instability
                    
                    for model_name in self.ensemble_weights.keys():
                        if model_name in performance_scores:
                            target_weight = performance_scores[model_name] / total_performance
                            current_weight = self.ensemble_weights[model_name]
                            
                            # Gradual adjustment
                            new_weight = current_weight * (1 - momentum) + target_weight * momentum
                            self.ensemble_weights[model_name] = new_weight
                    
                    # Normalize weights
                    self._normalize_weights()
                    self.performance_metrics["weight_adjustments"] += 1
                    
                    logger.debug("Ensemble weights adjusted", 
                               new_weights=self.ensemble_weights,
                               performance_scores=performance_scores)
        
        except Exception as e:
            logger.warning("Weight adjustment failed", error=str(e))

    def _update_metrics(self, strategy: str, inference_time: float, predictions: list[dict[str, Any]]) -> None:
        """Update performance metrics."""
        self.performance_metrics["predictions_total"] += 1
        self.performance_metrics["inference_times"].append(inference_time)
        self.performance_metrics["fallback_usage"][strategy] = self.performance_metrics["fallback_usage"].get(strategy, 0) + 1
        
        if predictions:
            scores = [p["score"] for p in predictions]
            self.performance_metrics["ensemble_consensus"].append(np.std(scores))
        
        self.prediction_count += 1

    async def warmup(self, warmup_samples: int = 10) -> None:
        """Warm up all models in the ensemble."""
        logger.info("Warming up ensemble models", samples=warmup_samples)
        
        dummy_features = np.random.rand(warmup_samples, 25)  # Standard feature count
        
        for i in range(warmup_samples):
            try:
                result = await self.predict(dummy_features[i])
                logger.debug(
                    f"Warmup sample {i+1}/{warmup_samples} completed",
                    strategy=result.get("strategy_used", "unknown"),
                    models_used=len(result.get("models_used", [])),
                    time_ms=round(result.get("total_processing_time_ms", 0), 1)
                )
            except Exception as e:
                logger.warning(f"Warmup sample {i+1} failed", error=str(e))
        
        logger.info("Ensemble warmup completed", 
                   performance_metrics=self.get_performance_metrics())

    def is_ready(self) -> bool:
        """Check if the ensemble is ready (at least one model loaded)."""
        return self.is_loaded and any(model.is_loaded for model in self.models.values())

    def get_loaded_models(self) -> list[str]:
        """Return list of successfully loaded models."""
        return [name for name, model in self.models.items() if model.is_loaded]

    def get_model_info(self) -> dict[str, Any]:
        """Get information about all models in the ensemble."""
        model_info = {}
        
        for model_name, model in self.models.items():
            if model.is_loaded:
                info = model.get_model_info()
                info["ensemble_weight"] = self.ensemble_weights.get(model_name, 0.0)
                info["performance"] = self.performance_metrics["model_performance"].get(model_name, {})
                model_info[model_name] = info
            else:
                model_info[model_name] = {
                    "name": model_name,
                    "is_loaded": False,
                    "ensemble_weight": 0.0,
                    "error": "Failed to load"
                }
        
        return model_info

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive ensemble performance metrics."""
        if not self.performance_metrics["inference_times"]:
            return {"predictions_total": 0, "note": "No predictions made yet"}
        
        times = self.performance_metrics["inference_times"]
        consensus_scores = self.performance_metrics["ensemble_consensus"]
        
        metrics = {
            "predictions_total": self.performance_metrics["predictions_total"],
            "average_response_time_ms": np.mean(times),
            "p95_response_time_ms": np.percentile(times, 95),
            "p99_response_time_ms": np.percentile(times, 99),
            "min_response_time_ms": np.min(times),
            "max_response_time_ms": np.max(times),
            "current_ensemble_weights": self.ensemble_weights.copy(),
            "weight_adjustments": self.performance_metrics["weight_adjustments"],
            "fallback_usage": self.performance_metrics["fallback_usage"].copy(),
            "model_performance": self.performance_metrics["model_performance"].copy(),
        }
        
        if consensus_scores:
            metrics["average_consensus"] = np.mean(consensus_scores)
            metrics["consensus_stability"] = 1.0 - np.std(consensus_scores)
        
        return metrics

    async def cleanup(self) -> None:
        """Clean up all ensemble resources."""
        logger.info("Cleaning up ensemble models")
        
        cleanup_tasks = []
        for model in self.models.values():
            cleanup_tasks.append(model.cleanup())
        
        # Clean up all models concurrently
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.models.clear()
        self.is_loaded = False
        
        logger.info("Ensemble cleanup completed")