"""
Adaptive Anomaly Detection Model
Online learning model that adapts to changing hydraulic system patterns
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import settings

from .base_model import BaseMLModel

logger = structlog.get_logger()


class AdaptiveModel(BaseMLModel):
    """
    Adaptive model that learns from streaming data and adjusts to system changes.

    Features:
    - Online learning with incremental updates
    - Sliding window for recent data patterns
    - Adaptive threshold adjustment
    - Concept drift detection
    - Memory-efficient streaming processing
    """

    def __init__(self, model_name: str = "adaptive"):
        super().__init__(model_name)
        self.model = None
        self.scaler = StandardScaler()

        # Adaptive components
        self.base_model = None  # Primary model (Isolation Forest)
        self.window_size = 1000  # Sliding window size
        self.data_window = deque(maxlen=self.window_size)
        self.label_window = deque(maxlen=self.window_size)  # For supervised adaptation
        self.prediction_history = deque(maxlen=100)  # Recent predictions

        # Adaptive thresholds
        self.base_threshold = settings.prediction_threshold
        self.adaptive_threshold = self.base_threshold
        self.threshold_history = deque(maxlen=50)

        # Drift detection
        self.drift_detector = {
            "window_scores": deque(maxlen=100),
            "drift_threshold": 0.3,
            "adaptation_rate": 0.1,
            "last_adaptation": 0,
            "adaptations_count": 0,
        }

        # Performance tracking
        self.adaptation_metrics = {
            "total_adaptations": 0,
            "drift_detections": 0,
            "threshold_adjustments": 0,
            "last_retrain_time": None,
            "window_accuracy": 0.0,
        }

        self.metadata["features_count"] = 25
        self.metadata["is_adaptive"] = True

    async def load(self) -> None:
        """Load Adaptive model from disk or create new adaptive model."""
        start_time = time.time()
        model_path = Path(settings.model_path) / "adaptive_model.joblib"
        logger.info("Loading Adaptive model", path=str(model_path))

        try:
            if model_path.exists():
                loaded_data = joblib.load(model_path)

                if isinstance(loaded_data, dict) and "base_model" in loaded_data:
                    # Load adaptive components
                    self.base_model = loaded_data["base_model"]
                    self.scaler = loaded_data.get("scaler", StandardScaler())
                    self.adaptive_threshold = loaded_data.get("adaptive_threshold", self.base_threshold)
                    self.adaptation_metrics = loaded_data.get("adaptation_metrics", self.adaptation_metrics)

                    # Restore windows (if saved)
                    if "data_window" in loaded_data:
                        self.data_window = deque(loaded_data["data_window"], maxlen=self.window_size)

                    if "features_count" in loaded_data:
                        self.metadata["features_count"] = int(loaded_data["features_count"])

                    logger.info(
                        "Real Adaptive model loaded",
                        adaptive_threshold=self.adaptive_threshold,
                        adaptations=self.adaptation_metrics.get("total_adaptations", 0),
                    )
                else:
                    # Legacy format or direct model
                    logger.warning("Legacy adaptive model format, creating new model")
                    await self._create_adaptive_model()
            else:
                logger.info("Adaptive model file not found, creating new adaptive model", path=str(model_path))
                await self._create_adaptive_model()

            self.is_loaded = True
            self.load_time = time.time() - start_time
            self.version = "v1.0.0-adaptive"

            logger.info(
                "Adaptive model loaded successfully",
                load_time_seconds=self.load_time,
                version=self.version,
                adaptive_threshold=self.adaptive_threshold,
                window_size=self.window_size,
                features_count=self.metadata.get("features_count"),
            )

        except Exception as e:
            logger.error("Adaptive loading failed, creating new model", error=str(e))
            await self._create_adaptive_model()

    async def _create_adaptive_model(self) -> None:
        """Create new adaptive model with initial training."""
        logger.info("Creating new adaptive model")

        try:
            # Generate initial training data
            n_samples = 2000
            n_features = self.metadata.get("features_count", 25)

            # Create realistic hydraulic data with temporal patterns
            X_init = self._generate_hydraulic_data(n_samples, n_features)

            # Fit initial scaler
            self.scaler.fit(X_init)
            X_scaled = self.scaler.transform(X_init)

            # Initialize base model (Isolation Forest for unsupervised anomaly detection)
            self.base_model = IsolationForest(
                n_estimators=100,
                contamination=0.05,  # 5% anomalies expected
                max_samples="auto",
                max_features=1.0,
                bootstrap=False,
                random_state=42,
                n_jobs=-1,
            )

            # Initial training
            self.base_model.fit(X_scaled)

            # Initialize data window with some initial data
            for sample in X_scaled[-100:]:  # Keep last 100 samples in window
                self.data_window.append(sample)

            # Calculate initial threshold based on training data
            initial_scores = self.base_model.decision_function(X_scaled)
            self.adaptive_threshold = np.percentile(initial_scores, 5)  # 5th percentile as threshold

            self.is_trained = True
            self.metadata["accuracy_score"] = 0.80  # Initial estimate

            logger.info(
                "Adaptive model created successfully",
                initial_threshold=self.adaptive_threshold,
                training_samples=n_samples,
                window_size=len(self.data_window),
            )

        except Exception as e:
            logger.error("Failed to create adaptive model", error=str(e))
            raise

    def _generate_hydraulic_data(self, n_samples: int, n_features: int) -> np.ndarray:
        """Generate realistic hydraulic system data with temporal patterns."""
        data = np.zeros((n_samples, n_features))

        # Create time-varying patterns
        time_steps = np.linspace(0, 10 * np.pi, n_samples)

        for i in range(n_features):
            # Base signal with trend and seasonality
            base_signal = np.sin(time_steps + i * 0.5) + 0.1 * time_steps

            # Add noise
            noise = np.random.normal(0, 0.3, n_samples)

            # Add some concept drift (gradual change)
            drift = 0.001 * time_steps * np.random.randn()

            data[:, i] = base_signal + noise + drift

            # Add occasional anomalies
            anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
            data[anomaly_indices, i] += np.random.uniform(2, 5, len(anomaly_indices))

        return data

    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Make adaptive prediction and update model."""
        if not self.is_loaded or self.base_model is None:
            raise RuntimeError("Adaptive model not loaded")

        # Ensure proper feature format
        features = self._ensure_vector(features)

        start_time = time.time()
        try:
            # Scale features
            features_2d = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features_2d)[0]

            # Get base prediction score
            decision_score = self.base_model.decision_function(features_2d)[0]

            # Convert decision score to probability-like score
            prediction_score = self._score_to_probability(decision_score)

            # Apply adaptive threshold
            is_anomaly = decision_score < self.adaptive_threshold

            # Calculate confidence based on distance from adaptive threshold
            distance_from_threshold = abs(decision_score - self.adaptive_threshold)
            confidence = min(0.6 + distance_from_threshold * 0.5, 0.95)

            # Update sliding window
            self.data_window.append(features_scaled)
            self.prediction_history.append(
                {
                    "score": prediction_score,
                    "decision_score": decision_score,
                    "timestamp": time.time(),
                    "is_anomaly": is_anomaly,
                }
            )

            # Detect concept drift and adapt if necessary
            await self._detect_and_adapt(features_scaled, decision_score)

            processing_time = (time.time() - start_time) * 1000
            self.update_stats(processing_time / 1000)

            result = {
                "score": prediction_score,
                "confidence": confidence,
                "is_anomaly": is_anomaly,
                "processing_time_ms": processing_time,
                "adaptive_info": {
                    "adaptive_threshold": float(self.adaptive_threshold),
                    "base_threshold": float(self.base_threshold),
                    "decision_score": float(decision_score),
                    "window_size": len(self.data_window),
                    "total_adaptations": self.adaptation_metrics["total_adaptations"],
                },
            }

            logger.debug(
                "Adaptive prediction completed",
                score=prediction_score,
                decision_score=decision_score,
                adaptive_threshold=self.adaptive_threshold,
                is_anomaly=is_anomaly,
                confidence=confidence,
            )

            return result

        except Exception as e:
            logger.error("Adaptive prediction failed", error=str(e), features_shape=features.shape)
            raise

    def _score_to_probability(self, decision_score: float) -> float:
        """Convert Isolation Forest decision score to probability-like score."""
        # Isolation Forest scores are typically in range [-1, 1]
        # Convert to [0, 1] where higher values indicate higher anomaly probability
        normalized_score = (1 - decision_score) / 2
        return np.clip(normalized_score, 0.0, 1.0)

    async def _detect_and_adapt(self, features: np.ndarray, decision_score: float) -> None:
        """Detect concept drift and adapt model if necessary."""
        try:
            # Add score to drift detection window
            self.drift_detector["window_scores"].append(decision_score)

            # Check for drift every 50 predictions
            if len(self.drift_detector["window_scores"]) >= 50:
                current_window = list(self.drift_detector["window_scores"])[-25:]
                previous_window = list(self.drift_detector["window_scores"])[-50:-25]

                # Statistical test for drift (Kolmogorov-Smirnov like)
                if len(previous_window) >= 25:
                    current_mean = np.mean(current_window)
                    previous_mean = np.mean(previous_window)

                    # Detect significant shift in score distribution
                    score_shift = abs(current_mean - previous_mean)

                    if score_shift > self.drift_detector["drift_threshold"]:
                        logger.info(
                            "Concept drift detected",
                            score_shift=score_shift,
                            threshold=self.drift_detector["drift_threshold"],
                        )

                        await self._adapt_model()
                        self.drift_detector["drift_detections"] += 1

            # Adaptive threshold adjustment
            await self._adjust_threshold()

        except Exception as e:
            logger.warning("Drift detection failed", error=str(e))

    async def _adapt_model(self) -> None:
        """Adapt the model based on recent data."""
        try:
            if len(self.data_window) < 100:
                return

            logger.info("Adapting model to recent data patterns")

            # Get recent data from window
            recent_data = np.array(list(self.data_window))

            # Retrain base model on recent data
            self.base_model.fit(recent_data)

            # Update adaptive threshold based on new model
            recent_scores = self.base_model.decision_function(recent_data)
            self.adaptive_threshold = np.percentile(recent_scores, 5)

            # Update metrics
            self.adaptation_metrics["total_adaptations"] += 1
            self.adaptation_metrics["last_retrain_time"] = time.time()

            logger.info(
                "Model adaptation completed",
                new_threshold=self.adaptive_threshold,
                training_samples=len(recent_data),
                total_adaptations=self.adaptation_metrics["total_adaptations"],
            )

        except Exception as e:
            logger.error("Model adaptation failed", error=str(e))

    async def _adjust_threshold(self) -> None:
        """Adjust threshold based on recent prediction patterns."""
        if len(self.prediction_history) < 20:
            return

        recent_predictions = list(self.prediction_history)[-20:]
        recent_scores = [p["decision_score"] for p in recent_predictions]

        # Calculate adaptive threshold based on recent score distribution
        score_std = np.std(recent_scores)
        score_mean = np.mean(recent_scores)

        # Adjust threshold if variance is high (system is changing)
        if score_std > 0.5:  # High variance threshold
            adjustment = self.drift_detector["adaptation_rate"] * score_std
            new_threshold = self.adaptive_threshold * (1 - adjustment)

            # Bound the adjustment
            max_adjustment = 0.2 * abs(self.base_threshold)
            new_threshold = np.clip(
                new_threshold, self.base_threshold - max_adjustment, self.base_threshold + max_adjustment
            )

            if abs(new_threshold - self.adaptive_threshold) > 0.01:
                self.adaptive_threshold = new_threshold
                self.adaptation_metrics["threshold_adjustments"] += 1

                logger.debug(
                    "Threshold adjusted",
                    old_threshold=self.adaptive_threshold,
                    new_threshold=new_threshold,
                    score_std=score_std,
                )

    def get_model_info(self) -> dict[str, Any]:
        """Get detailed Adaptive model information."""
        base_info = super().get_model_info()

        adaptive_specific = {
            "model_type": "Adaptive Online Learning",
            "base_model": "Isolation Forest",
            "window_size": self.window_size,
            "current_window_fill": len(self.data_window),
            "adaptive_threshold": float(self.adaptive_threshold),
            "base_threshold": float(self.base_threshold),
            "adaptation_metrics": self.adaptation_metrics,
            "drift_detection_enabled": True,
            "online_learning": True,
        }

        return {**base_info, **adaptive_specific}

    def get_adaptation_stats(self) -> dict[str, Any]:
        """Get adaptation and drift detection statistics."""
        recent_predictions = list(self.prediction_history)[-50:] if self.prediction_history else []

        return {
            "total_adaptations": self.adaptation_metrics["total_adaptations"],
            "drift_detections": self.adaptation_metrics.get("drift_detections", 0),
            "threshold_adjustments": self.adaptation_metrics.get("threshold_adjustments", 0),
            "current_threshold": float(self.adaptive_threshold),
            "threshold_drift": float(abs(self.adaptive_threshold - self.base_threshold)),
            "window_utilization": len(self.data_window) / self.window_size,
            "recent_anomaly_rate": sum(1 for p in recent_predictions if p["is_anomaly"])
            / max(len(recent_predictions), 1),
            "last_adaptation_time": self.adaptation_metrics.get("last_retrain_time"),
        }

    async def save_model(self, path: Path | None = None) -> None:
        """Save adaptive model state."""
        if not self.is_loaded:
            logger.warning("Cannot save unloaded adaptive model")
            return

        save_path = path or (Path(settings.model_path) / "adaptive_model.joblib")

        model_data = {
            "base_model": self.base_model,
            "scaler": self.scaler,
            "adaptive_threshold": self.adaptive_threshold,
            "adaptation_metrics": self.adaptation_metrics,
            "features_count": self.metadata["features_count"],
            "data_window": list(self.data_window),  # Save recent data
            "version": self.version,
            "save_time": time.time(),
        }

        try:
            joblib.dump(model_data, save_path)
            logger.info("Adaptive model saved", path=str(save_path))
        except Exception as e:
            logger.error("Failed to save adaptive model", error=str(e))

    async def cleanup(self) -> None:
        """Clean up Adaptive model resources."""
        logger.info("Cleaning up Adaptive model")

        # Save current state before cleanup
        await self.save_model()

        self.base_model = None
        self.data_window.clear()
        self.label_window.clear()
        self.prediction_history.clear()
        self.threshold_history.clear()
        self.drift_detector["window_scores"].clear()

        self.is_loaded = False
        self.is_trained = False

        logger.info("Adaptive model cleanup completed")
