"""
CatBoost Anomaly Detection Model
Enterprise production model for hydraulic systems (HELM replacement)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

from config import settings

from .base_model import BaseMLModel

logger = structlog.get_logger()


class CatBoostModel(BaseMLModel):
    def __init__(self, model_name: str = "catboost"):
        super().__init__(model_name)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance_ = None
        self.training_metrics = {}

        # По умолчанию 25, но при загрузке реальной модели обновим
        self.metadata["features_count"] = 25

    async def load(self) -> None:
        start_time = time.time()
        model_path = Path(settings.model_path) / "catboost_model.joblib"
        logger.info("Loading catboost model", path=str(model_path))

        try:
            if model_path.exists():
                model_data = joblib.load(model_path)
                self.model = model_data["model"]
                self.scaler = model_data["scaler"]
                self.feature_importance_ = model_data.get("feature_importance")
                self.training_metrics = model_data.get("training_metrics", {})

                # ✅ Проставляем корректное число признаков из артефакта, если есть
                if "features_count" in model_data:
                    self.metadata["features_count"] = int(model_data["features_count"])

                logger.info("Real CatBoost model loaded")
            else:
                logger.warning("Model file not found, creating mock model", path=str(model_path))
                await self._create_mock_model()

            self.is_loaded = True
            self.load_time = time.time() - start_time
            self.version = "v1.0.0-catboost"

            logger.info(
                "CatBoost model loaded",
                load_time_seconds=self.load_time,
                version=self.version,
                is_mock=not model_path.exists(),
                features_count=self.metadata.get("features_count"),
            )

        except Exception as e:
            logger.error("Failed to load CatBoost model", error=str(e))
            raise

    async def _create_mock_model(self) -> None:
        logger.info("Creating mock catboost model for development")
        try:
            mock_features = np.random.randn(100, self.metadata.get("features_count", 25))
            self.scaler.fit(mock_features)
            self.model = CatBoostClassifier(
                **{
                    "iterations": 10,
                    "depth": 3,
                    "learning_rate": 0.1,
                    "task_type": "CPU",
                    "random_seed": 42,
                    "logging_level": "Silent",
                    "allow_writing_files": False,
                }
            )
            mock_labels = np.random.binomial(1, 0.05, 100)
            mock_features_scaled = self.scaler.transform(mock_features)
            self.model.fit(mock_features_scaled, mock_labels, verbose=False)
            self.is_trained = True
            logger.info("Mock catboost model created")
        except Exception as e:
            logger.error("Failed to create mock CatBoost model", error=str(e))
            raise

    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        # ✅ Жестко приводим вход к 1D вектору нужной длины
        features = self._ensure_vector(features)

        start_time = time.time()
        try:
            features_2d = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features_2d)
            probabilities = self.model.predict_proba(features_scaled)
            if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                prediction_score = float(probabilities[0, 1])
            else:
                prediction_score = float(probabilities[0])

            threshold = settings.prediction_threshold
            distance_from_threshold = abs(prediction_score - threshold)
            confidence = min(0.8 + distance_from_threshold * 0.3, 0.95)
            processing_time = (time.time() - start_time) * 1000

            return {
                "score": prediction_score,
                "confidence": confidence,
                "is_anomaly": prediction_score > threshold,
                "processing_time_ms": processing_time,
            }
        except Exception as e:
            logger.error("CatBoost prediction failed", error=str(e))
            raise
