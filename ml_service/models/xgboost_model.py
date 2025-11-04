"""
XGBoost Anomaly Detection Model
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
from xgboost import XGBClassifier

from config import settings

from .base_model import BaseMLModel

logger = structlog.get_logger()


class XGBoostModel(BaseMLModel):
    def __init__(self, model_name: str = "xgboost"):
        super().__init__(model_name)
        self.model: XGBClassifier | None = None
        self.metadata["features_count"] = 25

    async def load(self) -> None:
        start_time = time.time()
        model_path = Path(settings.model_path) / "xgboost_model.joblib"
        logger.info("Loading xgboost model", path=str(model_path))

        try:
            if model_path.exists():
                model_data = joblib.load(model_path)
                self.model = model_data["model"]
                if "features_count" in model_data:
                    self.metadata["features_count"] = int(model_data["features_count"])
                logger.info("Real XGBoost model loaded")
            else:
                logger.warning("Model file not found, creating mock model", path=str(model_path))
                await self._create_mock_model()

            self.is_loaded = True
            self.load_time = time.time() - start_time
            self.version = "v1.0.0-xgb"
            logger.info(
                "XGBoost model loaded",
                load_time_seconds=self.load_time,
                version=self.version,
                features_count=self.metadata.get("features_count"),
            )
        except Exception as e:
            logger.error("Failed to load XGBoost model", error=str(e))
            raise

    async def _create_mock_model(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        logger.info("Creating mock xgboost model for development")
        # Используем простой RF как мок, чтобы не тянуть xgboost тренинг
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        mock_x = np.random.rand(200, self.metadata.get("features_count", 25))
        mock_y = np.random.binomial(1, 0.05, 200)
        self.model.fit(mock_x, mock_y)

    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        features = self._ensure_vector(features)
        start_time = time.time()
        try:
            proba = getattr(self.model, "predict_proba", None)
            if proba:
                p = proba(features.reshape(1, -1))
                score = float(p[0, 1] if p.shape[1] > 1 else p[0, 0])
            else:
                # fallback через predict
                pred = self.model.predict(features.reshape(1, -1))
                score = float(pred[0])

            threshold = settings.prediction_threshold
            confidence = min(0.95, 0.8 + abs(score - threshold) * 0.3)
            processing_time = (time.time() - start_time) * 1000
            return {"score": score, "confidence": confidence, "processing_time_ms": processing_time}
        except Exception as e:
            logger.error("XGBoost prediction failed", error=str(e))
            raise
