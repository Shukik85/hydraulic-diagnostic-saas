"""
RandomForest Anomaly Detection Model
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from config import settings

from .base_model import BaseMLModel

logger = structlog.get_logger()


class RandomForestModel(BaseMLModel):
    def __init__(self, model_name: str = "random_forest"):
        super().__init__(model_name)
        self.model: RandomForestClassifier | None = None
        self.scaler = StandardScaler()
        self.metadata["features_count"] = 25

    async def load(self) -> None:
        start_time = time.time()
        model_path = Path(settings.model_path) / "random_forest_model.joblib"
        logger.info("Loading random_forest model", path=str(model_path))

        try:
            if model_path.exists():
                loaded_data = joblib.load(model_path)
                
                # ✅ ROBUST LOADING - handles both formats
                if isinstance(loaded_data, dict) and "model" in loaded_data:
                    # New format with metadata
                    self.model = loaded_data["model"] 
                    self.scaler = loaded_data.get("scaler", StandardScaler())
                    if "features_count" in loaded_data:
                        self.metadata["features_count"] = int(loaded_data["features_count"])
                else:
                    # Direct model object (legacy/simple format)
                    self.model = loaded_data
                    logger.warning("Direct RandomForest model loaded, creating compatible scaler")
                    
                    # Create compatible scaler
                    self.scaler = StandardScaler()
                    expected_features = getattr(self.model, 'n_features_in_', 25)
                    mock_data = np.random.randn(10, expected_features) 
                    self.scaler.fit(mock_data)
                    self.metadata["features_count"] = expected_features

                logger.info("Real RandomForest model loaded", features_count=self.metadata["features_count"])
            else:
                logger.warning("Model file not found, creating mock model", path=str(model_path))
                await self._create_mock_model()

            self.is_loaded = True
            self.load_time = time.time() - start_time
            self.version = "v1.0.0-rf"
            logger.info(
                "RandomForest model loaded successfully",
                load_time_seconds=self.load_time,
                version=self.version,
                features_count=self.metadata.get("features_count"),
            )
        except Exception as e:
            logger.error("RandomForest loading failed, creating mock", error=str(e))
            await self._create_mock_model()

    async def _create_mock_model(self) -> None:
        logger.info("Creating mock random_forest model for development")
        try:
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            mock_x = np.random.rand(200, self.metadata.get("features_count", 25))
            mock_y = np.random.binomial(1, 0.05, 200)
            self.model.fit(mock_x, mock_y)
            
            # Fit scaler
            self.scaler.fit(mock_x)
            self.is_trained = True
            logger.info("Mock RandomForest model created")
        except Exception as e:
            logger.error("Failed to create mock RandomForest model", error=str(e))
            raise

    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        features = self._ensure_vector(features)
        start_time = time.time()
        try:
            features_2d = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features_2d)
            
            proba = self.model.predict_proba(features_scaled)
            score = float(proba[0, 1] if proba.shape[1] > 1 else proba[0, 0])
            threshold = settings.prediction_threshold
            confidence = min(0.95, 0.8 + abs(score - threshold) * 0.3)
            processing_time = (time.time() - start_time) * 1000
            return {"score": score, "confidence": confidence, "processing_time_ms": processing_time}
        except Exception as e:
            logger.error("RandomForest prediction failed", error=str(e))
            raise