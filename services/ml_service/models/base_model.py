"""
Base ML Model for Hydraulic Systems Diagnostics
Abstract base class для всех ML моделей
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog

from config import settings

logger = structlog.get_logger()


class BaseMLModel(ABC):
    """
    Абстрактный базовый класс для ML моделей.

    Контракт:
    - predict(features: np.ndarray 1D) -> dict[str, Any] с ключами
      {score: float, confidence: float, processing_time_ms: float}
    - Внутри метода модель сама приводит 1D → (1, -1) при необходимости
    - В случае некорректного формата входа — ValueError
    """

    def __init__(self, model_name: str, model_file: str | None = None):
        self.model_name = model_name
        self.model_file = model_file or f"{model_name}_model.joblib"
        self.version = "1.0.0"
        self.model = None
        self.is_loaded = False
        self.is_trained = False
        self.load_time = None
        self.prediction_count = 0
        self.total_inference_time = 0.0

        self.metadata: dict[str, Any] = {
            "features_count": 25,
            "model_size_mb": 0.0,
            "last_used": None,
            "accuracy_score": 0.0,
        }

    async def load(self) -> None:
        start_time = time.time()
        model_path = Path(settings.model_path) / self.model_file

        logger.info(f"Loading {self.model_name} model", path=str(model_path))

        try:
            if model_path.exists():
                self.model = joblib.load(model_path)
                self.is_loaded = True
                self.is_trained = True
                self.load_time = time.time() - start_time
                self.metadata["model_size_mb"] = model_path.stat().st_size / (1024 * 1024)
                logger.info(
                    f"{self.model_name} model loaded",
                    load_time_ms=self.load_time * 1000,
                    size_mb=self.metadata["model_size_mb"],
                )
            else:
                logger.warning("Model file not found, creating mock model", path=str(model_path))
                await self._create_mock_model()
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}", error=str(e))
            await self._create_mock_model()

    async def _create_mock_model(self) -> None:
        import numpy as np
        from sklearn.ensemble import IsolationForest

        logger.info(f"Creating mock {self.model_name} model for development")
        self.model = IsolationForest(contamination=0.1, random_state=42)
        mock_data = np.random.rand(1000, self.metadata.get("features_count", 25))  # nosec B311
        self.model.fit(mock_data)
        self.is_loaded = True
        self.is_trained = True
        self.load_time = 0.1
        self.metadata["accuracy_score"] = 0.85
        logger.info(f"Mock {self.model_name} model created")

    @abstractmethod
    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Возвращает {score, confidence, processing_time_ms}."""
        raise NotImplementedError

    def _ensure_vector(self, features: Any) -> np.ndarray:
        """Строго приводит вход к 1D numpy.ndarray[float]."""
        arr = np.asarray(features, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError("Invalid features shape; expected 1D feature vector")
        expected = int(self.metadata.get("features_count", 25))
        if len(arr) != expected:
            raise ValueError(f"Invalid features length: {len(arr)} != {expected}")
        return arr

    async def predict_batch(self, features_batch: np.ndarray) -> list[dict[str, Any]]:
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")

        results: list[dict[str, Any]] = []
        start_time = time.time()

        try:
            for features in features_batch:
                prediction = await self.predict(features)
                results.append(prediction)

            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(features_batch)
            logger.info(
                "Batch prediction completed",
                model=self.model_name,
                batch_size=len(features_batch),
                total_time_ms=total_time,
                avg_time_ms=avg_time,
            )
            return results
        except Exception as e:
            logger.error(f"Batch prediction failed for {self.model_name}", error=str(e))
            raise

    def update_stats(self, processing_time: float) -> None:
        self.prediction_count += 1
        self.total_inference_time += processing_time
        self.metadata["last_used"] = time.time()

    def get_stats(self) -> dict[str, Any]:
        avg_time = self.total_inference_time / self.prediction_count if self.prediction_count > 0 else 0.0
        return {
            "model_name": self.model_name,
            "version": self.version,
            "is_loaded": self.is_loaded,
            "is_trained": self.is_trained,
            "predictions_count": self.prediction_count,
            "average_inference_time_ms": avg_time * 1000,
            "load_time_seconds": self.load_time,
            **self.metadata,
        }

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "version": self.version,
            "description": f"{self.model_name} model for hydraulic anomaly detection",
            "accuracy": self.metadata.get("accuracy_score", 0.0),
            "last_trained": self.metadata.get("last_trained", None),
            "size_mb": self.metadata.get("model_size_mb", 0.0),
            "features_count": self.metadata.get("features_count", 0),
            "is_loaded": self.is_loaded,
            "load_time_ms": (self.load_time * 1000) if self.load_time else None,
        }

    def get_version(self) -> str:
        return self.version

    async def cleanup(self) -> None:
        logger.info(f"Cleaning up {self.model_name} model")
        self.model = None
        self.is_loaded = False
        self.is_trained = False

    def __str__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.model_name} ({status})"
