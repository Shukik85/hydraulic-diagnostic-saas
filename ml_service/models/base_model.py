"""
Base ML Model for Hydraulic Systems Diagnostics
Abstract base class для всех ML моделей
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any  # ✅ Полный typing импорт

import joblib
import numpy as np
import structlog

from config import settings

logger = structlog.get_logger()


class BaseMLModel(ABC):
    """
    Абстрактный базовый класс для ML моделей.

    Обеспечивает:
    - Единообразный интерфейс
    - Метрики производительности
    - Логирование и error handling
    - Валидацию данных
    """

    def __init__(self, model_name: str, model_file: str | None = None):
        self.model_name = model_name
        self.model_file = model_file or f"{model_name}_model.joblib"
        self.version = "1.0.0"
        self.model = None
        self.is_loaded = False
        self.is_trained = False  # ✅ Для компатибильности с CatBoost
        self.load_time = None
        self.prediction_count = 0
        self.total_inference_time = 0.0

        # Метаданные модели
        self.metadata: dict[str, Any] = {
            "features_count": 0,
            "model_size_mb": 0.0,
            "last_used": None,
            "accuracy_score": 0.0,
        }

    async def load(self) -> None:
        """Загрузка модели с диска."""
        start_time = time.time()
        model_path = Path(settings.model_path) / self.model_file

        logger.info(f"Loading {self.model_name} model", path=str(model_path))

        try:
            if model_path.exists():
                self.model = joblib.load(model_path)
                self.is_loaded = True
                self.is_trained = True
                self.load_time = time.time() - start_time

                # Обновление метаданных
                self.metadata["model_size_mb"] = model_path.stat().st_size / (1024 * 1024)

                logger.info(
                    f"{self.model_name} model loaded",
                    load_time_ms=self.load_time * 1000,
                    size_mb=self.metadata["model_size_mb"],
                )
            else:
                # Создаем мок модель для разработки
                logger.warning("Model file not found, creating mock model", path=str(model_path))
                await self._create_mock_model()

        except Exception as e:
            logger.error(f"Failed to load {self.model_name}", error=str(e))
            # Создаем мок модель как fallback
            await self._create_mock_model()

    async def _create_mock_model(self) -> None:
        """Создание мок модели для разработки и тестирования."""
        from sklearn.ensemble import IsolationForest

        logger.info(f"Creating mock {self.model_name} model for development")

        # Простая IsolationForest как mock
        self.model = IsolationForest(contamination=0.1, random_state=42)

        # Обучаем на случайных данных
        # bandit: skip (B311 - это только для mock model)
        mock_data = np.random.rand(1000, 25)  # nosec B311
        self.model.fit(mock_data)

        self.is_loaded = True
        self.is_trained = True
        self.load_time = 0.1  # Быстрая загрузка mock
        self.metadata["features_count"] = 25
        self.metadata["accuracy_score"] = 0.85  # Mock accuracy

        logger.info(f"Mock {self.model_name} model created")

    @abstractmethod
    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        """
        Абстрактный метод предсказания.

        Args:
            features: Массив признаков

        Returns:
            Dict с ключами: score, confidence, processing_time_ms
        """
        pass

    async def predict_batch(self, features_batch: np.ndarray) -> list[dict[str, Any]]:
        """Пакетное предсказание для оптимизации."""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")

        results: list[dict[str, Any]] = []
        start_time = time.time()

        try:
            # Пакетная обработка через основной метод
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

    def validate_features(self, features: np.ndarray) -> bool:
        """Валидация входных признаков."""
        if features is None:
            return False

        if not isinstance(features, np.ndarray):
            return False

        if features.ndim == 1:
            expected_features = self.metadata.get("features_count", 25)
            return len(features) == expected_features
        elif features.ndim == 2:
            expected_features = self.metadata.get("features_count", 25)
            return features.shape[1] == expected_features

        return False

    def update_stats(self, processing_time: float) -> None:
        """Обновление статистики модели."""
        self.prediction_count += 1
        self.total_inference_time += processing_time
        self.metadata["last_used"] = time.time()

    def get_stats(self) -> dict[str, Any]:
        """Получение статистики модели."""
        avg_time = (
            self.total_inference_time / self.prediction_count if self.prediction_count > 0 else 0.0
        )

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
        """Получение полной информации о модели."""
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
        """Получение версии модели."""
        return self.version

    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        logger.info(f"Cleaning up {self.model_name} model")

        self.model = None
        self.is_loaded = False
        self.is_trained = False

    def __str__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.model_name} ({status})"
