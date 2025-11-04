from __future__ import annotations

import json
import hashlib
from typing import Iterable

import numpy as np

from api.schemas import FeatureVector
from .feature_names_25 import EXPECTED_FEATURE_NAMES_25


def vector_from_feature_vector(fv: FeatureVector) -> np.ndarray:
    """Сохранено для обратной совместимости — просто собирает по fv.feature_names.
    Не использовать для моделей напрямую (используйте project_features_25).
    """
    if not fv.feature_names:
        raise ValueError("feature_names is empty")
    vector = np.array([fv.features[name] for name in fv.feature_names], dtype=float)
    return vector.ravel()


def project_features_25(fv: FeatureVector) -> np.ndarray:
    """Проецирует FeatureVector на фиксированный порядок из 25 признаков.
    Отсутствующие признаки заполняются 0.0.
    """
    out = np.zeros(len(EXPECTED_FEATURE_NAMES_25), dtype=float)
    for i, name in enumerate(EXPECTED_FEATURE_NAMES_25):
        out[i] = float(fv.features.get(name, 0.0))
    return out


def build_feature_cache_key(vector: np.ndarray, names: Iterable[str], prefix: str = "ml_pred:") -> str:
    if vector.ndim != 1:
        raise ValueError("expected 1D feature vector for cache key")
    rounded = [round(float(v), 6) for v in vector.tolist()]
    payload = {"names": list(names), "values": rounded}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return f"{prefix}{digest}"
