from __future__ import annotations

import json
import hashlib
from typing import Iterable

import numpy as np

from api.schemas import FeatureVector


def vector_from_feature_vector(fv: FeatureVector) -> np.ndarray:
    """Преобразует FeatureVector в 1D numpy.ndarray (стабильный порядок).
    - Всегда использует порядок из fv.feature_names
    - Приводит dtype=float
    - Поднимает ValueError при отсутствующих признаках
    """
    if not fv.feature_names:
        raise ValueError("feature_names is empty")

    try:
        vector = np.array([fv.features[name] for name in fv.feature_names], dtype=float)
    except KeyError as e:
        raise ValueError(f"missing feature in features dict: {e}") from e

    if vector.ndim != 1:
        vector = vector.ravel()
    return vector


def build_feature_cache_key(vector: np.ndarray, names: Iterable[str], prefix: str = "ml_pred:") -> str:
    """Генерирует устойчивый cache key по нормализованным признакам.
    - Округляет до 6 знаков для устойчивости
    - Хэширует JSON с именами и значениями
    """
    if vector.ndim != 1:
        raise ValueError("expected 1D feature vector for cache key")

    rounded = [round(float(v), 6) for v in vector.tolist()]
    payload = {"names": list(names), "values": rounded}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return f"{prefix}{digest}"
