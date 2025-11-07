from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import numpy as np

from api.schemas import FeatureVector


def adaptive_project(fv: FeatureVector, expected_size: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Адаптивно формирует вектор признаков из всего доступного FeatureVector.
    - Стабильный порядок: алфавитная сортировка имен
    - Если expected_size задан: pad/truncate до нужной длины
    - Возвращает (vector, used_feature_names)
    """
    names = sorted(fv.feature_names or list(fv.features.keys()))
    values = [float(fv.features.get(n, 0.0)) for n in names]
    vec = np.asarray(values, dtype=float).ravel()

    if expected_size is not None:
        if vec.size > expected_size:
            vec = vec[:expected_size]
            names = names[:expected_size]
        elif vec.size < expected_size:
            pad = expected_size - vec.size
            vec = np.pad(vec, (0, pad), constant_values=0.0)
            names = names + [f"__pad_{i}" for i in range(pad)]
    return vec, names


def build_feature_cache_key(vector: np.ndarray, names: Iterable[str], prefix: str = "ml_pred:") -> str:
    if vector.ndim != 1:
        raise ValueError("expected 1D feature vector for cache key")
    rounded = [round(float(v), 6) for v in vector.tolist()]
    payload = {"names": list(names), "values": rounded}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return f"{prefix}{digest}"
