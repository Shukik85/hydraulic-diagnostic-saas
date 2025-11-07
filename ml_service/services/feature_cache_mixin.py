from __future__ import annotations

import hashlib
import json

import numpy as np


class CacheServiceFeatureMixin:
    """Генерация cache key на основе признаков (вектор + имена).
    Может использоваться вместе с CacheService через множественное наследование или композицию.
    """

    @staticmethod
    def generate_cache_key_from_features(vector: np.ndarray, feature_names: list[str]) -> str:
        if vector.ndim != 1:
            raise ValueError("expected 1D feature vector for cache key")

        rounded = [round(float(v), 6) for v in vector.tolist()]
        payload = {"names": feature_names, "values": rounded}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
        return f"ml_pred:{digest}"
