"""
Redis Cache Service for ML Predictions
Enterprise кеширование предсказаний с TTL
"""

import hashlib
import json
import time
from typing import Any

import redis.asyncio as aioredis  # ✅ Замена aioredis на redis.asyncio
import structlog

from api.schemas import SensorDataBatch
from config import settings

logger = structlog.get_logger()


class CacheService:
    """
    Redis cache service для оптимизации ML inference.

    Особенности:
    - Кеширование предсказаний по хешу признаков
    - TTL 5 минут (настраиваемо)
    - Метрики cache hit rate
    - Connection pooling
    """

    def __init__(self):
        self.redis: aioredis.Redis | None = None
        self.connection_pool = None
        self.cache_prefix = "ml_pred:"
        self.hit_count = 0
        self.miss_count = 0
        self.is_mock = False
        self.mock_cache: dict[str, dict[str, Any]] = {}  # Mock cache fallback

    async def connect(self) -> None:
        """Подключение к Redis (с fallback на mock)."""
        try:
            self.connection_pool = aioredis.ConnectionPool.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )

            self.redis = aioredis.Redis(connection_pool=self.connection_pool)

            # Проверка подключения
            await self.redis.ping()
            self.is_mock = False
            logger.info("Redis cache connected", url=settings.redis_url)

        except Exception as e:
            logger.warning(
                "Redis connection failed, using mock cache",
                error=str(e),
                redis_url=settings.redis_url
            )
            # Fallback на mock cache
            self.redis = None
            self.is_mock = True
            self.mock_cache = {}

    async def disconnect(self) -> None:
        """Отключение от Redis."""
        if self.redis:
            await self.redis.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()
        self.mock_cache.clear()
        logger.info("Cache service disconnected")

    async def is_connected(self) -> bool:
        """Проверка подключения."""
        if self.is_mock:
            return True  # Mock cache всегда доступен
        
        if not self.redis:
            return False

        try:
            await self.redis.ping()
            return True
        except Exception:
            return False

    async def generate_cache_key(self, sensor_data: SensorDataBatch) -> str:
        """Генерация ключа кеша на основе данных."""
        # Создаем хеш от сенсорных данных
        cache_data = {
            "system_id": str(sensor_data.system_id),
            "readings_count": len(sensor_data.readings),
            "sensor_types": sorted(list(set(r.sensor_type for r in sensor_data.readings))),
            "values_hash": self._hash_values([r.value for r in sensor_data.readings]),
        }

        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.sha256(cache_string.encode()).hexdigest()[:16]

        return f"{self.cache_prefix}{cache_hash}"

    def _hash_values(self, values: list) -> str:
        """Хеширование числовых значений с округлением."""
        # Округляем до 2 знаков для устойчивости кеша
        rounded_values = [round(v, 2) for v in values]
        values_string = json.dumps(rounded_values, sort_keys=True)
        return hashlib.md5(values_string.encode()).hexdigest()[:8]

    async def get_prediction(self, cache_key: str) -> dict[str, Any] | None:
        """Получение предсказания из кеша."""
        if not settings.cache_predictions:
            return None

        try:
            if self.is_mock:
                # Mock cache
                if cache_key in self.mock_cache:
                    cached_data = self.mock_cache[cache_key]
                    # Проверяем TTL
                    if time.time() - cached_data["cached_at"] < settings.cache_ttl_seconds:
                        self.hit_count += 1
                        logger.debug("Mock cache hit", key=cache_key)
                        return cached_data["data"]
                    else:
                        # TTL истек
                        del self.mock_cache[cache_key]
                
                self.miss_count += 1
                return None
            
            # Real Redis
            cached_data = await self.redis.get(cache_key)

            if cached_data:
                self.hit_count += 1
                result = json.loads(cached_data)
                logger.debug("Cache hit", key=cache_key)
                return result
            else:
                self.miss_count += 1
                logger.debug("Cache miss", key=cache_key)
                return None

        except Exception as e:
            logger.warning("Cache get failed", error=str(e), key=cache_key)
            self.miss_count += 1
            return None

    async def save_prediction(self, cache_key: str, prediction: dict[str, Any]) -> bool:
        """Сохранение предсказания в кеш."""
        if not settings.cache_predictions:
            return False

        try:
            # Добавляем метаданные кеша
            cache_data = {
                **prediction,
                "cached_at": time.time(),
                "cache_ttl": settings.cache_ttl_seconds,
            }

            if self.is_mock:
                # Mock cache с TTL
                self.mock_cache[cache_key] = {
                    "data": cache_data,
                    "cached_at": time.time()
                }
                logger.debug("Prediction cached to mock", key=cache_key, ttl=settings.cache_ttl_seconds)
                return True
            
            # Real Redis
            await self.redis.set(
                cache_key, json.dumps(cache_data, default=str), ex=settings.cache_ttl_seconds
            )

            logger.debug("Prediction cached", key=cache_key, ttl=settings.cache_ttl_seconds)
            return True

        except Exception as e:
            logger.warning("Cache save failed", error=str(e), key=cache_key)
            return False

    def get_cache_stats(self) -> dict[str, Any]:
        """Статистика кеша."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0.0

        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "enabled": settings.cache_predictions,
            "ttl_seconds": settings.cache_ttl_seconds,
            "backend": "mock" if self.is_mock else "redis",
            "mock_entries": len(self.mock_cache) if self.is_mock else None,
        }

    async def clear_cache(self, pattern: str = None) -> int:
        """Очистка кеша."""
        try:
            if self.is_mock:
                if pattern:
                    # Очищаем по паттерну
                    keys_to_delete = [k for k in self.mock_cache if pattern in k]  # ✅ Исправил SIM118
                    for k in keys_to_delete:
                        del self.mock_cache[k]
                    return len(keys_to_delete)
                else:
                    count = len(self.mock_cache)
                    self.mock_cache.clear()
                    return count
            
            if not self.redis:
                return 0
            
            if pattern:
                keys = await self.redis.keys(f"{self.cache_prefix}{pattern}*")
                if keys:
                    return await self.redis.delete(*keys)
            else:
                # Очистка всех ML предсказаний
                keys = await self.redis.keys(f"{self.cache_prefix}*")
                if keys:
                    return await self.redis.delete(*keys)

            return 0

        except Exception as e:
            logger.error("Cache clear failed", error=str(e))
            return 0
