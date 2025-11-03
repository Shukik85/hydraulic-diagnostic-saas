"""
Redis Cache Service for ML Predictions
Enterprise кеширование предсказаний с TTL
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional

import aioredis
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
        self.redis: Optional[aioredis.Redis] = None
        self.connection_pool = None
        self.cache_prefix = "ml_pred:"
        self.hit_count = 0
        self.miss_count = 0
        
    async def connect(self) -> None:
        """Подключение к Redis."""
        try:
            self.connection_pool = aioredis.ConnectionPool.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            
            self.redis = aioredis.Redis(connection_pool=self.connection_pool)
            
            # Проверка подключения
            await self.redis.ping()
            
            logger.info("Redis cache connected", url=settings.redis_url)
            
        except Exception as e:
            logger.error("Redis connection failed", error=str(e))
            self.redis = None
            raise
    
    async def disconnect(self) -> None:
        """Отключение от Redis."""
        if self.redis:
            await self.redis.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()
            
        logger.info("Redis cache disconnected")
    
    async def is_connected(self) -> bool:
        """Проверка подключения."""
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
            "values_hash": self._hash_values([r.value for r in sensor_data.readings])
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
    
    async def get_prediction(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Получение предсказания из кеша."""
        if not self.redis or not settings.cache_predictions:
            return None
            
        try:
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
    
    async def save_prediction(self, cache_key: str, prediction: Dict[str, Any]) -> bool:
        """Сохранение предсказания в кеш."""
        if not self.redis or not settings.cache_predictions:
            return False
            
        try:
            # Добавляем метаданные кеша
            cache_data = {
                **prediction,
                "cached_at": time.time(),
                "cache_ttl": settings.cache_ttl_seconds
            }
            
            await self.redis.set(
                cache_key,
                json.dumps(cache_data, default=str),
                ex=settings.cache_ttl_seconds
            )
            
            logger.debug("Prediction cached", key=cache_key, ttl=settings.cache_ttl_seconds)
            return True
            
        except Exception as e:
            logger.warning("Cache save failed", error=str(e), key=cache_key)
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кеша."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0.0
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "enabled": settings.cache_predictions,
            "ttl_seconds": settings.cache_ttl_seconds
        }
    
    async def clear_cache(self, pattern: str = None) -> int:
        """Очистка кеша."""
        if not self.redis:
            return 0
            
        try:
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