"""Topology caching with LRU+TTL and stampede protection.

Provides:
- TopologyCache: Synchronous LRU cache with TTL
- AsyncTopologyCache: Async cache with stampede protection
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any

from src.schemas import GraphTopology

from .metrics import TOPOLOGY_CACHE_SIZE


class TopologyCache:
    """LRU cache with TTL for topology templates.

    Features:
    - LRU eviction policy
    - TTL-based expiration
    - Hit/miss statistics

    Examples:
        >>> cache = TopologyCache(max_size=100, ttl_seconds=300)
        >>> cache.put("topo_001", topology)
        >>> topo = cache.get("topo_001")  # Cache hit
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 300.0):
        """Initialize cache.

        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[GraphTopology, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> GraphTopology | None:
        """Get topology from cache.

        Args:
            key: Topology identifier

        Returns:
            Cached topology or None if not found/expired
        """
        if key not in self._cache:
            self._misses += 1
            TOPOLOGY_CACHE_SIZE.set(len(self._cache))
            return None

        topology, timestamp = self._cache[key]

        # Check TTL
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            self._misses += 1
            TOPOLOGY_CACHE_SIZE.set(len(self._cache))
            return None

        # LRU: Move to end
        self._cache.move_to_end(key)
        self._hits += 1
        return topology

    def put(self, key: str, topology: GraphTopology) -> None:
        """Put topology in cache.

        Args:
            key: Topology identifier
            topology: Topology to cache
        """
        # Evict oldest if full
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (topology, time.time())
        self._cache.move_to_end(key)
        TOPOLOGY_CACHE_SIZE.set(len(self._cache))

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        TOPOLOGY_CACHE_SIZE.set(0)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Statistics dictionary
        """
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "ttl_seconds": self.ttl_seconds,
        }


class AsyncTopologyCache:
    """Async topology cache with stampede protection.

    Prevents cache stampede by ensuring only one coroutine fetches
    a missing value while others wait for the result.

    Features:
    - LRU + TTL
    - Stampede protection via asyncio.Lock per key
    - Async-friendly interface

    Examples:
        >>> cache = AsyncTopologyCache()
        >>> async def fetch_topology(key: str) -> GraphTopology:
        ...     # Expensive operation
        ...     return build_topology(key)
        >>> topology = await cache.get_or_fetch("topo_001", fetch_topology)
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 300.0):
        """Initialize async cache.

        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[GraphTopology, float]] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._hits = 0
        self._misses = 0

    async def get_or_fetch(
        self,
        key: str,
        fetch_fn: Any,  # Callable[[str], Awaitable[GraphTopology]]
    ) -> GraphTopology:
        """Get topology from cache or fetch if missing.

        Stampede protection: If multiple coroutines request the same missing
        key, only one will call fetch_fn while others wait.

        Args:
            key: Topology identifier
            fetch_fn: Async function to fetch topology if missing

        Returns:
            Topology (from cache or freshly fetched)
        """
        # Fast path: Check cache without lock
        cached = self._get_from_cache(key)
        if cached is not None:
            return cached

        # Slow path: Acquire lock and fetch
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:
            # Double-check after acquiring lock
            cached = self._get_from_cache(key)
            if cached is not None:
                return cached

            # Fetch and cache
            topology = await fetch_fn(key)
            self._put_in_cache(key, topology)
            return topology

    def _get_from_cache(self, key: str) -> GraphTopology | None:
        """Internal cache get (no locking)."""
        if key not in self._cache:
            self._misses += 1
            return None

        topology, timestamp = self._cache[key]

        # Check TTL
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            self._misses += 1
            return None

        # LRU
        self._cache.move_to_end(key)
        self._hits += 1
        return topology

    def _put_in_cache(self, key: str, topology: GraphTopology) -> None:
        """Internal cache put (no locking)."""
        # Evict oldest if full
        if len(self._cache) >= self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            # Clean up lock for evicted key
            self._locks.pop(evicted_key, None)

        self._cache[key] = (topology, time.time())
        self._cache.move_to_end(key)
        TOPOLOGY_CACHE_SIZE.set(len(self._cache))

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Statistics dictionary
        """
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "ttl_seconds": self.ttl_seconds,
            "active_locks": len(self._locks),
        }
