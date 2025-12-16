import asyncio
import time
from unittest.mock import Mock

import pytest

from src.inference.cache import AsyncTopologyCache, TopologyCache
from src.schemas import GraphTopology


@pytest.fixture
def mock_topology() -> GraphTopology:
    topology = Mock(spec=GraphTopology)
    topology.topology_id = "topo_001"
    return topology


def test_topology_cache_lru_ttl(mock_topology: GraphTopology) -> None:
    cache = TopologyCache(max_size=2, ttl_seconds=1000.0)

    cache.put("key1", mock_topology)
    cache.put("key2", mock_topology)

    assert cache.get("key1") is mock_topology

    cache._cache["key1"] = (mock_topology, 0.0)
    assert cache.get("key1") is None

    cache._cache["key1"] = (mock_topology, time.time())
    cache.put("key3", mock_topology)
    assert "key2" not in cache._cache


@pytest.mark.asyncio
async def test_async_topology_cache_stampede_protection(mock_topology: GraphTopology) -> None:
    cache = AsyncTopologyCache(max_size=1, ttl_seconds=1000.0)
    call_count = 0

    async def fetch(_key: str) -> GraphTopology:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return mock_topology

    tasks = [cache.get_or_fetch("key1", fetch) for _ in range(3)]
    results = await asyncio.gather(*tasks)

    assert all(r is mock_topology for r in results)
    assert call_count == 1
