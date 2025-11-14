import pytest
import httpx

@pytest.mark.asyncio
async def test_rag_monitoring_endpoints():
    base_url = "http://localhost:8004"
    async with httpx.AsyncClient() as client:
        for ep in ["/health", "/ready", "/metrics"]:
            resp = await client.get(f"{base_url}{ep}")
            assert resp.status_code == 200
        metric_resp = await client.get(f"{base_url}/metrics")
        assert "rag_generation_latency" in metric_resp.text
