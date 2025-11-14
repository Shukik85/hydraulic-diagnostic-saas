import pytest
import httpx
import asyncio
from diagnosis_service import drift_ab_runner
import logging

@pytest.mark.asyncio
async def test_monitoring_health_ready_metrics():
    services = [
        ("diagnosis", 8003),
        ("gnn", 8002),
        ("rag", 8004),
    ]
    async with httpx.AsyncClient() as client:
        for name, port in services:
            base = f"http://localhost:{port}"
            resp = await client.get(f"{base}/health")
            assert resp.status_code == 200
            assert resp.json()["status"] in ["alive", "healthy", "ready", "started"]
            resp = await client.get(f"{base}/ready")
            assert resp.status_code in [200, 503]
            resp = await client.get(f"{base}/metrics")
            assert resp.status_code == 200
            assert "prometheus" not in resp.text.lower()

@pytest.mark.asyncio
async def test_drift_ab_runner_basic(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", lambda t: asyncio.sleep(0))
    logger = logging.getLogger("mlops.runner")
    logs = []
    logger.addHandler(logging.StreamHandler())
    monkeypatch.setattr(logger, "info", lambda msg, *a, **kw: logs.append(msg))
    monkeypatch.setattr(logger, "warning", lambda msg, *a, **kw: logs.append(msg))
    async def one_cycle():
        await drift_ab_runner.periodic_drift_and_ab(interval_sec=0.01)
    task = asyncio.create_task(one_cycle())
    await asyncio.sleep(0.02)
    task.cancel()
    assert any("Drift" in msg or "AB Test" in msg for msg in logs)

@pytest.mark.asyncio
async def test_drift_metrics_exposed():
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8003/metrics")
        assert resp.status_code == 200
        assert "model_drift_score" in resp.text

@pytest.mark.asyncio
async def test_admin_endpoints_auth():
    import jwt
    ADMIN_KEY = "your-secret-here"
    invalid_token = "invalid.jwt.token"
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8002/admin/model/info", headers={"Authorization": f"Bearer {invalid_token}"})
        assert resp.status_code in [401, 403]
    payload = {"sub": "testuser", "role": "admin"}
    valid_token = jwt.encode(payload, ADMIN_KEY, algorithm="HS256")
    resp = await client.get("http://localhost:8002/admin/model/info", headers={"Authorization": f"Bearer {valid_token}"})
    assert resp.status_code == 200
