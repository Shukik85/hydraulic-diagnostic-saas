"""Tests for FastAPI endpoints.

Covers:
- Health check endpoint
- Diagnostics prediction endpoint
- Request validation
- Response validation
- Error handling

Run with:
    pytest tests/test_api.py -v
"""

import pytest
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


class TestHealthCheck:
    """Tests for GET /api/v1/health endpoint."""

    def test_health_check_success(self):
        """Health check returns healthy status."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy", "degraded"]
        assert "model_loaded" in data
        assert "version" in data

    def test_health_check_model_loaded(self):
        """Health check indicates model is loaded."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert data["model_loaded"] is True


class TestDiagnosticsPredict:
    """Tests for POST /api/v1/diagnostics/predict endpoint."""

    def test_predict_valid_data(self):
        """Valid prediction request returns diagnostics."""
        payload = {
            "equipment_id": "pump_001",
            "sensor_readings": {
                "PS1": [100.5, 101.2, 100.8],
                "TS1": [45.3, 45.5, 45.4],
                "FS1": [8.5, 8.6, 8.5]
            },
            "lookback_minutes": 10
        }

        response = client.post("/api/v1/diagnostics/predict", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "equipment_id" in data
        assert data["equipment_id"] == "pump_001"
        assert "overall_health" in data
        assert "components" in data
        assert "recommendations" in data
        assert "model_version" in data
        assert "inference_time_ms" in data

        # Verify health score range
        assert 0 <= data["overall_health"] <= 1

        # Verify components
        assert len(data["components"]) > 0
        for comp in data["components"]:
            assert "component_name" in comp
            assert "health_score" in comp
            assert "severity_grade" in comp
            assert "confidence" in comp
            assert 0 <= comp["health_score"] <= 1
            assert 0 <= comp["confidence"] <= 1

    def test_predict_invalid_empty_sensors(self):
        """Empty sensor_readings returns 400 error."""
        payload = {
            "equipment_id": "pump_001",
            "sensor_readings": {}
        }

        response = client.post("/api/v1/diagnostics/predict", json=payload)
        assert response.status_code == 400

    def test_predict_mismatched_sensor_lengths(self):
        """Sensors with different sample counts return 400 error."""
        payload = {
            "equipment_id": "pump_001",
            "sensor_readings": {
                "PS1": [100.5, 101.2],          # 2 samples
                "TS1": [45.3, 45.5, 45.4]      # 3 samples - MISMATCH
            }
        }

        response = client.post("/api/v1/diagnostics/predict", json=payload)
        assert response.status_code == 400

    def test_predict_minimal_valid(self):
        """Minimal valid request works."""
        payload = {
            "equipment_id": "pump_001",
            "sensor_readings": {
                "PS1": [100.0]
            }
        }

        response = client.post("/api/v1/diagnostics/predict", json=payload)
        assert response.status_code == 200

    def test_predict_lookback_minutes_range(self):
        """lookback_minutes validation."""
        # Valid: 1-60
        payload_valid = {
            "equipment_id": "pump_001",
            "sensor_readings": {"PS1": [100.0]},
            "lookback_minutes": 10
        }
        response = client.post("/api/v1/diagnostics/predict", json=payload_valid)
        assert response.status_code == 200

        # Invalid: 0 (< 1)
        payload_low = {
            "equipment_id": "pump_001",
            "sensor_readings": {"PS1": [100.0]},
            "lookback_minutes": 0
        }
        response = client.post("/api/v1/diagnostics/predict", json=payload_low)
        assert response.status_code == 422  # Validation error

        # Invalid: 61 (> 60)
        payload_high = {
            "equipment_id": "pump_001",
            "sensor_readings": {"PS1": [100.0]},
            "lookback_minutes": 61
        }
        response = client.post("/api/v1/diagnostics/predict", json=payload_high)
        assert response.status_code == 422  # Validation error


class TestResponseValidation:
    """Tests for response schema validation."""

    def test_response_component_structure(self):
        """Component response has correct structure."""
        payload = {
            "equipment_id": "pump_001",
            "sensor_readings": {"PS1": [100.0]}
        }

        response = client.post("/api/v1/diagnostics/predict", json=payload)
        data = response.json()

        # Verify each component has required fields
        for comp in data["components"]:
            assert "component_name" in comp
            assert "health_score" in comp
            assert "severity_grade" in comp
            assert "confidence" in comp
            assert "contributing_sensors" in comp
            assert isinstance(comp["contributing_sensors"], list)

    def test_response_recommendations_type(self):
        """Recommendations is list of strings."""
        payload = {
            "equipment_id": "pump_001",
            "sensor_readings": {"PS1": [100.0]}
        }

        response = client.post("/api/v1/diagnostics/predict", json=payload)
        data = response.json()

        assert isinstance(data["recommendations"], list)
        for rec in data["recommendations"]:
            assert isinstance(rec, str)


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Full prediction workflow."""
        # 1. Check health
        health = client.get("/api/v1/health").json()
        assert health["status"] == "healthy"

        # 2. Make prediction
        payload = {
            "equipment_id": "pump_test_123",
            "sensor_readings": {
                "PS1": [99.0, 100.0, 101.0],
                "TS1": [44.0, 45.0, 46.0],
                "FS1": [8.0, 8.5, 9.0]
            }
        }
        response = client.post("/api/v1/diagnostics/predict", json=payload)
        assert response.status_code == 200

        # 3. Verify response
        data = response.json()
        assert data["equipment_id"] == "pump_test_123"
        assert len(data["components"]) == 4  # 4 components by default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
