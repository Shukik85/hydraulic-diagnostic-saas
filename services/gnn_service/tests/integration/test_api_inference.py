"""Integration tests for FastAPI inference API.

Tests:
- /health endpoint
- /stats endpoint
- /predict endpoint
- /predict/batch endpoint
- Request validation
- Error handling
- CORS
"""

import pytest
import torch
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from api.main import app
from src.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthPrediction,
    DegradationPrediction,
    AnomalyPrediction
)


# ==================== FIXTURES ====================

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_engine():
    """Mock InferenceEngine."""
    engine = Mock()
    
    # Mock predict
    async def mock_predict(request, topology):
        return PredictionResponse(
            equipment_id=request.equipment_id,
            health=HealthPrediction(score=0.87),
            degradation=DegradationPrediction(rate=0.12),
            anomaly=AnomalyPrediction(
                predictions={
                    "pressure_drop": 0.05,
                    "overheating": 0.03,
                    "cavitation": 0.02,
                    "leakage": 0.01,
                    "vibration_anomaly": 0.01,
                    "flow_restriction": 0.01,
                    "contamination": 0.01,
                    "seal_degradation": 0.01,
                    "valve_stiction": 0.01
                }
            ),
            inference_time_ms=45.3
        )
    
    engine.predict = AsyncMock(side_effect=mock_predict)
    
    # Mock predict_batch
    async def mock_predict_batch(requests, topology):
        return [
            await mock_predict(req, topology)
            for req in requests
        ]
    
    engine.predict_batch = AsyncMock(side_effect=mock_predict_batch)
    
    # Mock get_stats
    engine.get_stats = Mock(return_value={
        "model_path": "models/checkpoints/best.ckpt",
        "device": "cpu",
        "batch_size": 32,
        "model_device": "cpu",
        "model_parameters": 2500000,
        "queue_size": 0,
        "processing": False
    })
    
    return engine


# ==================== TESTS ====================

class TestHealthEndpoint:
    """Test /health endpoint."""
    
    def test_health_check_success(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
    
    @patch('api.main.engine', None)
    def test_health_check_unhealthy_when_engine_not_ready(self, client):
        """Test health returns unhealthy when engine not ready."""
        with patch('api.main.engine', None):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False


class TestStatsEndpoint:
    """Test /stats endpoint."""
    
    @patch('api.main.engine')
    def test_stats_success(self, mock_engine_global, client, mock_engine):
        """Test stats endpoint returns statistics."""
        mock_engine_global.return_value = mock_engine
        
        with patch('api.main.engine', mock_engine):
            response = client.get("/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert "model_path" in data
            assert "device" in data
            assert "batch_size" in data
    
    def test_stats_error_when_engine_not_ready(self, client):
        """Test stats returns 503 when engine not ready."""
        with patch('api.main.engine', None):
            response = client.get("/stats")
            
            assert response.status_code == 503
            assert "Service not ready" in response.json()["detail"]


class TestPredictEndpoint:
    """Test /predict endpoint."""
    
    @patch('api.main.engine')
    @patch('api.main.topology')
    def test_predict_success(
        self,
        mock_topology_global,
        mock_engine_global,
        client,
        mock_engine
    ):
        """Test successful single prediction."""
        mock_topology = Mock()
        
        with patch('api.main.engine', mock_engine), \
             patch('api.main.topology', mock_topology):
            
            request_data = {
                "equipment_id": "exc_001",
                "sensor_data": {
                    "pressure_pump_main": [100.0, 101.0, 102.0],
                    "temperature_pump_main": [60.0, 61.0, 62.0]
                }
            }
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check structure
            assert data["equipment_id"] == "exc_001"
            assert "health" in data
            assert "degradation" in data
            assert "anomaly" in data
            assert "inference_time_ms" in data
            
            # Check values
            assert 0 <= data["health"]["score"] <= 1
            assert 0 <= data["degradation"]["rate"] <= 1
            assert len(data["anomaly"]["predictions"]) == 9
    
    def test_predict_validation_error_missing_field(self, client):
        """Test predict returns 422 for missing fields."""
        request_data = {
            # Missing equipment_id
            "sensor_data": {
                "pressure_pump_main": [100.0, 101.0]
            }
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_validation_error_invalid_type(self, client):
        """Test predict returns 422 for invalid types."""
        request_data = {
            "equipment_id": 123,  # Should be string
            "sensor_data": {}
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422
    
    def test_predict_error_when_engine_not_ready(self, client):
        """Test predict returns 503 when engine not ready."""
        with patch('api.main.engine', None):
            request_data = {
                "equipment_id": "exc_001",
                "sensor_data": {}
            }
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 503
    
    def test_predict_error_when_topology_not_configured(self, client, mock_engine):
        """Test predict returns 500 when topology not configured."""
        with patch('api.main.engine', mock_engine), \
             patch('api.main.topology', None):
            
            request_data = {
                "equipment_id": "exc_001",
                "sensor_data": {}
            }
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 500
            assert "Topology not configured" in response.json()["detail"]
    
    @patch('api.main.engine')
    @patch('api.main.topology')
    def test_predict_handles_engine_error(
        self,
        mock_topology_global,
        mock_engine_global,
        client
    ):
        """Test predict handles engine errors gracefully."""
        # Mock engine that raises
        error_engine = Mock()
        error_engine.predict = AsyncMock(side_effect=RuntimeError("Model failed"))
        
        mock_topology = Mock()
        
        with patch('api.main.engine', error_engine), \
             patch('api.main.topology', mock_topology):
            
            request_data = {
                "equipment_id": "exc_001",
                "sensor_data": {}
            }
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 500
            assert "Prediction failed" in response.json()["detail"]


class TestPredictBatchEndpoint:
    """Test /predict/batch endpoint."""
    
    @patch('api.main.engine')
    @patch('api.main.topology')
    def test_predict_batch_success(
        self,
        mock_topology_global,
        mock_engine_global,
        client,
        mock_engine
    ):
        """Test successful batch prediction."""
        mock_topology = Mock()
        
        with patch('api.main.engine', mock_engine), \
             patch('api.main.topology', mock_topology):
            
            request_data = {
                "requests": [
                    {
                        "equipment_id": "exc_001",
                        "sensor_data": {"pressure": [100.0]}
                    },
                    {
                        "equipment_id": "exc_002",
                        "sensor_data": {"pressure": [105.0]}
                    },
                    {
                        "equipment_id": "exc_003",
                        "sensor_data": {"pressure": [110.0]}
                    }
                ]
            }
            
            response = client.post("/predict/batch", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check structure
            assert "predictions" in data
            assert "total_count" in data
            assert "total_time_ms" in data
            
            # Check values
            assert len(data["predictions"]) == 3
            assert data["total_count"] == 3
            assert data["total_time_ms"] > 0
            
            # Check each prediction
            for i, pred in enumerate(data["predictions"]):
                assert pred["equipment_id"] == f"exc_{i+1:03d}"
    
    def test_predict_batch_validation_error_empty_list(self, client):
        """Test batch predict returns 422 for empty request list."""
        request_data = {
            "requests": []  # Empty list
        }
        
        response = client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 422
    
    def test_predict_batch_validation_error_too_many(self, client):
        """Test batch predict returns 422 for too many requests."""
        request_data = {
            "requests": [
                {"equipment_id": f"exc_{i:03d}", "sensor_data": {}}
                for i in range(101)  # Exceeds max (100)
            ]
        }
        
        response = client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 422
    
    def test_predict_batch_error_when_engine_not_ready(self, client):
        """Test batch predict returns 503 when engine not ready."""
        with patch('api.main.engine', None):
            request_data = {
                "requests": [
                    {"equipment_id": "exc_001", "sensor_data": {}}
                ]
            }
            
            response = client.post("/predict/batch", json=request_data)
            
            assert response.status_code == 503


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers_present(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        # CORS should allow
        assert "access-control-allow-origin" in response.headers


class TestOpenAPIDocs:
    """Test OpenAPI documentation."""
    
    def test_openapi_json_available(self, client):
        """Test OpenAPI JSON is available."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_docs_ui_available(self, client):
        """Test Swagger UI is available."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
