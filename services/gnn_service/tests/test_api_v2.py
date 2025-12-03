"""Integration tests for FastAPI v2 endpoints.

Tests complete API workflow:
- Request validation
- Topology resolution
- Inference pipeline
- Response formatting
- Error handling

Author: GNN Service Team
Python: 3.14+
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient

from main import app
from src.schemas.requests import ComponentSensorReading


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client):
        """Test GET /health."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "gnn-service"
        assert data["version"] == "2.0.0"
    
    def test_healthz_alias(self, client):
        """Test GET /healthz (alias)."""
        response = client.get("/healthz")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_ready_check(self, client):
        """Test GET /ready."""
        response = client.get("/ready")
        
        # May be 200 or 503 depending on initialization
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data
            assert "topology_service_ready" in data


class TestTopologyEndpoints:
    """Test topology management endpoints."""
    
    def test_list_topologies(self, client):
        """Test GET /api/v2/topologies."""
        response = client.get("/api/v2/topologies")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "templates" in data
        assert isinstance(data["templates"], list)
        
        # Should have at least 3 built-in templates
        assert len(data["templates"]) >= 3
        
        # Check structure
        if data["templates"]:
            template = data["templates"][0]
            assert "template_id" in template
            assert "name" in template
            assert "description" in template
            assert "num_components" in template
            assert "num_edges" in template
    
    def test_get_topology_by_id(self, client):
        """Test GET /api/v2/topologies/{id}."""
        response = client.get("/api/v2/topologies/standard_pump_system")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["template_id"] == "standard_pump_system"
        assert "name" in data
        assert "components" in data
        assert "edges" in data
        assert len(data["components"]) > 0
        assert len(data["edges"]) > 0
    
    def test_get_nonexistent_topology(self, client):
        """Test GET /api/v2/topologies/{id} with invalid ID."""
        response = client.get("/api/v2/topologies/nonexistent_template")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_validate_topology_valid(self, client):
        """Test POST /api/v2/topologies/validate with valid topology."""
        valid_topology = {
            "equipment_id": "test_equipment",
            "components": {
                "pump_1": {
                    "component_id": "pump_1",
                    "component_type": "pump",
                    "manufacturer": "TestCo",
                    "model": "P1"
                },
                "valve_1": {
                    "component_id": "valve_1",
                    "component_type": "valve",
                    "manufacturer": "TestCo",
                    "model": "V1"
                }
            },
            "edges": [
                {
                    "source_id": "pump_1",
                    "target_id": "valve_1",
                    "edge_type": "pipe",
                    "diameter_mm": 25.0,
                    "length_m": 2.0,
                    "material": "steel"
                }
            ]
        }
        
        response = client.post(
            "/api/v2/topologies/validate",
            json=valid_topology
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["is_valid"] is True
        assert len(data["errors"]) == 0
        assert data["num_components"] == 2
        assert data["num_edges"] == 1
    
    def test_validate_topology_invalid(self, client):
        """Test POST /api/v2/topologies/validate with invalid topology."""
        invalid_topology = {
            "equipment_id": "test_equipment",
            "components": {
                "pump_1": {
                    "component_id": "pump_1",
                    "component_type": "pump",
                    "manufacturer": "TestCo",
                    "model": "P1"
                }
            },
            "edges": [
                {
                    "source_id": "pump_1",
                    "target_id": "nonexistent_valve",  # Doesn't exist
                    "edge_type": "pipe",
                    "diameter_mm": 25.0,
                    "length_m": 2.0,
                    "material": "steel"
                }
            ]
        }
        
        response = client.post(
            "/api/v2/topologies/validate",
            json=invalid_topology
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["is_valid"] is False
        assert len(data["errors"]) > 0


class TestInferenceEndpoints:
    """Test inference endpoints."""
    
    @pytest.fixture
    def minimal_request(self):
        """Create minimal inference request."""
        return {
            "equipment_id": "test_equipment_001",
            "timestamp": datetime.now().isoformat(),
            "sensor_readings": {
                "pump_main": {
                    "pressure_bar": 150.0,
                    "temperature_c": 65.0,
                    "vibration_g": 0.8,
                    "rpm": 1450
                },
                "filter_main": {
                    "pressure_bar": 148.0,
                    "temperature_c": 66.0
                },
                "valve_control": {
                    "pressure_bar": 145.0,
                    "temperature_c": 67.0
                },
                "cylinder_1": {
                    "pressure_bar": 140.0,
                    "temperature_c": 68.0
                }
            },
            "topology_id": "standard_pump_system"
        }
    
    def test_inference_minimal_valid(self, client, minimal_request):
        """Test POST /api/v2/inference/minimal with valid request."""
        response = client.post(
            "/api/v2/inference/minimal",
            json=minimal_request
        )
        
        # May be 503 if engine not initialized, or 200
        if response.status_code == 503:
            pytest.skip("Inference engine not initialized")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "equipment_id" in data
        assert "health" in data
        assert "degradation" in data
        assert "anomaly" in data
        assert "inference_time_ms" in data
        
        # Check health prediction
        assert "score" in data["health"]
        assert 0 <= data["health"]["score"] <= 1
        
        # Check degradation prediction
        assert "rate" in data["degradation"]
        assert 0 <= data["degradation"]["rate"] <= 1
        
        # Check anomaly predictions
        assert "predictions" in data["anomaly"]
        assert isinstance(data["anomaly"]["predictions"], dict)
    
    def test_inference_minimal_invalid_topology(self, client):
        """Test inference with invalid topology ID."""
        request = {
            "equipment_id": "test_equipment_001",
            "timestamp": datetime.now().isoformat(),
            "sensor_readings": {
                "pump_1": {
                    "pressure_bar": 150.0,
                    "temperature_c": 65.0
                }
            },
            "topology_id": "nonexistent_topology"
        }
        
        response = client.post(
            "/api/v2/inference/minimal",
            json=request
        )
        
        # Should return 400 (bad request) or 503 (not ready)
        assert response.status_code in [400, 503]
    
    def test_inference_minimal_missing_field(self, client):
        """Test inference with missing required field."""
        request = {
            "equipment_id": "test_equipment_001",
            # Missing timestamp
            "sensor_readings": {
                "pump_1": {
                    "pressure_bar": 150.0,
                    "temperature_c": 65.0
                }
            },
            "topology_id": "standard_pump_system"
        }
        
        response = client.post(
            "/api/v2/inference/minimal",
            json=request
        )
        
        # Should return 422 (validation error)
        assert response.status_code == 422
    
    def test_inference_minimal_invalid_sensor_value(self, client):
        """Test inference with invalid sensor values."""
        request = {
            "equipment_id": "test_equipment_001",
            "timestamp": datetime.now().isoformat(),
            "sensor_readings": {
                "pump_1": {
                    "pressure_bar": -100.0,  # Invalid: negative
                    "temperature_c": 65.0
                }
            },
            "topology_id": "standard_pump_system"
        }
        
        response = client.post(
            "/api/v2/inference/minimal",
            json=request
        )
        
        # Should return 422 (validation error)
        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 for nonexistent endpoint."""
        response = client.get("/api/v2/nonexistent")
        
        assert response.status_code == 404
    
    def test_422_validation_error(self, client):
        """Test 422 for validation errors."""
        # Missing required fields
        response = client.post(
            "/api/v2/inference/minimal",
            json={"equipment_id": "test"}  # Missing other required fields
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestBackwardCompatibility:
    """Test backward compatibility with v1 API."""
    
    def test_v1_endpoint_exists(self, client):
        """Test v1 endpoint still exists."""
        # Just check endpoint exists (may return 503 if not ready)
        response = client.post(
            "/api/v1/predict",
            json={}
        )
        
        # Should not return 404
        assert response.status_code != 404
    
    def test_v1_batch_endpoint_exists(self, client):
        """Test v1 batch endpoint still exists."""
        response = client.post(
            "/api/v1/batch/predict",
            json={}
        )
        
        # Should not return 404
        assert response.status_code != 404


class TestOpenAPIDocumentation:
    """Test OpenAPI documentation."""
    
    def test_openapi_schema(self, client):
        """Test GET /openapi.json."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check v2 endpoints in schema
        assert "/api/v2/inference/minimal" in schema["paths"]
        assert "/api/v2/topologies" in schema["paths"]
    
    def test_docs_endpoint(self, client):
        """Test GET /docs (Swagger UI)."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
    
    def test_redoc_endpoint(self, client):
        """Test GET /redoc (ReDoc)."""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "redoc" in response.text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
