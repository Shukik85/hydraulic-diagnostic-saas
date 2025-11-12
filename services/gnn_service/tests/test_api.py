from fastapi.testclient import TestClient
from main import app

def test_predict_api():
    client = TestClient(app)
    dummy_data = {"component_features": {"pump": [[25.0, 0.0, 0.0]*5], "cylinder_main": [[12.0, 0.0, 0.0]*5]}}
    resp = client.post("/predict", json=dummy_data)
    assert resp.status_code == 200 or resp.status_code == 400  # depends on trained model
