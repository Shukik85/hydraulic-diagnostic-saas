"""
Test script for the GNN API service.
"""

import requests
import time
import numpy as np

# API configuration
BASE_URL = "http://localhost:8003"
HEADERS = {"Content-Type": "application/json"}


def create_test_request() -> dict:
    """Create a test request with realistic hydraulic data."""

    # Use corrected physical norms for realistic test data
    test_data = {
        "pump": {
            "raw_features": [
                250.0,
                2000.0,
                60.0,
                2.5,
                75.0,
            ],  # pressure, rpm, temp, vibration, power
            "normalized_features": [0.7, 0.5, 0.6, 0.5, 0.7],  # normalized values
            "deviation_features": [0.0, 0.0, 0.0, 0.0, 0.1],  # slight power deviation
        },
        "cylinder_boom": {
            "raw_features": [200.0, 60.0, 50.0, 200.0, 160.0],
            "normalized_features": [0.6, 0.5, 0.5, 0.5, 0.6],
            "deviation_features": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "cylinder_stick": {
            "raw_features": [190.0, 55.0, 45.0, 180.0, 150.0],
            "normalized_features": [0.6, 0.5, 0.45, 0.5, 0.6],
            "deviation_features": [0.0, 0.0, 0.05, 0.0, 0.0],
        },
        "cylinder_bucket": {
            "raw_features": [180.0, 50.0, 40.0, 150.0, 140.0],
            "normalized_features": [0.6, 0.5, 0.4, 0.5, 0.6],
            "deviation_features": [0.0, 0.0, 0.1, 0.0, 0.0],
        },
        "motor_swing": {
            "raw_features": [1500.0, 650.0, 65.0, 210.0, 3.0],
            "normalized_features": [0.5, 0.5, 0.6, 0.5, 0.5],
            "deviation_features": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "motor_left": {
            "raw_features": [1450.0, 600.0, 63.0, 205.0, 3.2],
            "normalized_features": [0.5, 0.5, 0.6, 0.5, 0.5],
            "deviation_features": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "motor_right": {
            "raw_features": [1450.0, 600.0, 63.0, 205.0, 3.2],
            "normalized_features": [0.5, 0.5, 0.6, 0.5, 0.5],
            "deviation_features": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "equipment_id": "test_excavator_001",
        "timestamp": "2024-01-01T12:00:00Z",
    }

    return test_data


def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")


def test_single_prediction():
    """Test single prediction endpoint."""
    print("\nTesting single prediction...")

    test_request = create_test_request()

    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict", json=test_request, headers=HEADERS
        )
        processing_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            print("✅ Single prediction passed")
            print(f"Processing time: {processing_time:.2f}ms")
            print(f"System health: {result['system_health']:.3f}")

            # Print component statuses
            for component, diagnostics in result["components"].items():
                status_icon = (
                    "🔴"
                    if diagnostics["status"] == "critical"
                    else "🟡"
                    if diagnostics["status"] == "warning"
                    else "🟢"
                )
                print(
                    f"  {status_icon} {component}: {diagnostics['status']} "
                    f"(prob: {diagnostics['fault_probability']:.3f})"
                )

        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Prediction error: {e}")


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\nTesting batch prediction...")

    # Create multiple test requests
    batch_requests = []
    for i in range(3):  # Test with 3 graphs
        request = create_test_request()
        request["equipment_id"] = f"test_excavator_{i:03d}"
        batch_requests.append(request)

    batch_request = {"graphs": batch_requests, "batch_id": "test_batch_001"}

    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/batch_predict", json=batch_request, headers=HEADERS
        )
        processing_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction passed")
            print(f"Total processing time: {result['total_processing_time_ms']:.2f}ms")
            print(f"Processed {len(result['predictions'])} graphs")

        else:
            print(
                f"❌ Batch prediction failed: {response.status_code} - {response.text}"
            )

    except Exception as e:
        print(f"❌ Batch prediction error: {e}")


def test_model_config():
    """Test model configuration endpoint."""
    print("\nTesting model config...")

    try:
        response = requests.get(f"{BASE_URL}/model/config")
        if response.status_code == 200:
            config = response.json()
            print("✅ Model config retrieved")
            print(
                f"Model: {config['model_config']['num_gat_layers']} GAT layers, "
                f"{config['model_config']['num_heads']} heads"
            )
            print(f"Components: {', '.join(config['model_config']['component_names'])}")
        else:
            print(f"❌ Config retrieval failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Config error: {e}")


def performance_test():
    """Test API performance with multiple requests."""
    print("\nTesting performance...")

    test_request = create_test_request()
    times = []

    for i in range(10):  # 10 requests for performance testing
        start_time = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/predict", json=test_request, headers=HEADERS
            )
            if response.status_code == 200:
                elapsed = (time.time() - start_time) * 1000
                times.append(elapsed)
        except:
            pass

    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"✅ Performance: {avg_time:.2f}ms ± {std_time:.2f}ms per request")
        print(f"  Min: {min(times):.2f}ms, Max: {max(times):.2f}ms")

        if avg_time < 50:  # Our target
            print("🎉 Performance target (<50ms) achieved!")
        else:
            print("⚠️  Performance target not met")
    else:
        print("❌ Performance test failed")


def main():
    """Run all tests."""
    print("🧪 Hydraulic GNN API Test Suite")
    print("=" * 50)

    # Wait a bit for server to start
    print("Waiting for server to be ready...")
    time.sleep(2)

    test_health()
    test_model_config()
    test_single_prediction()
    test_batch_prediction()
    performance_test()

    print("\n" + "=" * 50)
    print("Test suite completed!")


if __name__ == "__main__":
    main()
