#!/bin/bash

echo "��� Installing Shared Package"
echo "============================"

# Go to shared directory
cd services/shared

# Install in editable mode
echo "Installing shared package in editable mode..."
pip install -e .

echo ""
echo "Verifying installation..."
pip list | grep hdx-shared

# Go back to root
cd ../..

# Test imports again
echo ""
echo "Testing imports..."
python << PYTHON
try:
    from shared.clients import GNNClient
    print("✅ GNNClient imported successfully")
except ImportError as e:
    print(f"❌ Failed: {e}")

try:
    from shared.schemas import SensorData, ComponentType
    print("✅ Schemas imported successfully")
except ImportError as e:
    print(f"❌ Failed: {e}")

try:
    from shared.validation import validate_sensor_batch
    print("✅ Validation imported successfully")
except ImportError as e:
    print(f"❌ Failed: {e}")

# Quick test
from shared.schemas import SensorData
from datetime import datetime

sensor = SensorData(
    sensor_id="TEST001",
    sensor_type="pressure",
    timestamp=datetime.now(),
    value=150.5,
    unit="bar"
)
print(f"\n✅ Test passed! Created sensor: {sensor.sensor_id} = {sensor.value} {sensor.unit}")

PYTHON

echo ""
echo "✅ Installation complete!"

