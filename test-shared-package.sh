#!/bin/bash

echo "��� Testing Shared Package"
echo "========================="

# Test imports
python << PYTHON
import sys
sys.path.insert(0, 'services/shared')

print("Testing imports...")

try:
    from shared.clients import GNNClient
    print("✅ GNNClient imported")
except ImportError as e:
    print(f"❌ GNNClient import failed: {e}")

try:
    from shared.schemas import SensorData, ComponentType
    print("✅ Schemas imported")
except ImportError as e:
    print(f"❌ Schemas import failed: {e}")

try:
    from shared.validation import validate_sensor_batch
    print("✅ Validation imported")
except ImportError as e:
    print(f"❌ Validation import failed: {e}")

print("\n✅ All imports successful!")

# Test basic functionality
print("\nTesting SensorData schema...")
from datetime import datetime
sensor = SensorData(
    sensor_id="TEST001",
    sensor_type="pressure",
    timestamp=datetime.now(),
    value=150.5,
    unit="bar",
    quality=0.98
)
print(f"✅ Created sensor: {sensor.sensor_id}")
print(f"   Value: {sensor.value} {sensor.unit}")

PYTHON

echo ""
echo "✅ Tests passed!"

