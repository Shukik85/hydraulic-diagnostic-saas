print("Testing shared package imports...")

try:
    from shared.clients import GNNClient
    print("OK: GNNClient works")
except ImportError as e:
    print(f"FAIL: GNNClient - {e}")

try:
    from shared.schemas import SensorData
    print("OK: SensorData works")
except ImportError as e:
    print(f"FAIL: SensorData - {e}")

try:
    from shared.validation import validate_sensor_batch
    print("OK: Validation works")
except ImportError as e:
    print(f"FAIL: Validation - {e}")

print("\nDone!")
