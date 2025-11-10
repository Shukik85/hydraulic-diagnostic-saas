"""
Simple test to check imports without dependencies.
"""

import sys
import os

print("🔍 Testing Python Paths")
print("=" * 50)

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# Check if gnn_service exists
gnn_service_path = os.path.join(current_dir, "gnn_service")
print(f"gnn_service path: {gnn_service_path}")
print(f"gnn_service exists: {os.path.exists(gnn_service_path)}")

# Check if config.py exists
config_path = os.path.join(gnn_service_path, "config.py")
print(f"config.py path: {config_path}")
print(f"config.py exists: {os.path.exists(config_path)}")

# Print Python path
print("\nPython path:")
for i, path in enumerate(sys.path):
    print(f"{i:2d}: {path}")

# Try to import directly
print("\nTrying direct import...")
try:
    # Add gnn_service to path
    sys.path.insert(0, gnn_service_path)

    # Try to import config
    import config

    print("✅ Successfully imported config directly")

    # Check what's in config
    print("Contents of config:")
    for attr in dir(config):
        if not attr.startswith("_"):
            print(f"  {attr}")

except ImportError as e:
    print(f"❌ Direct import failed: {e}")

print("=" * 50)
