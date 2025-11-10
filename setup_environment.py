"""
Setup script to configure Python path for the project.
Run this before any other scripts.
"""

import os
import sys


def setup_environment():
    """Add the current directory to Python path."""
    # Get the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))  # noqa: PTH100, PTH120

    # Add it to Python path if not already there
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"Added {current_dir} to Python path")

    # Check if gnn_service can be imported
    try:
        import gnn_service  # noqa: F401

        print("✅ gnn_service module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import gnn_service: {e}")
        return False


if __name__ == "__main__":
    success = setup_environment()
    if success:
        print("Environment setup completed successfully!")
    else:
        print("Environment setup failed!")
