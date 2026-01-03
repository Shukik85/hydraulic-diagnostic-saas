"""Pytest configuration and fixtures.

Sets up proper import paths for testing.
"""

import sys
from pathlib import Path

# Add src directory to Python path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
