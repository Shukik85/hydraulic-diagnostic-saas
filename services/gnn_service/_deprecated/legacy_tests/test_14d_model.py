"""DEPRECATED: Legacy test file (test_14d_model.py)

WARNING: This is an old standalone test not integrated into pytest.
Moved to archive for historical reference.

Current testing infrastructure:
- Location: ../tests/
- Framework: pytest
- Organization: test_api.py, test_inference.py, etc.

For current test execution:
    pytest ../tests/ -v

Historical note:
This file was used for early testing of 14D model features.
Functionality is now covered by integrated test suite.
"""

# NOTE: Original implementation archived.
# See ../tests/ for current test suite.
# Do NOT run this file directly.

import warnings

warnings.warn(
    "This test file is deprecated. Use ../tests/ instead.",
    DeprecationWarning,
    stacklevel=2
)

if __name__ == "__main__":
    print("❌ DEPRECATED: This test file should not be run.")
    print("\n📂 Current test suite location: ../tests/")
    print("📋 Run with: pytest ../tests/ -v")
    exit(1)
