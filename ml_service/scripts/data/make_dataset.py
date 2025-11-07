#!/usr/bin/env python3
"""
Make UCI dataset from RAW and export tidy CSV.
"""
import sys
from pathlib import Path

# Add ml_service to path for imports
sys.path.append(str(Path(__file__).parent))

from data.uci_raw_loader import build_raw_dataset

if __name__ == "__main__":
    print("🔧 Building UCI dataset from RAW files...")
    r = build_raw_dataset()
    print("📊 Build report:")
    print(f"   Status: {r['status']}")
    print(f"   Rows: {r['rows']}")
    if r['errors']:
        print(f"   Errors: {r['errors']}")
    else:
        print("   ✅ No errors")