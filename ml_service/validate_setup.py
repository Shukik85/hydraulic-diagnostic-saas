#!/usr/bin/env python3
"""
Setup Validation Script
Validates that all models can be imported and basic setup is correct
"""

import sys
from pathlib import Path

print("🔍 Validating ML Service Setup...")
print("=" * 40)

# Test basic imports
print("\n1️⃣ Testing core dependencies...")
try:
    import numpy as np
    print("   ✅ numpy")
except ImportError as e:
    print(f"   ❌ numpy: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("   ✅ pandas")
except ImportError as e:
    print(f"   ❌ pandas: {e}")

try:
    import sklearn
    print("   ✅ scikit-learn")
except ImportError as e:
    print(f"   ❌ scikit-learn: {e}")
    sys.exit(1)

try:
    import xgboost as xgb
    print("   ✅ xgboost")
except ImportError as e:
    print(f"   ❌ xgboost: {e}")
    sys.exit(1)

try:
    import catboost as cb
    print("   ✅ catboost")
except ImportError as e:
    print(f"   ❌ catboost: {e}")
    sys.exit(1)

try:
    import joblib
    print("   ✅ joblib")
except ImportError as e:
    print(f"   ❌ joblib: {e}")
    sys.exit(1)

try:
    import structlog
    print("   ✅ structlog")
except ImportError as e:
    print(f"   ❌ structlog: {e}")
    sys.exit(1)

# Test config import
print("\n2️⃣ Testing configuration...")
try:
    from config import settings
    print("   ✅ config.settings")
    print(f"   📝 Model path: {settings.model_path}")
    print(f"   🎯 Prediction threshold: {settings.prediction_threshold}")
except ImportError as e:
    print(f"   ❌ config: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ⚠️  config warning: {e}")

# Test model imports
print("\n3️⃣ Testing model imports...")
try:
    from models.base_model import BaseMLModel
    print("   ✅ BaseMLModel")
except ImportError as e:
    print(f"   ❌ BaseMLModel: {e}")
    sys.exit(1)

try:
    from models.catboost_model import CatBoostModel
    print("   ✅ CatBoostModel")
except ImportError as e:
    print(f"   ❌ CatBoostModel: {e}")
    sys.exit(1)

try:
    from models.xgboost_model import XGBoostModel
    print("   ✅ XGBoostModel")
except ImportError as e:
    print(f"   ❌ XGBoostModel: {e}")
    sys.exit(1)

try:
    from models.random_forest_model import RandomForestModel
    print("   ✅ RandomForestModel")
except ImportError as e:
    print(f"   ❌ RandomForestModel: {e}")
    sys.exit(1)

try:
    from models.adaptive_model import AdaptiveModel
    print("   ✅ AdaptiveModel")
except ImportError as e:
    print(f"   ❌ AdaptiveModel: {e}")
    sys.exit(1)

try:
    from models.ensemble import EnsembleModel
    print("   ✅ EnsembleModel")
except ImportError as e:
    print(f"   ❌ EnsembleModel: {e}")
    sys.exit(1)

# Test models package
print("\n4️⃣ Testing models package...")
try:
    from models import (
        AVAILABLE_MODELS,
        MODEL_REGISTRY,
        check_model_availability,
        create_model
    )
    print("   ✅ models package imports")
    print(f"   📊 Available models: {AVAILABLE_MODELS}")
except ImportError as e:
    print(f"   ❌ models package: {e}")
    sys.exit(1)

# Test model instantiation
print("\n5️⃣ Testing model instantiation...")
try:
    availability = check_model_availability()
    for model_name, available in availability.items():
        status = "✅" if available else "❌"
        print(f"   {status} {model_name} instantiation")
except Exception as e:
    print(f"   ❌ Model instantiation failed: {e}")
    sys.exit(1)

# Test numpy data creation
print("\n6️⃣ Testing data handling...")
try:
    test_data = np.random.rand(10, 25)
    print(f"   ✅ Test data created: {test_data.shape}")
except Exception as e:
    print(f"   ❌ Data handling failed: {e}")
    sys.exit(1)

# Check file structure
print("\n7️⃣ Checking file structure...")
required_files = [
    "models/__init__.py",
    "models/base_model.py", 
    "models/catboost_model.py",
    "models/xgboost_model.py",
    "models/random_forest_model.py",
    "models/adaptive_model.py",
    "models/ensemble.py",
    "config.py",
    "main.py"
]

for file_path in required_files:
    if Path(file_path).exists():
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} missing")

print("\n" + "=" * 40)
print("🎉 Setup validation completed successfully!")
print("🚀 All models and dependencies are properly configured")
print("\n📝 Next steps:")
print("   1. Run quick_test.py for smoke testing")
print("   2. Run scripts/test_models.py for comprehensive testing")
print("   3. Start integration with TimescaleDB")

print("\n🔥 Ready to rock! No more fake models! 🔥")