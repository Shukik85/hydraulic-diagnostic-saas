#!/usr/bin/env python3
"""
Quick Smoke Test for ML Models
Fast validation to ensure models work after implementation
"""

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from models import (
        CatBoostModel,
        XGBoostModel, 
        RandomForestModel,
        AdaptiveModel,
        EnsembleModel,
        check_model_availability
    )
    print("✅ All model imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


async def quick_test_model(model_class, model_name: str) -> bool:
    """Quick test of individual model."""
    print(f"🧪 Testing {model_name}...", end=" ")
    
    try:
        # Create and load model
        model = model_class()
        await model.load()
        
        if not model.is_loaded:
            print(f"❌ Failed to load")
            return False
        
        # Test single prediction
        test_features = np.random.rand(25)
        prediction = await model.predict(test_features)
        
        # Validate prediction structure
        required_keys = ['score', 'confidence', 'is_anomaly', 'processing_time_ms']
        if not all(key in prediction for key in required_keys):
            print(f"❌ Invalid prediction format")
            return False
        
        # Cleanup
        await model.cleanup()
        
        print(f"✅ PASS (score: {prediction['score']:.3f}, confidence: {prediction['confidence']:.3f})")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        return False


async def quick_test_ensemble() -> bool:
    """Quick test of ensemble model."""
    print(f"🎯 Testing ensemble...", end=" ")
    
    try:
        # Create and load ensemble
        ensemble = EnsembleModel()
        await ensemble.load_models()
        
        if not ensemble.is_loaded:
            print(f"❌ Failed to load")
            return False
        
        loaded_models = ensemble.get_loaded_models()
        print(f"\n   📊 Loaded models: {', '.join(loaded_models)}")
        
        # Test prediction
        test_features = np.random.rand(25)
        prediction = await ensemble.predict(test_features)
        
        # Validate ensemble prediction
        required_keys = ['ensemble_score', 'confidence', 'consensus_strength', 'individual_predictions']
        if not all(key in prediction for key in required_keys):
            print(f"❌ Invalid ensemble prediction format")
            return False
        
        individual_preds = prediction.get('individual_predictions', [])
        print(f"   🤖 Individual predictions: {len(individual_preds)}")
        print(f"   🎯 Ensemble score: {prediction['ensemble_score']:.3f}")
        print(f"   🤝 Consensus: {prediction.get('consensus_strength', 0):.3f}")
        
        # Cleanup
        await ensemble.cleanup()
        
        print(f"✅ ENSEMBLE PASS")
        return True
        
    except Exception as e:
        print(f"❌ ENSEMBLE FAIL: {str(e)}")
        return False


async def main():
    """Run quick smoke tests."""
    print("🚀 Quick ML Models Smoke Test")
    print("=" * 50)
    
    # Check availability first
    print("\n🔍 Checking model availability...")
    availability = check_model_availability()
    
    for model_name, available in availability.items():
        status = "✅" if available else "❌"
        print(f"   {status} {model_name}")
    
    print("\n🧪 Running individual model tests...")
    
    # Test individual models
    models_to_test = [
        (CatBoostModel, "catboost"),
        (XGBoostModel, "xgboost"),
        (RandomForestModel, "random_forest"),
        (AdaptiveModel, "adaptive")
    ]
    
    passed_models = 0
    total_models = len(models_to_test)
    
    for model_class, model_name in models_to_test:
        success = await quick_test_model(model_class, model_name)
        if success:
            passed_models += 1
    
    print(f"\n📊 Individual Models: {passed_models}/{total_models} passed")
    
    # Test ensemble
    print("\n🎯 Testing ensemble...")
    ensemble_success = await quick_test_ensemble()
    
    # Overall result
    print("\n" + "=" * 50)
    if passed_models >= 2 and ensemble_success:
        print("🎉 SMOKE TEST PASSED - Models are working!")
        print(f"✅ {passed_models} individual models functional")
        print("✅ Ensemble system operational")
        print("\n🚀 Ready for comprehensive testing and integration!")
        return 0
    else:
        print("❌ SMOKE TEST FAILED - Issues detected!")
        print(f"⚠️  Only {passed_models}/{total_models} models working")
        print(f"⚠️  Ensemble: {'✅' if ensemble_success else '❌'}")
        print("\n🔧 Check error messages above for debugging")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Smoke test failed: {e}")
        sys.exit(1)