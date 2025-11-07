#!/usr/bin/env python3
"""
UCI Data Loader Test
Quick validation of real data loading and preprocessing
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from data.uci_loader import UCIHydraulicLoader, load_uci_hydraulic_data
    print("✅ UCI loader import successful")
except ImportError as e:
    print(f"❌ UCI loader import failed: {e}")
    sys.exit(1)

def main():
    print("📊 Testing UCI Hydraulic Data Loader")
    print("=" * 50)
    
    try:
        # Test the loader
        loader = UCIHydraulicLoader()
        
        # Check if data files exist
        data_path = Path("./data/industrial_iot")
        if not data_path.exists():
            print(f"❌ Data directory not found: {data_path}")
            return False
            
        files = list(data_path.glob("*.csv"))
        print(f"\n📁 Available data files:")
        for f in files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"   ✅ {f.name} ({size_mb:.1f} MB)")
        
        if not files:
            print("❌ No CSV files found in data directory")
            return False
        
        # Test loading smaller file first
        small_file = "industrial_fault_detection_data_1000.csv"
        
        print(f"\n📊 Testing with {small_file}...")
        
        # Load and prepare data
        data = load_uci_hydraulic_data(
            filename=small_file,
            window_minutes=5
        )
        
        print(f"\n✅ Data loaded and prepared successfully!")
        
        # Display data info
        info = data['data_info']
        print(f"\n📈 Dataset Information:")
        print(f"   Total samples: {info['total_samples']:,}")
        print(f"   Features: {info['n_features']}")
        print(f"   Window: {info['window_minutes']} minutes")
        print(f"   Binary classification: {info['binary_classification']}")
        print(f"   Class distribution: {info['class_distribution']}")
        print(f"   Date range: {info['date_range']['start']} to {info['date_range']['end']}")
        
        # Display data splits
        print(f"\n📊 Data Splits:")
        print(f"   Training: {len(data['X_train']):,} samples")
        print(f"   Validation: {len(data['X_val']):,} samples")
        print(f"   Test: {len(data['X_test']):,} samples")
        
        # Show feature engineering results
        print(f"\n🔍 Engineered Features (first 15):")
        for i, name in enumerate(data['feature_names'][:15]):
            sample_val = data['X_train'][0, i]
            print(f"   {i+1:2d}. {name:40s}: {sample_val:8.3f}")
        
        if len(data['feature_names']) > 15:
            print(f"   ... and {len(data['feature_names']) - 15} more features")
        
        # Data quality checks
        print(f"\n🧪 Data Quality Checks:")
        
        # Check for NaN values
        nan_count = np.sum(np.isnan(data['X_train']))
        print(f"   NaN values in training data: {nan_count}")
        
        # Check feature ranges
        X_train = data['X_train']
        feature_mins = np.min(X_train, axis=0)
        feature_maxs = np.max(X_train, axis=0)
        feature_stds = np.std(X_train, axis=0)
        
        print(f"   Feature ranges:")
        print(f"     Min values: [{feature_mins.min():.3f}, {feature_mins.max():.3f}]")
        print(f"     Max values: [{feature_maxs.min():.3f}, {feature_maxs.max():.3f}]")
        print(f"     Std values: [{feature_stds.min():.3f}, {feature_stds.max():.3f}]")
        
        # Check class balance
        y_train = data['y_train']
        class_counts = np.bincount(y_train)
        class_ratio = class_counts[1] / class_counts[0] if len(class_counts) > 1 else 0
        print(f"   Class balance (fault/normal): {class_ratio:.3f}")
        
        # Test with larger file if available
        large_file = "Industrial_fault_detection.csv"
        if (data_path / large_file).exists():
            print(f"\n📊 Testing with LARGE dataset {large_file}...")
            try:
                large_data = load_uci_hydraulic_data(
                    filename=large_file,
                    window_minutes=5
                )
                print(f"   ✅ Large dataset loaded: {large_data['data_info']['total_samples']:,} samples")
                print(f"   ✅ Features: {large_data['data_info']['n_features']}")
                print(f"   ✅ Training samples: {len(large_data['X_train']):,}")
                
            except Exception as e:
                print(f"   ⚠️  Large dataset test failed: {e}")
        
        print(f"\n🎉 UCI Data Loader test completed successfully!")
        print(f"💪 Ready to train models on REAL data!")
        
        # Save test results
        reports_dir = Path("./reports")
        reports_dir.mkdir(exist_ok=True)
        
        import json
        test_report = {
            "test_timestamp": time.time(),
            "test_status": "SUCCESS",
            "data_files_found": [f.name for f in files],
            "small_dataset_info": info,
            "feature_count": len(data['feature_names']),
            "sample_features": data['feature_names'][:10],
            "data_quality": {
                "nan_count": int(nan_count),
                "feature_ranges": {
                    "min": float(feature_mins.min()),
                    "max": float(feature_maxs.max())
                },
                "class_balance_ratio": float(class_ratio)
            }
        }
        
        with open(reports_dir / "uci_loader_test_report.json", 'w') as f:
            json.dump(test_report, f, indent=2)
        
        print(f"\n💾 Test report saved to reports/uci_loader_test_report.json")
        
        return True
        
    except Exception as e:
        print(f"\n💥 UCI loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import time
    import numpy as np
    
    success = main()
    
    if success:
        print(f"\n🚀 Ready to run: python train_real_production_models.py")
        sys.exit(0)
    else:
        print(f"\n❌ Fix issues before running training")
        sys.exit(1)