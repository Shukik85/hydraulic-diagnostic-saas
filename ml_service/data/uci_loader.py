#!/usr/bin/env python3
"""
UCI Hydraulic Data Loader
Load and preprocess real industrial IoT sensor data for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import structlog
from datetime import datetime

logger = structlog.get_logger()


class UCIHydraulicLoader:
    """Load and preprocess UCI Hydraulic dataset for ML training."""
    
    def __init__(self, data_path: str = "./data/industrial_iot"):
        self.data_path = Path(data_path)
        self.df = None
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def load_data(self, filename: str = "industrial_fault_detection_data_1000.csv") -> pd.DataFrame:
        """Load industrial IoT data from CSV."""
        
        file_path = self.data_path / filename
        
        if not file_path.exists():
            logger.error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading UCI Hydraulic data from {file_path}")
        
        # Load CSV data
        df = pd.read_csv(file_path)
        
        # Parse timestamp
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        logger.info(f"Loaded {len(df)} samples with columns: {list(df.columns)}")
        logger.info(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        
        # Check fault label distribution
        fault_counts = df['Fault Label'].value_counts().sort_index()
        logger.info(f"Fault label distribution: {dict(fault_counts)}")
        
        self.df = df
        return df
    
    def engineer_features(self, window_minutes: int = 5) -> Tuple[np.ndarray, List[str]]:
        """Engineer features from time series data."""
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info(f"Engineering features with {window_minutes}-minute windows")
        
        # Sort by timestamp
        df_sorted = self.df.sort_values('Timestamp').reset_index(drop=True)
        
        # Create rolling windows for feature engineering
        feature_data = []
        feature_names = []
        
        # Base sensor features (current values)
        sensor_features = {
            'vibration': 'Vibration (mm/s)',
            'temperature': 'Temperature (°C)', 
            'pressure': 'Pressure (bar)',
            'rms_vibration': 'RMS Vibration',
            'mean_temp': 'Mean Temp'
        }
        
        # Add current sensor values
        for feature_name, column_name in sensor_features.items():
            feature_names.append(f"current_{feature_name}")
        
        # Add rolling statistics for each sensor
        rolling_window = f"{window_minutes}min"
        
        for feature_name, column_name in sensor_features.items():
            # Rolling mean
            feature_names.append(f"rolling_mean_{feature_name}_{window_minutes}min")
            # Rolling std
            feature_names.append(f"rolling_std_{feature_name}_{window_minutes}min")
            # Rolling min
            feature_names.append(f"rolling_min_{feature_name}_{window_minutes}min")
            # Rolling max  
            feature_names.append(f"rolling_max_{feature_name}_{window_minutes}min")
        
        # Add derived features
        derived_features = [
            'pressure_vibration_ratio',
            'temp_pressure_ratio',
            'vibration_change_rate',
            'pressure_stability',
            'temperature_stability'
        ]
        feature_names.extend(derived_features)
        
        # Set timestamp as index for rolling operations
        df_sorted.set_index('Timestamp', inplace=True)
        
        # Calculate rolling statistics
        rolling_stats = {}
        for feature_name, column_name in sensor_features.items():
            rolling_series = df_sorted[column_name].rolling(rolling_window, min_periods=1)
            rolling_stats[f"rolling_mean_{feature_name}"] = rolling_series.mean()
            rolling_stats[f"rolling_std_{feature_name}"] = rolling_series.std().fillna(0)
            rolling_stats[f"rolling_min_{feature_name}"] = rolling_series.min()
            rolling_stats[f"rolling_max_{feature_name}"] = rolling_series.max()
        
        # Build feature matrix
        n_samples = len(df_sorted)
        n_features = len(feature_names)
        X = np.zeros((n_samples, n_features))
        
        feature_idx = 0
        
        # Current sensor values
        for feature_name, column_name in sensor_features.items():
            X[:, feature_idx] = df_sorted[column_name].values
            feature_idx += 1
        
        # Rolling statistics
        for feature_name, column_name in sensor_features.items():
            for stat_type in ['mean', 'std', 'min', 'max']:
                stat_key = f"rolling_{stat_type}_{feature_name}"
                X[:, feature_idx] = rolling_stats[stat_key].values
                feature_idx += 1
        
        # Derived features
        vibration_col = df_sorted['Vibration (mm/s)'].values
        pressure_col = df_sorted['Pressure (bar)'].values
        temperature_col = df_sorted['Temperature (°C)'].values
        
        # Pressure-vibration ratio
        X[:, feature_idx] = pressure_col / (vibration_col + 1e-6)
        feature_idx += 1
        
        # Temperature-pressure ratio
        X[:, feature_idx] = temperature_col / (pressure_col + 1e-6)
        feature_idx += 1
        
        # Vibration change rate (derivative approximation)
        vibration_diff = np.diff(vibration_col, prepend=vibration_col[0])
        X[:, feature_idx] = vibration_diff
        feature_idx += 1
        
        # Pressure stability (inverse of std in window)
        pressure_rolling_std = rolling_stats["rolling_std_pressure"].values
        X[:, feature_idx] = 1.0 / (pressure_rolling_std + 1e-6)
        feature_idx += 1
        
        # Temperature stability
        temp_rolling_std = rolling_stats["rolling_std_temperature"].values
        X[:, feature_idx] = 1.0 / (temp_rolling_std + 1e-6)
        feature_idx += 1
        
        # Handle any NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        logger.info(f"Engineered {X.shape[1]} features from {len(sensor_features)} base sensors")
        
        self.feature_names = feature_names
        return X, feature_names
    
    def prepare_labels(self, binary_classification: bool = True) -> np.ndarray:
        """Prepare labels for classification."""
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        labels = self.df['Fault Label'].values
        
        if binary_classification:
            # Convert multi-class to binary: 0 = normal, 1 = any fault
            binary_labels = (labels > 0).astype(int)
            
            logger.info(f"Binary classification: Normal={np.sum(binary_labels==0)}, Fault={np.sum(binary_labels==1)}")
            return binary_labels
        else:
            # Keep multi-class labels
            unique_labels = np.unique(labels)
            logger.info(f"Multi-class classification: {dict(zip(unique_labels, np.bincount(labels.astype(int))))}")
            return labels
    
    def load_and_prepare_production_data(self, 
                                       filename: str = "industrial_fault_detection_data_1000.csv",
                                       window_minutes: int = 5,
                                       binary_classification: bool = True,
                                       test_size: float = 0.2,
                                       val_size: float = 0.1) -> Dict[str, np.ndarray]:
        """Complete data loading and preparation pipeline."""
        
        logger.info("Starting UCI Hydraulic data preparation pipeline")
        
        # Load raw data
        self.load_data(filename)
        
        # Engineer features
        X, feature_names = self.engineer_features(window_minutes=window_minutes)
        
        # Prepare labels
        y = self.prepare_labels(binary_classification=binary_classification)
        
        # Ensure we have the same number of samples
        assert len(X) == len(y), f"Feature-label mismatch: {len(X)} != {len(y)}"
        
        # Split data: train/val/test
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: from the 80%, split into train and validation
        val_from_temp = val_size / (1 - test_size)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_from_temp, random_state=42, stratify=y_temp
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Features: {len(feature_names)} engineered features")
        
        return {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled, 
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "X_train_raw": X_train,  # Unscaled for reference
            "feature_names": feature_names,
            "scaler": scaler,
            "data_info": {
                "total_samples": len(X),
                "n_features": len(feature_names),
                "window_minutes": window_minutes,
                "binary_classification": binary_classification,
                "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
                "date_range": {
                    "start": str(self.df['Timestamp'].min()),
                    "end": str(self.df['Timestamp'].max())
                }
            }
        }
    
    def save_feature_info(self, feature_names: List[str], output_path: str = "./reports") -> None:
        """Save feature information for production use."""
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        feature_info = {
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "timestamp": datetime.now().isoformat(),
            "data_source": "UCI Hydraulic Industrial IoT Dataset"
        }
        
        # Save as JSON
        import json
        with open(output_dir / "feature_contract.json", 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logger.info(f"Feature contract saved to {output_dir / 'feature_contract.json'}")


def load_uci_hydraulic_data(data_path: str = "./data/industrial_iot",
                           filename: str = "industrial_fault_detection_data_1000.csv",
                           window_minutes: int = 5) -> Dict[str, np.ndarray]:
    """Convenience function to load UCI hydraulic data."""
    
    loader = UCIHydraulicLoader(data_path)
    return loader.load_and_prepare_production_data(
        filename=filename,
        window_minutes=window_minutes,
        binary_classification=True  # Convert multi-class to binary for anomaly detection
    )


if __name__ == "__main__":
    # Test the loader
    try:
        print("📊 Testing UCI Hydraulic Data Loader")
        print("=" * 50)
        
        loader = UCIHydraulicLoader()
        
        # Load and prepare data
        data = loader.load_and_prepare_production_data()
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   Train samples: {len(data['X_train'])}")
        print(f"   Validation samples: {len(data['X_val'])}")
        print(f"   Test samples: {len(data['X_test'])}")
        print(f"   Features: {data['data_info']['n_features']}")
        print(f"   Classes: {data['data_info']['class_distribution']}")
        
        # Show feature names
        print(f"\n🔍 First 10 features:")
        for i, name in enumerate(data['feature_names'][:10]):
            print(f"   {i+1:2d}. {name}")
        
        print(f"\n📊 Sample feature values (first sample):")
        sample_features = data['X_train'][0]
        for i in range(min(10, len(sample_features))):
            print(f"   {data['feature_names'][i]:30s}: {sample_features[i]:8.3f}")
        
        # Save feature info
        loader.save_feature_info(data['feature_names'])
        
        print(f"\n🎉 UCI Hydraulic loader test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Loader test failed: {e}")
        import traceback
        traceback.print_exc()