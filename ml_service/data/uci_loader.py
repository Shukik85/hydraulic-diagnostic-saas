#!/usr/bin/env python3
"""
UCI Hydraulic Data Loader
Load and preprocess real industrial IoT sensor data for model training
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()


class UCIHydraulicLoader:
    """Load and preprocess UCI Hydraulic dataset for ML training."""

    def __init__(self, data_path: str = "./data/industrial_iot"):
        self.data_path = Path(data_path)
        self.df = None
        self.feature_names = []
        self.scaler = StandardScaler()

    def find_data_file(self, filename: str) -> Path:
        """Auto-search for data file in multiple locations."""

        # Search paths in priority order
        search_paths = [
            self.data_path / filename,  # Original path
            Path("ml_service/data/industrial_iot") / filename,  # RAW export path
            Path("data/industrial_iot") / filename,  # Alternative path
            Path("./") / filename,  # Current directory
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Found data file at: {path}")
                return path

        # If not found, show all attempted paths
        attempted_paths = [str(p) for p in search_paths]
        logger.error(f"Data file '{filename}' not found in any of: {attempted_paths}")
        raise FileNotFoundError(f"Data file '{filename}' not found. Searched: {attempted_paths}")

    def load_data(self, filename: str = "Industrial_fault_detection.csv") -> pd.DataFrame:
        """Load industrial IoT data from CSV with auto-path detection."""

        # First try the newer large dataset, then fallback to smaller
        fallback_files = [
            "Industrial_fault_detection.csv",
            "industrial_fault_detection_data_1000.csv",
        ]

        if filename not in fallback_files:
            fallback_files.insert(0, filename)

        df = None
        for try_filename in fallback_files:
            try:
                file_path = self.find_data_file(try_filename)
                logger.info(f"Loading UCI Hydraulic data from {file_path}")

                # Load CSV data
                df = pd.read_csv(file_path)

                # Parse timestamp
                if "Timestamp" in df.columns:
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

                logger.info(f"✅ Loaded {len(df)} samples from {try_filename}")
                logger.info(f"   Columns: {list(df.columns)}")

                if "Timestamp" in df.columns:
                    logger.info(f"   Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

                # Check fault label distribution
                if "Fault Label" in df.columns:
                    fault_counts = df["Fault Label"].value_counts().sort_index()
                    logger.info(f"   Fault distribution: {dict(fault_counts)}")

                self.df = df
                return df

            except FileNotFoundError:
                logger.warning(f"File not found: {try_filename}, trying next...")
                continue
            except Exception as e:
                logger.error(f"Error loading {try_filename}: {e}")
                continue

        # If we get here, no file was found
        raise FileNotFoundError(f"No valid data files found from: {fallback_files}")

    def engineer_features(self, window_minutes: int = 5) -> tuple[np.ndarray, list[str]]:
        """Engineer features from time series data."""

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info(f"Engineering features with {window_minutes}-minute windows")

        # Sort by timestamp if available
        if "Timestamp" in self.df.columns:
            df_sorted = self.df.sort_values("Timestamp").reset_index(drop=True)
        else:
            df_sorted = self.df.copy()
            # Create synthetic timestamp if missing
            df_sorted["Timestamp"] = pd.date_range("2025-01-01", periods=len(df_sorted), freq="100ms")

        # Create rolling windows for feature engineering
        feature_data = []
        feature_names = []

        # Detect sensor features from columns
        sensor_features = {}
        for col in df_sorted.columns:
            if col not in ["Timestamp", "Fault Label"]:
                # Map column names to standardized names
                if "vibration" in col.lower() or "VS" in col:
                    sensor_features["vibration"] = col
                elif "temperature" in col.lower() or "TS" in col:
                    sensor_features["temperature"] = col
                elif "pressure" in col.lower() or "PS" in col:
                    sensor_features["pressure"] = col
                elif "rms" in col.lower():
                    sensor_features["rms_vibration"] = col
                elif "mean" in col.lower() and "temp" in col.lower():
                    sensor_features["mean_temp"] = col

        # If standard mapping fails, use first few numeric columns
        if not sensor_features:
            numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ["Fault Label"]]
            for i, col in enumerate(numeric_cols[:5]):  # Take first 5 numeric columns
                sensor_features[f"sensor_{i + 1}"] = col

        logger.info(f"Detected sensor features: {sensor_features}")

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
            "pressure_vibration_ratio",
            "temp_pressure_ratio",
            "vibration_change_rate",
            "pressure_stability",
            "temperature_stability",
        ]
        feature_names.extend(derived_features)

        # Set timestamp as index for rolling operations if available
        if "Timestamp" in df_sorted.columns:
            df_sorted.set_index("Timestamp", inplace=True)

        # Calculate rolling statistics
        rolling_stats = {}
        for feature_name, column_name in sensor_features.items():
            if column_name in df_sorted.columns:
                if "Timestamp" in self.df.columns:
                    rolling_series = df_sorted[column_name].rolling(rolling_window, min_periods=1)
                else:
                    # Use sample-based window if no timestamp
                    window_samples = int(window_minutes * 60 * 10)  # Assume 10Hz
                    rolling_series = df_sorted[column_name].rolling(window_samples, min_periods=1)

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
            if column_name in df_sorted.columns:
                X[:, feature_idx] = df_sorted[column_name].values
            feature_idx += 1

        # Rolling statistics
        for feature_name, column_name in sensor_features.items():
            for stat_type in ["mean", "std", "min", "max"]:
                stat_key = f"rolling_{stat_type}_{feature_name}"
                if stat_key in rolling_stats:
                    X[:, feature_idx] = rolling_stats[stat_key].values
                feature_idx += 1

        # Derived features
        # Get main sensor columns or use first available
        vibration_col = None
        pressure_col = None
        temperature_col = None

        for feature_name, column_name in sensor_features.items():
            if "vibration" in feature_name and column_name in df_sorted.columns:
                vibration_col = df_sorted[column_name].values
            elif "pressure" in feature_name and column_name in df_sorted.columns:
                pressure_col = df_sorted[column_name].values
            elif "temperature" in feature_name and column_name in df_sorted.columns:
                temperature_col = df_sorted[column_name].values

        # Use fallbacks if main sensors not found
        if vibration_col is None and sensor_features:
            first_sensor = list(sensor_features.values())[0]
            vibration_col = df_sorted[first_sensor].values if first_sensor in df_sorted.columns else np.ones(n_samples)

        if pressure_col is None and len(sensor_features) > 1:
            second_sensor = list(sensor_features.values())[1]
            pressure_col = (
                df_sorted[second_sensor].values if second_sensor in df_sorted.columns else np.ones(n_samples) * 160
            )
        elif pressure_col is None:
            pressure_col = np.ones(n_samples) * 160

        if temperature_col is None and len(sensor_features) > 2:
            third_sensor = list(sensor_features.values())[2]
            temperature_col = (
                df_sorted[third_sensor].values if third_sensor in df_sorted.columns else np.ones(n_samples) * 60
            )
        elif temperature_col is None:
            temperature_col = np.ones(n_samples) * 60

        # Pressure-vibration ratio
        X[:, feature_idx] = pressure_col / (np.abs(vibration_col) + 1e-6)
        feature_idx += 1

        # Temperature-pressure ratio
        X[:, feature_idx] = temperature_col / (pressure_col + 1e-6)
        feature_idx += 1

        # Vibration change rate (derivative approximation)
        vibration_diff = np.diff(vibration_col, prepend=vibration_col[0])
        X[:, feature_idx] = vibration_diff
        feature_idx += 1

        # Pressure stability (inverse of std in window)
        pressure_key = None
        for feature_name, column_name in sensor_features.items():
            if "pressure" in feature_name:
                pressure_key = f"rolling_std_{feature_name}"
                break

        if pressure_key and pressure_key in rolling_stats:
            pressure_rolling_std = rolling_stats[pressure_key].values
        else:
            pressure_rolling_std = np.ones(n_samples) * 0.1

        X[:, feature_idx] = 1.0 / (pressure_rolling_std + 1e-6)
        feature_idx += 1

        # Temperature stability
        temp_key = None
        for feature_name, column_name in sensor_features.items():
            if "temperature" in feature_name:
                temp_key = f"rolling_std_{feature_name}"
                break

        if temp_key and temp_key in rolling_stats:
            temp_rolling_std = rolling_stats[temp_key].values
        else:
            temp_rolling_std = np.ones(n_samples) * 0.1

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

        if "Fault Label" not in self.df.columns:
            logger.warning("No 'Fault Label' column found, creating dummy labels")
            labels = np.zeros(len(self.df))
        else:
            labels = self.df["Fault Label"].values

        if binary_classification:
            # Convert multi-class to binary: 0 = normal, 1 = any fault
            binary_labels = (labels > 0).astype(int)

            logger.info(
                f"Binary classification: Normal={np.sum(binary_labels == 0)}, Fault={np.sum(binary_labels == 1)}"
            )
            return binary_labels
        else:
            # Keep multi-class labels
            unique_labels = np.unique(labels)
            logger.info(f"Multi-class classification: {dict(zip(unique_labels, np.bincount(labels.astype(int))))}")
            return labels

    def load_and_prepare_production_data(
        self,
        filename: str = "Industrial_fault_detection.csv",
        window_minutes: int = 5,
        binary_classification: bool = True,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> dict[str, np.ndarray]:
        """Complete data loading and preparation pipeline."""

        logger.info("Starting UCI Hydraulic data preparation pipeline")

        # Load raw data with auto-search
        self.load_data(filename)

        # Engineer features
        X, feature_names = self.engineer_features(window_minutes=window_minutes)

        # Prepare labels
        y = self.prepare_labels(binary_classification=binary_classification)

        # Ensure we have the same number of samples
        assert len(X) == len(y), f"Feature-label mismatch: {len(X)} != {len(y)}"

        # Split data: train/val/test
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

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
                    "start": str(self.df["Timestamp"].min()) if "Timestamp" in self.df.columns else "N/A",
                    "end": str(self.df["Timestamp"].max()) if "Timestamp" in self.df.columns else "N/A",
                },
            },
        }

    def save_feature_info(self, feature_names: list[str], output_path: str = "./reports") -> None:
        """Save feature information for production use."""

        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        feature_info = {
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "timestamp": datetime.now().isoformat(),
            "data_source": "UCI Hydraulic Industrial IoT Dataset",
        }

        # Save as JSON
        import json

        with open(output_dir / "feature_contract.json", "w") as f:
            json.dump(feature_info, f, indent=2)

        logger.info(f"Feature contract saved to {output_dir / 'feature_contract.json'}")


def load_uci_hydraulic_data(
    data_path: str = "./data/industrial_iot", filename: str = "Industrial_fault_detection.csv", window_minutes: int = 5
) -> dict[str, np.ndarray]:
    """Convenience function to load UCI hydraulic data."""

    loader = UCIHydraulicLoader(data_path)
    return loader.load_and_prepare_production_data(
        filename=filename,
        window_minutes=window_minutes,
        binary_classification=True,  # Convert multi-class to binary for anomaly detection
    )


if __name__ == "__main__":
    # Test the loader
    try:
        print("📊 Testing UCI Hydraulic Data Loader")
        print("=" * 50)

        loader = UCIHydraulicLoader()

        # Load and prepare data
        data = loader.load_and_prepare_production_data()

        print("\n✅ Data loaded successfully!")
        print(f"   Train samples: {len(data['X_train'])}")
        print(f"   Validation samples: {len(data['X_val'])}")
        print(f"   Test samples: {len(data['X_test'])}")
        print(f"   Features: {data['data_info']['n_features']}")
        print(f"   Classes: {data['data_info']['class_distribution']}")

        # Show feature names
        print("\n🔍 First 10 features:")
        for i, name in enumerate(data["feature_names"][:10]):
            print(f"   {i + 1:2d}. {name}")

        print("\n📊 Sample feature values (first sample):")
        sample_features = data["X_train"][0]
        for i in range(min(10, len(sample_features))):
            print(f"   {data['feature_names'][i]:30s}: {sample_features[i]:8.3f}")

        # Save feature info
        loader.save_feature_info(data["feature_names"])

        print("\n🎉 UCI Hydraulic loader test completed successfully!")

    except Exception as e:
        print(f"\n❌ Loader test failed: {e}")
        import traceback

        traceback.print_exc()
