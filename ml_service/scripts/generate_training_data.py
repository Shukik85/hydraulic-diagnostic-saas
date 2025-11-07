#!/usr/bin/env python3
"""
Generate Synthetic Training Data for Hydraulic System Models
Creates realistic sensor data with known anomaly patterns
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_hydraulic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Генерирует реалистичные данные гидравлической системы"""
    np.random.seed(42)

    data = []

    for i in range(n_samples):
        # Normal operation (80% of samples)
        is_anomaly = np.random.random() < 0.2

        if is_anomaly:
            # Anomaly patterns
            pressure_base = np.random.uniform(150, 180)  # high pressure
            temp_base = np.random.uniform(65, 85)  # high temp
            flow_base = np.random.uniform(5, 8)  # low flow
            vibration_base = np.random.uniform(2.5, 4.0)  # high vibration
        else:
            # Normal operation
            pressure_base = np.random.uniform(90, 120)  # normal pressure
            temp_base = np.random.uniform(35, 50)  # normal temp
            flow_base = np.random.uniform(10, 15)  # normal flow
            vibration_base = np.random.uniform(0.3, 1.5)  # low vibration

        # Add noise
        sample = {
            # Pressure sensors (6 channels)
            "ps1": pressure_base + np.random.normal(0, 3),
            "ps2": pressure_base + np.random.normal(0, 2),
            "ps3": pressure_base * 0.95 + np.random.normal(0, 2),
            "ps4": pressure_base * 1.05 + np.random.normal(0, 2),
            "ps5": pressure_base * 0.9 + np.random.normal(0, 3),
            "ps6": pressure_base + np.random.normal(0, 2),
            # Temperature sensors (4 channels)
            "ts1": temp_base + np.random.normal(0, 2),
            "ts2": temp_base + np.random.normal(0, 1.5),
            "ts3": temp_base * 1.1 + np.random.normal(0, 2),
            "ts4": temp_base + np.random.normal(0, 1),
            # Flow sensors (2 channels)
            "fs1": flow_base + np.random.normal(0, 0.5),
            "fs2": flow_base + np.random.normal(0, 0.3),
            # Vibration sensor
            "vs1": vibration_base + np.random.normal(0, 0.1),
            # Motor power
            "eps1": np.random.uniform(1.5, 3.0),
            # System efficiency metrics
            "ce": np.random.uniform(15, 25),  # cooling efficiency
            "cp": np.random.uniform(1.8, 2.8),  # cooling power
            "se": np.random.uniform(12, 20),  # system efficiency
            # Label
            "label": 1 if is_anomaly else 0,
        }

        data.append(sample)

    df = pd.DataFrame(data)
    print(f"Generated {n_samples} samples, {df['label'].sum()} anomalies ({100 * df['label'].mean():.1f}%)")

    return df


if __name__ == "__main__":
    # Create data directory
    data_dir = Path("data/uci_hydraulic")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate training data
    df = generate_synthetic_hydraulic_data(n_samples=1500)

    # Save as parquet
    output_path = data_dir / "cycles_sample_100.parquet"
    df.to_parquet(output_path, index=False)

    print(f"✅ Synthetic training data saved: {output_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"📈 Anomaly rate: {100 * df['label'].mean():.1f}%")
    print("🎯 Ready for training: python scripts/train_production_models.py")
