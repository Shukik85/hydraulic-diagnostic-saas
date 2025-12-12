#!/usr/bin/env python3
"""Inspect CSV sensor data for Phase 3 validation.

Inspects bim_comprehensive.csv to identify available sensors
and their suitability for Phase 3 dynamic edge features.

Author: GNN Service Team
Python: 3.14+
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def inspect_csv_columns(csv_path: Path) -> dict:
    """Inspect CSV columns to identify sensors.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Dictionary with sensor inventory
    """
    print(f"Reading CSV: {csv_path}")
    print(f"File size: {csv_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Read first few rows to inspect structure
    df_head = pd.read_csv(csv_path, nrows=1000)

    print(f"\nDataset shape (first 1000 rows): {df_head.shape}")
    print(f"Columns: {len(df_head.columns)}")

    # Identify sensor types by column names
    sensors = {
        "pressure": [],
        "temperature": [],
        "vibration": [],
        "flow": [],
        "rpm": [],
        "other": []
    }

    for col in df_head.columns:
        col_lower = col.lower()

        if "pressure" in col_lower or "press" in col_lower:
            sensors["pressure"].append(col)
        elif "temperature" in col_lower or "temp" in col_lower:
            sensors["temperature"].append(col)
        elif "vibration" in col_lower or "vib" in col_lower:
            sensors["vibration"].append(col)
        elif "flow" in col_lower:
            sensors["flow"].append(col)
        elif "rpm" in col_lower or "speed" in col_lower:
            sensors["rpm"].append(col)
        else:
            sensors["other"].append(col)

    # Print summary
    print("\n" + "="*60)
    print("SENSOR INVENTORY")
    print("="*60)

    for sensor_type, columns in sensors.items():
        if columns:
            print(f"\n{sensor_type.upper()} ({len(columns)} columns):")
            for col in columns[:5]:  # Show first 5
                # Get basic stats
                col_data = df_head[col]
                missing = col_data.isna().sum()
                range_str = f"{col_data.min():.2f} to {col_data.max():.2f}"

                print(f"  - {col:40s} [{range_str}] (missing: {missing})")

            if len(columns) > 5:
                print(f"  ... and {len(columns) - 5} more")

    # Analyze data quality
    print("\n" + "="*60)
    print("DATA QUALITY")
    print("="*60)

    total_missing = df_head.isna().sum().sum()
    total_cells = df_head.shape[0] * df_head.shape[1]
    missing_pct = (total_missing / total_cells) * 100

    print(f"\nMissing values: {total_missing:,} / {total_cells:,} ({missing_pct:.2f}%)")

    # Check if timestamp column exists
    timestamp_cols = [c for c in df_head.columns if "time" in c.lower()]
    print(f"\nTimestamp columns: {timestamp_cols}")

    if timestamp_cols:
        # Estimate sampling rate
        ts_col = timestamp_cols[0]
        if pd.api.types.is_datetime64_any_dtype(df_head[ts_col]):
            time_diffs = df_head[ts_col].diff().dropna()
            avg_diff = time_diffs.mean()
            print(f"Average time step: {avg_diff}")

    return sensors


def check_phase3_compatibility(sensors: dict) -> dict:
    """Check if sensors support Phase 3 features.
    
    Args:
        sensors: Sensor inventory
    
    Returns:
        Compatibility report
    """
    print("\n" + "="*60)
    print("PHASE 3 COMPATIBILITY")
    print("="*60)

    compatibility = {
        "static_edge_features": {
            "status": "manual_input",
            "note": "Diameter, length, material from edge_specifications.json"
        },
        "dynamic_edge_features": {}
    }

    # Check each dynamic feature
    features = [
        ("pressure_drop_bar", "pressure", True),
        ("flow_rate_lpm", "flow", False),
        ("temperature_delta_c", "temperature", False),
        ("vibration_level_g", "vibration", False),
        ("age_hours", None, False),  # From install_date
        ("maintenance_score", None, False)  # From maintenance logs
    ]

    for feature_name, sensor_type, can_compute in features:
        if sensor_type:
            has_sensor = len(sensors.get(sensor_type, [])) > 0

            if has_sensor:
                status = "✅ AVAILABLE"
            elif can_compute:
                status = "⚠️ COMPUTABLE (from other sensors)"
            else:
                status = "❌ MISSING (use synthetic)"
        else:
            status = "⚠️ FROM METADATA (install_date, maintenance logs)"

        compatibility["dynamic_edge_features"][feature_name] = {
            "status": status,
            "sensor_required": sensor_type,
            "can_compute": can_compute
        }

        print(f"  {feature_name:25s} → {status}")

    return compatibility


def generate_report(sensors: dict, compatibility: dict, output_dir: Path):
    """Generate sensor inventory and compatibility report.
    
    Args:
        sensors: Sensor inventory
        compatibility: Compatibility report
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sensor inventory
    inventory_path = output_dir / "sensor_inventory.json"
    with open(inventory_path, "w") as f:
        json.dump(sensors, f, indent=2)
    print(f"\n✅ Saved sensor inventory: {inventory_path}")

    # Save compatibility report
    compat_path = output_dir / "phase3_compatibility.json"
    with open(compat_path, "w") as f:
        json.dump(compatibility, f, indent=2)
    print(f"✅ Saved compatibility report: {compat_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_sensors = sum(len(v) for v in sensors.values())
    print(f"\nTotal sensors: {total_sensors}")
    print(f"  - Pressure: {len(sensors['pressure'])}")
    print(f"  - Temperature: {len(sensors['temperature'])}")
    print(f"  - Vibration: {len(sensors['vibration'])}")
    print(f"  - Flow: {len(sensors['flow'])}")
    print(f"  - RPM: {len(sensors['rpm'])}")
    print(f"  - Other: {len(sensors['other'])}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if len(sensors["pressure"]) > 0:
        print("\n✅ Pressure sensors AVAILABLE")
        print("   → Can compute pressure_drop_bar")

    if len(sensors["temperature"]) > 0:
        print("\n✅ Temperature sensors AVAILABLE")
        print("   → Can compute temperature_delta_c")
    else:
        print("\n⚠️ Temperature sensors MISSING")
        print("   → Use synthetic data (60-70°C typical)")

    if len(sensors["vibration"]) > 0:
        print("\n✅ Vibration sensors AVAILABLE")
        print("   → Can compute vibration_level_g")
    else:
        print("\n⚠️ Vibration sensors MISSING")
        print("   → Use synthetic data (0.5-1.0g typical)")

    if len(sensors["flow"]) > 0:
        print("\n✅ Flow sensors AVAILABLE")
        print("   → Can use direct measurements")
    else:
        print("\n⚠️ Flow sensors MISSING")
        print("   → Use Darcy-Weisbach estimation (Phase 3.1)")

    print("\n✅ READY FOR QUICK VALIDATION")
    print("   Next: Generate 14D edge graphs")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect CSV sensor data for Phase 3 validation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/bim_comprehensive.csv"),
        help="Path to CSV file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis"),
        help="Output directory for reports"
    )

    args = parser.parse_args()

    # Inspect
    sensors = inspect_csv_columns(args.input)

    # Check compatibility
    compatibility = check_phase3_compatibility(sensors)

    # Generate reports
    generate_report(sensors, compatibility, args.output)


if __name__ == "__main__":
    main()
