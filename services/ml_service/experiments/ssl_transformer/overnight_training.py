"""
🌙 OVERNIGHT FULL SYSTEM TRAINING PIPELINE
Обучение ВСЕХ компонентов на полных BIM данных
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
log_file = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

logging.info("=" * 60)
logging.info("🌙 OVERNIGHT TRAINING PIPELINE STARTED")
logging.info("=" * 60)


# ==== ВСТРОЕННЫЙ BIM СИМУЛЯТОР ====
class QuickBIMSimulator:
    """Быстрый симулятор полной гидравлической системы"""

    def generate_comprehensive_dataset(self, n_samples=100000, fault_rate=0.6):
        logging.info(f"Generating {n_samples:,} synthetic samples...")

        np.random.seed(42)
        data = {}

        # Time
        data["time_s"] = np.linspace(0, n_samples / 10, n_samples)

        # === PUMP ===
        pump_normal = np.random.randint(0, 2, n_samples) > fault_rate
        data["pump_pressure_outlet"] = np.where(
            pump_normal,
            np.random.uniform(170, 190, n_samples),  # Normal
            np.random.uniform(80, 160, n_samples),  # Fault: low pressure
        )
        data["pump_speed_rpm"] = np.where(
            pump_normal, np.random.uniform(1750, 1850, n_samples), np.random.uniform(1400, 1700, n_samples)
        )
        data["pump_temperature"] = np.where(
            pump_normal,
            np.random.uniform(55, 70, n_samples),
            np.random.uniform(75, 95, n_samples),  # Fault: overheating
        )
        data["pump_vibration"] = np.where(
            pump_normal, np.random.uniform(1.5, 2.5, n_samples), np.random.uniform(3.5, 6.0, n_samples)
        )
        data["pump_power"] = data["pump_pressure_outlet"] * data["pump_speed_rpm"] / 100
        data["fault_pump"] = (~pump_normal).astype(int)

        # === CYLINDERS (Boom, Stick, Bucket) ===
        for cyl_name in ["boom", "stick", "bucket"]:
            cyl_normal = np.random.randint(0, 2, n_samples) > fault_rate

            data[f"{cyl_name}_pressure_extend"] = np.where(
                cyl_normal, np.random.uniform(150, 180, n_samples), np.random.uniform(80, 140, n_samples)
            )
            data[f"{cyl_name}_pressure_retract"] = np.where(
                cyl_normal, np.random.uniform(50, 70, n_samples), np.random.uniform(20, 50, n_samples)
            )
            data[f"{cyl_name}_position"] = np.cumsum(np.random.randn(n_samples) * 0.01) % 2.5
            data[f"{cyl_name}_velocity"] = np.gradient(data[f"{cyl_name}_position"])
            data[f"{cyl_name}_pressure_diff"] = (
                data[f"{cyl_name}_pressure_extend"] - data[f"{cyl_name}_pressure_retract"]
            )
            data[f"fault_cylinder_{cyl_name}"] = (~cyl_normal).astype(int)

        # === MOTORS (Swing, Left, Right) ===
        for motor_name in ["swing", "left", "right"]:
            motor_normal = np.random.randint(0, 2, n_samples) > fault_rate

            data[f"{motor_name}_speed_rpm"] = np.where(
                motor_normal, np.random.uniform(450, 550, n_samples), np.random.uniform(200, 400, n_samples)
            )
            data[f"{motor_name}_torque"] = np.where(
                motor_normal, np.random.uniform(100, 150, n_samples), np.random.uniform(50, 90, n_samples)
            )
            data[f"{motor_name}_temperature"] = np.where(
                motor_normal, np.random.uniform(60, 75, n_samples), np.random.uniform(80, 100, n_samples)
            )
            data[f"{motor_name}_pressure_inlet"] = data["pump_pressure_outlet"] * np.random.uniform(
                0.9, 0.95, n_samples
            )
            data[f"{motor_name}_vibration"] = np.where(
                motor_normal, np.random.uniform(2.0, 3.5, n_samples), np.random.uniform(4.5, 7.0, n_samples)
            )
            data[f"fault_motor_{motor_name}"] = (~motor_normal).astype(int)

        df = pd.DataFrame(data)
        logging.info(f"✅ Generated {len(df):,} samples")
        return df


# 1. GENERATE BIM DATA
logging.info("\n📊 Step 1: Generating comprehensive BIM data...")

try:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    bim_file = data_dir / "bim_comprehensive.csv"

    if bim_file.exists():
        logging.info(f"Loading existing BIM data from {bim_file}...")
        df_full = pd.read_csv(bim_file)
    else:
        simulator = QuickBIMSimulator()
        df_full = simulator.generate_comprehensive_dataset(n_samples=100000, fault_rate=0.64)
        df_full.to_csv(bim_file, index=False)

    logging.info(f"Dataset: {df_full.shape}")

except Exception as e:
    logging.error(f"❌ Failed to generate data: {e}")
    raise

# 2. EXTRACT COMPONENT-SPECIFIC DATA
logging.info("\n📦 Step 2: Extracting component datasets...")

component_configs = {
    "cylinder_boom": {
        "features": ["pressure_extend", "pressure_retract", "position", "velocity", "pressure_diff"],
        "target": "fault_cylinder_boom",
        "prefix": "boom_",
    },
    "cylinder_stick": {
        "features": ["pressure_extend", "pressure_retract", "position", "velocity", "pressure_diff"],
        "target": "fault_cylinder_stick",
        "prefix": "stick_",
    },
    "cylinder_bucket": {
        "features": ["pressure_extend", "pressure_retract", "position", "velocity", "pressure_diff"],
        "target": "fault_cylinder_bucket",
        "prefix": "bucket_",
    },
    "pump": {
        "features": ["pressure_outlet", "speed_rpm", "temperature", "vibration", "power"],
        "target": "fault_pump",
        "prefix": "pump_",
    },
    "motor_swing": {
        "features": ["speed_rpm", "torque", "temperature", "pressure_inlet", "vibration"],
        "target": "fault_motor_swing",
        "prefix": "swing_",
    },
    "motor_left": {
        "features": ["speed_rpm", "torque", "temperature", "pressure_inlet", "vibration"],
        "target": "fault_motor_left",
        "prefix": "left_",
    },
    "motor_right": {
        "features": ["speed_rpm", "torque", "temperature", "pressure_inlet", "vibration"],
        "target": "fault_motor_right",
        "prefix": "right_",
    },
}

extracted_components = {}

for comp_name, config in component_configs.items():
    try:
        feature_cols = [config["prefix"] + f for f in config["features"]]

        # Check columns
        available_cols = [c for c in feature_cols if c in df_full.columns]

        if len(available_cols) < 3:
            logging.warning(f"⚠️  Skipping {comp_name}: insufficient columns")
            continue

        # Extract
        df_comp = df_full[available_cols + [config["target"]]].copy()

        # Rename
        rename_map = {old: old.replace(config["prefix"], "") for old in available_cols}
        rename_map[config["target"]] = "fault_any"
        df_comp = df_comp.rename(columns=rename_map)

        # Clean
        df_comp = df_comp.dropna()

        # Remove zero-variance
        for col in df_comp.columns:
            if df_comp[col].dtype in [np.float64, np.float32, np.int64]:
                if df_comp[col].std() == 0:
                    df_comp = df_comp.drop(columns=[col])

        # Save
        output_file = data_dir / f"component_{comp_name}.csv"
        df_comp.to_csv(output_file, index=False)

        extracted_components[comp_name] = {
            "file": str(output_file),
            "samples": len(df_comp),
            "features": list(df_comp.columns)[:-1],
            "fault_rate": df_comp["fault_any"].mean(),
        }

        logging.info(f"✅ {comp_name}: {len(df_comp):,} samples, {df_comp['fault_any'].mean() * 100:.1f}% faults")

    except Exception as e:
        logging.error(f"❌ {comp_name}: {e}")

# 3. TRAIN ALL MODELS
logging.info("\n🚀 Step 3: Training all component models...")
logging.info("This will take several hours...")

from config.equipment_schema import create_cat336_config
from models.component_models import CylinderModel, PumpModel
from train_physics_informed import train_component_model

config = create_cat336_config()
results = {}


def get_model_class(comp_name):
    if "cylinder" in comp_name:
        return CylinderModel
    elif "pump" in comp_name or "motor" in comp_name:
        return PumpModel
    return CylinderModel


total_start = time.time()

for comp_name, comp_data in extracted_components.items():
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Training {comp_name.upper()}")
    logging.info(f"{'=' * 60}")

    comp_start = time.time()

    try:
        comp_type = "cylinder" if "cylinder" in comp_name else "pump"

        model, acc, f1 = train_component_model(
            model_class=get_model_class(comp_name),
            data_path=comp_data["file"],
            model_name=f"{comp_name.capitalize()}Model",
            feature_cols=comp_data["features"],
            equipment_config=config,
            component_type=comp_type,
            epochs=100,
            batch_size=128,
            lr=0.0001,
        )

        comp_time = time.time() - comp_start

        results[comp_name] = {
            "status": "success",
            "accuracy": float(acc),
            "f1_score": float(f1),
            "training_time_s": comp_time,
            "samples": comp_data["samples"],
        }

        logging.info(f"✅ Complete: Acc={acc * 100:.2f}%, F1={f1 * 100:.2f}%, Time={comp_time / 60:.1f}min")

    except Exception as e:
        results[comp_name] = {"status": "failed", "error": str(e)}
        logging.error(f"❌ Failed: {e}")

total_time = time.time() - total_start

# 4. SAVE RESULTS
results_dir = Path("training_results")
results_dir.mkdir(exist_ok=True)

summary = {
    "timestamp": datetime.now().isoformat(),
    "total_training_time_h": total_time / 3600,
    "components_trained": len([r for r in results.values() if r["status"] == "success"]),
    "results": results,
}

results_file = results_dir / f"overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_file, "w") as f:
    json.dump(summary, f, indent=2)

# 5. SUMMARY
logging.info("\n" + "=" * 60)
logging.info("🎉 OVERNIGHT TRAINING COMPLETE!")
logging.info("=" * 60)
logging.info(f"Total time: {total_time / 3600:.2f} hours")
logging.info(f"Success: {summary['components_trained']}")

for comp, res in results.items():
    if res["status"] == "success":
        logging.info(f"✅ {comp:20s} - Acc: {res['accuracy'] * 100:5.2f}%, F1: {res['f1_score'] * 100:5.2f}%")
    else:
        logging.info(f"❌ {comp:20s} - {res.get('error', 'Failed')}")

logging.info(f"\n📁 Results: {results_file}")
logging.info("🎯 Ready for production!")
