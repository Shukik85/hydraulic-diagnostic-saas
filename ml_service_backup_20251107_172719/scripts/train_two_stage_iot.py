#!/usr/bin/env python3
"""
Two-Stage Production Training Pipeline for Industrial IoT Dataset
Stage 1: Binary anomaly detection (normal vs fault) - High Recall
Stage 2: Multi-class fault classification (pump/valve/cooling) - High Precision
With SMOTE balancing, model versioning, and MLOps hooks
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import structlog
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Add ml_service to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logger = structlog.get_logger()

# Stage 1: Binary Classification Parameters (optimized for recall)
STAGE1_PARAMS = {
    "n_estimators": 800,
    "max_depth": 6,
    "learning_rate": 0.03,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": 8,
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "logloss"
}

# Stage 2: Multi-class Parameters (optimized for precision)
STAGE2_PARAMS = {
    "iterations": 600,
    "depth": 10,
    "learning_rate": 0.03,
    "loss_function": "MultiClass",
    "l2_leaf_reg": 8,
    "bootstrap_type": "Bayesian",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 8,
    "eval_metric": "MultiClass"
}

# Component mapping for actionable insights
FAULT_TYPE_MAPPING = {
    0: "normal",
    1: "pump_fault", 
    2: "valve_fault",
    3: "cooling_fault"
}

COMPONENT_MAPPING = {
    1: ["pump_main", "pump_motor"],
    2: ["valve_main", "valve_control"], 
    3: ["cooler", "heat_exchanger", "cooling_system"]
}


class TwoStageTrainer:
    def __init__(self, csv_path: str = "data/industrial_iot/Industrial_fault_detection.csv"):
        self.csv_path = Path(csv_path)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Create versioned model directory
        self.version = datetime.now().strftime("v%Y%m%d_%H%M")
        self.version_dir = self.models_dir / self.version
        self.version_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        
        logger.info("TwoStageTrainer initialized", 
                   csv_path=str(self.csv_path), 
                   models_dir=str(self.models_dir),
                   version=self.version)

    def load_and_prepare_data(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Загрузить Industrial IoT CSV и подготовить для обучения"""
        logger.info("Loading Industrial IoT dataset", path=str(self.csv_path))
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        logger.info("Dataset loaded", shape=df.shape, columns=list(df.columns))
        
        if "Fault_Type" not in df.columns:
            raise ValueError("Missing 'Fault_Type' column in dataset")
        
        # Separate features and target
        feature_columns = [c for c in df.columns if c != "Fault_Type"]
        X = df[feature_columns].values
        y = df["Fault_Type"].values
        
        logger.info("Data prepared", 
                   samples=X.shape[0],
                   features=X.shape[1], 
                   feature_names=len(feature_columns),
                   fault_distribution=dict(zip(*np.unique(y, return_counts=True))))
        
        return X, y, feature_columns

    def train_stage1_binary(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Stage 1: Binary anomaly detection (normal vs fault)"""
        logger.info("=== STAGE 1: Binary Anomaly Detection ===")
        
        # Convert to binary labels
        y_train_binary = (y_train != 0).astype(int)
        y_val_binary = (y_val != 0).astype(int)
        
        # Calculate scale_pos_weight for imbalanced data
        neg_samples = np.sum(y_train_binary == 0)
        pos_samples = np.sum(y_train_binary == 1)
        scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1.0
        
        logger.info("Binary class distribution", 
                   normal=int(neg_samples),
                   fault=int(pos_samples), 
                   scale_pos_weight=float(scale_pos_weight))
        
        start_time = time.time()
        
        # Train binary classifier
        stage1_params = STAGE1_PARAMS.copy()
        stage1_params["scale_pos_weight"] = scale_pos_weight
        
        model = XGBClassifier(**stage1_params)
        model.fit(X_train, y_train_binary, eval_set=[(X_val, y_val_binary)], verbose=False)
        
        # Validation
        y_pred_binary = model.predict(X_val)
        y_proba_binary = model.predict_proba(X_val)[:, 1]  # P(fault)
        
        accuracy = accuracy_score(y_val_binary, y_pred_binary)
        f1 = f1_score(y_val_binary, y_pred_binary)
        
        training_time = time.time() - start_time
        
        # Find optimal threshold for F1
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            y_pred_thresh = (y_proba_binary >= thresh).astype(int)
            f1_thresh = f1_score(y_val_binary, y_pred_thresh)
            if f1_thresh > best_f1:
                best_f1 = f1_thresh
                best_threshold = thresh
        
        logger.info("Stage 1 validation report:\n" + 
                   classification_report(y_val_binary, y_pred_binary, digits=3))
        logger.info("Optimal threshold", threshold=best_threshold, f1_score=best_f1)
        
        # Save Stage 1 model
        model_path = self.version_dir / "binary_detector_xgb.joblib"
        joblib.dump(model, model_path)
        
        metadata = {
            "stage": 1,
            "name": "binary_detector",
            "model_type": "xgboost",
            "version": self.version,
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "best_f1": float(best_f1),
            "optimal_threshold": float(best_threshold),
            "training_time_seconds": training_time,
            "features_count": int(X_train.shape[1]),
            "training_samples": int(X_train.shape[0]),
            "model_size_mb": float(model_path.stat().st_size / 1024 / 1024),
            "hyperparameters": stage1_params
        }
        
        return metadata, best_threshold

    def train_stage2_multiclass(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray, 
                               use_smote: bool = False) -> dict:
        """Stage 2: Multi-class fault type classification"""
        stage_name = "STAGE 2 (SMOTE)" if use_smote else "STAGE 2 (Plain)"
        logger.info(f"=== {stage_name}: Fault Type Classification ===")
        
        # Filter only fault samples (y != 0)
        fault_mask_train = y_train != 0
        fault_mask_val = y_val != 0
        
        if fault_mask_train.sum() == 0 or fault_mask_val.sum() == 0:
            logger.warning("No fault samples for Stage 2 training")
            return {"error": "No fault samples available"}
        
        X_train_faults = X_train[fault_mask_train]
        y_train_faults = y_train[fault_mask_train] 
        X_val_faults = X_val[fault_mask_val]
        y_val_faults = y_val[fault_mask_val]
        
        logger.info("Stage 2 fault samples",
                   train_faults=X_train_faults.shape[0],
                   val_faults=X_val_faults.shape[0],
                   fault_classes=sorted(np.unique(y_train_faults)))
        
        # Apply SMOTE if requested
        if use_smote:
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y_train_faults)) - 1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_faults, y_train_faults)
                logger.info("SMOTE applied", 
                           original_samples=X_train_faults.shape[0],
                           balanced_samples=X_train_balanced.shape[0])
            except Exception as e:
                logger.warning("SMOTE failed, using original data", error=str(e))
                X_train_balanced, y_train_balanced = X_train_faults, y_train_faults
        else:
            X_train_balanced, y_train_balanced = X_train_faults, y_train_faults
        
        start_time = time.time()
        
        # Compute sample weights for class balancing
        classes = np.unique(y_train_balanced)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_balanced)
        cw_map = {int(c): w for c, w in zip(classes, class_weights)}
        sample_weight = np.vectorize(cw_map.get)(y_train_balanced)
        
        # Train Stage 2 model
        model = CatBoostClassifier(**STAGE2_PARAMS)
        model.fit(X_train_balanced, y_train_balanced, 
                 sample_weight=sample_weight,
                 eval_set=(X_val_faults, y_val_faults))
        
        # Validation
        y_pred_faults = model.predict(X_val_faults)
        accuracy = accuracy_score(y_val_faults, y_pred_faults)
        macro_f1 = f1_score(y_val_faults, y_pred_faults, average='macro')
        
        training_time = time.time() - start_time
        
        logger.info(f"{stage_name} validation report:\n" + 
                   classification_report(y_val_faults, y_pred_faults, digits=3))
        
        # Save Stage 2 model
        model_suffix = "_smote" if use_smote else ""
        model_path = self.version_dir / f"fault_classifier_catboost{model_suffix}.joblib"
        joblib.dump(model, model_path)
        
        metadata = {
            "stage": 2,
            "name": f"fault_classifier{'_smote' if use_smote else ''}",
            "model_type": "catboost",
            "version": self.version,
            "use_smote": use_smote,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "training_time_seconds": training_time,
            "features_count": int(X_train_balanced.shape[1]),
            "training_samples": int(X_train_balanced.shape[0]),
            "original_fault_samples": int(X_train_faults.shape[0]),
            "model_size_mb": float(model_path.stat().st_size / 1024 / 1024),
            "hyperparameters": STAGE2_PARAMS
        }
        
        return metadata

    def train_two_stage_pipeline(self) -> dict:
        """Complete two-stage training pipeline"""
        logger.info("=== STARTING TWO-STAGE IOT MODEL TRAINING ===")
        
        # 1. Load and prepare data
        X, y, feature_names = self.load_and_prepare_data()
        
        if len(X) == 0:
            raise ValueError("No training data available")
        
        # 2. Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info("Train/validation split",
                   train_samples=X_train.shape[0],
                   val_samples=X_val.shape[0],
                   train_fault_ratio=float((y_train != 0).sum() / len(y_train)),
                   val_fault_ratio=float((y_val != 0).sum() / len(y_val)))
        
        # 3. Feature scaling
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        scaler_path = self.version_dir / "feature_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info("Feature scaler saved", path=str(scaler_path))
        
        # 4. Train Stage 1 (Binary)
        stage1_metadata, optimal_threshold = self.train_stage1_binary(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        # 5. Train Stage 2 (Multi-class) - Both variants
        stage2_plain = self.train_stage2_multiclass(
            X_train_scaled, y_train, X_val_scaled, y_val, use_smote=False
        )
        
        stage2_smote = self.train_stage2_multiclass(
            X_train_scaled, y_train, X_val_scaled, y_val, use_smote=True
        )
        
        # 6. Choose best Stage 2 model
        if "error" not in stage2_smote and "error" not in stage2_plain:
            smote_f1 = stage2_smote.get("macro_f1", 0.0)
            plain_f1 = stage2_plain.get("macro_f1", 0.0)
            
            if smote_f1 > plain_f1:
                best_stage2 = "smote"
                best_stage2_metadata = stage2_smote
                logger.info("SMOTE variant selected", smote_f1=smote_f1, plain_f1=plain_f1)
            else:
                best_stage2 = "plain" 
                best_stage2_metadata = stage2_plain
                logger.info("Plain variant selected", smote_f1=smote_f1, plain_f1=plain_f1)
        else:
            best_stage2 = "plain"
            best_stage2_metadata = stage2_plain if "error" not in stage2_plain else stage2_smote
        
        # 7. Create symlinks to current best models in models/ root
        try:
            best_binary = self.version_dir / "binary_detector_xgb.joblib"
            best_multiclass = self.version_dir / f"fault_classifier_catboost{'_smote' if best_stage2 == 'smote' else ''}.joblib"
            
            # Copy to root models/ for API loading
            import shutil
            shutil.copy2(best_binary, self.models_dir / "catboost_model.joblib")  # compatibility
            shutil.copy2(best_multiclass, self.models_dir / "xgboost_model.joblib")  # compatibility
            shutil.copy2(scaler_path, self.models_dir / "feature_scaler.joblib")
            
        except Exception as e:
            logger.warning("Failed to copy models to root", error=str(e))
        
        # 8. Save complete training summary
        training_summary = {
            "training_timestamp": time.time(),
            "version": self.version,
            "dataset_path": str(self.csv_path),
            "total_samples": int(X.shape[0]),
            "features_count": int(X.shape[1]),
            "feature_names": feature_names,
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "fault_type_mapping": FAULT_TYPE_MAPPING,
            "component_mapping": COMPONENT_MAPPING,
            "optimal_binary_threshold": optimal_threshold,
            "best_stage2_variant": best_stage2,
            "stage1": stage1_metadata,
            "stage2_plain": stage2_plain,
            "stage2_smote": stage2_smote,
            "best_stage2": best_stage2_metadata
        }
        
        summary_path = self.version_dir / "two_stage_summary.json"
        with summary_path.open("w") as f:
            json.dump(training_summary, f, indent=2)
        
        # Also save to root for API
        root_summary_path = self.models_dir / "training_summary.json"
        with root_summary_path.open("w") as f:
            json.dump(training_summary, f, indent=2)
        
        logger.info("=== TWO-STAGE TRAINING COMPLETED ===",
                   version=self.version,
                   stage1_f1=stage1_metadata.get("best_f1", 0.0),
                   stage2_macro_f1=best_stage2_metadata.get("macro_f1", 0.0),
                   best_variant=best_stage2,
                   summary_path=str(summary_path))
        
        return training_summary


def main():
    """Main training function"""
    trainer = TwoStageTrainer()
    
    try:
        results = trainer.train_two_stage_pipeline()
        
        print("\n🎉 Two-Stage Training completed successfully!")
        print(f"📦 Version: {results['version']}")
        print(f"📊 Dataset: {results['total_samples']} samples, {results['features_count']} features")
        
        stage1 = results['stage1']
        stage2 = results['best_stage2']
        
        print(f"\n🔥 Stage 1 (Binary Anomaly Detection):")
        print(f"  Model: XGBoost")
        print(f"  Accuracy: {stage1['accuracy']:.3f}")
        print(f"  F1 Score: {stage1['f1_score']:.3f}")
        print(f"  Best F1: {stage1['best_f1']:.3f} (threshold: {stage1['optimal_threshold']:.3f})")
        
        if "error" not in stage2:
            print(f"\n🎯 Stage 2 (Fault Classification):")
            print(f"  Model: CatBoost ({results['best_stage2_variant']})")
            print(f"  Accuracy: {stage2['accuracy']:.3f}")
            print(f"  Macro F1: {stage2['macro_f1']:.3f}")
            print(f"  Training samples: {stage2['training_samples']} ({stage2['original_fault_samples']} original)")
        
        print(f"\n📁 Models saved to: models/{results['version']}/")
        print(f"📄 Summary: models/{results['version']}/two_stage_summary.json")
        
        print(f"\n🚀 Ready to restart API with two-stage models!")
        print(f"Next steps:")
        print(f"  1. python main.py")
        print(f"  2. python scripts/push_to_api.py --cycles 10 --api http://localhost:8001")
        
        return 0
        
    except Exception as e:
        logger.error("Two-stage training failed", error=str(e))
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())