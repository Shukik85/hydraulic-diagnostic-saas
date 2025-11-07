#!/usr/bin/env python3
"""
Train Production Models on Industrial IoT Dataset (Single-Stage for Ensemble)
Saves models with strict filenames expected by EnsembleLoader:
- models/catboost_model.joblib      (CatBoostClassifier)
- models/xgboost_model.joblib       (XGBClassifier)
- models/random_forest_model.joblib (RandomForestClassifier)
- models/feature_scaler.joblib      (StandardScaler)
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# Add ml_service to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logger = structlog.get_logger()

# Production hyperparameters tuned for your hardware (i5-10400F, 16GB)
PRODUCTION_PARAMS = {
    "catboost": {
        "iterations": 400,
        "depth": 8,
        "learning_rate": 0.03,
        "loss_function": "MultiClass",
        "l2_leaf_reg": 6,
        "bootstrap_type": "Bayesian",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 8,
    },
    "xgboost": {
        "n_estimators": 800,
        "max_depth": 6,
        "learning_rate": 0.03,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 8,
        "objective": "multi:softprob",
        "tree_method": "hist",
    },
    "random_forest": {
        "n_estimators": 800,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": 8,
        "class_weight": "balanced",
    },
}

DATASET_PATH = Path("data/industrial_iot/Industrial_fault_detection.csv")
MODELS_DIR = Path("models")


def load_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    if "Fault_Type" not in df.columns:
        raise ValueError("Missing 'Fault_Type' column in dataset")
    feature_columns = [c for c in df.columns if c != "Fault_Type"]
    X = df[feature_columns].values
    y = df["Fault_Type"].values
    return X, y, feature_columns


def save_scaler(scaler: StandardScaler):
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / "feature_scaler.joblib"
    joblib.dump(scaler, path)
    logger.info("Feature scaler saved", path=str(path))


def train_and_save_models():
    logger.info("=== SINGLE-STAGE TRAINING FOR ENSEMBLE ===")
    X, y, feature_names = load_data()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    save_scaler(scaler)

    results = {}

    # CatBoost
    try:
        logger.info("Training CatBoost (for Ensemble)")
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        cw_map = {int(c): w for c, w in zip(classes, class_weights)}
        sample_weight = np.vectorize(cw_map.get)(y_train)

        cat = CatBoostClassifier(**PRODUCTION_PARAMS["catboost"])
        cat.fit(X_train_s, y_train, sample_weight=sample_weight, eval_set=(X_val_s, y_val))
        y_pred = cat.predict(X_val_s)
        acc = accuracy_score(y_val, y_pred)
        logger.info("CatBoost validation report\n" + classification_report(y_val, y_pred, digits=3))
        path = MODELS_DIR / "catboost_model.joblib"
        joblib.dump(cat, path)
        results["catboost"] = {"accuracy": float(acc), "path": str(path)}
    except Exception as e:
        logger.error("CatBoost training failed", error=str(e))
        results["catboost"] = {"error": str(e)}

    # XGBoost
    try:
        logger.info("Training XGBoost (for Ensemble)")
        xgb = XGBClassifier(**PRODUCTION_PARAMS["xgboost"])
        xgb.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
        y_pred = xgb.predict(X_val_s)
        acc = accuracy_score(y_val, y_pred)
        logger.info("XGBoost validation report\n" + classification_report(y_val, y_pred, digits=3))
        path = MODELS_DIR / "xgboost_model.joblib"
        joblib.dump(xgb, path)
        results["xgboost"] = {"accuracy": float(acc), "path": str(path)}
    except Exception as e:
        logger.error("XGBoost training failed", error=str(e))
        results["xgboost"] = {"error": str(e)}

    # RandomForest
    try:
        logger.info("Training RandomForest (for Ensemble)")
        rf = RandomForestClassifier(**PRODUCTION_PARAMS["random_forest"])
        rf.fit(X_train_s, y_train)
        y_pred = rf.predict(X_val_s)
        acc = accuracy_score(y_val, y_pred)
        logger.info("RandomForest validation report\n" + classification_report(y_val, y_pred, digits=3))
        path = MODELS_DIR / "random_forest_model.joblib"
        joblib.dump(rf, path)
        results["random_forest"] = {"accuracy": float(acc), "path": str(path)}
    except Exception as e:
        logger.error("RandomForest training failed", error=str(e))
        results["random_forest"] = {"error": str(e)}

    # Save summary
    summary = {
        "timestamp": time.time(),
        "features_count": int(X.shape[1]),
        "feature_names": feature_names,
        "results": results,
    }
    with (MODELS_DIR / "training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("=== ENSEMBLE TRAINING COMPLETED ===", results=results)


if __name__ == "__main__":
    sys.exit(train_and_save_models() or 0)
