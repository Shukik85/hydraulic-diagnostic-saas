#!/usr/bin/env python3
"""
GPU-Accelerated REAL Production Model Training
- CatBoost (GPU)
- XGBoost (gpu_hist)
- RandomForest (cuML zero-code-change acceleration)

Requires:
  pip install --upgrade pip setuptools wheel
  pip install --upgrade catboost xgboost
  pip install cuml-cu12 --extra-index-url https://pypi.nvidia.com
"""
import os
import sys
from pathlib import Path
import time
import structlog
from rich.console import Console
from rich.panel import Panel

# Enable cuML zero-code-change acceleration for sklearn
try:
    import cuml.experimental.accel  # noqa: F401
    CUMLOK = True
except Exception:
    CUMLOK = False

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

# Local imports
sys.path.append(str(Path(__file__).parent))
from data.uci_loader import load_uci_hydraulic_data

structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
log = structlog.get_logger()
console = Console()


def train_catboost_gpu(X_train, y_train, X_val, y_val):
    console.print("\n🐱 CatBoost (GPU)...")
    base = CatBoostClassifier(
        random_seed=42,
        logging_level='Silent',
        allow_writing_files=False,
        early_stopping_rounds=50,
        task_type='GPU',
        devices='0'
    )
    grid = {
        'iterations': [300, 500],
        'depth': [6, 8],
        'learning_rate': [0.05, 0.1]
    }
    gs = GridSearchCV(base, grid, cv=3, scoring='roc_auc', n_jobs=1, verbose=0)
    t0 = time.time()
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    yv = model.predict(X_val)
    pv = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pv)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, yv, average='binary')
    console.print(f"   ⏱️ {time.time()-t0:.1f}s  AUC={auc:.4f}  F1={f1:.4f}")
    return {'model': model, 'training_metrics': {'best_params': gs.best_params_, 'best_cv_score': gs.best_score_, 'validation_auc': auc, 'validation_f1': f1}}


def train_xgb_gpu(X_train, y_train, X_val, y_val):
    console.print("\n🚀 XGBoost (gpu_hist)...")
    base = xgb.XGBClassifier(
        random_state=42,
        eval_metric='auc',
        verbosity=0,
        tree_method='gpu_hist',
        gpu_id=0
    )
    grid = {
        'n_estimators': [300, 500],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.9],
        'colsample_bytree': [0.9]
    }
    gs = GridSearchCV(base, grid, cv=3, scoring='roc_auc', n_jobs=1, verbose=0)
    t0 = time.time()
    gs.fit(X_train, y_train)
    best_params = gs.best_params_
    final = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric='auc',
        verbosity=0,
        tree_method='gpu_hist',
        gpu_id=0
    )
    try:
        final.set_params(early_stopping_rounds=50)
    except Exception:
        pass
    final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    yv = final.predict(X_val)
    pv = final.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pv)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, yv, average='binary')
    console.print(f"   ⏱️ {time.time()-t0:.1f}s  AUC={auc:.4f}  F1={f1:.4f}")
    return {'model': final, 'training_metrics': {'best_params': best_params, 'best_cv_score': gs.best_score_, 'validation_auc': auc, 'validation_f1': f1}}


def train_rf_gpu_or_cpu(X_train, y_train, X_val, y_val):
    console.print("\n🌲 RandomForest (cuML accel if available)...")
    grid = {
        'n_estimators': [200, 300],
        'max_depth': [15, None],
        'min_samples_split': [5, 10],
        'max_features': ['sqrt', 'log2']
    }
    # cuML accel hooks sklearn under the hood, so we keep sklearn API
    base = RandomForestClassifier(random_state=42, n_jobs=4, oob_score=True, class_weight='balanced')
    gs = GridSearchCV(base, grid, cv=3, scoring='roc_auc', n_jobs=1, verbose=0)
    t0 = time.time()
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    yv = model.predict(X_val)
    pv = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pv)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_val, yv)
    console.print(f"   ⏱️ {time.time()-t0:.1f}s  AUC={auc:.4f}  F1={f1:.4f}  (cuML={'ON' if CUMLOK else 'OFF'})")
    return {'model': model, 'training_metrics': {'best_params': gs.best_params_, 'best_cv_score': gs.best_score_, 'validation_auc': auc, 'validation_f1': f1, 'oob_score': getattr(model, 'oob_score_', None)}}


def main():
    console.print(Panel.fit("🔥 GPU Ensemble Training on REAL UCI Hydraulic Data", style="bold green"))
    data = load_uci_hydraulic_data(filename="Industrial_fault_detection.csv", window_minutes=5)
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    models = {}
    models['catboost'] = train_catboost_gpu(X_train, y_train, X_val, y_val)
    models['xgboost'] = train_xgb_gpu(X_train, y_train, X_val, y_val)
    models['random_forest'] = train_rf_gpu_or_cpu(X_train, y_train, X_val, y_val)

    # Evaluate on test
    console.print("\n📊 Test evaluation:")
    for name, bundle in models.items():
        mdl = bundle['model']
        if name == 'random_forest':
            ypred = mdl.predict(X_test)
            yproba = mdl.predict_proba(X_test)[:, 1]
        else:
            ypred = mdl.predict(X_test)
            yproba = mdl.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, yproba)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, ypred, average='binary')
        console.print(f"   {name:12s}  AUC={auc:.4f}  F1={f1:.4f}")
        bundle['test_metrics'] = {'test_auc': auc, 'test_f1': f1}

    # Save models
    models_dir = Path('./models')
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, bundle in models.items():
        joblib.dump(bundle, models_dir / f"{name}_model_gpu.joblib")
    console.print("\n💾 Saved GPU models to ./models/*_model_gpu.joblib")


if __name__ == '__main__':
    sys.exit(main())
