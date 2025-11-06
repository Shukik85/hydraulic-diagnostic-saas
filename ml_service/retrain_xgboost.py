#!/usr/bin/env python3
"""
Quick XGBoost Retrain - добавить недостающую модель
"""
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import structlog

sys.path.insert(0, str(Path(__file__).parent))
from data.uci_loader import load_uci_hydraulic_data

logger = structlog.get_logger()
structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

def retrain_xgboost():
    print("🚀 Quick XGBoost retrain on REAL data...")
    
    # Load same data
    data = load_uci_hydraulic_data(filename="Industrial_fault_detection.csv")
    X_train, X_val = data['X_train'], data['X_val']
    y_train, y_val = data['y_train'], data['y_val']
    
    # Simplified param grid (faster)
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [6, 8],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.9, 1.0]
    }
    
    # Base model WITHOUT early_stopping_rounds
    xgb_base = xgb.XGBClassifier(
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )
    
    print("   🔍 GridSearch without early stopping...")
    grid_search = GridSearchCV(
        xgb_base, param_grid, cv=3, scoring='roc_auc',
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Final model with best params + early stopping
    best_params = grid_search.best_params_
    final_model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )
    
    print("   🎯 Training final model with early stopping...")
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Metrics
    y_val_pred_proba = final_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred_proba)
    
    print(f"   ✅ XGBoost CV AUC: {grid_search.best_score_:.4f}")
    print(f"   ✅ XGBoost Val AUC: {auc:.4f}")
    
    # Save model
    save_data = {
        "model": final_model,
        "training_metrics": {
            "best_params": best_params,
            "best_cv_score": grid_search.best_score_,
            "validation_auc": auc,
            "data_source": "REAL_UCI_HYDRAULIC_DATA"
        },
        "model_version": "1.0.0-production-REAL",
        "is_mock_model": False
    }
    
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "xgboost_model.joblib"
    joblib.dump(save_data, model_path)
    
    print(f"   💾 Saved XGBoost to {model_path}")
    print(f"   🎉 XGBoost REAL model ready!")
    
    return auc

if __name__ == "__main__":
    retrain_xgboost()
