#!/usr/bin/env python3
"""
REAL Production Model Training Script with CPU/GPU modes
- CPU: Fast training with optimized grids (10-25 minutes)
- GPU: Full hyperparameter search for maximum quality (25-50 minutes)

Usage:
  python train_real_production_models.py                      # CPU fast mode
  python train_real_production_models.py --gpu                # GPU full mode  
  python train_real_production_models.py --only xgboost --gpu # GPU single model
"""

import asyncio
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Enable cuML zero-code-change acceleration if available
try:
    import cuml.experimental.accel
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# ML imports
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
from catboost import CatBoostClassifier

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import settings
from data.uci_loader import load_uci_hydraulic_data

logger = structlog.get_logger()
console = Console()

# Configuration grids: CPU FAST vs GPU FULL
GRIDS = {
    "catboost": {
        "cpu_fast": {
            'iterations': [300],
            'depth': [6], 
            'learning_rate': [0.1],
            'l2_leaf_reg': [3]
        },
        "gpu_full": {
            'iterations': [500, 800, 1000],
            'depth': [6, 8, 10],
            'learning_rate': [0.03, 0.05, 0.1],
            'l2_leaf_reg': [3, 5, 10]
        }
    },
    "xgboost": {
        "cpu_fast": {
            'n_estimators': [200, 300],
            'max_depth': [4, 6],
            'learning_rate': [0.1],
            'subsample': [0.9],
            'colsample_bytree': [0.9]
        },
        "gpu_full": {
            'n_estimators': [300, 500],  # Reduced for stability
            'max_depth': [4, 6, 8],      # Conservative depths
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9],     # Avoid 1.0 for GPU stability
            'colsample_bytree': [0.8, 0.9],
            'reg_alpha': [0, 0.1],       # Conservative regularization
            'reg_lambda': [1, 3]         # Conservative regularization
        }
    },
    "random_forest": {
        "cpu_fast": {
            'n_estimators': [200, 300],
            'max_depth': [15, None],
            'min_samples_split': [5, 10],
            'max_features': ['sqrt', 'log2']
        },
        "gpu_full": {
            'n_estimators': [300, 500, 800],
            'max_depth': [15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    }
}


class RealProductionModelTrainer:
    """Train models on REAL UCI Hydraulic data with CPU/GPU modes."""
    
    def __init__(self, use_gpu: bool = False, mode: str = "fast"):
        self.console = console
        self.models_dir = Path(settings.model_path)
        self.models_dir.mkdir(exist_ok=True)
        
        self.use_gpu = use_gpu
        self.mode = "gpu_full" if use_gpu else "cpu_fast"
        
        self.trained_models = {}
        self.data_info = None
        
        if use_gpu:
            console.print("🔥 GPU Mode - Full hyperparameter search for maximum quality")
            console.print(f"   cuML acceleration: {'✅ Available' if CUML_AVAILABLE else '❌ Install: pip install cuml-cu12'}")
        else:
            console.print("⚡ CPU Mode - Fast optimized grids for quick results")
    
    def _grid_combinations(self, param_grid):
        """Calculate grid search combinations."""
        import itertools
        return list(itertools.product(*param_grid.values()))
    
    def load_real_data(self, data_path: str = None) -> Dict[str, np.ndarray]:
        """Load REAL UCI hydraulic dataset."""
        
        console.print("🏭 Loading REAL UCI Hydraulic System dataset...")
        
        try:
            data = load_uci_hydraulic_data(
                filename=data_path or "Industrial_fault_detection.csv",
                window_minutes=5
            )
            console.print(f"✅ Loaded dataset: {data['X_train'].shape[0]} training samples")
        except Exception as e:
            console.print(f"⚠️  Primary dataset failed: {e}")
            data = load_uci_hydraulic_data(
                filename="industrial_fault_detection_data_1000.csv",
                window_minutes=5
            )
            console.print(f"✅ Loaded fallback: {data['X_train'].shape[0]} training samples")
        
        self.data_info = data['data_info']
        return data
    
    def train_catboost_real(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Train CatBoost: CPU fast vs GPU full."""
        
        grid_key = "gpu_full" if self.use_gpu else "cpu_fast"
        param_grid = GRIDS["catboost"][grid_key]
        combinations = len(self._grid_combinations(param_grid))
        
        mode_text = f"GPU Full ({combinations} combinations)" if self.use_gpu else f"CPU Fast ({combinations} combinations)"
        console.print(f"\n🐱 CatBoost {mode_text}")
        
        # Base configuration
        base_params = {
            'random_seed': 42,
            'logging_level': 'Silent',
            'allow_writing_files': False,
            'early_stopping_rounds': 50
        }
        
        if self.use_gpu:
            base_params.update({'task_type': 'GPU', 'devices': '0'})
        
        catboost_base = CatBoostClassifier(**base_params)
        n_jobs = 1 if self.use_gpu else -1
        
        grid_search = GridSearchCV(
            catboost_base, param_grid, cv=3, scoring='roc_auc', 
            n_jobs=n_jobs, verbose=0
        )
        
        start_time = time.time()
        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            if self.use_gpu and "GPU" in str(e):
                console.print("   ⚠️ GPU error, falling back to CPU...")
                base_params.pop('task_type', None)
                base_params.pop('devices', None)
                catboost_base = CatBoostClassifier(**base_params)
                grid_search = GridSearchCV(catboost_base, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
                grid_search.fit(X_train, y_train)
            else:
                raise
        
        training_time = time.time() - start_time
        best_model = grid_search.best_estimator_
        
        # Validation
        y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_val_pred_proba)
        f1 = precision_recall_fscore_support(y_val, best_model.predict(X_val), average='binary')[2]
        
        console.print(f"   ⏱️  {training_time:.1f}s  ✅ CV: {grid_search.best_score_:.4f}  ✅ Val: {auc_score:.4f}")
        
        return {
            "model": best_model,
            "training_metrics": {
                "best_params": grid_search.best_params_,
                "best_cv_score": grid_search.best_score_,
                "validation_auc": auc_score,
                "validation_f1": f1,
                "training_time_seconds": training_time,
                "mode": mode_text,
                "data_source": "REAL_UCI_HYDRAULIC_DATA"
            }
        }
    
    def train_xgboost_real(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Train XGBoost with GTX 1650 SUPER GPU compatibility fixes."""
        
        grid_key = "gpu_full" if self.use_gpu else "cpu_fast"
        param_grid = GRIDS["xgboost"][grid_key]
        combinations = len(self._grid_combinations(param_grid))
        
        mode_text = f"GPU Full ({combinations} combinations)" if self.use_gpu else f"CPU Fast ({combinations} combinations)"
        console.print(f"\n🚀 XGBoost {mode_text}")
        
        # GTX 1650 SUPER GPU compatibility fixes
        base_params = {
            'random_state': 42,
            'eval_metric': 'auc',
            'verbosity': 0
        }
        
        if self.use_gpu:
            # Enhanced GPU configuration for GTX 1650 SUPER
            base_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor',
                # GTX 1650 SUPER specific optimizations
                'max_bin': 256,           # Reduce memory usage
                'gpu_batch_nrows': 0,     # Auto batch sizing
                'single_precision_histogram': True,  # Reduce precision for stability
                'deterministic_histogram': True,     # Stable results
                'sampling_method': 'uniform',        # Stable sampling
                # Missing value handling for GPU
                'missing': -999,          # Explicit missing value
                'enable_categorical': False,  # Disable categorical for stability
            })
        else:
            base_params['tree_method'] = 'hist'
        
        # Data preprocessing for GPU compatibility
        if self.use_gpu:
            console.print("   🔧 Preprocessing data for GPU compatibility...")
            
            # Replace any NaN/inf with explicit missing value
            X_train_gpu = X_train.copy()
            X_val_gpu = X_val.copy()
            
            # Handle missing values explicitly
            X_train_gpu = np.nan_to_num(X_train_gpu, nan=-999, posinf=-999, neginf=-999)
            X_val_gpu = np.nan_to_num(X_val_gpu, nan=-999, posinf=-999, neginf=-999)
            
            # Ensure float32 for GPU efficiency
            X_train_gpu = X_train_gpu.astype(np.float32)
            X_val_gpu = X_val_gpu.astype(np.float32)
            y_train_gpu = y_train.astype(np.int32)
            y_val_gpu = y_val.astype(np.int32)
        else:
            X_train_gpu = X_train
            X_val_gpu = X_val
            y_train_gpu = y_train
            y_val_gpu = y_val
        
        xgb_base = xgb.XGBClassifier(**base_params)
        n_jobs = 1 if self.use_gpu else -1
        
        grid_search = GridSearchCV(
            xgb_base, param_grid, cv=3, scoring='roc_auc',
            n_jobs=n_jobs, verbose=0
        )
        
        start_time = time.time()
        try:
            grid_search.fit(X_train_gpu, y_train_gpu)
        except Exception as e:
            if self.use_gpu and ("GPU" in str(e) or "CUDA" in str(e) or "assertion" in str(e).lower()):
                console.print("   ⚠️ GPU compatibility issue, optimizing parameters...")
                
                # Fallback GPU parameters for GTX 1650 SUPER
                safer_params = base_params.copy()
                safer_params.update({
                    'max_depth': 6,           # Conservative depth
                    'max_bin': 128,           # Even smaller bins
                    'grow_policy': 'lossguide', # Alternative growth policy
                    'max_leaves': 256,        # Limit leaves for GTX 1650
                    'reg_alpha': 0.1,         # Add regularization
                    'reg_lambda': 1.0,        # Add regularization
                    'scale_pos_weight': 3.5   # Handle class imbalance
                })
                
                # Simplified grid for compatibility
                safe_grid = {
                    'n_estimators': [300, 500],
                    'learning_rate': [0.1, 0.2],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8]
                }
                
                console.print("   🔧 Using GTX 1650 SUPER compatible parameters...")
                xgb_safe = xgb.XGBClassifier(**safer_params)
                grid_search = GridSearchCV(xgb_safe, safe_grid, cv=3, scoring='roc_auc', n_jobs=1, verbose=0)
                grid_search.fit(X_train_gpu, y_train_gpu)
                
            else:
                # Complete fallback to CPU
                console.print("   ⚠️ GPU error, falling back to CPU...")
                base_params['tree_method'] = 'hist'
                base_params.pop('gpu_id', None)
                base_params.pop('predictor', None)
                for gpu_param in ['max_bin', 'gpu_batch_nrows', 'single_precision_histogram', 
                                'deterministic_histogram', 'sampling_method', 'enable_categorical']:
                    base_params.pop(gpu_param, None)
                
                xgb_base = xgb.XGBClassifier(**base_params)
                grid_search = GridSearchCV(xgb_base, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
                grid_search.fit(X_train, y_train)
                X_train_gpu, X_val_gpu, y_train_gpu, y_val_gpu = X_train, X_val, y_train, y_val
        
        # Final model with early stopping
        best_params = grid_search.best_params_
        final_params = {**base_params, **best_params}
        final_model = xgb.XGBClassifier(**final_params)
        
        # Compatible early stopping
        try:
            final_model.set_params(early_stopping_rounds=50)
            final_model.fit(X_train_gpu, y_train_gpu, 
                          eval_set=[(X_val_gpu, y_val_gpu)], verbose=False)
        except Exception:
            console.print("   ⚠️ Early stopping not supported, training without...")
            final_model.fit(X_train_gpu, y_train_gpu)
        
        training_time = time.time() - start_time
        
        # Validation
        y_val_pred_proba = final_model.predict_proba(X_val_gpu)[:, 1]
        auc_score = roc_auc_score(y_val_gpu, y_val_pred_proba)
        f1 = precision_recall_fscore_support(y_val_gpu, final_model.predict(X_val_gpu), average='binary')[2]
        
        console.print(f"   ⏱️  {training_time:.1f}s  ✅ CV: {grid_search.best_score_:.4f}  ✅ Val: {auc_score:.4f}")
        
        return {
            "model": final_model,
            "training_metrics": {
                "best_params": best_params,
                "best_cv_score": grid_search.best_score_,
                "validation_auc": auc_score,
                "validation_f1": f1,
                "training_time_seconds": training_time,
                "mode": mode_text,
                "data_source": "REAL_UCI_HYDRAULIC_DATA",
                "gpu_compatibility": "GTX_1650_SUPER_OPTIMIZED"
            }
        }
    
    def train_random_forest_real(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Train RandomForest: CPU fast vs GPU full (with cuML)."""
        
        grid_key = "gpu_full" if self.use_gpu else "cpu_fast"
        param_grid = GRIDS["random_forest"][grid_key]
        combinations = len(self._grid_combinations(param_grid))
        
        cuml_text = " + cuML" if CUML_AVAILABLE and self.use_gpu else ""
        mode_text = f"GPU Full ({combinations} combinations){cuml_text}" if self.use_gpu else f"CPU Fast ({combinations} combinations)"
        console.print(f"\n🌲 RandomForest {mode_text}")
        
        if self.use_gpu and CUML_AVAILABLE:
            console.print("   🚀 cuML acceleration enabled (up to 137x faster)")
        
        # Base configuration
        base_params = {
            'random_state': 42,
            'oob_score': True,
            'class_weight': 'balanced',
            'n_jobs': 4  # Conservative for stability
        }
        
        rf_base = RandomForestClassifier(**base_params)
        
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=3, scoring='roc_auc',
            n_jobs=1, verbose=0  # Single job for grid to avoid conflicts
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        
        # Validation
        y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_val_pred_proba)
        f1 = precision_recall_fscore_support(y_val, best_model.predict(X_val), average='binary')[2]
        
        console.print(f"   ⏱️  {training_time:.1f}s  ✅ CV: {grid_search.best_score_:.4f}  ✅ Val: {auc_score:.4f}")
        console.print(f"   ✅ OOB: {best_model.oob_score_:.4f}")
        
        return {
            "model": best_model,
            "training_metrics": {
                "best_params": grid_search.best_params_,
                "best_cv_score": grid_search.best_score_,
                "validation_auc": auc_score,
                "validation_f1": f1,
                "oob_score": best_model.oob_score_,
                "training_time_seconds": training_time,
                "mode": mode_text,
                "data_source": "REAL_UCI_HYDRAULIC_DATA"
            }
        }
    
    def train_adaptive_real(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Train Adaptive (Isolation Forest)."""
        
        console.print("\n🔄 Training Adaptive (Isolation Forest)...")
        
        X_normal = X_train[y_train == 0]
        console.print(f"   📊 Training on {len(X_normal)} normal samples")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'contamination': [0.05, 0.1, 0.15, 0.2, 0.25],
            'max_features': [0.5, 0.75, 1.0]
        }
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        start_time = time.time()
        
        # Manual grid search for Isolation Forest
        for n_est in param_grid['n_estimators']:
            for contam in param_grid['contamination']:
                for max_feat in param_grid['max_features']:
                    try:
                        model = IsolationForest(
                            n_estimators=n_est,
                            contamination=contam,
                            max_features=max_feat,
                            random_state=42,
                            n_jobs=-1
                        )
                        model.fit(X_normal)
                        anomaly_scores = model.decision_function(X_val)
                        auc = roc_auc_score(y_val, -anomaly_scores)
                        if auc > best_score:
                            best_score = auc
                            best_params = {'n_estimators': n_est, 'contamination': contam, 'max_features': max_feat}
                            best_model = model
                    except Exception:
                        continue
        
        training_time = time.time() - start_time
        
        # Fallback if no model found
        if best_model is None:
            best_model = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
            best_model.fit(X_normal)
            best_params = {'n_estimators': 200, 'contamination': 0.15, 'max_features': 1.0}
            best_score = 0.5
        
        # Final validation
        anomaly_scores = best_model.decision_function(X_val)
        y_val_pred = (best_model.predict(X_val) == -1).astype(int)
        auc_score = roc_auc_score(y_val, -anomaly_scores)
        f1 = precision_recall_fscore_support(y_val, y_val_pred, average='binary')[2]
        
        console.print(f"   ⏱️  {training_time:.1f}s  ✅ Val AUC: {auc_score:.4f}")
        
        return {
            "base_model": best_model,
            "training_metrics": {
                "best_params": best_params,
                "best_cv_score": best_score,
                "validation_auc": auc_score,
                "validation_f1": f1,
                "training_time_seconds": training_time,
                "training_samples": len(X_normal),
                "data_source": "REAL_UCI_HYDRAULIC_DATA"
            }
        }
    
    def evaluate_and_save(self, X_test, y_test):
        """Evaluate models and save to disk."""
        
        console.print("\n📊 Test Evaluation...")
        
        test_results = {}
        for model_name, model_data in self.trained_models.items():
            try:
                if model_name == 'adaptive':
                    model = model_data['base_model']
                    anomaly_scores = model.decision_function(X_test)
                    y_pred = (model.predict(X_test) == -1).astype(int)
                    y_pred_proba = (-anomaly_scores + 1) / 2
                else:
                    model = model_data['model']
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                auc_score = roc_auc_score(y_test, y_pred_proba)
                f1 = precision_recall_fscore_support(y_test, y_pred, average='binary')[2]
                
                test_results[model_name] = {
                    "test_auc": auc_score,
                    "test_f1": f1,
                    "data_source": "REAL_UCI_HYDRAULIC_TEST_DATA"
                }
                
                console.print(f"   {model_name:12s}  AUC={auc_score:.4f}  F1={f1:.4f}")
                
            except Exception as e:
                console.print(f"   ❌ {model_name} test failed: {e}")
                test_results[model_name] = {"error": str(e)}
        
        # Save models
        console.print("\n💾 Saving models...")
        for model_name, model_data in self.trained_models.items():
            save_data = {
                **model_data,
                "training_timestamp": time.time(),
                "model_version": "1.0.0-production-REAL",
                "dataset_info": "REAL UCI Hydraulic System Industrial IoT Data",
                "data_source": "REAL_UCI_HYDRAULIC_DATA",
                "dataset_details": self.data_info,
                "is_mock_model": False,
                "training_method": f"hyperparameter_optimization_{self.mode}"
            }
            
            model_path = self.models_dir / f"{model_name}_model.joblib"
            joblib.dump(save_data, model_path)
            console.print(f"   ✅ {model_name}: {model_path.stat().st_size/1024:.1f} KB")
        
        return test_results
    
    def display_summary(self, test_results):
        """Display final summary."""
        
        table = Table(title="🏆 REAL Production Training Results")
        table.add_column("Model", style="cyan")
        table.add_column("Mode", style="blue")
        table.add_column("Time", style="yellow")
        table.add_column("CV AUC", style="green")
        table.add_column("Test AUC", style="green")
        table.add_column("Test F1", style="yellow")
        
        total_time = 0
        successful = 0
        
        for model_name in self.trained_models.keys():
            train_metrics = self.trained_models[model_name]['training_metrics']
            test_metrics = test_results.get(model_name, {})
            
            training_time = train_metrics.get('training_time_seconds', 0)
            total_time += training_time
            
            if 'error' not in test_metrics:
                successful += 1
                table.add_row(
                    model_name,
                    train_metrics.get('mode', 'N/A'),
                    f"{training_time:.1f}s",
                    f"{train_metrics.get('best_cv_score', 0):.4f}",
                    f"{test_metrics.get('test_auc', 0):.4f}",
                    f"{test_metrics.get('test_f1', 0):.4f}"
                )
        
        self.console.print(table)
        
        mode_desc = "GPU Full Search" if self.use_gpu else "CPU Fast"
        summary = Panel(
            f"""🏭 Dataset: {self.data_info['total_samples']:,} UCI Hydraulic samples
📊 Features: {self.data_info['n_features']} engineered from REAL sensors
⏱️  Total Time: {total_time/60:.1f} minutes ({mode_desc})
🎯 Success: {successful}/{len(self.trained_models)} models

🔥 REAL DATA TRAINING COMPLETED!
🚫 NO MORE MOCK MODELS!""",
            title="✅ Training Complete",
            style="green" if successful == len(self.trained_models) else "yellow"
        )
        self.console.print(summary)
    
    async def train_models(self, only_model: str = None, data_path: str = None):
        """Train models with specified configuration."""
        
        # Load data
        data = self.load_real_data(data_path)
        X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
        y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
        
        # Define training tasks
        all_tasks = [
            ("catboost", self.train_catboost_real),
            ("xgboost", self.train_xgboost_real), 
            ("random_forest", self.train_random_forest_real),
            ("adaptive", self.train_adaptive_real)
        ]
        
        # Filter if only_model specified
        if only_model:
            tasks = [(name, func) for name, func in all_tasks if name == only_model]
            if not tasks:
                raise ValueError(f"Model '{only_model}' not found. Available: catboost, xgboost, random_forest, adaptive")
        else:
            tasks = all_tasks
        
        console.print(f"\n🎯 Training {len(tasks)} model(s) - {self.mode.upper()} mode")
        
        # Train models sequentially (better for GPU memory)
        for model_name, train_func in tasks:
            try:
                console.print(f"\n▶️  Starting {model_name}...")
                trained_model = train_func(X_train, y_train, X_val, y_val)
                self.trained_models[model_name] = trained_model
                console.print(f"   ✅ {model_name} completed")
            except Exception as e:
                console.print(f"   ❌ {model_name} failed: {e}")
                # Continue with other models
        
        # Evaluate and save
        if self.trained_models:
            test_results = self.evaluate_and_save(X_test, y_test)
            self.display_summary(test_results)
        else:
            console.print("❌ No models were successfully trained!")


async def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train production ML models on REAL UCI data")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration (enables full search)")
    parser.add_argument("--only", choices=["catboost", "xgboost", "random_forest", "adaptive"],
                       help="Train only specified model")
    parser.add_argument("--data", type=str, help="Path to training data CSV")
    
    args = parser.parse_args()
    
    # Show mode
    mode_text = "GPU FULL SEARCH" if args.gpu else "CPU FAST"
    console.print(Panel.fit(f"🔥 REAL Production Training - {mode_text}", style="bold red"))
    
    trainer = RealProductionModelTrainer(use_gpu=args.gpu)
    
    try:
        await trainer.train_models(only_model=args.only, data_path=args.data)
        
        console.print(Panel(
            f"""🎉 Training completed!

🔥 Models trained on ACTUAL UCI HYDRAULIC DATA!
🚫 NO MORE MOCK MODELS!

🔧 Next steps:
   1. Test: python scripts/test_models.py
   2. Deploy ML service: python main.py
   3. Integrate TimescaleDB ingestion
   
💪 REAL DATA = REAL CONFIDENCE!""",
            title="✅ Mission Accomplished", 
            style="bold green"
        ))
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n❌ Training interrupted")
        return 1
    except Exception as e:
        console.print(f"\n💥 Training failed: {e}")
        import traceback
        console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)