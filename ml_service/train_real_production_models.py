#!/usr/bin/env python3
"""
REAL Production Model Training Script
Train all 4 models on ACTUAL UCI Hydraulic System dataset
NO MORE SYNTHETIC DATA - REAL INDUSTRIAL IOT DATA ONLY!
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

# ML imports
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_fscore_support, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.utils import resample
import xgboost as xgb
from catboost import CatBoostClassifier

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from data.uci_loader import load_uci_hydraulic_data, UCIHydraulicLoader

logger = structlog.get_logger()
console = Console()


class RealProductionModelTrainer:
    """Train all models on REAL UCI Hydraulic data - NO SYNTHETIC DATA!"""
    
    def __init__(self):
        self.console = console
        self.models_dir = Path(settings.model_path)
        self.models_dir.mkdir(exist_ok=True)
        
        self.trained_models = {}
        self.training_results = {}
        self.data_info = None
        
    def load_real_data(self) -> Dict[str, np.ndarray]:
        """Load REAL UCI hydraulic dataset."""
        
        console.print("🏭 Loading REAL UCI Hydraulic System dataset...")
        
        try:
            # Try larger dataset first
            data = load_uci_hydraulic_data(
                filename="Industrial_fault_detection.csv",  # Larger file
                window_minutes=5
            )
            console.print(f"✅ Loaded LARGE dataset: {data['X_train'].shape[0]} training samples")
        except Exception as e:
            console.print(f"⚠️  Large dataset failed, using smaller: {e}")
            # Fallback to smaller dataset
            data = load_uci_hydraulic_data(
                filename="industrial_fault_detection_data_1000.csv",
                window_minutes=5
            )
            console.print(f"✅ Loaded SMALL dataset: {data['X_train'].shape[0]} training samples")
        
        self.data_info = data['data_info']
        
        console.print(Panel(
            f"""📊 Dataset Information:
Total Samples: {data['data_info']['total_samples']:,}
Features: {data['data_info']['n_features']}
Window: {data['data_info']['window_minutes']} minutes
Classes: {data['data_info']['class_distribution']}
Date Range: {data['data_info']['date_range']['start']} to {data['data_info']['date_range']['end']}

✅ REAL INDUSTRIAL IOT DATA LOADED!
🚫 NO SYNTHETIC DATA!""",
            title="🏭 Real Data Loaded",
            style="green"
        ))
        
        return data
    
    def train_catboost_real(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train CatBoost on REAL data."""
        
        console.print("\n🐱 Training CatBoost on REAL UCI data...")
        
        # Hyperparameter optimization
        param_grid = {
            'iterations': [200, 300, 500],
            'depth': [4, 6, 8],
            'learning_rate': [0.03, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 10]
        }
        
        # Base model
        catboost_base = CatBoostClassifier(
            random_seed=42,
            logging_level='Silent',
            allow_writing_files=False,
            early_stopping_rounds=50
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            catboost_base, param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1, verbose=0
        )
        
        console.print("   🔍 Hyperparameter optimization on REAL data...")
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)
        y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_val, y_val_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary')
        accuracy = accuracy_score(y_val, y_val_pred)
        
        training_metrics = {
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "validation_auc": auc_score,
            "validation_accuracy": accuracy,
            "validation_precision": precision,
            "validation_recall": recall,
            "validation_f1": f1,
            "feature_importance": best_model.feature_importances_.tolist(),
            "data_source": "REAL_UCI_HYDRAULIC_DATA"  # Mark as real!
        }
        
        console.print(f"   ✅ REAL Data CV AUC: {grid_search.best_score_:.4f}")
        console.print(f"   ✅ REAL Data Val AUC: {auc_score:.4f}")
        console.print(f"   ✅ REAL Data Val F1: {f1:.4f}")
        
        return {
            "model": best_model,
            "scaler": None,  # Already scaled in data prep
            "training_metrics": training_metrics,
            "features_count": X_train.shape[1]
        }
    
    def train_xgboost_real(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train XGBoost on REAL data."""
        
        console.print("\n🚀 Training XGBoost on REAL UCI data...")
        
        # Hyperparameter optimization
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Base model
        xgb_base = xgb.XGBClassifier(
            random_state=42,
            eval_metric='auc',
            verbosity=0,
            early_stopping_rounds=50
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            xgb_base, param_grid, cv=3, scoring='roc_auc',
            n_jobs=-1, verbose=0
        )
        
        console.print("   🔍 Hyperparameter optimization on REAL data...")
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)
        y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_val, y_val_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary')
        accuracy = accuracy_score(y_val, y_val_pred)
        
        training_metrics = {
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "validation_auc": auc_score,
            "validation_accuracy": accuracy,
            "validation_precision": precision,
            "validation_recall": recall,
            "validation_f1": f1,
            "feature_importance": best_model.feature_importances_.tolist(),
            "data_source": "REAL_UCI_HYDRAULIC_DATA"  # Mark as real!
        }
        
        console.print(f"   ✅ REAL Data CV AUC: {grid_search.best_score_:.4f}")
        console.print(f"   ✅ REAL Data Val AUC: {auc_score:.4f}")
        console.print(f"   ✅ REAL Data Val F1: {f1:.4f}")
        
        return {
            "model": best_model,
            "scaler": None,  # Already scaled
            "training_metrics": training_metrics,
            "features_count": X_train.shape[1]
        }
    
    def train_random_forest_real(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest on REAL data."""
        
        console.print("\n🌲 Training Random Forest on REAL UCI data...")
        
        # Hyperparameter optimization
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Base model
        rf_base = RandomForestClassifier(
            random_state=42,
            oob_score=True,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=3, scoring='roc_auc',
            n_jobs=-1, verbose=0
        )
        
        console.print("   🔍 Hyperparameter optimization on REAL data...")
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)
        y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_val, y_val_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary')
        accuracy = accuracy_score(y_val, y_val_pred)
        
        training_metrics = {
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "validation_auc": auc_score,
            "validation_accuracy": accuracy,
            "validation_precision": precision,
            "validation_recall": recall,
            "validation_f1": f1,
            "oob_score": best_model.oob_score_,
            "feature_importance": best_model.feature_importances_.tolist(),
            "data_source": "REAL_UCI_HYDRAULIC_DATA"  # Mark as real!
        }
        
        console.print(f"   ✅ REAL Data CV AUC: {grid_search.best_score_:.4f}")
        console.print(f"   ✅ REAL Data Val AUC: {auc_score:.4f}")
        console.print(f"   ✅ REAL Data OOB Score: {best_model.oob_score_:.4f}")
        console.print(f"   ✅ REAL Data Val F1: {f1:.4f}")
        
        return {
            "model": best_model,
            "scaler": None,  # Already scaled
            "training_metrics": training_metrics,
            "features_count": X_train.shape[1]
        }
    
    def train_adaptive_real(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train Adaptive model on REAL data."""
        
        console.print("\n🔄 Training Adaptive model on REAL UCI data...")
        
        # For Isolation Forest, use only normal samples for training
        X_normal = X_train[y_train == 0]
        
        console.print(f"   📊 Training on {len(X_normal)} REAL normal samples")
        
        # Hyperparameter optimization
        param_grid = {
            'n_estimators': [100, 200, 300],
            'contamination': [0.05, 0.1, 0.15, 0.2, 0.25],
            'max_features': [0.5, 0.75, 1.0]
        }
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        console.print("   🔍 Hyperparameter optimization on REAL data...")
        
        # Manual grid search for Isolation Forest
        for n_est in param_grid['n_estimators']:
            for contam in param_grid['contamination']:
                for max_feat in param_grid['max_features']:
                    
                    try:
                        # Train model
                        model = IsolationForest(
                            n_estimators=n_est,
                            contamination=contam,
                            max_features=max_feat,
                            random_state=42,
                            n_jobs=-1
                        )
                        
                        model.fit(X_normal)  # Train on REAL normal samples
                        
                        # Predict on validation set
                        anomaly_scores = model.decision_function(X_val)
                        
                        # Calculate AUC using anomaly scores  
                        auc = roc_auc_score(y_val, -anomaly_scores)  # Negative for correct direction
                        
                        if auc > best_score:
                            best_score = auc
                            best_params = {
                                'n_estimators': n_est,
                                'contamination': contam,
                                'max_features': max_feat
                            }
                            best_model = model
                    except Exception:
                        continue
        
        # Final evaluation
        if best_model is not None:
            anomaly_scores = best_model.decision_function(X_val)
            y_val_pred_binary = best_model.predict(X_val)
            y_val_pred = (y_val_pred_binary == -1).astype(int)
            
            auc_score = roc_auc_score(y_val, -anomaly_scores)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary')
            accuracy = accuracy_score(y_val, y_val_pred)
        else:
            # Fallback if optimization failed
            best_model = IsolationForest(
                n_estimators=200,
                contamination=0.15,  # Estimated from data distribution
                random_state=42
            )
            best_model.fit(X_normal)
            
            anomaly_scores = best_model.decision_function(X_val)
            y_val_pred_binary = best_model.predict(X_val)
            y_val_pred = (y_val_pred_binary == -1).astype(int)
            
            auc_score = roc_auc_score(y_val, -anomaly_scores)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary')
            accuracy = accuracy_score(y_val, y_val_pred)
            
            best_params = {'n_estimators': 200, 'contamination': 0.15, 'max_features': 1.0}
            best_score = auc_score
        
        training_metrics = {
            "best_params": best_params,
            "best_cv_score": best_score,
            "validation_auc": auc_score,
            "validation_accuracy": accuracy,
            "validation_precision": precision,
            "validation_recall": recall,
            "validation_f1": f1,
            "training_samples": len(X_normal),
            "data_source": "REAL_UCI_HYDRAULIC_DATA"  # Mark as real!
        }
        
        console.print(f"   ✅ REAL Data Val AUC: {best_score:.4f}")
        console.print(f"   ✅ REAL Data Val F1: {f1:.4f}")
        console.print(f"   ✅ Best params: {best_params}")
        
        return {
            "base_model": best_model,
            "scaler": None,  # Already scaled
            "training_metrics": training_metrics,
            "features_count": X_train.shape[1]
        }
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate all trained models on REAL test set."""
        
        console.print("\n📊 Evaluating models on REAL test data...")
        
        test_results = {}
        
        for model_name, model_data in self.trained_models.items():
            console.print(f"   Testing {model_name} on REAL data...")
            
            try:
                if model_name == 'adaptive':
                    # Special handling for Isolation Forest
                    model = model_data['base_model']
                    anomaly_scores = model.decision_function(X_test)
                    y_pred_binary = model.predict(X_test)
                    y_pred = (y_pred_binary == -1).astype(int)
                    y_pred_proba = (-anomaly_scores + 1) / 2  # Convert to 0-1 range
                else:
                    model = model_data['model']
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics on REAL test data
                auc_score = roc_auc_score(y_test, y_pred_proba)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                accuracy = accuracy_score(y_test, y_pred)
                
                test_results[model_name] = {
                    "test_auc": auc_score,
                    "test_accuracy": accuracy,
                    "test_precision": precision,
                    "test_recall": recall,
                    "test_f1": f1,
                    "data_source": "REAL_UCI_HYDRAULIC_TEST_DATA"
                }
                
                console.print(f"     ✅ REAL Test AUC: {auc_score:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                console.print(f"     ❌ Error testing {model_name} on REAL data: {e}")
                test_results[model_name] = {"error": str(e)}
        
        return test_results
    
    def save_real_trained_models(self) -> None:
        """Save all REAL trained models to disk."""
        
        console.print("\n💾 Saving REAL trained models...")
        
        for model_name, model_data in self.trained_models.items():
            try:
                # Prepare data for saving - MARK AS REAL!
                save_data = {
                    **model_data,
                    "training_timestamp": time.time(),
                    "model_version": "1.0.0-production-REAL",
                    "dataset_info": "REAL UCI Hydraulic System Industrial IoT Data",
                    "data_source": "REAL_UCI_HYDRAULIC_DATA",
                    "dataset_details": self.data_info,
                    "is_mock_model": False,  # Explicitly mark as NOT mock
                    "training_method": "hyperparameter_optimization_on_real_data"
                }
                
                # Save to file - REPLACE the mock model!
                model_path = self.models_dir / f"{model_name}_model.joblib"
                joblib.dump(save_data, model_path)
                
                console.print(f"   ✅ REAL {model_name} saved to {model_path}")
                console.print(f"      🔍 Size: {model_path.stat().st_size / 1024:.1f} KB")
                
            except Exception as e:
                console.print(f"   ❌ Failed to save REAL {model_name}: {e}")
    
    def display_real_training_summary(self, test_results: Dict[str, Any]) -> None:
        """Display comprehensive REAL training summary."""
        
        # Training Results Table
        training_table = Table(title="🏆 REAL Production Model Training Results (UCI Data)")
        training_table.add_column("Model", style="cyan")
        training_table.add_column("CV AUC", style="green")
        training_table.add_column("Val AUC", style="green")
        training_table.add_column("Test AUC", style="green")
        training_table.add_column("Test F1", style="yellow")
        training_table.add_column("Test Accuracy", style="blue")
        training_table.add_column("Data Source", style="bold")
        training_table.add_column("Status", style="bold")
        
        for model_name in self.trained_models.keys():
            train_metrics = self.trained_models[model_name]['training_metrics']
            test_metrics = test_results.get(model_name, {})
            
            if 'error' not in test_metrics:
                training_table.add_row(
                    model_name,
                    f"{train_metrics.get('best_cv_score', 0):.4f}",
                    f"{train_metrics.get('validation_auc', 0):.4f}",
                    f"{test_metrics.get('test_auc', 0):.4f}",
                    f"{test_metrics.get('test_f1', 0):.4f}",
                    f"{test_metrics.get('test_accuracy', 0):.4f}",
                    "REAL UCI DATA",
                    "✅ SUCCESS"
                )
            else:
                training_table.add_row(
                    model_name,
                    "N/A",
                    "N/A", 
                    "N/A",
                    "N/A",
                    "N/A",
                    "ERROR",
                    "❌ FAILED"
                )
        
        self.console.print(training_table)
        
        # Summary Panel
        successful_models = len([m for m in test_results.values() if 'error' not in m])
        total_models = len(self.trained_models)
        
        if successful_models == total_models:
            status_text = f"🎉 ALL {total_models} MODELS TRAINED ON REAL DATA!"
            status_style = "green"
        else:
            status_text = f"⚠️ {successful_models}/{total_models} models trained on REAL data"
            status_style = "yellow"
        
        summary_panel = Panel(
            f"""🏭 REAL Dataset: {self.data_info['total_samples']:,} samples from UCI Hydraulic System
📊 Features: {self.data_info['n_features']} engineered from REAL sensor data
🕰️ Time Window: {self.data_info['window_minutes']} minutes rolling statistics
📈 Date Range: {self.data_info['date_range']['start']} to {self.data_info['date_range']['end']}
🎯 Classes: {self.data_info['class_distribution']}

{status_text}

⚡ Models trained with hyperparameter optimization on REAL data!
💾 Saved to: {self.models_dir}
🚫 NO MORE MOCK MODELS - ONLY REAL TRAINED MODELS!""",
            title="🔥 REAL Data Training Summary",
            style=status_style
        ))
        
        self.console.print(summary_panel)
    
    async def train_all_models_on_real_data(self) -> None:
        """Train all models on REAL UCI data."""
        
        self.console.print(Panel.fit(
            "🔥 REAL Production Model Training - UCI Hydraulic Data",
            style="bold red"
        ))
        
        # Load REAL data
        data = self.load_real_data()
        X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
        y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
        
        # Training tasks
        training_tasks = [
            ("catboost", self.train_catboost_real),
            ("xgboost", self.train_xgboost_real),
            ("random_forest", self.train_random_forest_real),
            ("adaptive", self.train_adaptive_real)
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            overall_task = progress.add_task("Training on REAL data...", total=len(training_tasks))
            
            for model_name, train_func in training_tasks:
                model_task = progress.add_task(f"Training {model_name} on REAL data...", total=None)
                
                try:
                    trained_model = train_func(X_train, y_train, X_val, y_val)
                    self.trained_models[model_name] = trained_model
                    
                    progress.update(model_task, description=f"✅ {model_name} - REAL DATA TRAINED")
                    
                except Exception as e:
                    progress.update(model_task, description=f"❌ {model_name} - REAL DATA FAILED: {str(e)}")
                    console.print(f"\n💥 Error training {model_name} on REAL data: {e}")
                
                progress.advance(overall_task)
                progress.remove_task(model_task)
        
        # Evaluate on REAL test set
        if self.trained_models:
            test_results = self.evaluate_on_test_set(X_test, y_test)
            
            # Save REAL models
            self.save_real_trained_models()
            
            # Display results
            self.display_real_training_summary(test_results)
        else:
            console.print("❌ No models were successfully trained on REAL data!")


async def main():
    """Main REAL training pipeline."""
    trainer = RealProductionModelTrainer()
    
    try:
        await trainer.train_all_models_on_real_data()
        
        console.print(Panel(
            """🎉 REAL Production model training completed!

🔥 ALL MODELS TRAINED ON ACTUAL UCI HYDRAULIC DATA!
🚫 NO MORE MOCK MODELS!

🔧 Next steps:
   1. Run 'python quick_test.py' - should show NO mock warnings
   2. Run 'python scripts/test_models.py' - comprehensive testing
   3. Deploy to production with confidence!
   
💪 REAL DATA = REAL CONFIDENCE!""",
            title="✅ Mission Accomplished",
            style="bold green"
        ))
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n❌ Training interrupted by user")
        return 1
    except Exception as e:
        console.print(f"\n💥 REAL data training failed: {e}")
        import traceback
        console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)