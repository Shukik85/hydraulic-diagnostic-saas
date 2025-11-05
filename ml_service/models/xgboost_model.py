"""
XGBoost Anomaly Detection Model
High-performance gradient boosting for hydraulic systems diagnostics
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from config import settings

from .base_model import BaseMLModel

logger = structlog.get_logger()


class XGBoostModel(BaseMLModel):
    """
    Production XGBoost model optimized for hydraulic anomaly detection.
    
    Features:
    - Early stopping to prevent overfitting
    - Optimized hyperparameters for time-series anomaly detection
    - Built-in feature importance analysis
    - Support for probability and binary predictions
    """
    
    def __init__(self, model_name: str = "xgboost"):
        super().__init__(model_name)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance_ = None
        self.training_metrics = {}
        
        # XGBoost optimized hyperparameters for anomaly detection
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.1,
            'reg_alpha': 0.01,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,  # Use all CPU cores
            'verbosity': 0,  # Silent mode
        }
        
        self.metadata["features_count"] = 25

    async def load(self) -> None:
        """Load XGBoost model from disk or create mock model."""
        start_time = time.time()
        model_path = Path(settings.model_path) / "xgboost_model.joblib"
        logger.info("Loading XGBoost model", path=str(model_path))

        try:
            if model_path.exists():
                loaded_data = joblib.load(model_path)
                
                if isinstance(loaded_data, dict) and "model" in loaded_data:
                    # New format with metadata
                    self.model = loaded_data["model"]
                    self.scaler = loaded_data.get("scaler", StandardScaler())
                    self.feature_importance_ = loaded_data.get("feature_importance")
                    self.training_metrics = loaded_data.get("training_metrics", {})
                    if "features_count" in loaded_data:
                        self.metadata["features_count"] = int(loaded_data["features_count"])
                else:
                    # Direct model object (legacy format)
                    self.model = loaded_data
                    logger.warning("Direct XGBoost model loaded, creating compatible scaler")
                    
                    # Create compatible scaler
                    self.scaler = StandardScaler()
                    expected_features = getattr(self.model, 'n_features_in_', 25)
                    mock_data = np.random.randn(100, expected_features)
                    self.scaler.fit(mock_data)
                    self.metadata["features_count"] = expected_features

                logger.info("Real XGBoost model loaded", features_count=self.metadata["features_count"])
            else:
                logger.warning("XGBoost model file not found, creating mock model", path=str(model_path))
                await self._create_mock_model()

            self.is_loaded = True
            self.load_time = time.time() - start_time
            self.version = "v1.0.0-xgboost"

            logger.info(
                "XGBoost model loaded successfully",
                load_time_seconds=self.load_time,
                version=self.version,
                is_mock=not model_path.exists(),
                features_count=self.metadata.get("features_count"),
            )

        except Exception as e:
            logger.error("XGBoost loading failed, creating mock", error=str(e))
            await self._create_mock_model()

    async def _create_mock_model(self) -> None:
        """Create production-quality mock XGBoost model for development."""
        logger.info("Creating production-quality mock XGBoost model")
        
        try:
            # Generate realistic hydraulic system data
            n_samples = 2000
            n_features = self.metadata.get("features_count", 25)
            
            # Create diverse training data
            normal_data = np.random.normal(0, 1, (int(n_samples * 0.95), n_features))
            # Add some correlations typical for hydraulic systems
            normal_data[:, 1] = normal_data[:, 0] * 0.7 + np.random.normal(0, 0.3, int(n_samples * 0.95))
            normal_data[:, 2] = normal_data[:, 0] * -0.5 + np.random.normal(0, 0.4, int(n_samples * 0.95))
            
            # Anomalous data with different patterns
            anomaly_data = np.random.normal(0, 2.5, (int(n_samples * 0.05), n_features))
            anomaly_data[:, 0] += 3  # Shift first feature significantly
            
            # Combine data
            X_train = np.vstack([normal_data, anomaly_data])
            y_train = np.hstack([
                np.zeros(int(n_samples * 0.95)),
                np.ones(int(n_samples * 0.05))
            ])
            
            # Shuffle data
            shuffle_idx = np.random.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]
            
            # Fit scaler
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            
            # Create XGBoost model with production parameters
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                early_stopping_rounds=10,
                **self.xgb_params
            )
            
            # Split for validation (for early stopping)
            split_idx = int(len(X_train_scaled) * 0.8)
            X_val = X_train_scaled[split_idx:]
            y_val = y_train[split_idx:]
            X_train_scaled = X_train_scaled[:split_idx]
            y_train = y_train[:split_idx]
            
            # Train with early stopping
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Store feature importance
            self.feature_importance_ = self.model.feature_importances_
            
            # Mock training metrics
            from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
            
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            y_pred = self.model.predict(X_val)
            
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            self.training_metrics = {
                "auc_score": float(auc_score),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "n_estimators_used": int(self.model.n_features_in_),
                "best_iteration": getattr(self.model, 'best_iteration', 100),
            }
            
            self.metadata["accuracy_score"] = float(auc_score)
            self.is_trained = True
            
            logger.info(
                "Mock XGBoost model created successfully",
                auc_score=auc_score,
                precision=precision,
                recall=recall,
                f1_score=f1,
                n_features=n_features
            )
            
        except Exception as e:
            logger.error("Failed to create mock XGBoost model", error=str(e))
            raise

    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Make prediction using XGBoost model."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("XGBoost model not loaded")

        # Ensure proper feature format
        features = self._ensure_vector(features)

        start_time = time.time()
        try:
            # Reshape for model input
            features_2d = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features_2d)
            
            # Get probability predictions
            probabilities = self.model.predict_proba(features_scaled)
            
            if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                prediction_score = float(probabilities[0, 1])  # Anomaly probability
            else:
                prediction_score = float(probabilities[0])

            # Calculate confidence based on distance from decision boundary
            threshold = settings.prediction_threshold
            distance_from_threshold = abs(prediction_score - threshold)
            
            # Higher confidence for predictions farther from threshold
            base_confidence = 0.7
            confidence_boost = min(distance_from_threshold * 1.5, 0.25)
            confidence = min(base_confidence + confidence_boost, 0.95)

            processing_time = (time.time() - start_time) * 1000

            self.update_stats(processing_time / 1000)

            result = {
                "score": prediction_score,
                "confidence": confidence,
                "is_anomaly": prediction_score > threshold,
                "processing_time_ms": processing_time,
            }
            
            # Add feature importance for top features if available
            if self.feature_importance_ is not None:
                top_features_idx = np.argsort(self.feature_importance_)[-5:]
                result["important_features"] = {
                    f"feature_{idx}": {
                        "importance": float(self.feature_importance_[idx]),
                        "value": float(features[idx])
                    }
                    for idx in top_features_idx
                }

            logger.debug(
                "XGBoost prediction completed",
                score=prediction_score,
                confidence=confidence,
                processing_time_ms=processing_time
            )

            return result

        except Exception as e:
            logger.error("XGBoost prediction failed", error=str(e), features_shape=features.shape)
            raise

    def get_model_info(self) -> dict[str, Any]:
        """Get detailed XGBoost model information."""
        base_info = super().get_model_info()
        
        xgb_specific = {
            "model_type": "XGBoost Gradient Boosting",
            "objective": self.xgb_params.get("objective", "binary:logistic"),
            "max_depth": self.xgb_params.get("max_depth", 6),
            "learning_rate": self.xgb_params.get("learning_rate", 0.1),
            "n_estimators": getattr(self.model, 'n_estimators', 100) if self.model else 100,
            "training_metrics": self.training_metrics,
            "feature_importance_available": self.feature_importance_ is not None,
        }
        
        return {**base_info, **xgb_specific}

    def get_feature_importance(self, top_k: int = 10) -> dict[str, float]:
        """Get top-k most important features."""
        if self.feature_importance_ is None:
            return {}
        
        feature_names = [f"feature_{i}" for i in range(len(self.feature_importance_))]
        importance_dict = dict(zip(feature_names, self.feature_importance_))
        
        # Sort by importance and return top-k
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_k])

    async def cleanup(self) -> None:
        """Clean up XGBoost model resources."""
        logger.info("Cleaning up XGBoost model")
        
        if self.model is not None:
            # XGBoost models can be memory intensive
            del self.model
        
        self.model = None
        self.feature_importance_ = None
        self.training_metrics = {}
        self.is_loaded = False
        self.is_trained = False
        
        logger.info("XGBoost model cleanup completed")