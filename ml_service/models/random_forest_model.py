"""
Random Forest Anomaly Detection Model
Ensemble learning with multiple decision trees for hydraulic systems diagnostics
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from config import settings

from .base_model import BaseMLModel

logger = structlog.get_logger()


class RandomForestModel(BaseMLModel):
    """
    Production RandomForest model optimized for hydraulic anomaly detection.

    Features:
    - Ensemble of 100 decision trees
    - Automatic feature importance ranking
    - Out-of-bag error estimation
    - Balanced class weights for anomaly detection
    - Robust against overfitting
    """

    def __init__(self, model_name: str = "random_forest"):
        super().__init__(model_name)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance_ = None
        self.training_metrics = {}
        self.oob_score_ = None

        # RandomForest optimized hyperparameters for anomaly detection
        self.rf_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",  # Good default for classification
            "bootstrap": True,
            "oob_score": True,  # Enable out-of-bag error estimation
            "class_weight": "balanced",  # Handle imbalanced anomaly data
            "random_state": 42,
            "n_jobs": -1,  # Use all CPU cores
            "verbose": 0,  # Silent mode
        }

        self.metadata["features_count"] = 25

    async def load(self) -> None:
        """Load RandomForest model from disk or create mock model."""
        start_time = time.time()
        model_path = Path(settings.model_path) / "random_forest_model.joblib"
        logger.info("Loading RandomForest model", path=str(model_path))

        try:
            if model_path.exists():
                loaded_data = joblib.load(model_path)

                if isinstance(loaded_data, dict) and "model" in loaded_data:
                    # New format with metadata
                    self.model = loaded_data["model"]
                    self.scaler = loaded_data.get("scaler", StandardScaler())
                    self.feature_importance_ = loaded_data.get("feature_importance")
                    self.training_metrics = loaded_data.get("training_metrics", {})
                    self.oob_score_ = loaded_data.get("oob_score")
                    if "features_count" in loaded_data:
                        self.metadata["features_count"] = int(loaded_data["features_count"])
                else:
                    # Direct model object (legacy format)
                    self.model = loaded_data
                    logger.warning("Direct RandomForest model loaded, creating compatible scaler")

                    # Create compatible scaler
                    self.scaler = StandardScaler()
                    expected_features = getattr(self.model, "n_features_in_", 25)
                    mock_data = np.random.randn(100, expected_features)
                    self.scaler.fit(mock_data)
                    self.metadata["features_count"] = expected_features
                    self.oob_score_ = getattr(self.model, "oob_score_", None)

                logger.info(
                    "Real RandomForest model loaded",
                    features_count=self.metadata["features_count"],
                    oob_score=self.oob_score_,
                )
            else:
                logger.warning("RandomForest model file not found, creating mock model", path=str(model_path))
                await self._create_mock_model()

            self.is_loaded = True
            self.load_time = time.time() - start_time
            self.version = "v1.0.0-randomforest"

            logger.info(
                "RandomForest model loaded successfully",
                load_time_seconds=self.load_time,
                version=self.version,
                is_mock=not model_path.exists(),
                features_count=self.metadata.get("features_count"),
                n_estimators=getattr(self.model, "n_estimators", 100) if self.model else 100,
            )

        except Exception as e:
            logger.error("RandomForest loading failed, creating mock", error=str(e))
            await self._create_mock_model()

    async def _create_mock_model(self) -> None:
        """Create production-quality mock RandomForest model for development."""
        logger.info("Creating production-quality mock RandomForest model")

        try:
            # Generate realistic hydraulic system data with more complexity
            n_samples = 3000  # More samples for better forest training
            n_features = self.metadata.get("features_count", 25)

            # Create realistic normal operating conditions
            normal_data = np.random.normal(0, 1, (int(n_samples * 0.95), n_features))

            # Add realistic correlations between hydraulic parameters
            # Pressure and flow rate correlation
            normal_data[:, 1] = normal_data[:, 0] * 0.8 + np.random.normal(0, 0.2, int(n_samples * 0.95))
            # Temperature and pressure correlation
            normal_data[:, 2] = normal_data[:, 0] * 0.6 + np.random.normal(0, 0.3, int(n_samples * 0.95))
            # Vibration patterns
            normal_data[:, 3] = np.sin(normal_data[:, 0]) + np.random.normal(0, 0.1, int(n_samples * 0.95))

            # Create various anomaly patterns
            n_anomalies = int(n_samples * 0.05)
            anomaly_data = np.zeros((n_anomalies, n_features))

            # Different types of anomalies
            for i in range(n_anomalies):
                if i < n_anomalies // 3:
                    # Pressure spike anomalies
                    anomaly_data[i] = np.random.normal(0, 1, n_features)
                    anomaly_data[i, 0] += np.random.uniform(2, 4)  # High pressure
                    anomaly_data[i, 1] -= np.random.uniform(1, 2)  # Low flow
                elif i < 2 * n_anomalies // 3:
                    # Temperature anomalies
                    anomaly_data[i] = np.random.normal(0, 1.5, n_features)
                    anomaly_data[i, 2] += np.random.uniform(3, 5)  # High temp
                else:
                    # Vibration anomalies
                    anomaly_data[i] = np.random.normal(0, 1, n_features)
                    anomaly_data[i, 3] += np.random.uniform(2, 4)  # High vibration
                    anomaly_data[i, 4] += np.random.uniform(1, 3)  # Secondary effect

            # Combine data
            X_train = np.vstack([normal_data, anomaly_data])
            y_train = np.hstack([np.zeros(int(n_samples * 0.95)), np.ones(n_anomalies)])

            # Shuffle data
            shuffle_idx = np.random.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]

            # Fit scaler
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)

            # Create RandomForest model with production parameters
            self.model = RandomForestClassifier(**self.rf_params)

            # Train the model
            self.model.fit(X_train_scaled, y_train)

            # Store feature importance and OOB score
            self.feature_importance_ = self.model.feature_importances_
            self.oob_score_ = self.model.oob_score_

            # Calculate training metrics
            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
            from sklearn.model_selection import cross_val_score

            # Use cross-validation for more reliable metrics
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring="roc_auc", n_jobs=-1)

            # Get predictions for detailed metrics
            y_pred_proba = self.model.predict_proba(X_train_scaled)[:, 1]
            y_pred = self.model.predict(X_train_scaled)

            precision, recall, f1, _ = precision_recall_fscore_support(y_train, y_pred, average="binary")
            auc_score = roc_auc_score(y_train, y_pred_proba)

            self.training_metrics = {
                "auc_score": float(auc_score),
                "cv_auc_mean": float(np.mean(cv_scores)),
                "cv_auc_std": float(np.std(cv_scores)),
                "oob_score": float(self.oob_score_),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "n_estimators": int(self.model.n_estimators),
                "max_depth": self.rf_params["max_depth"],
            }

            self.metadata["accuracy_score"] = float(auc_score)
            self.is_trained = True

            logger.info(
                "Mock RandomForest model created successfully",
                auc_score=auc_score,
                oob_score=self.oob_score_,
                cv_auc_mean=np.mean(cv_scores),
                precision=precision,
                recall=recall,
                f1_score=f1,
                n_features=n_features,
                n_estimators=self.model.n_estimators,
            )

        except Exception as e:
            logger.error("Failed to create mock RandomForest model", error=str(e))
            raise

    async def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Make prediction using RandomForest model."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("RandomForest model not loaded")

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

            # Calculate confidence based on ensemble consensus
            # RandomForest confidence can be estimated from tree vote distribution
            tree_predictions = np.array([tree.predict_proba(features_scaled)[0, 1] for tree in self.model.estimators_])

            # Calculate consensus - how many trees agree on the prediction
            threshold = settings.prediction_threshold
            positive_votes = np.sum(tree_predictions > threshold)
            negative_votes = len(tree_predictions) - positive_votes

            # Confidence based on consensus strength
            consensus_ratio = max(positive_votes, negative_votes) / len(tree_predictions)
            base_confidence = 0.6
            confidence = min(base_confidence + consensus_ratio * 0.35, 0.95)

            processing_time = (time.time() - start_time) * 1000

            self.update_stats(processing_time / 1000)

            result = {
                "score": prediction_score,
                "confidence": confidence,
                "is_anomaly": prediction_score > threshold,
                "processing_time_ms": processing_time,
                "tree_consensus": {
                    "positive_votes": int(positive_votes),
                    "negative_votes": int(negative_votes),
                    "consensus_ratio": float(consensus_ratio),
                },
            }

            # Add feature importance for top features if available
            if self.feature_importance_ is not None:
                top_features_idx = np.argsort(self.feature_importance_)[-5:]
                result["important_features"] = {
                    f"feature_{idx}": {
                        "importance": float(self.feature_importance_[idx]),
                        "value": float(features[idx]),
                    }
                    for idx in top_features_idx
                }

            logger.debug(
                "RandomForest prediction completed",
                score=prediction_score,
                confidence=confidence,
                consensus_ratio=consensus_ratio,
                processing_time_ms=processing_time,
            )

            return result

        except Exception as e:
            logger.error("RandomForest prediction failed", error=str(e), features_shape=features.shape)
            raise

    def get_model_info(self) -> dict[str, Any]:
        """Get detailed RandomForest model information."""
        base_info = super().get_model_info()

        rf_specific = {
            "model_type": "Random Forest Ensemble",
            "n_estimators": self.rf_params.get("n_estimators", 100),
            "max_depth": self.rf_params.get("max_depth", 10),
            "max_features": self.rf_params.get("max_features", "sqrt"),
            "class_weight": self.rf_params.get("class_weight", "balanced"),
            "oob_score": float(self.oob_score_) if self.oob_score_ is not None else None,
            "training_metrics": self.training_metrics,
            "feature_importance_available": self.feature_importance_ is not None,
        }

        return {**base_info, **rf_specific}

    def get_feature_importance(self, top_k: int = 10) -> dict[str, float]:
        """Get top-k most important features."""
        if self.feature_importance_ is None:
            return {}

        feature_names = [f"feature_{i}" for i in range(len(self.feature_importance_))]
        importance_dict = dict(zip(feature_names, self.feature_importance_))

        # Sort by importance and return top-k
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_k])

    def get_tree_stats(self) -> dict[str, Any]:
        """Get statistics about the trees in the forest."""
        if not self.is_loaded or self.model is None:
            return {}

        tree_depths = [tree.get_depth() for tree in self.model.estimators_]
        tree_leaves = [tree.get_n_leaves() for tree in self.model.estimators_]

        return {
            "n_estimators": len(self.model.estimators_),
            "avg_tree_depth": float(np.mean(tree_depths)),
            "max_tree_depth": int(np.max(tree_depths)),
            "min_tree_depth": int(np.min(tree_depths)),
            "avg_tree_leaves": float(np.mean(tree_leaves)),
            "total_tree_nodes": int(np.sum(tree_leaves)),
        }

    async def cleanup(self) -> None:
        """Clean up RandomForest model resources."""
        logger.info("Cleaning up RandomForest model")

        if self.model is not None:
            # RandomForest can have many trees, clean up properly
            del self.model

        self.model = None
        self.feature_importance_ = None
        self.training_metrics = {}
        self.oob_score_ = None
        self.is_loaded = False
        self.is_trained = False

        logger.info("RandomForest model cleanup completed")
