#!/usr/bin/env python3
"""
Two-Stage Classifier Service for Enhanced Anomaly Detection
Stage 1: Binary anomaly detection (normal vs fault) - High Recall
Stage 2: Multi-class fault classification (pump/valve/cooling) - High Precision
Provides actionable insights with component mapping and confidence scoring

Enterprise Features:
- Feature contract validation for production compatibility
- Graceful degradation on contract mismatches
- Detailed compatibility reporting
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

# Component mapping for actionable maintenance insights
COMPONENT_MAPPING = {
    1: ["pump_main", "pump_motor"],
    2: ["valve_main", "valve_control"],
    3: ["cooler", "heat_exchanger", "cooling_system"],
}

FAULT_TYPE_NAMES = {0: "normal", 1: "pump_fault", 2: "valve_fault", 3: "cooling_fault"}


class TwoStageClassifier:
    """Two-stage classifier for enhanced anomaly detection and fault typing"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.binary_model = None
        self.multiclass_model = None
        self.scaler = None
        self.config = None
        self.is_loaded = False
        self.version = "v20251105_0011"

        # Enterprise feature contract validation
        self.feature_contract = None
        self.contract_compatible = False
        self.compatibility_reason = "not_validated"

        # Performance tracking
        self.prediction_count = 0
        self.load_time = None

        logger.info("TwoStageClassifier initialized", models_dir=str(self.models_dir))

    def _load_feature_contract(self) -> dict[str, Any]:
        """Load enterprise feature contract for compatibility validation"""
        contract_path = self.models_dir / "features_contract.json"

        try:
            if contract_path.exists():
                with contract_path.open("r", encoding="utf-8") as f:
                    contract = json.load(f)

                logger.info(
                    "Feature contract loaded",
                    contract_version=contract.get("contract_version", "unknown"),
                    features_count=contract.get("features_count", 0),
                )

                return contract
            else:
                logger.warning("No feature contract found", path=str(contract_path))
                return None

        except Exception as e:
            logger.error("Feature contract loading failed", path=str(contract_path), error=str(e))
            return None

    def _validate_contract_compatibility(self, contract: dict[str, Any]) -> tuple[bool, str]:
        """Validate Two-Stage models compatibility with feature contract"""
        if not contract:
            return False, "feature_contract_missing"

        try:
            # Check feature count compatibility
            contract_features = contract.get("features_count", 0)
            if contract_features != 25:
                return False, f"feature_count_mismatch: contract={contract_features}, expected=25"

            # Check scaler compatibility if loaded
            if self.scaler and hasattr(self.scaler, "n_features_in_"):
                scaler_features = self.scaler.n_features_in_
                if scaler_features != contract_features:
                    return False, f"scaler_mismatch: scaler={scaler_features}, contract={contract_features}"

            # Check config compatibility
            if self.config:
                config_features = self.config.get("features_count", 0)
                if config_features > 0 and config_features != contract_features:
                    return False, f"config_mismatch: config={config_features}, contract={contract_features}"

            return True, "compatible"

        except Exception as e:
            return False, f"validation_error: {str(e)}"

    def _robust_model_load(self, model_path: Path, model_name: str):
        """Robust model loading with dict/direct format compatibility"""
        try:
            logger.info(f"Loading {model_name} model", path=str(model_path))

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            loaded_data = joblib.load(model_path)

            # Handle both dict and direct model formats
            if isinstance(loaded_data, dict):
                if "model" in loaded_data:
                    model = loaded_data["model"]
                    logger.info(f"{model_name} model loaded from dict format")
                else:
                    # Dict without 'model' key - use as-is
                    model = loaded_data
                    logger.info(f"{model_name} model loaded as dict object")
            else:
                # Direct model object
                model = loaded_data
                logger.info(f"{model_name} model loaded as direct object")

            # Validate model has required methods
            if not hasattr(model, "predict"):
                raise ValueError(f"{model_name} model missing predict method")

            return model

        except Exception as e:
            logger.error(f"Failed to load {model_name} model", path=str(model_path), error=str(e))
            logger.error(f"Traceback for {model_name}:", traceback=traceback.format_exc())
            return None

    def load_models(self) -> bool:
        """Load two-stage models with enterprise feature contract validation"""
        try:
            start_time = time.time()
            logger.info("Starting Two-Stage model loading", version=self.version)

            # Load and validate feature contract first
            self.feature_contract = self._load_feature_contract()

            # Try multiple config locations
            config_paths = [
                self.models_dir / self.version / "two_stage_summary.json",
                self.models_dir / "training_summary.json",
                self.models_dir / "two_stage_summary.json",
            ]

            config_loaded = False
            for config_path in config_paths:
                if config_path.exists():
                    try:
                        with config_path.open("r", encoding="utf-8") as f:
                            self.config = json.load(f)
                        logger.info(
                            "Configuration loaded", path=str(config_path), version=self.config.get("version", "unknown")
                        )
                        config_loaded = True
                        break
                    except Exception as e:
                        logger.warning("Config loading failed", path=str(config_path), error=str(e))
                        continue

            if not config_loaded:
                logger.warning("No configuration found, using defaults")
                self.config = {
                    "version": self.version,
                    "optimal_binary_threshold": 0.1,
                    "best_stage2_variant": "plain",
                    "features_count": 25,
                }

            # Load feature scaler with fallback locations
            scaler_paths = [
                self.models_dir / self.version / "feature_scaler.joblib",
                self.models_dir / "feature_scaler.joblib",
            ]

            scaler_loaded = False
            for scaler_path in scaler_paths:
                if scaler_path.exists():
                    try:
                        self.scaler = joblib.load(scaler_path)
                        logger.info(
                            "Feature scaler loaded",
                            path=str(scaler_path),
                            n_features=getattr(self.scaler, "n_features_in_", "unknown"),
                        )
                        scaler_loaded = True
                        break
                    except Exception as e:
                        logger.warning("Scaler loading failed", path=str(scaler_path), error=str(e))
                        continue

            if not scaler_loaded:
                logger.error("No feature scaler found, creating default")
                self.scaler = StandardScaler()
                # Fit on dummy data with contract features count
                features_count = self.feature_contract.get("features_count", 25) if self.feature_contract else 25
                dummy_data = np.random.randn(10, features_count)
                self.scaler.fit(dummy_data)

            # Validate feature contract compatibility
            self.contract_compatible, self.compatibility_reason = self._validate_contract_compatibility(
                self.feature_contract
            )

            if not self.contract_compatible:
                logger.error("Feature contract validation failed", reason=self.compatibility_reason)
                # Continue loading for graceful degradation, but mark as incompatible

            # Load Stage 1 binary model (multiple locations)
            binary_paths = [
                self.models_dir / self.version / "binary_detector_xgb.joblib",
                self.models_dir / "binary_detector_xgb.joblib",
                self.models_dir / "catboost_model.joblib",  # fallback
            ]

            for binary_path in binary_paths:
                self.binary_model = self._robust_model_load(binary_path, "Stage1-Binary")
                if self.binary_model:
                    break

            # Load Stage 2 multiclass model
            best_variant = self.config.get("best_stage2_variant", "plain")
            multiclass_suffix = "_smote" if best_variant == "smote" else ""

            multiclass_paths = [
                self.models_dir / self.version / f"fault_classifier_catboost{multiclass_suffix}.joblib",
                self.models_dir / f"fault_classifier_catboost{multiclass_suffix}.joblib",
            ]

            for multiclass_path in multiclass_paths:
                self.multiclass_model = self._robust_model_load(multiclass_path, "Stage2-Multiclass")
                if self.multiclass_model:
                    break

            # Determine loading success
            binary_ok = self.binary_model is not None
            multiclass_ok = self.multiclass_model is not None
            scaler_ok = self.scaler is not None

            # Loading success requires models + scaler + contract compatibility
            if binary_ok and scaler_ok and self.contract_compatible:
                self.is_loaded = True
                self.load_time = time.time() - start_time

                logger.info(
                    "Two-stage classifier loaded successfully",
                    load_time_ms=self.load_time * 1000,
                    binary_available=binary_ok,
                    multiclass_available=multiclass_ok,
                    contract_compatible=self.contract_compatible,
                )
                return True
            else:
                logger.error(
                    "Two-stage loading failed",
                    binary_ok=binary_ok,
                    multiclass_ok=multiclass_ok,
                    scaler_ok=scaler_ok,
                    contract_compatible=self.contract_compatible,
                    compatibility_reason=self.compatibility_reason,
                )
                return False

        except Exception as e:
            logger.error("Two-stage model loading exception", error=str(e), traceback=traceback.format_exc())
            self.is_loaded = False
            self.compatibility_reason = f"loading_exception: {str(e)}"
            return False

    def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Two-stage prediction with actionable insights"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if not self.contract_compatible:
            raise RuntimeError(f"Feature contract incompatible: {self.compatibility_reason}")

        start_time = time.time()

        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Validate input feature count
        expected_features = self.feature_contract.get("features_count", 25) if self.feature_contract else 25
        if features.shape[1] != expected_features:
            raise ValueError(f"Input feature count mismatch: got {features.shape[1]}, expected {expected_features}")

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Stage 1: Binary anomaly detection
        binary_proba = self.binary_model.predict_proba(features_scaled)
        fault_probability = float(binary_proba[0, 1])  # P(fault)

        # Get optimal threshold from config
        threshold = self.config.get("optimal_binary_threshold", 0.1)
        is_anomaly = fault_probability >= threshold

        # Initialize result
        result = {
            "is_anomaly": is_anomaly,
            "anomaly_score": fault_probability,
            "binary_confidence": float(max(binary_proba[0])),
            "stage1_processing_time_ms": (time.time() - start_time) * 1000,
            "threshold_used": threshold,
            "anomaly_type": None,
            "fault_class": 0,
            "multiclass_confidence": 0.0,
            "affected_components": [],
            "stage2_processing_time_ms": 0.0,
        }

        # Stage 2: Multi-class fault classification (only if anomaly detected)
        if is_anomaly and self.multiclass_model is not None:
            stage2_start = time.time()

            try:
                multiclass_proba = self.multiclass_model.predict_proba(features_scaled)
                predicted_class = int(self.multiclass_model.predict(features_scaled)[0])
                class_confidence = float(max(multiclass_proba[0]))

                # Map to fault type and components
                anomaly_type = FAULT_TYPE_NAMES.get(predicted_class, "unknown_fault")
                affected_components = COMPONENT_MAPPING.get(predicted_class, [])

                result.update(
                    {
                        "anomaly_type": anomaly_type,
                        "fault_class": predicted_class,
                        "multiclass_confidence": class_confidence,
                        "affected_components": affected_components,
                        "stage2_processing_time_ms": (time.time() - stage2_start) * 1000,
                    }
                )

                logger.debug(
                    "Stage 2 prediction completed",
                    fault_class=predicted_class,
                    anomaly_type=anomaly_type,
                    confidence=class_confidence,
                    components=len(affected_components),
                )

            except Exception as e:
                logger.error("Stage 2 prediction failed", error=str(e))
                result["stage2_error"] = str(e)

        elif is_anomaly and self.multiclass_model is None:
            # Binary-only mode: generic fault classification
            result.update(
                {
                    "anomaly_type": "generic_fault",
                    "fault_class": 1,
                    "multiclass_confidence": fault_probability,
                    "affected_components": ["system_generic"],
                }
            )

        # Update performance tracking
        self.prediction_count += 1
        total_processing_time = (time.time() - start_time) * 1000
        result["total_processing_time_ms"] = total_processing_time

        logger.debug(
            "Two-stage prediction completed",
            is_anomaly=is_anomaly,
            anomaly_score=fault_probability,
            anomaly_type=result.get("anomaly_type"),
            processing_time_ms=total_processing_time,
        )

        return result

    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive model information including contract compatibility"""
        return {
            "is_loaded": self.is_loaded,
            "load_time_ms": self.load_time * 1000 if self.load_time else None,
            "prediction_count": self.prediction_count,
            "binary_model_available": self.binary_model is not None,
            "multiclass_model_available": self.multiclass_model is not None,
            "scaler_available": self.scaler is not None,
            "contract_compatibility": {
                "compatible": self.contract_compatible,
                "reason": self.compatibility_reason,
                "contract_version": self.feature_contract.get("contract_version") if self.feature_contract else None,
                "expected_features": self.feature_contract.get("features_count") if self.feature_contract else None,
                "scaler_features": getattr(self.scaler, "n_features_in_", None) if self.scaler else None,
            },
            "configuration": {
                "version": self.config.get("version", self.version) if self.config else self.version,
                "optimal_threshold": self.config.get("optimal_binary_threshold", 0.1) if self.config else 0.1,
                "best_stage2_variant": self.config.get("best_stage2_variant", "plain") if self.config else "plain",
                "component_mapping": COMPONENT_MAPPING,
                "fault_type_names": FAULT_TYPE_NAMES,
            },
        }

    def validate_prediction(self, features: np.ndarray, expected_class: int | None = None) -> dict[str, Any]:
        """Validate prediction and provide feedback for model improvement"""
        prediction = self.predict(features)

        validation_result = {"prediction": prediction, "validation_timestamp": time.time()}

        if expected_class is not None:
            predicted_class = prediction["fault_class"]
            is_correct = predicted_class == expected_class

            validation_result.update(
                {
                    "expected_class": expected_class,
                    "predicted_class": predicted_class,
                    "is_correct": is_correct,
                    "stage1_correct": (prediction["is_anomaly"]) == (expected_class != 0),
                    "stage2_correct": is_correct if expected_class != 0 else True,
                }
            )

            logger.info(
                "Prediction validated",
                expected=expected_class,
                predicted=predicted_class,
                correct=is_correct,
                anomaly_score=prediction["anomaly_score"],
            )

        return validation_result

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for monitoring"""
        return {
            "prediction_count": self.prediction_count,
            "is_loaded": self.is_loaded,
            "contract_compatible": self.contract_compatible,
            "load_time_ms": self.load_time * 1000 if self.load_time else None,
            "models_available": {
                "binary": self.binary_model is not None,
                "multiclass": self.multiclass_model is not None,
                "scaler": self.scaler is not None,
            },
            "configuration": self.config.get("version", self.version) if self.config else self.version,
            "compatibility_status": {"compatible": self.contract_compatible, "reason": self.compatibility_reason},
        }


# Global instance for API usage
_two_stage_classifier: TwoStageClassifier | None = None


def get_two_stage_classifier() -> TwoStageClassifier:
    """Get or create global two-stage classifier instance"""
    global _two_stage_classifier

    if _two_stage_classifier is None:
        _two_stage_classifier = TwoStageClassifier()

        # Try to load models with detailed logging
        logger.info("Initializing global Two-Stage classifier")
        if not _two_stage_classifier.load_models():
            logger.warning("Two-stage classifier models not loaded, will use fallback")

    return _two_stage_classifier


def reload_two_stage_models() -> bool:
    """Reload two-stage models (useful for model updates)"""
    global _two_stage_classifier

    if _two_stage_classifier is None:
        _two_stage_classifier = TwoStageClassifier()

    success = _two_stage_classifier.load_models()
    logger.info(
        "Two-Stage models reload attempt",
        success=success,
        contract_compatible=_two_stage_classifier.contract_compatible,
        compatibility_reason=_two_stage_classifier.compatibility_reason,
    )
    return success
