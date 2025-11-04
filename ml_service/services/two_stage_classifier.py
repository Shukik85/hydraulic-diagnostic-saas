#!/usr/bin/env python3
"""
Two-Stage Classifier Service for Enhanced Anomaly Detection
Stage 1: Binary anomaly detection (normal vs fault) - High Recall
Stage 2: Multi-class fault classification (pump/valve/cooling) - High Precision
Provides actionable insights with component mapping and confidence scoring
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import structlog
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

# Component mapping for actionable maintenance insights
COMPONENT_MAPPING = {
    1: ["pump_main", "pump_motor"],
    2: ["valve_main", "valve_control"],
    3: ["cooler", "heat_exchanger", "cooling_system"]
}

FAULT_TYPE_NAMES = {
    0: "normal",
    1: "pump_fault",
    2: "valve_fault", 
    3: "cooling_fault"
}


class TwoStageClassifier:
    """Two-stage classifier for enhanced anomaly detection and fault typing"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.binary_model = None
        self.multiclass_model = None
        self.scaler = None
        self.config = None
        self.is_loaded = False
        
        # Performance tracking
        self.prediction_count = 0
        self.load_time = None
        
        logger.info("TwoStageClassifier initialized", models_dir=str(self.models_dir))
    
    def load_models(self) -> bool:
        """Load two-stage models and configuration"""
        try:
            start_time = time.time()
            
            # Load configuration
            config_path = self.models_dir / "training_summary.json"
            if config_path.exists():
                with config_path.open("r") as f:
                    self.config = json.load(f)
                logger.info("Configuration loaded", 
                           version=self.config.get("version", "unknown"),
                           optimal_threshold=self.config.get("optimal_binary_threshold", 0.5),
                           best_stage2=self.config.get("best_stage2_variant", "plain"))
            else:
                logger.warning("No configuration found, using defaults")
                self.config = {
                    "optimal_binary_threshold": 0.35,
                    "best_stage2_variant": "plain"
                }
            
            # Load feature scaler
            scaler_path = self.models_dir / "feature_scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded", path=str(scaler_path))
            else:
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            
            # Load Stage 1 binary model (compatibility: try multiple names)
            binary_paths = [
                self.models_dir / "binary_detector_xgb.joblib",
                self.models_dir / "catboost_model.joblib",  # fallback compatibility
                self.models_dir / "xgboost_model.joblib"  # fallback compatibility
            ]
            
            binary_loaded = False
            for binary_path in binary_paths:
                if binary_path.exists():
                    self.binary_model = joblib.load(binary_path)
                    logger.info("Stage 1 binary model loaded", path=str(binary_path))
                    binary_loaded = True
                    break
            
            if not binary_loaded:
                raise FileNotFoundError("No Stage 1 binary model found")
            
            # Load Stage 2 multiclass model
            best_variant = self.config.get("best_stage2_variant", "plain")
            multiclass_suffix = "_smote" if best_variant == "smote" else ""
            multiclass_path = self.models_dir / f"fault_classifier_catboost{multiclass_suffix}.joblib"
            
            if multiclass_path.exists():
                self.multiclass_model = joblib.load(multiclass_path)
                logger.info("Stage 2 multiclass model loaded", 
                           path=str(multiclass_path),
                           variant=best_variant)
            else:
                logger.warning("Stage 2 model not found, binary-only mode", path=str(multiclass_path))
                self.multiclass_model = None
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info("Two-stage classifier loaded successfully",
                       load_time_ms=self.load_time * 1000,
                       binary_available=self.binary_model is not None,
                       multiclass_available=self.multiclass_model is not None)
            
            return True
            
        except Exception as e:
            logger.error("Failed to load two-stage models", error=str(e))
            self.is_loaded = False
            return False
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Two-stage prediction with actionable insights"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        start_time = time.time()
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Stage 1: Binary anomaly detection
        binary_proba = self.binary_model.predict_proba(features_scaled)
        fault_probability = float(binary_proba[0, 1])  # P(fault)
        
        # Get optimal threshold from config
        threshold = self.config.get("optimal_binary_threshold", 0.35)
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
            "stage2_processing_time_ms": 0.0
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
                
                result.update({
                    "anomaly_type": anomaly_type,
                    "fault_class": predicted_class,
                    "multiclass_confidence": class_confidence,
                    "affected_components": affected_components,
                    "stage2_processing_time_ms": (time.time() - stage2_start) * 1000
                })
                
                logger.debug("Stage 2 prediction completed",
                           fault_class=predicted_class,
                           anomaly_type=anomaly_type,
                           confidence=class_confidence,
                           components=len(affected_components))
                
            except Exception as e:
                logger.error("Stage 2 prediction failed", error=str(e))
                result["stage2_error"] = str(e)
        
        elif is_anomaly and self.multiclass_model is None:
            # Binary-only mode: generic fault classification
            result.update({
                "anomaly_type": "generic_fault",
                "fault_class": 1,
                "multiclass_confidence": fault_probability,
                "affected_components": ["system_generic"]
            })
        
        # Update performance tracking
        self.prediction_count += 1
        total_processing_time = (time.time() - start_time) * 1000
        result["total_processing_time_ms"] = total_processing_time
        
        logger.debug("Two-stage prediction completed",
                    is_anomaly=is_anomaly,
                    anomaly_score=fault_probability,
                    anomaly_type=result.get("anomaly_type"),
                    processing_time_ms=total_processing_time)
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "is_loaded": self.is_loaded,
            "load_time_ms": self.load_time * 1000 if self.load_time else None,
            "prediction_count": self.prediction_count,
            "binary_model_available": self.binary_model is not None,
            "multiclass_model_available": self.multiclass_model is not None,
            "scaler_available": self.scaler is not None,
            "configuration": {
                "version": self.config.get("version", "unknown") if self.config else "unknown",
                "optimal_threshold": self.config.get("optimal_binary_threshold", 0.35) if self.config else 0.35,
                "best_stage2_variant": self.config.get("best_stage2_variant", "plain") if self.config else "plain",
                "component_mapping": COMPONENT_MAPPING,
                "fault_type_names": FAULT_TYPE_NAMES
            }
        }
    
    def validate_prediction(self, features: np.ndarray, 
                           expected_class: Optional[int] = None) -> Dict[str, Any]:
        """Validate prediction and provide feedback for model improvement"""
        prediction = self.predict(features)
        
        validation_result = {
            "prediction": prediction,
            "validation_timestamp": time.time()
        }
        
        if expected_class is not None:
            predicted_class = prediction["fault_class"]
            is_correct = predicted_class == expected_class
            
            validation_result.update({
                "expected_class": expected_class,
                "predicted_class": predicted_class,
                "is_correct": is_correct,
                "stage1_correct": (prediction["is_anomaly"]) == (expected_class != 0),
                "stage2_correct": is_correct if expected_class != 0 else True
            })
            
            logger.info("Prediction validated",
                       expected=expected_class,
                       predicted=predicted_class,
                       correct=is_correct,
                       anomaly_score=prediction["anomaly_score"])
        
        return validation_result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        return {
            "prediction_count": self.prediction_count,
            "is_loaded": self.is_loaded,
            "load_time_ms": self.load_time * 1000 if self.load_time else None,
            "models_available": {
                "binary": self.binary_model is not None,
                "multiclass": self.multiclass_model is not None,
                "scaler": self.scaler is not None
            },
            "configuration": self.config.get("version", "unknown") if self.config else "unknown"
        }


# Global instance for API usage
_two_stage_classifier: Optional[TwoStageClassifier] = None


def get_two_stage_classifier() -> TwoStageClassifier:
    """Get or create global two-stage classifier instance"""
    global _two_stage_classifier
    
    if _two_stage_classifier is None:
        _two_stage_classifier = TwoStageClassifier()
        
        # Try to load models
        if not _two_stage_classifier.load_models():
            logger.warning("Two-stage classifier models not loaded, will use fallback")
    
    return _two_stage_classifier


def reload_two_stage_models() -> bool:
    """Reload two-stage models (useful for model updates)"""
    global _two_stage_classifier
    
    if _two_stage_classifier is None:
        _two_stage_classifier = TwoStageClassifier()
    
    return _two_stage_classifier.load_models()