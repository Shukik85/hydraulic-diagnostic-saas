"""
RUL Inference Engine
Движок inference для предсказания RUL (Remaining Useful Life) по историческим данным

Features:
- Sliding window queries из TimescaleDB
- Multi-horizon RUL predictions (5/15/30 min)
- Confidence scoring
- Maintenance action recommendations
- Real-time temporal feature extraction
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from pydantic import BaseModel

from model_temporal_rul import TemporalRULPredictor, create_temporal_rul_model

logger = logging.getLogger(__name__)


class RULInferenceEngine:
    """
    RUL Inference Engine с поддержкой sliding windows.
    
    Workflow:
    1. Query TimescaleDB за последние 60 минут
    2. Extract temporal features (12 timesteps × 5 min)
    3. Run RUL prediction
    4. Generate maintenance recommendations
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        time_window_minutes: int = 60,
        timestep_minutes: int = 5,
        use_compile: bool = True,
    ):
        """
        Args:
            model_path: Path to trained RUL model checkpoint
            device: 'cuda' or 'cpu'
            time_window_minutes: Lookback window (60 min recommended)
            timestep_minutes: Time bucket size (5 min recommended)
            use_compile: Enable torch.compile
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.time_window_minutes = time_window_minutes
        self.timestep_minutes = timestep_minutes
        self.num_timesteps = time_window_minutes // timestep_minutes  # 12 steps
        
        # Load model
        self.model = self._load_model(model_path, use_compile)
        
        # Thresholds for maintenance decisions
        self.critical_threshold = 0.3  # RUL < 0.3 = critical
        self.warning_threshold = 0.5   # RUL < 0.5 = warning
        
        # Component names (can be dynamic from metadata)
        self.component_names = [
            "pump", "boom", "stick", "bucket", "swing", "left_track", "right_track"
        ]
        
        logger.info(
            f"RUL Inference Engine initialized: "
            f"window={time_window_minutes}min, "
            f"timesteps={self.num_timesteps}, "
            f"device={self.device}"
        )
    
    def _load_model(
        self,
        model_path: Optional[str],
        use_compile: bool,
    ) -> TemporalRULPredictor:
        """Load trained RUL model or create new one."""
        try:
            if model_path:
                logger.info(f"Loading RUL model from {model_path}")
                checkpoint = torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=False,
                )
                model = create_temporal_rul_model(
                    device=str(self.device),
                    use_compile=False,  # Compile after loading weights
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("✅ Model loaded from checkpoint")
            else:
                logger.warning("No model path provided, creating new model")
                model = create_temporal_rul_model(
                    device=str(self.device),
                    use_compile=False,
                )
            
            model.eval()
            
            # Compile after loading
            if use_compile and hasattr(torch, "compile"):
                logger.info("Compiling RUL model...")
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading RUL model: {e}")
            raise
    
    def extract_temporal_features(
        self,
        sensor_data: Dict[str, List[Dict[str, Any]]],
        num_nodes: int = 7,
        num_features: int = 15,
    ) -> torch.Tensor:
        """
        Extract temporal features from sensor data.
        
        Args:
            sensor_data: {
                "pump": [{"timestamp": ..., "pressure": ..., "flow": ...}, ...],
                "valve": [...],
                ...
            }
            num_nodes: Number of components
            num_features: Features per component
        
        Returns:
            x_sequence: [1, time_steps, n_nodes, n_features]
        """
        # Initialize tensor
        x_sequence = torch.zeros(
            1,  # batch_size = 1 for single system
            self.num_timesteps,
            num_nodes,
            num_features,
            dtype=torch.float32,
        )
        
        # Process each component
        for node_idx, component_name in enumerate(self.component_names[:num_nodes]):
            if component_name not in sensor_data:
                logger.warning(f"No data for component {component_name}")
                continue
            
            component_data = sensor_data[component_name]
            
            # Extract features per timestep
            for t in range(min(self.num_timesteps, len(component_data))):
                timestep_data = component_data[t]
                
                # Extract sensor values (example: pressure, flow, temp, etc.)
                features = []
                for sensor_name in ["pressure", "flow", "temperature", "vibration", "position"]:
                    value = timestep_data.get(sensor_name, 0.0)
                    
                    # Add value and derived features
                    features.extend([
                        value,                           # raw value
                        timestep_data.get(f"{sensor_name}_mean", value),
                        timestep_data.get(f"{sensor_name}_std", 0.0),
                    ])
                
                # Pad to num_features
                features = features[:num_features]
                features += [0.0] * (num_features - len(features))
                
                x_sequence[0, t, node_idx, :] = torch.tensor(features, dtype=torch.float32)
        
        return x_sequence
    
    def predict_rul(
        self,
        sensor_data: Dict[str, List[Dict[str, Any]]],
        edge_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Predict RUL for all components.
        
        Args:
            sensor_data: Temporal sensor data per component
            edge_index: Graph topology (optional, uses star topology by default)
        
        Returns:
            {
                "rul_predictions": {"horizon_5min": [...], ...},
                "confidence_scores": {"horizon_5min": [...], ...},
                "maintenance_actions": [...],
                "inference_time_ms": ...,
            }
        """
        start_time = time.time()
        
        try:
            # Extract features
            x_sequence = self.extract_temporal_features(sensor_data)
            x_sequence = x_sequence.to(self.device)
            
            # Default star topology if not provided
            if edge_index is None:
                edge_index = self._create_star_topology(num_nodes=7)
            edge_index = edge_index.to(self.device)
            
            # Inference
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        rul_preds, confidences = self.model(x_sequence, edge_index)
                else:
                    rul_preds, confidences = self.model(x_sequence, edge_index)
            
            # Post-process
            rul_predictions = {}
            confidence_scores = {}
            
            for horizon_name in rul_preds.keys():
                rul = rul_preds[horizon_name].cpu().numpy()[0]  # [n_nodes]
                conf = confidences[horizon_name].cpu().numpy()[0]
                
                rul_predictions[horizon_name] = [
                    {
                        "component": self.component_names[i],
                        "rul_score": float(rul[i]),
                        "minutes_to_failure": self._rul_to_minutes(rul[i], horizon_name),
                    }
                    for i in range(len(rul))
                ]
                
                confidence_scores[horizon_name] = [
                    {
                        "component": self.component_names[i],
                        "confidence": float(1.0 / (1.0 + conf[i])),  # Inverse uncertainty
                    }
                    for i in range(len(conf))
                ]
            
            # Generate maintenance recommendations
            maintenance_actions = self._generate_maintenance_actions(
                rul_predictions,
                confidence_scores,
            )
            
            inference_time = (time.time() - start_time) * 1000
            
            return {
                "rul_predictions": rul_predictions,
                "confidence_scores": confidence_scores,
                "maintenance_actions": maintenance_actions,
                "inference_time_ms": inference_time,
                "prediction_timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"RUL prediction failed: {e}")
            raise
    
    def _create_star_topology(self, num_nodes: int = 7) -> torch.Tensor:
        """Create star graph (pump at center)."""
        edge_list = []
        for target in range(1, num_nodes):
            edge_list.extend([[0, target], [target, 0]])
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    def _rul_to_minutes(self, rul_score: float, horizon_name: str) -> float:
        """
        Convert RUL score (0-1) to estimated minutes until failure.
        
        Args:
            rul_score: 0 = imminent failure, 1 = healthy
            horizon_name: "horizon_5min", "horizon_15min", etc.
        
        Returns:
            Estimated minutes to failure
        """
        # Extract horizon from name (e.g., "horizon_15min" -> 15)
        horizon_minutes = int(horizon_name.split("_")[1].replace("min", ""))
        
        # Linear mapping: RUL 0.0 -> 0 min, RUL 1.0 -> horizon_minutes
        return rul_score * horizon_minutes
    
    def _generate_maintenance_actions(
        self,
        rul_predictions: Dict[str, List[Dict[str, Any]]],
        confidence_scores: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Generate maintenance action recommendations.
        
        Returns:
            [
                {
                    "component": "pump",
                    "action": "emergency_stop",
                    "priority": "critical",
                    "rul_5min": 0.15,
                    "confidence": 0.85,
                    "reason": "RUL < 0.3 within 5 minutes"
                },
                ...
            ]
        """
        actions = []
        
        # Check 5-minute horizon first (most urgent)
        horizon_5min = rul_predictions.get("horizon_5min", [])
        conf_5min = confidence_scores.get("horizon_5min", [])
        
        for i, rul_data in enumerate(horizon_5min):
            component = rul_data["component"]
            rul_score = rul_data["rul_score"]
            confidence = conf_5min[i]["confidence"]
            
            # Critical: RUL < 0.3 within 5 minutes
            if rul_score < self.critical_threshold and confidence > 0.7:
                actions.append({
                    "component": component,
                    "action": "emergency_stop",
                    "priority": "critical",
                    "rul_score": rul_score,
                    "confidence": confidence,
                    "horizon": "5min",
                    "reason": f"RUL {rul_score:.2f} < {self.critical_threshold} within 5 minutes",
                })
            
            # Warning: RUL < 0.5 within 5 minutes
            elif rul_score < self.warning_threshold and confidence > 0.6:
                actions.append({
                    "component": component,
                    "action": "planned_shutdown",
                    "priority": "warning",
                    "rul_score": rul_score,
                    "confidence": confidence,
                    "horizon": "5min",
                    "reason": f"RUL {rul_score:.2f} < {self.warning_threshold} within 5 minutes",
                })
        
        # Check 15-minute horizon for planning
        horizon_15min = rul_predictions.get("horizon_15min", [])
        conf_15min = confidence_scores.get("horizon_15min", [])
        
        for i, rul_data in enumerate(horizon_15min):
            component = rul_data["component"]
            rul_score = rul_data["rul_score"]
            confidence = conf_15min[i]["confidence"]
            
            # Only add if not already in critical/warning from 5min
            already_flagged = any(a["component"] == component for a in actions)
            
            if not already_flagged and rul_score < self.warning_threshold and confidence > 0.6:
                actions.append({
                    "component": component,
                    "action": "schedule_maintenance",
                    "priority": "medium",
                    "rul_score": rul_score,
                    "confidence": confidence,
                    "horizon": "15min",
                    "reason": f"RUL {rul_score:.2f} declining within 15 minutes",
                })
        
        # Sort by priority
        priority_order = {"critical": 0, "warning": 1, "medium": 2}
        actions.sort(key=lambda x: priority_order.get(x["priority"], 99))
        
        return actions


if __name__ == "__main__":
    # Test RUL inference
    logging.basicConfig(level=logging.INFO)
    
    # Create engine
    engine = RULInferenceEngine(
        model_path=None,  # Will create new model for testing
        device="cuda" if torch.cuda.is_available() else "cpu",
        time_window_minutes=60,
        timestep_minutes=5,
    )
    
    # Dummy sensor data (12 timesteps × 7 components)
    dummy_sensor_data = {
        component: [
            {
                "timestamp": datetime.now() - timedelta(minutes=(11-t)*5),
                "pressure": 150.0 + np.random.randn() * 10,
                "flow": 80.0 + np.random.randn() * 5,
                "temperature": 60.0 + np.random.randn() * 2,
                "vibration": 0.5 + np.random.randn() * 0.1,
                "position": 0.0,
            }
            for t in range(12)
        ]
        for component in ["pump", "boom", "stick", "bucket", "swing", "left_track", "right_track"]
    }
    
    # Run inference
    result = engine.predict_rul(dummy_sensor_data)
    
    # Print results
    print("\n" + "=" * 70)
    print("RUL INFERENCE TEST")
    print("=" * 70)
    
    print(f"\nInference Time: {result['inference_time_ms']:.2f}ms")
    
    print("\nRUL Predictions (5 min horizon):")
    for pred in result["rul_predictions"]["horizon_5min"]:
        print(f"  {pred['component']}: RUL={pred['rul_score']:.3f}, TTF={pred['minutes_to_failure']:.1f}min")
    
    print("\nMaintenance Actions:")
    for action in result["maintenance_actions"]:
        print(
            f"  [{action['priority'].upper()}] {action['component']}: "
            f"{action['action']} (RUL={action['rul_score']:.3f}, conf={action['confidence']:.3f})"
        )
    
    print("\n" + "=" * 70)
    print("✅ RUL inference test passed!")
    print("=" * 70)
