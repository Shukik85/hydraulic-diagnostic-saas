"""
Celery tasks for ML inference with equipment-aware normalization
"""
from celery import Celery
import torch
import json
from datetime import datetime
from django.core.cache import cache

app = Celery("ml_tasks", broker="redis://localhost:6379/0")

# Model cache
MODEL_CACHE = {}


@app.task(name="ml.predict_anomaly", bind=True, max_retries=3)
def predict_anomaly(self, sensor_data, equipment_id, component_type, config_json):
    """
    Equipment-aware anomaly prediction
    
    Args:
        sensor_data: Raw sensor readings dict
        equipment_id: Equipment identifier
        component_type: Component type (cylinder, pump, valve)
        config_json: Equipment configuration dict
    """
    try:
        # Normalize sensor data using equipment config
        normalized_data = normalize_sensor_data(
            sensor_data, config_json, component_type
        )
        
        # Load model (cached)
        model = _get_model(equipment_id, component_type)
        
        # ML prediction
        with torch.no_grad():
            tensor_data = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0)
            ml_output = model(tensor_data)
            ml_confidence = torch.sigmoid(ml_output).item()
        
        # Physics-based threshold checking
        physics_alerts = check_physics_thresholds(
            sensor_data, config_json, component_type
        )
        
        # Combine ML + Physics
        anomaly_detected = ml_confidence > 0.7 or len(physics_alerts) > 0
        
        result = {
            "anomaly_detected": anomaly_detected,
            "ml_confidence": ml_confidence,
            "physics_alerts": physics_alerts,
            "equipment_id": equipment_id,
            "component_type": component_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store result (async)
        store_inference_result.delay(result)
        
        # Send alert if anomaly
        if anomaly_detected:
            send_alert.delay(result)
        
        return result
        
    except Exception as e:
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)


def normalize_sensor_data(sensor_data, config_json, component_type):
    """Normalize using equipment-specific thresholds"""
    normalized = {}
    
    if component_type == "cylinder":
        cyl_config = config_json["components"]["boom_cylinder"]
        
        normalized["pressure_extend"] = (
            sensor_data["pressure_extend"] / cyl_config["max_pressure"]
        )
        normalized["pressure_retract"] = (
            sensor_data["pressure_retract"] / cyl_config["max_pressure"]
        )
        normalized["position"] = (
            sensor_data["position"] / (cyl_config["stroke"] / 1000)
        )
        normalized["velocity"] = (
            sensor_data["velocity"] / cyl_config["thresholds"]["velocity_max"]
        )
        # Add other features...
    
    elif component_type == "pump":
        pump_config = config_json["components"]["pump"]
        
        normalized["pressure_outlet"] = (
            sensor_data["pressure_outlet"] / pump_config["max_pressure"]
        )
        normalized["speed_rpm"] = (
            sensor_data["speed_rpm"] / pump_config["nominal_rpm"]
        )
        # Add other features...
    
    return list(normalized.values())


def check_physics_thresholds(sensor_data, config_json, component_type):
    """Physics-based threshold checking"""
    alerts = []
    
    if component_type == "cylinder":
        cyl_config = config_json["components"]["boom_cylinder"]
        thresholds = cyl_config["thresholds"]
        
        # Pressure checks
        if sensor_data["pressure_extend"] > thresholds["pressure_extend"]["max"]:
            alerts.append({
                "type": "overpressure_extend",
                "value": sensor_data["pressure_extend"],
                "threshold": thresholds["pressure_extend"]["max"],
                "severity": "critical"
            })
        
        # Pressure diff
        pressure_diff = abs(
            sensor_data["pressure_extend"] - sensor_data["pressure_retract"]
        )
        if pressure_diff > thresholds["pressure_diff_max"]:
            alerts.append({
                "type": "pressure_imbalance",
                "value": pressure_diff,
                "threshold": thresholds["pressure_diff_max"],
                "severity": "warning"
            })
    
    return alerts


def _get_model(equipment_id, component_type):
    """Load model from cache or MLflow"""
    cache_key = f"{equipment_id}:{component_type}"
    
    if cache_key not in MODEL_CACHE:
        # TODO: Load from MLflow
        # For now, load from checkpoint
        from ml_service.experiments.ssl_transformer.models.component_models import (
            CylinderModel, PumpModel
        )
        
        if component_type == "cylinder":
            model = CylinderModel(hidden_dim=64)
        else:
            model = PumpModel(hidden_dim=64)
        
        # Load weights
        checkpoint_path = f"ml_service/experiments/ssl_transformer/checkpoints/{component_type}_best.pt"
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        MODEL_CACHE[cache_key] = model
    
    return MODEL_CACHE[cache_key]


@app.task(name="ml.store_result")
def store_inference_result(result):
    """Store inference result in TimescaleDB"""
    # TODO: Implement TimescaleDB storage
    pass


@app.task(name="alerts.send")
def send_alert(result):
    """Send alert via multiple channels"""
    # TODO: Implement WebSocket, email, SMS alerts
    pass
