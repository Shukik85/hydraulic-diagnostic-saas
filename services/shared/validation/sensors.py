"""
Sensor data validation utilities.
"""

from typing import Dict, List
from ..schemas import SensorData, SensorType


def validate_sensor_data(data: Dict) -> bool:
    """
    Validate sensor data structure.
    
    Args:
        data: Raw sensor data dict
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["sensor_id", "timestamp", "value", "unit"]
    return all(field in data for field in required_fields)


def validate_pressure_range(pressure: float, unit: str = "bar") -> bool:
    """
    Validate pressure is within acceptable range.
    
    Args:
        pressure: Pressure value
        unit: Unit of measurement (bar, psi, kPa)
        
    Returns:
        True if within range
    """
    ranges = {
        "bar": (0, 500),
        "psi": (0, 7250),
        "kPa": (0, 50000)
    }
    
    min_val, max_val = ranges.get(unit, (0, 1000))
    return min_val <= pressure <= max_val


def validate_temperature_range(temp: float, unit: str = "C") -> bool:
    """
    Validate temperature is within acceptable range.
    
    Args:
        temp: Temperature value
        unit: Unit (C, F, K)
        
    Returns:
        True if within range
    """
    ranges = {
        "C": (-40, 150),
        "F": (-40, 302),
        "K": (233, 423)
    }
    
    min_val, max_val = ranges.get(unit, (-100, 200))
    return min_val <= temp <= max_val


def validate_sensor_quality(quality: float) -> bool:
    """Validate sensor quality score."""
    return 0.0 <= quality <= 1.0


def validate_sensor_batch(sensors: List[SensorData]) -> Dict[str, List[str]]:
    """
    Validate batch of sensor data.
    
    Returns:
        Dict with 'valid' and 'invalid' sensor IDs
    """
    valid = []
    invalid = []
    
    for sensor in sensors:
        try:
            # Type-specific validation
            if sensor.sensor_type == SensorType.PRESSURE:
                is_valid = validate_pressure_range(sensor.value, sensor.unit)
            elif sensor.sensor_type == SensorType.TEMPERATURE:
                is_valid = validate_temperature_range(sensor.value, sensor.unit)
            else:
                is_valid = True
            
            # Quality validation
            is_valid = is_valid and validate_sensor_quality(sensor.quality)
            
            if is_valid:
                valid.append(sensor.sensor_id)
            else:
                invalid.append(sensor.sensor_id)
                
        except Exception:
            invalid.append(sensor.sensor_id)
    
    return {
        "valid": valid,
        "invalid": invalid,
        "total": len(sensors),
        "valid_count": len(valid),
        "invalid_count": len(invalid)
    }
