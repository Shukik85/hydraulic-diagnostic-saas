"""
Sensor validation configuration for Hydraulic Diagnostic Platform.

Industry-standard ranges for hydraulic system sensors based on ISO 4413:2010.
"""

# Sensor value ranges by unit type
# Based on ISO 4413:2010 and industrial hydraulic system standards

SENSOR_VALUE_RANGES = {
    # Pressure sensors (bar)
    "bar": {
        "min": 0.0,
        "max": 700.0,  # Max for high-pressure hydraulic systems
        "typical_min": 50.0,
        "typical_max": 350.0,
    },
    
    # Temperature sensors (Celsius)
    "celsius": {
        "min": -40.0,  # Cold start conditions
        "max": 120.0,  # Max safe oil temperature
        "typical_min": 20.0,
        "typical_max": 80.0,
    },
    
    # Flow rate sensors (liters per minute)
    "lpm": {
        "min": 0.0,
        "max": 1000.0,  # High-capacity pump systems
        "typical_min": 10.0,
        "typical_max": 500.0,
    },
    
    # Rotational speed (RPM)
    "rpm": {
        "min": 0.0,
        "max": 6000.0,  # High-speed pump motors
        "typical_min": 500.0,
        "typical_max": 3000.0,
    },
}

# Quality score thresholds
QUALITY_THRESHOLDS = {
    "good": 90,      # Quality >= 90: Good data
    "acceptable": 70,  # Quality >= 70: Acceptable
    "poor": 50,      # Quality >= 50: Poor quality
    # Quality < 50: Reject and quarantine
}

# Timestamp validation settings (in seconds)
TIMESTAMP_VALIDATION = {
    "max_future_offset": 300,  # 5 minutes into future allowed
    "max_past_offset": 157680000,  # 5 years into past (retention policy)
}

# Duplicate detection window (in seconds)
DUPLICATE_DETECTION_WINDOW = 1  # 1 second window for duplicate detection

# Quarantine settings
QUARANTINE_CONFIG = {
    "auto_quarantine_reasons": [
        "out_of_range",
        "invalid_timestamp",
        "parse_error",
        "system_not_found",
    ],
    "require_manual_review": [
        "duplicate",
        "invalid_unit",
    ],
}
