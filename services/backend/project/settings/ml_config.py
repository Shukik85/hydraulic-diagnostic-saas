"""
ML Service Configuration for UCI Hydraulic Models.

Configuration settings for integration with retrained ML models
based on real UCI Machine Learning Repository hydraulic dataset.
"""

# ML Service URLs and Endpoints
ML_SERVICE_URL = 'http://ml-service:8001'
ML_PREDICT_ENDPOINT = '/predict'  # Can be updated to /predict/v2 or /predict/uci
ML_HEALTH_ENDPOINT = '/health'
ML_MODELS_STATUS_ENDPOINT = '/models/status'

# Model Preferences and Thresholds
ML_MODEL_PREFERENCE = 'ensemble'  # Options: 'catboost', 'xgboost', 'rf', 'adaptive', 'ensemble'
ML_CONFIDENCE_THRESHOLD = 0.75     # Minimum confidence for "good" predictions
ML_ANOMALY_THRESHOLD = 0.8         # Threshold for anomaly detection

# Timeouts and Retries
ML_REQUEST_TIMEOUT_SECONDS = 15.0  # Extended for real model inference
ML_REQUEST_RETRIES = 2
ML_COLD_START_TIMEOUT = 30.0       # First request after model loading

# Feature Engineering Configuration
UCI_FEATURE_MAPPING = {
    # Our sensor names -> UCI feature names
    'system_pressure': 'PS1',
    'pressure_secondary': 'PS2', 
    'oil_temperature': 'TS1',
    'coolant_temperature': 'TS2',
    'flow_rate': 'FS1',
    'return_flow': 'FS2',
    'vibration_level': 'VS1',
    'motor_speed': 'EPS1',
    'motor_power': 'EPS1'
}

# Sensor Value Normalization
NORMALIZATION_RANGES = {
    # UCI dataset typical ranges for feature scaling
    'PS1': (0, 300),      # Pressure: 0-300 bar
    'PS2': (0, 300),
    'PS3': (0, 300), 
    'PS4': (0, 300),
    'PS5': (0, 300),
    'PS6': (0, 300),
    'TS1': (-10, 120),    # Temperature: -10 to 120°C
    'TS2': (-10, 120),
    'TS3': (-10, 120),
    'TS4': (-10, 120),
    'FS1': (0, 100),      # Flow: 0-100 L/min
    'FS2': (0, 100),
    'VS1': (0, 20),       # Vibration: 0-20 mm/s
    'EPS1': (0, 100),     # Motor power: 0-100%
    'CE': (0, 100),       # Cooling efficiency: 0-100%
    'CP': (0, 100),       # Cooling power: 0-100%
    'SE': (0, 100)        # Stability efficiency: 0-100%
}

# Quality Gates for Production
MIN_FEATURES_FOR_ML = 3           # Minimum number of features required
REQUIRED_UCI_FEATURES = ['PS1']   # Features that must be present
PREFERRED_UCI_FEATURES = ['PS1', 'TS1', 'FS1', 'VS1']  # Optimal feature set

# Alert Thresholds
FAULT_PROBABILITY_ALERT = 0.8     # Send alert if fault probability > 80%
CONFIDENCE_WARNING = 0.6          # Warn if confidence < 60%
ANOMALY_SCORE_CRITICAL = 0.9      # Critical alert threshold

# Model Performance Monitoring
MODEL_ACCURACY_THRESHOLD = 0.85   # Expected minimum accuracy
PREDICTION_LATENCY_WARNING_MS = 5000  # Warn if ML prediction takes >5s
MODEL_DRIFT_CHECK_INTERVAL_HOURS = 24 # Check for model drift daily

# Batch Processing Configuration  
ML_BATCH_SIZE = 50                # Maximum readings to analyze in one ML call
ML_BATCH_TIMEOUT_SECONDS = 30.0   # Timeout for batch predictions

# Feature Engineering Settings
COMPUTE_DERIVED_FEATURES = True   # Compute CE, CP, SE from base sensors
FEATURE_SMOOTHING_WINDOW = 3      # Number of recent readings to smooth
OUTLIER_DETECTION_ENABLED = True  # Enable statistical outlier detection

# Development Settings
ML_MOCK_MODE = False              # Use mock predictions (set True for testing)
ML_LOG_REQUESTS = True            # Log all ML service requests in debug
ML_CACHE_PREDICTIONS = False      # Cache predictions (not recommended for real-time)
