# Canonical 25 features (UCI Hydraulic cycles)
# Используется для проекции FeatureVector → фиксированный порядок признаков (25)
EXPECTED_FEATURE_NAMES_25 = [
    # Pressure (6)
    "pressure_mean",
    "pressure_std",
    "pressure_max",
    "pressure_min",
    "pressure_gradient",
    "autocorrelation_lag1",
    # Temperature (5)
    "temperature_mean",
    "temperature_std",
    "temperature_max",
    "temperature_min",
    "temperature_gradient",
    # Flow (5)
    "flow_mean",
    "flow_std",
    "flow_max",
    "flow_min",
    "flow_gradient",
    # Vibration & relations (7)
    "vibration_rms",
    "trend_slope",
    "temp_pressure_correlation",
    "pressure_flow_ratio",
    "system_efficiency",
    "cooling_efficiency",
    "cooling_power",
    # Motor (1)
    "motor_power_mean",
]
