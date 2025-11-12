# services/gnn_service/feature_engineering.py
"""
Feature engineering utilities (normalization, imputation, trend extraction) for dynamic GNN pipeline.
"""
def normalize(value, nominal, min_val, max_val):
    return (value - nominal) / (max_val - min_val) if max_val > min_val else 0.0

def deviation(value, nominal):
    return (value - nominal) / nominal if nominal != 0 else 0.0

def fill_missing(sensor_values, default=0.0):
    return [v if v is not None else default for v in sensor_values]

def rolling_trend(sequence, window=3):
    # Simple trend for small samples
    if len(sequence) < window:
        return 'stable'
    trend = sequence[-1] - sequence[-window]
    if trend > 0.05:
        return 'increasing'
    elif trend < -0.05:
        return 'decreasing'
    else:
        return 'stable'
