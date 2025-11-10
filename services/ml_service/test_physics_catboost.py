"""
Test CatBoost on Physics-Based Hydraulic Data
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("="*70)
print("TESTING CATBOOST ON PHYSICS-BASED DATA")
print("="*70)
print()

# Load model
print("[1/3] Loading CatBoost model...")
model_data = joblib.load("models/catboost_model.joblib")
model = model_data["model"]
print(f"  ✅ Model loaded (25 features expected)")
print()

# Load physics test data
print("[2/3] Loading physics-based test data...")
df = pd.read_csv("physics_test.csv")
print(f"  ✅ Loaded {len(df)} samples")
print(f"  ✅ Normal: {len(df[df.label==0])}, Fault: {len(df[df.label==1])}")
print()

# Prepare features (model expects 25, we have 5 → need padding!)
print("[3/3] Preparing features...")
base_features = ['pressure_bar', 'temperature_celsius', 'vibration_mms', 'flow_lpm', 'speed_rpm']

# Create 25 features by engineering from base 5
X_test = df[base_features].values
print(f"  Base features: {X_test.shape}")

# Engineer 20 more features (realistic!)
X_engineered = np.zeros((len(df), 25))
X_engineered[:, :5] = X_test

# Feature engineering (based on physics)
X_engineered[:, 5] = X_test[:, 0]**2  # Pressure squared
X_engineered[:, 6] = X_test[:, 1]**2  # Temp squared
X_engineered[:, 7] = X_test[:, 2]**2  # Vib squared
X_engineered[:, 8] = X_test[:, 0] * X_test[:, 1]  # Pressure * Temp
X_engineered[:, 9] = X_test[:, 0] * X_test[:, 3]  # Pressure * Flow
X_engineered[:, 10] = X_test[:, 1] * X_test[:, 3]  # Temp * Flow
X_engineered[:, 11] = X_test[:, 0] / (X_test[:, 3] + 1)  # Pressure / Flow
X_engineered[:, 12] = X_test[:, 1] / (X_test[:, 4] + 1)  # Temp / Speed
X_engineered[:, 13] = X_test[:, 2] / (X_test[:, 4] + 1)  # Vib / Speed

# Statistical features
for i in range(5):
    X_engineered[:, 14+i] = np.log1p(X_test[:, i])  # Log transforms
    X_engineered[:, 19+i] = np.sqrt(np.abs(X_test[:, i]))  # Sqrt transforms

print(f"  Engineered features: {X_engineered.shape}")
print()

# Predict
print("PREDICTION:")
y_true = df['label'].values
y_pred = model.predict(X_engineered)
y_proba = model.predict_proba(X_engineered)[:, 1]

# Metrics
auc = roc_auc_score(y_true, y_proba)
cm = confusion_matrix(y_true, y_pred)

print()
print(f"  AUC: {auc:.4f}")
print()
print("Confusion Matrix:")
print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
print()

print(classification_report(y_true, y_pred, target_names=['Normal', 'Fault'], digits=4))

# Sample predictions
print()
print("SAMPLE PREDICTIONS (first 10):")
print()
for i in range(min(10, len(df))):
    row = df.iloc[i]
    print(f"Sample {i+1}:")
    print(f"  Pressure: {row.pressure_bar:.1f} bar, Temp: {row.temperature_celsius:.1f}°C, Vib: {row.vibration_mms:.1f} mm/s")
    print(f"  TRUE: {'FAULT' if y_true[i]==1 else 'NORMAL'} | PRED: {'FAULT' if y_pred[i]==1 else 'NORMAL'} (prob: {y_proba[i]:.3f})")
    print()

print("="*70)
if auc > 0.9:
    print("✅ MODEL PERFORMANCE: EXCELLENT!")
elif auc > 0.8:
    print("✅ MODEL PERFORMANCE: GOOD")
elif auc > 0.7:
    print("⚠️  MODEL PERFORMANCE: ACCEPTABLE")
else:
    print("❌ MODEL PERFORMANCE: NEEDS IMPROVEMENT")
print("="*70)
