"""Quick sanity check - CatBoost without temporal features"""
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

print("="*60)
print("CATBOOST SANITY CHECK - Simple Features Only")
print("="*60)
print()

# Load model
print("Loading CatBoost...")
model_data = joblib.load("models/catboost_model.joblib")
model = model_data["model"] if isinstance(model_data, dict) else model_data

# Load simple data (NO temporal features!)
print("Loading data...")
df = pd.read_csv("data/industrial_iot/Industrial_fault_detection.csv")

# Only numeric, NO timestamp
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
target_col = numeric_cols[-1]
feature_cols = numeric_cols[:-1]

print(f"  Features: {len(feature_cols)} (simple, NO lag/rolling)")
print(f"  Target: {target_col}")

X = df[feature_cols].values
y = df[target_col].values

# Simple split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print()

# Predict
print("Testing...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)

print()
print("RESULTS:")
print(f"  AUC: {auc:.4f}")
print()
print(classification_report(y_test, y_pred, digits=4))
print()

if auc > 0.95:
    print("✅ Model looks GOOD (but might still have some leakage)")
elif auc > 0.85:
    print("✅ Model is ACCEPTABLE")
elif auc > 0.70:
    print("⚠️  Model is MODERATE (needs improvement)")
else:
    print("❌ Model is POOR (data leakage or bad training)")

print()
print("="*60)
