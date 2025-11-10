"""Check CatBoost expected features"""
import joblib

model_data = joblib.load("models/catboost_model.joblib")
model = model_data["model"] if isinstance(model_data, dict) else model_data

# CatBoost методы
print(f"Feature names: {model.feature_names_}")
print(f"Number of features: {len(model.feature_names_)}")
print()

# Также проверим metadata если есть
if isinstance(model_data, dict):
    print("Model metadata:")
    for key, value in model_data.items():
        if key != 'model':
            print(f"  {key}: {value}")
