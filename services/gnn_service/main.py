# services/gnn_service/main.py
"""
FastAPI entrypoint для Universal Dynamic GNN Service.
"""
from fastapi import FastAPI, HTTPException
from schemas import PredictFromFeaturesRequest, GNNPredictionResponse
from inference_dynamic import DynamicGNNInference
import torch
import os

app = FastAPI(title="Universal GNN Diagnostics API (Dynamic)")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/universal_dynamic_best.ckpt")
METADATA_PATH = os.environ.get("METADATA_PATH", "data/system_metadata.json")
inference_engine = DynamicGNNInference(MODEL_PATH, METADATA_PATH, device="cuda" if torch.cuda.is_available() else "cpu")

@app.post("/predict", response_model=GNNPredictionResponse)
def predict_endpoint(req: PredictFromFeaturesRequest):
    try:
        feats = {k: torch.tensor(v).unsqueeze(0) for k,v in req.component_features.items()}
        result = inference_engine.predict(feats)
        # Используйте postprocessor чтобы собрать GNNPredictionResponse
        # Здесь можете вставить постпроцессинг под вашу задачу
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
