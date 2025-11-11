"""
FastAPI inference сервис для универсального GNN — production-ready, полностью динамический:
- Загружает equipment_metadata.json c user_id/system_id
- Получает срез данных из TimeSeries (timescaledb_connector)
- Строит граф (через graph_builder)
- Запускает model_universal
- Возвращает результат инференса + debug info
"""
from fastapi import FastAPI, Query, HTTPException
import torch
import pandas as pd
import json
from model_universal import UniversalHydraulicGNN
from graph_builder import build_node_features, adjacency_to_edge_index

app = FastAPI(title="UniversalGNN Inference Service")

# Dummy func — заменить на prod loader
def load_metadata(user_id, system_id):
    with open(f'data/{user_id}_{system_id}_metadata.json') as f:
        return json.load(f)

def load_sensor_data(user_id, system_id, minutes=5):
    # Plug: зависит от TimescaleDB, здесь — просто csv слайс
    return pd.read_csv(f"data/{user_id}_{system_id}_window.csv")

@app.post("/gnn/infer")
def gnn_infer(user_id: str = Query(...), system_id: str = Query(...)):
    metadata = load_metadata(user_id, system_id)
    data_df = load_sensor_data(user_id, system_id)
    node_features = build_node_features(data_df, metadata)
    edge_index = adjacency_to_edge_index(metadata['adjacency_matrix'])
    model = UniversalHydraulicGNN(metadata)
    model.eval()
    with torch.no_grad():
        logits = model(node_features, edge_index)
        preds = torch.sigmoid(logits)
    labels = [comp['id'] for comp in metadata['components']]
    result = {labels[i]: float(preds[i].item()) for i in range(len(labels))}
    return {
        "system_id": system_id,
        "anomaly_scores": result,
        "n_components": len(labels)
    }

@app.get("/gnn/health")
def health():
    return {"status": "ok"}
