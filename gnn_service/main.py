"""FastAPI application for GNN inference."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
import os

from .inference import GNNInference
from .config import config

app = FastAPI(
    title="GNN Anomaly Detection Service",
    version="0.1.0",
    description="Temporal Graph Attention Network for hydraulic system anomaly detection",
)

# Load model on startup
model_path = os.path.join(config.model_save_dir, "gnn_classifier_best.ckpt")
if not os.path.exists(model_path):
    print(f"⚠ Model not found at {model_path}. Please train the model first.")
    inference_engine = None
else:
    inference_engine = GNNInference(model_path)


class PredictionRequest(BaseModel):
    """Request schema for anomaly prediction."""
    node_features: List[List[float]]  # [num_nodes, num_features]
    edge_index: List[List[int]]  # [2, num_edges]
    edge_attr: Optional[List[List[float]]] = None  # [num_edges, edge_dim]
    component_names: Optional[List[str]] = None  # For explainability


class PredictionResponse(BaseModel):
    """Response schema for anomaly prediction."""
    prediction: int  # 0 = normal, 1 = anomaly
    probability: float  # Confidence [0, 1]
    anomaly_score: float  # P(anomaly)
    explanation: Optional[dict] = None  # Explainability insights


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "device": config.device,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Run anomaly detection on hydraulic system graph.
    
    Args:
        request: Graph data with node features and edge connectivity
    
    Returns:
        Prediction with anomaly score and explainability
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )
    
    try:
        # Convert to numpy
        node_features = np.array(request.node_features, dtype=np.float32)
        edge_index = np.array(request.edge_index, dtype=np.int64)
        edge_attr = (
            np.array(request.edge_attr, dtype=np.float32)
            if request.edge_attr is not None
            else None
        )
        
        # Run inference
        result = inference_engine.predict(node_features, edge_index, edge_attr)
        
        # Add explainability if component names provided
        explanation = None
        if request.component_names:
            explanation = inference_engine.explain_prediction(
                result, request.component_names
            )
        
        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            anomaly_score=result["anomaly_score"],
            explanation=explanation,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Run batch inference on multiple graphs."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        graphs = [
            (
                np.array(req.node_features, dtype=np.float32),
                np.array(req.edge_index, dtype=np.int64),
                np.array(req.edge_attr, dtype=np.float32) if req.edge_attr else None,
            )
            for req in requests
        ]
        
        results = inference_engine.batch_predict(graphs)
        
        return {
            "predictions": [
                {
                    "prediction": r["prediction"],
                    "probability": r["probability"],
                    "anomaly_score": r["anomaly_score"],
                }
                for r in results
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "gnn_service.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
    )
