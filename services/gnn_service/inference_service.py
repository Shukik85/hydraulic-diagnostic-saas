"""
Universal GNN Inference Service - PyTorch 2.5.1
Dynamic fields supported: node count, feature count, edge topology, model rebuilt at load
Optimized for torch.compile + mixed precision FP16 + dynamic batch sizes
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import logging
from typing import List, Optional
import time
from pathlib import Path

from model_universal import UniversalHydraulicGNN, get_feature_dim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Universal GNN Inference Service",
    description="Universal Graph Neural Network for hydraulic diagnostics (dynamic components/sensors)",
    version="3.0.0-universal"
)

inference_engine = None
model_metadata = None

class InferenceRequest(BaseModel):
    user_id: str
    system_id: str
    node_features: List[List[float]]  # динамический размер: n_nodes x n_features
    edge_index: List[List[int]]       # динамический topology
    metadata: dict                    # явно передаем equipment metadata: components, sensors

class InferenceResponse(BaseModel):
    system_id: str
    overall_anomaly_score: float
    component_scores: List[dict]
    inference_time_ms: float
    pytorch_version: str
    gpu_memory_mb: float
    n_nodes: int
    n_features: int
    n_classes: int

@app.on_event("startup")
async def startup_event():
    """Initialize universal dynamic model"""
    global inference_engine, model_metadata
    logger.info("Starting Universal GNN Service...")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    # Загрузка метадаты по умолчанию (пример)
    try:
        metadata_path = Path("models/default_metadata.json")
        if metadata_path.exists():
            import json
            model_metadata = json.load(metadata_path.open())
        else:
            model_metadata = {
                "system_id": "default_system",
                "components": [
                    {"component_id":"pump","sensors":["pressure","flow"]},
                    {"component_id":"valve","sensors":["position","pressure"]}
                ]
            }
        logger.info(f"Loaded metadata: n_nodes={len(model_metadata['components'])} n_features={get_feature_dim(model_metadata)}")
        # Universal GNN (dynamic)
        model = UniversalHydraulicGNN(model_metadata)
        if torch.cuda.is_available():
            device = "cuda"
            model = model.to(device)
        else:
            device = "cpu"
        # torch.compile оптимизация
        if hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile (reduce-overhead, dynamic shape)...")
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            logger.info("Model compiled!")
        global inference_engine
        inference_engine = model
        logger.info("✅ Universal GNN loaded and ready!")
    except Exception as e:
        logger.error(f"Failed to load universal model: {e}")
        raise

@app.get("/health/")
async def health_check():
    """Health check with GPU info"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3
        }
    return {
        "status": "ok",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        **gpu_info
    }

@app.post("/gnn/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Universal dynamic inference (variable nodes/features/edges)"""
    global inference_engine, model_metadata
    if inference_engine is None:
        raise HTTPException(503, "Model not initialized")
    try:
        start = time.time()
        # Dynamic feature dim per request
        if request.metadata is not None:
            model_metadata = request.metadata
            n_nodes = len(model_metadata['components'])
            n_features = get_feature_dim(model_metadata)
            n_classes = n_nodes
            # Rebuild model per dynamic metadata
            model = UniversalHydraulicGNN(model_metadata)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            # Compile new structure
            if hasattr(torch, "compile"):
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            inference_engine = model
        else:
            model = inference_engine
            n_nodes = len(model_metadata['components'])
            n_features = get_feature_dim(model_metadata)
            n_classes = n_nodes
        # Inputs
        x = torch.tensor(request.node_features, dtype=torch.float32)
        edge_index = torch.tensor(request.edge_index, dtype=torch.long)
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = edge_index.cuda()
        # Inference with mixed precision where possible
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(x, edge_index)
            else:
                logits = model(x, edge_index)
        scores = torch.sigmoid(logits).squeeze()
        inference_time = (time.time() - start) * 1000
        component_scores = [
            {"component_index": i, "score": float(s)}
            for i, s in enumerate(scores)
        ]
        gpu_mem_mb = torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0
        return {
            "system_id": request.system_id,
            "overall_anomaly_score": float(scores.mean()),
            "component_scores": component_scores,
            "inference_time_ms": inference_time,
            "pytorch_version": torch.__version__,
            "gpu_memory_mb": gpu_mem_mb,
            "n_nodes": n_nodes,
            "n_features": n_features,
            "n_classes": n_classes
        }
    except Exception as e:
        logger.error(f"Universal inference failed: {e}")
        raise HTTPException(500, str(e))

@app.get("/memory-stats")
async def memory_stats():
    if torch.cuda.is_available():
        return {"gpu_memory_mb": torch.cuda.memory_allocated(0) / 1024**2}
    return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
