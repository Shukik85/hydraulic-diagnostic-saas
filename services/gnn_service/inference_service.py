"""
FastAPI Inference Service
Optimized for PyTorch 2.3 + GTX 1650 SUPER
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import logging
from typing import List
import time

from models.lightweight_gnn import LightweightGNN, MemoryEfficientInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GNN Inference Service",
    description="Graph Neural Network for hydraulic diagnostics",
    version="2.0.0-gtx1650"
)

inference_engine = None


class InferenceRequest(BaseModel):
    user_id: str
    system_id: str
    node_features: List[List[float]]
    edge_index: List[List[int]]


class InferenceResponse(BaseModel):
    system_id: str
    overall_anomaly_score: float
    component_scores: List[dict]
    inference_time_ms: float
    pytorch_version: str
    gpu_memory_mb: float


@app.on_event("startup")
async def startup_event():
    """Initialize model"""
    global inference_engine

    logger.info("Starting GNN Service...")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: 4GB (GTX 1650 SUPER)")

    try:
        model = LightweightGNN()
        inference_engine = MemoryEfficientInference(model)
        logger.info("✅ Model loaded and optimized!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
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
    """Run GNN inference"""
    if inference_engine is None:
        raise HTTPException(503, "Model not initialized")

    try:
        start = time.time()

        x = torch.tensor(request.node_features, dtype=torch.float32)
        edge_index = torch.tensor(request.edge_index, dtype=torch.long)

        scores = inference_engine.predict(x, edge_index)

        inference_time = (time.time() - start) * 1000

        component_scores = [
            {"component_index": i, "score": float(s)}
            for i, s in enumerate(scores.squeeze())
        ]

        # Get memory stats
        mem_stats = inference_engine.get_memory_stats()
        gpu_mem_mb = mem_stats.get("allocated_gb", 0) * 1024

        return {
            "system_id": request.system_id,
            "overall_anomaly_score": float(scores.mean()),
            "component_scores": component_scores,
            "inference_time_ms": inference_time,
            "pytorch_version": torch.__version__,
            "gpu_memory_mb": gpu_mem_mb
        }

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/memory-stats")
async def memory_stats():
    """Get GPU memory statistics"""
    if inference_engine:
        return inference_engine.get_memory_stats()
    return {}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
