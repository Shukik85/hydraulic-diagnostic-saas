#!/usr/bin/env python3
"""
Simplified ML Prediction Service
Standalone FastAPI service for containerized deployment
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog
import numpy as np
import joblib
from prometheus_client import Counter, Histogram, generate_latest
from contextlib import asynccontextmanager

# Configure logging
structlog.configure(processors=[structlog.dev.ConsoleRenderer(colors=True)])
logger = structlog.get_logger()

# Metrics
prediction_counter = Counter('ml_predictions_total', ['model', 'result'])
prediction_latency = Histogram('ml_prediction_duration_seconds')


class PredictRequest(BaseModel):
    """Prediction request schema."""
    features: List[float] = Field(..., min_items=25, max_items=25)
    sensor_id: Optional[str] = None
    trace_id: Optional[str] = None


class PredictResponse(BaseModel):
    """Prediction response schema."""
    prediction: int = Field(..., description="0=normal, 1=fault")
    probability: float = Field(..., description="Fault probability 0-1")
    confidence: float = Field(..., description="Prediction confidence 0-1")
    models_used: List[str] = Field(..., description="Active models")
    processing_time_ms: float = Field(..., description="Latency")
    model_version: str = Field(..., description="Model version")


class SimplePredictor:
    """Simplified ensemble predictor for containers."""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.ready = False
        self.model_version = "unknown"
    
    def load_models(self, models_dir: str = "./models"):
        """Load models from directory."""
        
        logger.info(f"Loading models from {models_dir}")
        models_path = Path(models_dir)
        
        if not models_path.exists():
            logger.error(f"Models directory not found: {models_dir}")
            return
        
        # Load primary models
        model_files = {
            'catboost': 'catboost_model.joblib',
            'xgboost': 'xgboost_model.joblib',
            'random_forest': 'random_forest_model.joblib'
        }
        
        loaded = 0
        for name, filename in model_files.items():
            path = models_path / filename
            if path.exists():
                try:
                    data = joblib.load(path)
                    
                    # Check if REAL model
                    if data.get('data_source') != 'REAL_UCI_HYDRAULIC_DATA':
                        logger.warning(f"Skipping mock model: {name}")
                        continue
                    
                    self.models[name] = data['model']
                    self.model_version = data.get('model_version', 'unknown')
                    loaded += 1
                    
                    logger.info(f"✅ Loaded {name} model")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {name}: {e}")
        
        # Load scaler
        scaler_path = models_path / 'scaler.joblib'
        if scaler_path.exists():
            try:
                scaler_data = joblib.load(scaler_path)
                self.scaler = scaler_data if hasattr(scaler_data, 'transform') else scaler_data.get('scaler')
                logger.info("✅ Loaded scaler")
            except Exception as e:
                logger.error(f"❌ Failed to load scaler: {e}")
        
        self.ready = loaded >= 2
        logger.info(f"Predictor ready: {loaded} models loaded")
    
    async def predict(self, features: List[float]) -> Dict:
        """Make ensemble prediction."""
        
        if not self.ready:
            raise HTTPException(status_code=503, detail="Models not ready")
        
        start_time = time.time()
        
        # Prepare input
        X = np.array(features).reshape(1, -1)
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Get predictions
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0, 1]
                
                predictions[name] = pred
                probabilities[name] = proba
                
                prediction_counter.labels(model=name, result=str(pred)).inc()
                
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        
        if not probabilities:
            raise HTTPException(status_code=500, detail="All models failed")
        
        # Ensemble (weighted average)
        weights = {'catboost': 0.5, 'xgboost': 0.3, 'random_forest': 0.2}
        
        ensemble_proba = sum(
            weights.get(name, 0.1) * proba 
            for name, proba in probabilities.items()
        )
        ensemble_proba /= sum(weights.get(name, 0.1) for name in probabilities.keys())
        
        ensemble_pred = 1 if ensemble_proba > 0.5 else 0
        
        # Confidence based on model agreement
        proba_values = list(probabilities.values())
        confidence = 1.0 - np.std(proba_values) if len(proba_values) > 1 else 0.8
        confidence = max(0.0, min(1.0, confidence))
        
        processing_time = (time.time() - start_time) * 1000
        prediction_latency.observe(processing_time / 1000)
        
        return {
            "prediction": ensemble_pred,
            "probability": round(ensemble_proba, 4),
            "confidence": round(confidence, 4),
            "models_used": list(probabilities.keys()),
            "processing_time_ms": round(processing_time, 2),
            "model_version": self.model_version
        }


# Global predictor
predictor = SimplePredictor()


@asynccontextmanager
async def lifespan_simple(app: FastAPI):
    """Simple lifespan for standalone service."""
    logger.info("🚀 Starting simple ML service")
    
    try:
        predictor.load_models()
        logger.info(f"✅ Ready with {len(predictor.models)} models")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
    
    yield
    logger.info("🛑 Shutting down")


# FastAPI app
app = FastAPI(
    title="Hydraulic ML Service",
    description="Real-time hydraulic fault prediction",
    version="1.0.0-simple",
    lifespan=lifespan_simple
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict hydraulic fault."""
    
    try:
        result = await predictor.predict(request.features)
        return PredictResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def models_info():
    """Get models information."""
    return {
        "models_loaded": list(predictor.models.keys()),
        "ready": predictor.ready,
        "model_version": predictor.model_version,
        "data_source": "REAL_UCI_HYDRAULIC_DATA"
    }


@app.get("/healthz")
async def health():
    """Health check."""
    return {"status": "healthy", "ready": predictor.ready}


@app.get("/readyz")
async def readiness():
    """Readiness check."""
    if not predictor.ready:
        raise HTTPException(status_code=503, detail="Not ready")
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(
        "simple_predict:app",
        host="0.0.0.0",
        port=8001,
        reload=False
    )