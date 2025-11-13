"""
Unified GNN Service Client
Used by: backend_fastapi, diagnosis_service
"""

import httpx
from typing import Dict, List, Optional
from pydantic import BaseModel


class GNNPredictionRequest(BaseModel):
    """Request schema for GNN prediction."""
    sensor_data: List[Dict]
    graph_structure: Optional[Dict] = None
    timestamp: str


class GNNPredictionResponse(BaseModel):
    """Response schema from GNN service."""
    predictions: List[float]
    confidence: float
    attention_weights: Dict[str, float]
    inference_time_ms: float


class GNNClient:
    """
    Unified client for GNN Service.
    
    Usage:
        from shared.clients import GNNClient
        
        client = GNNClient(base_url="http://gnn-service:8001")
        result = await client.predict(data)
    """
    
    def __init__(
        self,
        base_url: str = "http://gnn-service:8001",
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def health(self) -> Dict:
        """Check GNN service health."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def predict(
        self,
        request: GNNPredictionRequest
    ) -> GNNPredictionResponse:
        """
        Run GNN prediction.
        
        Args:
            request: Prediction request with sensor data
            
        Returns:
            GNNPredictionResponse with predictions and metadata
            
        Raises:
            httpx.HTTPError: If service call fails
        """
        response = await self.client.post(
            f"{self.base_url}/predict",
            json=request.model_dump()
        )
        response.raise_for_status()
        return GNNPredictionResponse(**response.json())
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()
