"""GNN Service async client."""

import httpx
import structlog

logger = structlog.get_logger(__name__)


class GNNClient:
    """Async HTTP client для GNN Service."""
    
    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-Internal-API-Key": self.api_key},
            timeout=timeout,
        )
    
    async def predict(
        self,
        node_features: list[list[float]],
        edge_index: list[list[int]],
        edge_attr: list[list[float]] | None = None,
        component_names: list[str] | None = None,
    ) -> dict:
        """Single equipment prediction."""
        payload = {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "component_names": component_names,
        }
        
        response = await self.client.post("/predict", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    async def batch_predict(self, graphs: list[dict]) -> dict:
        """Batch prediction для fleet."""
        payload = {"graphs": graphs}
        
        response = await self.client.post("/batch_predict", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    async def health(self) -> dict:
        """Health check."""
        response = await self.client.get("/health")
        response.raise_for_status()
        
        return response.json()
    
    async def close(self):
        """Close client."""
        await self.client.aclose()
