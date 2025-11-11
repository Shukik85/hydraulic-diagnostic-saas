"""
GNN Service client for ML inference
"""
import httpx
import structlog
from typing import Dict, Any

from config import settings

logger = structlog.get_logger()


class GNNClient:
    """Client for GNN inference service"""

    def __init__(self):
        self.base_url = settings.GNN_SERVICE_URL
        self.timeout = 30.0

    async def infer(self, user_id: str, system_id: str) -> Dict[str, Any]:
        """
        Run GNN inference for system
        Returns anomaly scores per component
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/gnn/infer",
                    params={"user_id": user_id, "system_id": system_id}
                )
                response.raise_for_status()

                result = response.json()
                logger.info(
                    "gnn_inference_completed",
                    user_id=user_id,
                    system_id=system_id,
                    n_components=result.get("n_components")
                )

                return result

            except httpx.HTTPError as e:
                logger.error("gnn_inference_failed", exc_info=e)
                raise RuntimeError(f"GNN inference failed: {str(e)}")

    async def health_check(self) -> bool:
        """Check if GNN service is healthy"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/gnn/health")
                return response.status_code == 200
            except:
                return False
