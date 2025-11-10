"""Client for internal ML service communication."""

import httpx
from django.conf import settings
from rest_framework.exceptions import APIException
import structlog
from typing import Dict, Any, Optional, List

logger = structlog.get_logger()


class MLServiceClient:
    """HTTP client for ml_service internal API."""

    def __init__(self):
        self.base_url = settings.ML_SERVICE_URL
        self.api_key = settings.ML_INTERNAL_API_KEY
        self.timeout = httpx.Timeout(10.0, connect=5.0)

    async def predict(
        self,
        sensor_data: Dict[str, Any],
        system_id: int,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Send anomaly prediction request to ML service.

        Args:
            sensor_data: Sensor data for prediction
            system_id: System ID
            use_cache: Enable cache for predictions

        Returns:
            ML prediction response with anomaly detection results

        Raises:
            APIException: If ML service is unreachable or returns error
        """
        headers = {"X-Internal-API-Key": self.api_key}
        payload = {
            "sensor_data": {
                "system_id": system_id,
                **sensor_data
            },
            "use_cache": use_cache,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/predict", json=payload, headers=headers
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                "ML service HTTP error",
                status_code=e.response.status_code,
                response_body=e.response.text[:200],
            )
            raise APIException(
                f"ML service error: {e.response.status_code}", code="ml_service_error"
            )

        except httpx.RequestError as e:
            logger.error("ML service unreachable", error=str(e))
            raise APIException(
                "ML service unreachable", code="ml_service_unreachable"
            )

        except Exception as e:
            logger.error("Unexpected ML client error", error=str(e))
            raise APIException(
                f"ML client error: {str(e)}", code="ml_client_error"
            )

    async def predict_batch(
        self,
        requests: List[Dict[str, Any]],
        parallel_processing: bool = True,
    ) -> Dict[str, Any]:
        """Send batch prediction request to ML service.

        Args:
            requests: List of prediction requests
            parallel_processing: Enable parallel processing

        Returns:
            Batch prediction response

        Raises:
            APIException: If ML service is unreachable or returns error
        """
        headers = {"X-Internal-API-Key": self.api_key}
        payload = {
            "requests": requests,
            "parallel_processing": parallel_processing,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/predict/batch", json=payload, headers=headers
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                "ML service batch HTTP error",
                status_code=e.response.status_code,
            )
            raise APIException(
                f"ML service batch error: {e.response.status_code}", code="ml_service_error"
            )

        except Exception as e:
            logger.error("ML batch prediction failed", error=str(e))
            raise APIException(
                f"ML batch prediction failed: {str(e)}", code="ml_batch_error"
            )

    async def get_model_performance(self) -> Dict[str, Any]:
        """Get ML model performance statistics.

        Returns:
            Model performance metrics
        """
        headers = {"X-Internal-API-Key": self.api_key}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/models/performance", headers=headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning("ML model performance check failed", error=str(e))
            return {"error": str(e), "available": False}

    async def health_check(self) -> Dict[str, Any]:
        """Check ML service health.

        Returns:
            Health status response
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning("ML service health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}


# Singleton instance
_ml_client: Optional[MLServiceClient] = None


def get_ml_client() -> MLServiceClient:
    """Get or create ML client singleton.

    Returns:
        MLServiceClient instance
    """
    global _ml_client
    if _ml_client is None:
        _ml_client = MLServiceClient()
    return _ml_client
