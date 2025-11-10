"""Client for internal RAG service communication."""

import httpx
from django.conf import settings
from rest_framework.exceptions import APIException
import structlog
from typing import Dict, Any, Optional

logger = structlog.get_logger()


class RagServiceClient:
    """HTTP client for rag_service internal API."""

    def __init__(self):
        self.base_url = settings.RAG_SERVICE_URL
        self.api_key = settings.RAG_INTERNAL_API_KEY
        self.timeout = httpx.Timeout(30.0, connect=5.0)

    async def query(
        self,
        query_text: str,
        system_id: int,
        user_context: Optional[Dict[str, Any]] = None,
        max_results: int = 3,
    ) -> Dict[str, Any]:
        """Send query to RAG service.

        Args:
            query_text: User query text
            system_id: RAG system ID
            user_context: User context (user_id, username, etc.)
            max_results: Max number of relevant documents

        Returns:
            RAG response with answer and sources

        Raises:
            APIException: If RAG service is unreachable or returns error
        """
        headers = {"X-Internal-API-Key": self.api_key}
        payload = {
            "query": query_text,
            "system_id": system_id,
            "context": user_context or {},
            "max_results": max_results,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/query", json=payload, headers=headers
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                "RAG service HTTP error",
                status_code=e.response.status_code,
                response_body=e.response.text[:200],
            )
            raise APIException(
                f"RAG service error: {e.response.status_code}", code="rag_service_error"
            )

        except httpx.RequestError as e:
            logger.error("RAG service unreachable", error=str(e))
            raise APIException(
                "RAG service unreachable", code="rag_service_unreachable"
            )

        except Exception as e:
            logger.error("Unexpected RAG client error", error=str(e))
            raise APIException(
                f"RAG client error: {str(e)}", code="rag_client_error"
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check RAG service health.

        Returns:
            Health status response
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning("RAG service health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}


# Singleton instance
_rag_client: Optional[RagServiceClient] = None


def get_rag_client() -> RagServiceClient:
    """Get or create RAG client singleton.

    Returns:
        RagServiceClient instance
    """
    global _rag_client
    if _rag_client is None:
        _rag_client = RagServiceClient()
    return _rag_client
