"""
ML Service Client - Production Integration
Connects Django backend to FastAPI ML service (ONNX models)
"""
import httpx
import logging
from typing import Any
from django.conf import settings

logger = logging.getLogger(__name__)

class MLServiceClient:
    """HTTP client for ML service communication"""
    
    def __init__(self):
        self.base_url = getattr(settings, 'ML_SERVICE_URL', 'http://localhost:8001')
        self.timeout = 5.0
        logger.info(f'MLServiceClient initialized: {self.base_url}')
        
    async def predict_anomaly(self, features: list[float]) -> dict[str, Any]:
        """
        Call ML service for anomaly prediction (ONNX CatBoost)
        
        Args:
            features: List of sensor features [pressure, temp, vibration, ...]
            
        Returns:
            {
                'prediction': 0 or 1 (0=normal, 1=fault),
                'probability': float (0.0-1.0),
                'model': 'catboost_onnx',
                'inference_time_ms': float
            }
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f'{self.base_url}/predict',
                    json={'features': features}
                )
                response.raise_for_status()
                result = response.json()
                
                logger.info(
                    f'ML prediction: {result.get("prediction")} '
                    f'(prob: {result.get("probability"):.3f}, '
                    f'time: {result.get("inference_time_ms"):.1f}ms)'
                )
                
                return result
                
        except httpx.TimeoutException:
            logger.error(f'ML service timeout after {self.timeout}s')
            raise Exception('ML service timeout')
        except httpx.HTTPStatusError as e:
            logger.error(f'ML service HTTP error: {e.response.status_code}')
            raise Exception(f'ML service error: {e.response.status_code}')
        except Exception as e:
            logger.error(f'ML service error: {e}')
            raise
            
    async def health_check(self) -> dict[str, Any]:
        """Check ML service health status"""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f'{self.base_url}/health')
                return {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'response': response.json() if response.status_code == 200 else None
                }
        except Exception as e:
            logger.warning(f'ML service health check failed: {e}')
            return {'status': 'unreachable', 'error': str(e)}
            
    def predict_anomaly_sync(self, features: list[float]) -> dict[str, Any]:
        """Synchronous prediction for non-async contexts"""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f'{self.base_url}/predict',
                    json={'features': features}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f'ML service sync error: {e}')
            raise

# Global client instance
ml_client = MLServiceClient()
