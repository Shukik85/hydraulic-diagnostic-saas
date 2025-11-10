"""
ML Prediction API Endpoint
POST /api/v1/predict/ - Real-time anomaly detection
"""
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
import logging

from .ml_client import ml_client
from .models import HydraulicSystem

logger = logging.getLogger(__name__)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
async def predict_anomaly(request):
    """
    Predict anomaly using ML service (ONNX CatBoost)
    
    Request:
    {
        "system_id": "uuid",
        "features": [100.5, 60.2, 8.5, ...] (25 float values)
    }
    
    Response:
    {
        "job_id": "uuid",
        "prediction": 0 or 1,
        "probability": 0.95,
        "confidence": "high",
        "model": "catboost_onnx",
        "inference_time_ms": 0.08,
        "timestamp": "2025-11-08T01:59:00Z"
    }
    """
    
    # Validate input
    system_id = request.data.get('system_id')
    features = request.data.get('features')
    
    if not system_id:
        return Response(
            {'error': 'system_id is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if not features or not isinstance(features, list):
        return Response(
            {'error': 'features must be a list of floats'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Validate features length
    if len(features) != 25:
        return Response(
            {'error': f'Expected 25 features, got {len(features)}'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Validate system exists
    try:
        system = await HydraulicSystem.objects.aget(id=system_id)
    except HydraulicSystem.DoesNotExist:
        return Response(
            {'error': f'System {system_id} not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Call ML service
    try:
        ml_result = await ml_client.predict_anomaly(features)
        
        # Determine confidence level
        prob = ml_result.get('probability', 0.5)
        if prob >= 0.9 or prob <= 0.1:
            confidence = 'high'
        elif prob >= 0.7 or prob <= 0.3:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Build response
        response_data = {
            'system_id': str(system_id),
            'prediction': ml_result.get('prediction'),
            'probability': prob,
            'confidence': confidence,
            'model': ml_result.get('model', 'unknown'),
            'inference_time_ms': ml_result.get('inference_time_ms', 0),
            'timestamp': timezone.now().isoformat(),
        }
        
        logger.info(
            f'Prediction for {system.name}: '
            f'{response_data["prediction"]} '
            f'(prob: {prob:.3f}, time: {response_data["inference_time_ms"]:.1f}ms)'
        )
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f'ML prediction failed: {e}')
        return Response(
            {'error': 'ML service unavailable', 'details': str(e)},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
async def ml_service_health(request):
    """
    Check ML service health
    
    GET /api/v1/ml/health
    
    Response:
    {
        "status": "healthy" or "unhealthy",
        "ml_service": {...}
    }
    """
    health_status = await ml_client.health_check()
    
    return Response({
        'status': 'healthy' if health_status.get('status') == 'healthy' else 'unhealthy',
        'ml_service': health_status,
        'timestamp': timezone.now().isoformat()
    })
