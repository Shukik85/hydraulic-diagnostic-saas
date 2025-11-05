"""
DRF API Views for Sensor Data Management.

Provides REST API endpoints for:
- Sensor nodes and configurations CRUD
- Real-time sensor readings with filtering
- Time-series data aggregation
- Health monitoring and metrics
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from django.db.models import Q, Avg, Min, Max, Count
from django.utils import timezone
from django.http import JsonResponse
from django.core.cache import cache
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import OrderingFilter, SearchFilter

from .models import SensorNode, SensorConfig, SensorReading
from .serializers import (
    SensorNodeSerializer,
    SensorConfigSerializer, 
    SensorReadingSerializer,
    SensorReadingListSerializer,
    SensorDataTimeSeriesSerializer,
    SensorStatsSerializer,
    NodeHealthSerializer
)
from .tasks import poll_sensor_nodes


class SensorNodeViewSet(viewsets.ModelViewSet):
    """ViewSet for managing sensor nodes."""
    
    queryset = SensorNode.objects.all()
    serializer_class = SensorNodeSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['protocol', 'is_active', 'connection_status']
    search_fields = ['name', 'host_address']
    ordering_fields = ['name', 'created_at', 'last_poll_time']
    ordering = ['-last_poll_time']
    
    @action(detail=True, methods=['post'])
    def poll_now(self, request, pk=None):
        """Trigger immediate polling of a specific sensor node."""
        try:
            node = self.get_object()
            
            # Trigger Celery task for this specific node
            task = poll_sensor_nodes.delay(node_ids=[node.id])
            
            return Response({
                'status': 'polling_started',
                'task_id': task.id,
                'node_name': node.name,
                'message': f'Polling initiated for node {node.name}'
            })
            
        except Exception as e:
            return Response({
                'status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['get'])
    def health(self, request, pk=None):
        """Get detailed health status for a sensor node."""
        node = self.get_object()
        
        # Get recent readings (last 1 hour)
        recent_cutoff = timezone.now() - timedelta(hours=1)
        recent_readings = SensorReading.objects.filter(
            sensor_config__node=node,
            timestamp__gte=recent_cutoff
        )
        
        # Calculate health metrics
        total_readings = recent_readings.count()
        good_readings = recent_readings.filter(quality='good').count()
        
        health_data = {
            'node_id': node.id,
            'node_name': node.name,
            'protocol': node.protocol,
            'is_active': node.is_active,
            'connection_status': node.connection_status,
            'last_poll_time': node.last_poll_time,
            'last_poll_success': node.last_poll_success,
            'sensors_count': node.sensors.filter(is_active=True).count(),
            'recent_readings_count': total_readings,
            'avg_collection_latency_ms': recent_readings.aggregate(
                avg_latency=Avg('collection_latency_ms')
            )['avg_latency'],
            'error_rate_percent': (
                (total_readings - good_readings) / total_readings * 100
                if total_readings > 0 else 0
            )
        }
        
        serializer = NodeHealthSerializer(health_data)
        return Response(serializer.data)


class SensorConfigViewSet(viewsets.ModelViewSet):
    """ViewSet for managing sensor configurations."""
    
    queryset = SensorConfig.objects.all()
    serializer_class = SensorConfigSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['node', 'data_type', 'is_active']
    search_fields = ['name', 'unit']
    ordering_fields = ['name', 'register_address', 'created_at']
    ordering = ['register_address']


class SensorReadingViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for accessing sensor readings (read-only)."""
    
    queryset = SensorReading.objects.select_related(
        'sensor_config', 'sensor_config__node'
    ).all()
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_fields = ['sensor_config', 'quality']
    ordering_fields = ['timestamp']
    ordering = ['-timestamp']
    
    def get_serializer_class(self):
        """Use lightweight serializer for list view."""
        if self.action == 'list':
            return SensorReadingListSerializer
        return SensorReadingSerializer
    
    def get_queryset(self):
        """Filter readings with time range and sensor parameters."""
        queryset = super().get_queryset()
        
        # Time range filtering
        from_time = self.request.query_params.get('from')
        to_time = self.request.query_params.get('to')
        
        if from_time:
            try:
                from_dt = datetime.fromisoformat(from_time.replace('Z', '+00:00'))
                queryset = queryset.filter(timestamp__gte=from_dt)
            except ValueError:
                pass  # Invalid datetime format, ignore
        
        if to_time:
            try:
                to_dt = datetime.fromisoformat(to_time.replace('Z', '+00:00'))
                queryset = queryset.filter(timestamp__lte=to_dt)
            except ValueError:
                pass  # Invalid datetime format, ignore
        
        # Node filtering
        node_id = self.request.query_params.get('node')
        if node_id:
            queryset = queryset.filter(sensor_config__node_id=node_id)
        
        # Sensor filtering
        sensor_id = self.request.query_params.get('sensor')
        if sensor_id:
            queryset = queryset.filter(sensor_config_id=sensor_id)
        
        # Limit to recent data by default (last 24 hours)
        if not from_time and not to_time:
            default_cutoff = timezone.now() - timedelta(hours=24)
            queryset = queryset.filter(timestamp__gte=default_cutoff)
        
        return queryset[:10000]  # Limit to 10k records for performance
    
    @action(detail=False, methods=['get'])
    def time_series(self, request):
        """Get time-series data with optional aggregation."""
        # Get query parameters
        sensor_id = request.query_params.get('sensor')
        node_id = request.query_params.get('node')
        interval = request.query_params.get('interval', '1m')  # 1m, 5m, 15m, 1h
        from_time = request.query_params.get('from')
        to_time = request.query_params.get('to')
        
        if not sensor_id and not node_id:
            return Response({
                'error': 'Either sensor or node parameter is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Build base query
        queryset = SensorReading.objects.filter(quality='good')
        
        if sensor_id:
            queryset = queryset.filter(sensor_config_id=sensor_id)
        elif node_id:
            queryset = queryset.filter(sensor_config__node_id=node_id)
        
        # Apply time filtering
        if from_time:
            try:
                from_dt = datetime.fromisoformat(from_time.replace('Z', '+00:00'))
                queryset = queryset.filter(timestamp__gte=from_dt)
            except ValueError:
                return Response({
                    'error': 'Invalid from time format. Use ISO format.'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        if to_time:
            try:
                to_dt = datetime.fromisoformat(to_time.replace('Z', '+00:00'))
                queryset = queryset.filter(timestamp__lte=to_dt)
            except ValueError:
                return Response({
                    'error': 'Invalid to time format. Use ISO format.'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Default to last 4 hours if no time specified
        if not from_time and not to_time:
            default_from = timezone.now() - timedelta(hours=4)
            queryset = queryset.filter(timestamp__gte=default_from)
        
        # Get raw data points (no aggregation for now - can add later)
        readings = queryset.order_by('timestamp')[:1000]  # Limit for performance
        
        # Serialize data
        time_series_data = []
        for reading in readings:
            time_series_data.append({
                'timestamp': reading.timestamp,
                'value': float(reading.processed_value),
                'quality': reading.quality,
                'sensor_name': reading.sensor_config.name
            })
        
        return Response({
            'data': time_series_data,
            'count': len(time_series_data),
            'interval': interval,
            'sensor_id': sensor_id,
            'node_id': node_id
        })
    
    @action(detail=False, methods=['get'])
    def latest(self, request):
        """Get latest readings from all sensors."""
        # Cache key for latest readings
        cache_key = 'sensor_readings_latest'
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return Response(cached_data)
        
        # Get latest reading for each active sensor
        latest_readings = []
        
        active_sensors = SensorConfig.objects.filter(
            is_active=True,
            node__is_active=True
        ).select_related('node')
        
        for sensor in active_sensors:
            latest_reading = SensorReading.objects.filter(
                sensor_config=sensor
            ).order_by('-timestamp').first()
            
            if latest_reading:
                latest_readings.append({
                    'sensor_id': sensor.id,
                    'sensor_name': sensor.name,
                    'node_name': sensor.node.name,
                    'timestamp': latest_reading.timestamp,
                    'value': float(latest_reading.processed_value),
                    'unit': sensor.unit,
                    'quality': latest_reading.quality,
                    'register_address': sensor.register_address
                })
        
        response_data = {
            'readings': latest_readings,
            'count': len(latest_readings),
            'timestamp': timezone.now()
        }
        
        # Cache for 5 seconds
        cache.set(cache_key, response_data, 5)
        
        return Response(response_data)
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get sensor statistics and data quality metrics."""
        # Time range for statistics (default: last 24 hours)
        hours = int(request.query_params.get('hours', 24))
        cutoff_time = timezone.now() - timedelta(hours=hours)
        
        # Aggregate statistics per sensor
        sensor_stats = []
        
        active_sensors = SensorConfig.objects.filter(
            is_active=True,
            node__is_active=True
        ).select_related('node')
        
        for sensor in active_sensors:
            readings = SensorReading.objects.filter(
                sensor_config=sensor,
                timestamp__gte=cutoff_time
            )
            
            total_count = readings.count()
            if total_count == 0:
                continue
            
            good_count = readings.filter(quality='good').count()
            bad_count = readings.filter(quality='bad').count()
            uncertain_count = readings.filter(quality='uncertain').count()
            
            # Statistical aggregation on good readings only
            good_readings = readings.filter(quality='good')
            stats = good_readings.aggregate(
                avg_value=Avg('processed_value'),
                min_value=Min('processed_value'),
                max_value=Max('processed_value')
            )
            
            latest_reading = readings.order_by('-timestamp').first()
            
            sensor_stats.append({
                'sensor_config_id': sensor.id,
                'sensor_name': sensor.name,
                'total_readings': total_count,
                'good_readings': good_count,
                'bad_readings': bad_count,
                'uncertain_readings': uncertain_count,
                'avg_value': float(stats['avg_value']) if stats['avg_value'] else None,
                'min_value': float(stats['min_value']) if stats['min_value'] else None,
                'max_value': float(stats['max_value']) if stats['max_value'] else None,
                'last_reading_time': latest_reading.timestamp if latest_reading else None,
                'data_quality_percent': (good_count / total_count * 100) if total_count > 0 else 0
            })
        
        serializer = SensorStatsSerializer(sensor_stats, many=True)
        
        return Response({
            'stats': serializer.data,
            'period_hours': hours,
            'generated_at': timezone.now()
        })


class SensorConfigViewSet(viewsets.ModelViewSet):
    """ViewSet for managing sensor configurations."""
    
    queryset = SensorConfig.objects.select_related('node').all()
    serializer_class = SensorConfigSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['node', 'data_type', 'is_active']
    search_fields = ['name', 'unit']
    ordering_fields = ['name', 'register_address', 'created_at']
    ordering = ['register_address']


# Health check endpoint (function-based view)
def sensor_health_status(request):
    """
    Quick health status endpoint for monitoring systems.
    
    Returns JSON with overall sensor system health.
    """
    try:
        # Quick health metrics
        total_nodes = SensorNode.objects.filter(is_active=True).count()
        connected_nodes = SensorNode.objects.filter(
            is_active=True,
            connection_status='connected'
        ).count()
        
        # Recent readings check (last 5 minutes)
        recent_cutoff = timezone.now() - timedelta(minutes=5)
        recent_readings = SensorReading.objects.filter(
            timestamp__gte=recent_cutoff
        ).count()
        
        # Overall health status
        if total_nodes == 0:
            health_status = 'no_nodes'
        elif connected_nodes == 0:
            health_status = 'all_disconnected'
        elif connected_nodes / total_nodes >= 0.8:  # 80% threshold
            health_status = 'healthy'
        else:
            health_status = 'degraded'
        
        return JsonResponse({
            'status': health_status,
            'total_nodes': total_nodes,
            'connected_nodes': connected_nodes,
            'recent_readings': recent_readings,
            'connection_rate': (connected_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        }, status=500)
