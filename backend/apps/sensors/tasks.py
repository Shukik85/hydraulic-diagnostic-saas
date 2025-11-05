"""
Celery Tasks for Sensor Data Collection.

Implements periodic Modbus polling, data validation, storage in TimescaleDB,
and integration with ML service for anomaly detection.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal

from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.core.cache import cache
import requests
from prometheus_client import Counter, Histogram, Gauge

from .models import SensorNode, SensorConfig, SensorReading
from .protocols.factory import ProtocolHandlerFactory

logger = logging.getLogger(__name__)

# Prometheus metrics
sensor_readings_total = Counter(
    'sensor_readings_total',
    'Total number of sensor readings collected',
    ['node_name', 'protocol', 'quality']
)

sensor_read_duration = Histogram(
    'sensor_read_duration_seconds',
    'Time spent reading sensor data',
    ['node_name', 'protocol']
)

ml_predictions_total = Counter(
    'ml_predictions_total',
    'Total number of ML predictions made',
    ['status']
)

active_sensor_nodes = Gauge(
    'active_sensor_nodes',
    'Number of active sensor nodes being polled'
)


@shared_task(bind=True, max_retries=3)
def poll_sensor_nodes(self, node_ids: Optional[List[int]] = None):
    """
    Poll all active sensor nodes for data collection.
    
    Args:
        node_ids: Optional list of specific node IDs to poll.
                 If None, polls all active nodes.
    """
    logger.info("🔄 Starting sensor nodes polling task")
    
    try:
        # Get active sensor nodes
        if node_ids:
            nodes = SensorNode.objects.filter(
                id__in=node_ids, 
                is_active=True
            ).prefetch_related('sensors')
        else:
            nodes = SensorNode.objects.filter(
                is_active=True
            ).prefetch_related('sensors')
        
        if not nodes.exists():
            logger.warning("⚠️ No active sensor nodes found")
            return {"status": "no_active_nodes", "nodes_count": 0}
        
        # Update metrics
        active_sensor_nodes.set(nodes.count())
        
        # Poll each node
        results = []
        for node in nodes:
            try:
                result = poll_single_node(node)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed to poll node {node.name}: {e}")
                results.append({
                    "node_id": node.id,
                    "node_name": node.name,
                    "status": "error",
                    "error": str(e)
                })
        
        # Aggregate results
        total_readings = sum(r.get('readings_count', 0) for r in results)
        success_count = sum(1 for r in results if r['status'] == 'success')
        
        logger.info(
            f"✅ Polling completed: {success_count}/{len(results)} nodes successful, "
            f"{total_readings} total readings"
        )
        
        return {
            "status": "completed",
            "nodes_polled": len(results),
            "nodes_successful": success_count,
            "total_readings": total_readings,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"💥 Sensor polling task failed: {e}")
        # Retry with exponential backoff
        raise self.retry(
            countdown=2 ** self.request.retries,
            exc=e,
            max_retries=3
        )


def poll_single_node(node: SensorNode) -> Dict[str, Any]:
    """
    Poll a single sensor node and collect all sensor readings.
    
    Args:
        node: SensorNode instance to poll
        
    Returns:
        Dict with polling results and metrics
    """
    start_time = time.time()
    
    logger.info(f"📡 Polling node: {node.name} ({node.protocol})")
    
    try:
        # Get protocol handler
        handler = ProtocolHandlerFactory.create_handler(
            protocol=node.protocol,
            host=node.host_address,
            port=node.port,
            protocol_config=node.protocol_config or {}
        )
        
        # Get active sensor configurations
        sensor_configs = list(
            node.sensors.filter(is_active=True).values(
                'id', 'register_address', 'data_type', 'scale_factor', 
                'offset', 'unit', 'name', 'validation_min', 'validation_max'
            )
        )
        
        if not sensor_configs:
            logger.warning(f"⚠️ No active sensors for node {node.name}")
            return {
                "node_id": node.id,
                "node_name": node.name,
                "status": "no_sensors",
                "readings_count": 0
            }
        
        # Collect sensor readings using asyncio
        readings = asyncio.run(
            collect_sensor_readings(handler, sensor_configs, node)
        )
        
        # Store readings in TimescaleDB
        stored_count = store_sensor_readings(readings, node)
        
        # Update metrics
        duration = time.time() - start_time
        sensor_read_duration.labels(
            node_name=node.name, 
            protocol=node.protocol
        ).observe(duration)
        
        # Count by quality
        for reading in readings:
            sensor_readings_total.labels(
                node_name=node.name,
                protocol=node.protocol,
                quality=reading.quality
            ).inc()
        
        # Update node status
        node.last_poll_time = timezone.now()
        node.last_poll_success = True
        node.connection_status = 'connected'
        node.save(update_fields=['last_poll_time', 'last_poll_success', 'connection_status'])
        
        logger.info(
            f"✅ Node {node.name} polled successfully: "
            f"{stored_count}/{len(readings)} readings stored"
        )
        
        # Trigger ML analysis if we have good readings
        good_readings = [r for r in readings if r.quality == 'good']
        if good_readings and len(good_readings) >= 3:  # Minimum for ML
            # Trigger ML analysis asynchronously
            analyze_sensor_data_ml.delay(
                node_id=node.id,
                reading_ids=[r.id for r in good_readings if hasattr(r, 'id')]
            )
        
        return {
            "node_id": node.id,
            "node_name": node.name,
            "status": "success",
            "readings_count": len(readings),
            "stored_count": stored_count,
            "good_readings": len(good_readings),
            "duration_ms": int(duration * 1000)
        }
        
    except Exception as e:
        logger.error(f"❌ Error polling node {node.name}: {e}")
        
        # Update node error status
        node.last_poll_time = timezone.now()
        node.last_poll_success = False
        node.connection_status = 'error'
        node.last_error = str(e)[:500]  # Truncate long errors
        node.save(update_fields=[
            'last_poll_time', 'last_poll_success', 
            'connection_status', 'last_error'
        ])
        
        raise


async def collect_sensor_readings(
    handler, 
    sensor_configs: List[Dict[str, Any]], 
    node: SensorNode
) -> List[SensorReading]:
    """
    Collect readings from all sensors on a node using the protocol handler.
    
    Args:
        handler: Protocol handler instance
        sensor_configs: List of sensor configuration dicts
        node: SensorNode instance
        
    Returns:
        List of SensorReading instances (not yet saved to DB)
    """
    readings = []
    
    try:
        # Connect to the device
        connected = await handler.connect()
        if not connected:
            logger.error(f"❌ Failed to connect to {node.name}: {handler.last_error}")
            # Create error readings for all sensors
            for config in sensor_configs:
                readings.append(create_error_reading(config, "Connection failed", node))
            return readings
        
        logger.debug(f"✅ Connected to {node.name}")
        
        # Read each sensor
        for config in sensor_configs:
            timestamp = timezone.now()
            
            try:
                # Read raw value from device
                raw_value = await handler.read_sensor(
                    register_address=config['register_address'],
                    data_type=config['data_type']
                )
                
                # Apply scaling and offset
                scaled_value = apply_sensor_scaling(
                    raw_value, 
                    config['scale_factor'], 
                    config['offset']
                )
                
                # Validate reading
                quality, error_msg = validate_sensor_reading(
                    scaled_value,
                    config['validation_min'],
                    config['validation_max']
                )
                
                # Create reading instance
                reading = SensorReading(
                    sensor_config_id=config['id'],
                    timestamp=timestamp,
                    raw_value=raw_value,
                    processed_value=scaled_value,
                    quality=quality,
                    error_message=error_msg,
                    collection_latency_ms=0  # Will be set during storage
                )
                
                readings.append(reading)
                
                logger.debug(
                    f"📊 {config['name']}: raw={raw_value:.3f} -> "
                    f"processed={scaled_value:.3f} {config['unit']} (quality={quality})"
                )
                
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to read sensor {config['name']} "
                    f"at register {config['register_address']}: {e}"
                )
                
                readings.append(create_error_reading(config, str(e), node))
        
    finally:
        # Always disconnect
        try:
            await handler.disconnect()
            logger.debug(f"🔌 Disconnected from {node.name}")
        except Exception as e:
            logger.warning(f"⚠️ Disconnect error for {node.name}: {e}")
    
    return readings


def create_error_reading(config: Dict[str, Any], error_message: str, node: SensorNode) -> SensorReading:
    """Create an error sensor reading."""
    return SensorReading(
        sensor_config_id=config['id'],
        timestamp=timezone.now(),
        raw_value=0.0,
        processed_value=0.0,
        quality='bad',
        error_message=error_message[:500],  # Truncate long errors
        collection_latency_ms=0
    )


def apply_sensor_scaling(raw_value: float, scale_factor: Optional[Decimal], offset: Optional[Decimal]) -> float:
    """Apply scale factor and offset to raw sensor value."""
    value = float(raw_value)
    
    if scale_factor is not None:
        value *= float(scale_factor)
    
    if offset is not None:
        value += float(offset)
    
    return value


def validate_sensor_reading(
    value: float, 
    min_value: Optional[Decimal], 
    max_value: Optional[Decimal]
) -> tuple[str, Optional[str]]:
    """
    Validate sensor reading against configured limits.
    
    Returns:
        Tuple of (quality, error_message)
    """
    if min_value is not None and value < float(min_value):
        return 'bad', f'Value {value} below minimum {min_value}'
    
    if max_value is not None and value > float(max_value):
        return 'bad', f'Value {value} above maximum {max_value}'
    
    # Additional heuristic checks
    if abs(value) > 1e6:  # Extremely large values
        return 'uncertain', f'Unusually large value: {value}'
    
    return 'good', None


def store_sensor_readings(readings: List[SensorReading], node: SensorNode) -> int:
    """
    Store sensor readings in TimescaleDB with bulk insert for performance.
    
    Args:
        readings: List of SensorReading instances
        node: SensorNode for context
        
    Returns:
        Number of successfully stored readings
    """
    if not readings:
        return 0
    
    try:
        # Bulk create with ignore_conflicts for robustness
        with transaction.atomic():
            stored_readings = SensorReading.objects.bulk_create(
                readings,
                batch_size=1000,
                ignore_conflicts=False  # We want to know about conflicts
            )
        
        logger.info(
            f"💾 Stored {len(stored_readings)} readings for node {node.name} "
            f"to TimescaleDB"
        )
        
        return len(stored_readings)
        
    except Exception as e:
        logger.error(f"💥 Failed to store readings for node {node.name}: {e}")
        
        # Try individual inserts as fallback
        success_count = 0
        for reading in readings:
            try:
                reading.save()
                success_count += 1
            except Exception as individual_error:
                logger.warning(
                    f"⚠️ Failed to store individual reading: {individual_error}"
                )
        
        logger.info(f"💾 Fallback storage: {success_count}/{len(readings)} readings saved")
        return success_count


@shared_task(bind=True, max_retries=2)
def analyze_sensor_data_ml(self, node_id: int, reading_ids: List[int]):
    """
    Analyze sensor readings using ML service for anomaly detection.
    
    Args:
        node_id: SensorNode ID
        reading_ids: List of SensorReading IDs to analyze
    """
    logger.info(f"🤖 Starting ML analysis for node {node_id}")
    
    try:
        # Get node and readings
        node = SensorNode.objects.get(id=node_id)
        readings = SensorReading.objects.filter(
            id__in=reading_ids,
            quality='good'
        ).select_related('sensor_config')
        
        if not readings.exists():
            logger.warning(f"⚠️ No good quality readings found for ML analysis")
            return {"status": "no_good_readings", "node_id": node_id}
        
        # Prepare features for ML service
        features = prepare_ml_features(readings)
        
        # Call ML service
        ml_result = call_ml_service(features, node.name)
        
        if ml_result.get('success'):
            # Store ML results (if you have DiagnosticReport model)
            # store_ml_results(node, readings, ml_result)
            
            ml_predictions_total.labels(status='success').inc()
            
            logger.info(
                f"✅ ML analysis completed for node {node.name}: "
                f"prediction={ml_result.get('prediction')}, "
                f"confidence={ml_result.get('confidence'):.3f}"
            )
            
            return {
                "status": "success",
                "node_id": node_id,
                "prediction": ml_result.get('prediction'),
                "confidence": ml_result.get('confidence'),
                "readings_analyzed": readings.count()
            }
        else:
            ml_predictions_total.labels(status='failed').inc()
            logger.warning(f"⚠️ ML service returned error: {ml_result.get('error')}")
            
            return {
                "status": "ml_service_error",
                "node_id": node_id,
                "error": ml_result.get('error')
            }
            
    except Exception as e:
        ml_predictions_total.labels(status='error').inc()
        logger.error(f"💥 ML analysis failed for node {node_id}: {e}")
        
        # Retry with backoff
        raise self.retry(
            countdown=2 ** self.request.retries,
            exc=e,
            max_retries=2
        )


def prepare_ml_features(readings) -> Dict[str, Any]:
    """
    Convert sensor readings to ML service input format.
    
    Args:
        readings: QuerySet of SensorReading instances
        
    Returns:
        Dict with features for ML service
    """
    features = {
        "timestamp": readings.first().timestamp.isoformat(),
        "sensors": {}
    }
    
    for reading in readings:
        sensor_name = reading.sensor_config.name.lower().replace(' ', '_')
        features["sensors"][sensor_name] = {
            "value": float(reading.processed_value),
            "unit": reading.sensor_config.unit,
            "quality": reading.quality,
            "register_address": reading.sensor_config.register_address
        }
    
    logger.debug(f"🧮 Prepared ML features: {len(features['sensors'])} sensors")
    return features


def call_ml_service(features: Dict[str, Any], node_name: str) -> Dict[str, Any]:
    """
    Call ML service for anomaly detection.
    
    Args:
        features: Prepared sensor features
        node_name: Node name for context
        
    Returns:
        ML service response dict
    """
    try:
        # ML service endpoint (from settings or default)
        ml_service_url = getattr(
            settings, 
            'ML_SERVICE_URL', 
            'http://ml-service:8001'
        )
        
        response = requests.post(
            f"{ml_service_url}/predict",
            json=features,
            timeout=5.0,
            headers={
                "Content-Type": "application/json",
                "X-Node-Name": node_name
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        logger.debug(f"🤖 ML service response: {result}")
        
        return {
            "success": True,
            "prediction": result.get('prediction', 'normal'),
            "confidence": result.get('confidence', 0.0),
            "anomaly_score": result.get('anomaly_score', 0.0),
            "features_used": len(features['sensors'])
        }
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"⚠️ ML service request failed: {e}")
        return {
            "success": False,
            "error": f"Request failed: {e}"
        }
    except Exception as e:
        logger.error(f"💥 Unexpected ML service error: {e}")
        return {
            "success": False,
            "error": f"Unexpected error: {e}"
        }


@shared_task
def cleanup_old_readings():
    """
    Clean up old sensor readings based on retention policy.
    Runs daily to maintain TimescaleDB performance.
    """
    logger.info("🧹 Starting sensor readings cleanup task")
    
    try:
        # Default retention: 5 years (as per requirements)
        retention_days = getattr(settings, 'SENSOR_DATA_RETENTION_DAYS', 5 * 365)
        cutoff_date = timezone.now() - timedelta(days=retention_days)
        
        # Delete old readings (TimescaleDB will handle this efficiently)
        deleted_count = SensorReading.objects.filter(
            timestamp__lt=cutoff_date
        ).delete()[0]
        
        if deleted_count > 0:
            logger.info(f"🗑️ Cleaned up {deleted_count} old sensor readings")
        else:
            logger.debug("✅ No old readings to cleanup")
        
        return {
            "status": "completed",
            "deleted_count": deleted_count,
            "cutoff_date": cutoff_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"💥 Cleanup task failed: {e}")
        raise


@shared_task
def sensor_health_check():
    """
    Perform health checks on all sensor nodes.
    Updates connection status and triggers alerts if needed.
    """
    logger.info("🏥 Starting sensor nodes health check")
    
    try:
        nodes = SensorNode.objects.filter(is_active=True)
        
        results = {
            "healthy": 0,
            "unhealthy": 0,
            "total": nodes.count()
        }
        
        for node in nodes:
            # Check if node was polled recently (within 5 minutes)
            if node.last_poll_time:
                time_since_poll = timezone.now() - node.last_poll_time
                if time_since_poll <= timedelta(minutes=5) and node.last_poll_success:
                    results["healthy"] += 1
                    continue
            
            # Node is unhealthy
            results["unhealthy"] += 1
            
            # Update status if not already marked as error
            if node.connection_status != 'error':
                node.connection_status = 'timeout'
                node.save(update_fields=['connection_status'])
            
            logger.warning(
                f"⚠️ Unhealthy node detected: {node.name} "
                f"(last poll: {node.last_poll_time})"
            )
        
        logger.info(
            f"🏥 Health check completed: {results['healthy']}/{results['total']} nodes healthy"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"💥 Health check failed: {e}")
        raise