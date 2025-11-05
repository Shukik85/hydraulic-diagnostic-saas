"""
Celery Tasks for Sensor Data Collection.

Implements periodic Modbus polling, data validation, storage in TimescaleDB,
and integration with ML service using UCI Hydraulic dataset format.
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
    ['status', 'model_type']
)

ml_prediction_confidence = Histogram(
    'ml_prediction_confidence',
    'ML prediction confidence scores',
    ['model_type', 'prediction']
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
    Analyze sensor readings using ML service with UCI Hydraulic format.
    
    Args:
        node_id: SensorNode ID
        reading_ids: List of SensorReading IDs to analyze
    """
    logger.info(f"🤖 Starting UCI Hydraulic ML analysis for node {node_id}")
    
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
        
        # Prepare features in UCI Hydraulic format
        features = prepare_ml_features_uci(readings)
        
        # Call ML service with UCI format
        ml_result = call_ml_service_uci(features, node.name)
        
        if ml_result.get('success'):
            # Update metrics
            model_type = ml_result.get('model_used', 'ensemble')
            prediction = ml_result.get('prediction', 'unknown')
            confidence = ml_result.get('confidence', 0.0)
            
            ml_predictions_total.labels(
                status='success', 
                model_type=model_type
            ).inc()
            
            ml_prediction_confidence.labels(
                model_type=model_type,
                prediction=prediction
            ).observe(confidence)
            
            # Store ML results (enhanced with UCI metadata)
            # store_uci_ml_results(node, readings, ml_result)
            
            logger.info(
                f"✅ UCI ML analysis completed for node {node.name}: "
                f"prediction={prediction}, confidence={confidence:.3f}, "
                f"model={model_type}"
            )
            
            return {
                "status": "success",
                "node_id": node_id,
                "prediction": prediction,
                "confidence": confidence,
                "model_used": model_type,
                "uci_features_count": len(features.get('features', {})),
                "readings_analyzed": readings.count()
            }
        else:
            ml_predictions_total.labels(
                status='failed', 
                model_type='unknown'
            ).inc()
            
            logger.warning(f"⚠️ ML service returned error: {ml_result.get('error')}")
            
            return {
                "status": "ml_service_error",
                "node_id": node_id,
                "error": ml_result.get('error')
            }
            
    except Exception as e:
        ml_predictions_total.labels(
            status='error', 
            model_type='unknown'
        ).inc()
        
        logger.error(f"💥 ML analysis failed for node {node_id}: {e}")
        
        # Retry with backoff
        raise self.retry(
            countdown=2 ** self.request.retries,
            exc=e,
            max_retries=2
        )


def prepare_ml_features_uci(readings) -> Dict[str, Any]:
    """
    Convert sensor readings to UCI Hydraulic dataset format.
    
    UCI Hydraulic features:
    - PS1..PS6: Pressure sensors (bar)
    - TS1..TS4: Temperature sensors (°C)  
    - FS1..FS2: Flow sensors (L/min)
    - VS1: Vibration sensor (mm/s)
    - EPS1: Motor power (% or scaled)
    - CE, CP, SE: Efficiency metrics (optional)
    
    Args:
        readings: QuerySet of SensorReading instances
        
    Returns:
        Dict with UCI-formatted features for ML service
    """
    # Initialize UCI feature structure
    uci_features = {
        # Pressure sensors (6 channels in UCI dataset)
        "PS1": None, "PS2": None, "PS3": None, "PS4": None, "PS5": None, "PS6": None,
        # Temperature sensors (4 channels in UCI dataset)  
        "TS1": None, "TS2": None, "TS3": None, "TS4": None,
        # Flow sensors (2 channels in UCI dataset)
        "FS1": None, "FS2": None,
        # Vibration sensor (1 channel)
        "VS1": None,
        # Motor/Power metrics
        "EPS1": None,  # Motor power consumption
        # Efficiency metrics (optional, can be computed)
        "CE": None,    # Cooling efficiency  
        "CP": None,    # Cooling power
        "SE": None     # Stability efficiency
    }
    
    # Map our demo sensor readings to UCI format
    for reading in readings:
        sensor_name = reading.sensor_config.name.lower().replace(' ', '_')
        value = float(reading.processed_value)
        unit = reading.sensor_config.unit.lower()
        
        # Map based on sensor name and characteristics
        if "pressure" in sensor_name:
            # System Pressure -> PS1 (primary pressure sensor)
            if uci_features["PS1"] is None:
                uci_features["PS1"] = value
            elif uci_features["PS2"] is None:  # Additional pressure sensors
                uci_features["PS2"] = value
                
        elif "temperature" in sensor_name:
            # Oil Temperature -> TS1 (primary temperature sensor)  
            if uci_features["TS1"] is None:
                uci_features["TS1"] = value
            elif uci_features["TS2"] is None:  # Additional temperature sensors
                uci_features["TS2"] = value
                
        elif "flow" in sensor_name:
            # Flow Rate -> FS1 (primary flow sensor)
            if uci_features["FS1"] is None:
                uci_features["FS1"] = value
            elif uci_features["FS2"] is None:  # Secondary flow sensor
                uci_features["FS2"] = value
                
        elif "vibration" in sensor_name:
            # Vibration Level -> VS1
            uci_features["VS1"] = value
            
        elif "speed" in sensor_name or "motor" in sensor_name:
            # Motor Speed -> EPS1 (convert RPM to power % approximation)
            # Normalize typical motor speed (1500 RPM) to 0-100% scale
            normalized_power = min(100.0, max(0.0, value / 1500.0 * 100.0))
            uci_features["EPS1"] = normalized_power
    
    # Compute efficiency metrics if we have enough data
    if uci_features["PS1"] and uci_features["FS1"] and uci_features["TS1"]:
        # Cooling Efficiency (CE): simplified metric based on temp/pressure ratio
        if uci_features["PS1"] > 0:
            uci_features["CE"] = min(100.0, (100 - uci_features["TS1"]) / uci_features["PS1"] * 1000)
        
        # Cooling Power (CP): flow * pressure efficiency proxy
        if uci_features["FS1"] > 0:
            uci_features["CP"] = uci_features["PS1"] * uci_features["FS1"] / 100.0
            
        # Stability Efficiency (SE): inverse of vibration if available
        if uci_features["VS1"] is not None and uci_features["VS1"] > 0:
            uci_features["SE"] = max(0.0, 100.0 - (uci_features["VS1"] * 10))
    
    # Remove None values (some models may require this)
    clean_features = {k: v for k, v in uci_features.items() if v is not None}
    
    logger.debug(f"🧠 Prepared UCI features: {len(clean_features)} active channels")
    logger.debug(f"📊 Features: {list(clean_features.keys())}")
    
    return {
        "tag": "REAL_UCI_HYDRAULIC_DATA",
        "timestamp": readings.first().timestamp.isoformat(),
        "node_id": readings.first().sensor_config.node_id,
        "features": clean_features,
        "metadata": {
            "total_sensors": len(readings),
            "feature_mapping": {
                "pressure_sensors": [k for k in clean_features.keys() if k.startswith('PS')],
                "temperature_sensors": [k for k in clean_features.keys() if k.startswith('TS')], 
                "flow_sensors": [k for k in clean_features.keys() if k.startswith('FS')],
                "vibration_sensors": [k for k in clean_features.keys() if k.startswith('VS')],
                "power_sensors": [k for k in clean_features.keys() if k.startswith('EPS')]
            }
        }
    }


def call_ml_service_uci(features: Dict[str, Any], node_name: str) -> Dict[str, Any]:
    """
    Call ML service for UCI Hydraulic anomaly detection.
    
    Enhanced for real UCI models with proper error handling and retries.
    
    Args:
        features: Prepared UCI features
        node_name: Node name for context
        
    Returns:
        ML service response dict with UCI-specific fields
    """
    try:
        # ML service configuration
        ml_service_url = getattr(
            settings, 
            'ML_SERVICE_URL', 
            'http://ml-service:8001'
        )
        
        # Check if service supports UCI-specific endpoint
        predict_endpoint = getattr(
            settings,
            'ML_PREDICT_ENDPOINT', 
            '/predict'  # Could be /predict/uci or /predict/v2
        )
        
        # Prepare request payload for real models
        payload = {
            **features,  # Includes tag, timestamp, features, metadata
            "request_id": f"{node_name}_{int(time.time())}",
            "model_preference": getattr(settings, 'ML_MODEL_PREFERENCE', 'ensemble'),
            "confidence_threshold": getattr(settings, 'ML_CONFIDENCE_THRESHOLD', 0.7)
        }
        
        # Make request with extended timeout for model inference
        response = requests.post(
            f"{ml_service_url}{predict_endpoint}",
            json=payload,
            timeout=15.0,  # Extended for real model inference
            headers={
                "Content-Type": "application/json",
                "X-Node-Name": node_name,
                "X-Data-Format": "UCI-Hydraulic",
                "X-Request-Source": "celery-ingestion"
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Enhanced logging for real models
        model_info = result.get('model_info', {})
        logger.info(
            f"🤖 UCI ML response: prediction={result.get('prediction')}, "
            f"confidence={result.get('confidence', 0):.3f}, "
            f"model={model_info.get('primary_model', 'unknown')}"
        )
        
        # Log additional UCI-specific metrics if available
        if 'feature_importance' in result:
            top_features = result['feature_importance'][:3]  # Top 3 features
            logger.debug(f"📊 Top features: {top_features}")
        
        return {
            "success": True,
            "prediction": result.get('prediction', 'normal'),
            "confidence": result.get('confidence', 0.0),
            "anomaly_score": result.get('anomaly_score', 0.0),
            "model_used": model_info.get('primary_model', 'ensemble'),
            "model_version": model_info.get('version', 'unknown'),
            "features_used": len(features.get('features', {})),
            "processing_time_ms": result.get('processing_time_ms', 0),
            "uci_specific": {
                "fault_probability": result.get('fault_probability', 0.0),
                "component_health": result.get('component_health', {}),
                "maintenance_recommendation": result.get('maintenance_recommendation'),
                "feature_importance": result.get('feature_importance', [])
            }
        }
        
    except requests.exceptions.Timeout:
        logger.warning(f"⏰ ML service timeout for {node_name} (models may be training)")
        return {
            "success": False,
            "error": "ML service timeout - models may be training",
            "retry_recommended": True
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


@shared_task
def timescaledb_maintenance():
    """
    Perform TimescaleDB maintenance operations.
    
    Includes:
    - Hypertable compression policy updates
    - Statistics refresh
    - Chunk analysis
    """
    logger.info("🔧 Starting TimescaleDB maintenance")
    
    try:
        from django.db import connection
        
        maintenance_results = {}
        
        # Update compression policies
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT compress_chunk(i, if_not_compressed => true) 
                FROM show_chunks('sensors_reading') i
                WHERE age(now(), range_end) > INTERVAL '1 day';
            """)
            
            # Get hypertable stats
            cursor.execute("""
                SELECT 
                    hypertable_name,
                    num_chunks,
                    table_size,
                    index_size,
                    total_size
                FROM timescaledb_information.hypertables_size
                WHERE hypertable_name = 'sensors_reading';
            """)
            
            stats = cursor.fetchone()
            if stats:
                maintenance_results['hypertable_stats'] = {
                    'num_chunks': stats[1],
                    'table_size_bytes': stats[2],
                    'index_size_bytes': stats[3],
                    'total_size_bytes': stats[4]
                }
        
        logger.info(f"✅ TimescaleDB maintenance completed: {maintenance_results}")
        return maintenance_results
        
    except Exception as e:
        logger.error(f"💥 TimescaleDB maintenance failed: {e}")
        raise


@shared_task  
def debug_sensor_status():
    """Debug task for development - logs sensor system status."""
    if not settings.DEBUG:
        return {"status": "debug_only"}
    
    try:
        nodes_count = SensorNode.objects.filter(is_active=True).count()
        recent_readings = SensorReading.objects.filter(
            timestamp__gte=timezone.now() - timedelta(minutes=5)
        ).count()
        
        logger.info(
            f"🐛 Debug status: {nodes_count} active nodes, "
            f"{recent_readings} readings in last 5 min"
        )
        
        return {
            "active_nodes": nodes_count,
            "recent_readings": recent_readings,
            "timestamp": timezone.now().isoformat()
        }
        
    except Exception as e:
        logger.warning(f"🐛 Debug status error: {e}")
        return {"status": "error", "error": str(e)}


# Legacy function for backward compatibility
def prepare_ml_features(readings) -> Dict[str, Any]:
    """
    Legacy feature preparation - redirects to UCI format.
    Maintains backward compatibility while using new UCI format.
    """
    return prepare_ml_features_uci(readings)


def call_ml_service(features: Dict[str, Any], node_name: str) -> Dict[str, Any]:
    """
    Legacy ML service call - redirects to UCI format.
    Maintains backward compatibility while using new UCI format.
    """
    return call_ml_service_uci(features, node_name)