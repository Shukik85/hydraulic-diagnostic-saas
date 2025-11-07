"""
HTTP JSON Protocol Handler for Sensor Data Collection.

Provides REST/HTTP JSON interface for sensor data collection.
Useful for devices that expose sensor data via HTTP APIs.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
import aiohttp
import time
import logging

from .base import (
    BaseProtocolHandler, 
    SensorReading, 
    ProtocolError, 
    ConnectionError, 
    ReadError, 
    ValidationError
)

logger = logging.getLogger(__name__)


class HTTPJSONHandler(BaseProtocolHandler):
    """HTTP JSON protocol handler for REST-based sensor data collection."""
    
    @property
    def protocol_name(self) -> str:
        return 'http_json'
    
    def __init__(self, host: str, port: int, protocol_config: Optional[Dict[str, Any]] = None):
        super().__init__(host, port, protocol_config)
        
        # HTTP-specific configuration
        self.use_ssl = self.protocol_config.get('use_ssl', False)
        self.base_path = self.protocol_config.get('base_path', '/api/sensors')
        self.auth_type = self.protocol_config.get('auth_type', None)  # None, 'basic', 'bearer'
        self.username = self.protocol_config.get('username')
        self.password = self.protocol_config.get('password')
        self.api_key = self.protocol_config.get('api_key')
        self.headers = self.protocol_config.get('headers', {})
        
        self.session: Optional[aiohttp.ClientSession] = None
    
    def _get_base_url(self) -> str:
        """Construct base URL for HTTP requests."""
        scheme = 'https' if self.use_ssl else 'http'
        return f"{scheme}://{self.host}:{self.port}{self.base_path}"
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        auth_headers = {}
        
        if self.auth_type == 'bearer' and self.api_key:
            auth_headers['Authorization'] = f'Bearer {self.api_key}'
        elif self.auth_type == 'basic' and self.username and self.password:
            import base64
            credentials = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            auth_headers['Authorization'] = f'Basic {credentials}'
        
        # Add custom headers
        auth_headers.update(self.headers)
        
        return auth_headers
    
    async def connect(self) -> bool:
        """Establish HTTP session."""
        try:
            timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._get_auth_headers()
            )
            
            # Test connection with a health check request
            health_url = f"{self._get_base_url()}/health"
            
            try:
                async with self.session.get(health_url) as response:
                    if response.status < 400:
                        self.is_connected = True
                        logger.info(f"Connected to HTTP JSON API at {self._get_base_url()}")
                        return True
                    else:
                        logger.warning(f"HTTP health check returned status {response.status}")
                        
            except aiohttp.ClientError:
                # Health endpoint might not exist, try a simple GET to base path
                try:
                    async with self.session.get(self._get_base_url()) as response:
                        if response.status < 500:  # Accept any non-server-error status
                            self.is_connected = True
                            logger.info(f"Connected to HTTP JSON API at {self._get_base_url()}")
                            return True
                except aiohttp.ClientError as e:
                    self.last_error = f"HTTP connection test failed: {e}"
                    logger.error(self.last_error)
                    await self.session.close()
                    self.session = None
                    return False
            
            self.is_connected = False
            self.last_error = "HTTP connection test failed"
            await self.session.close()
            self.session = None
            return False
            
        except Exception as e:
            self.is_connected = False
            self._log_error("connect", e)
            if self.session:
                await self.session.close()
                self.session = None
            return False
    
    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self.session:
            try:
                await self.session.close()
                logger.info(f"Disconnected from HTTP JSON API at {self._get_base_url()}")
            except Exception as e:
                logger.warning(f"Error during HTTP disconnect: {e}")
            finally:
                self.session = None
                self.is_connected = False
    
    async def read_sensor(
        self, 
        register_address: int, 
        data_type: str,
        **kwargs
    ) -> float:
        """Read a single sensor value via HTTP JSON API.
        
        Args:
            register_address: Used as sensor ID in the HTTP request
            data_type: Expected data type (for validation)
            **kwargs: Additional parameters (endpoint, field_name, etc.)
            
        Returns:
            float: Sensor value
            
        Raises:
            ReadError: If reading fails
        """
        if not self.is_connected or not self.session:
            raise ReadError("Not connected to HTTP JSON API")
        
        self._validate_data_type(data_type)
        
        # Allow custom endpoint or use sensor ID
        endpoint = kwargs.get('endpoint', f'/sensor/{register_address}')
        field_name = kwargs.get('field_name', 'value')
        
        url = f"{self._get_base_url()}{endpoint}"
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    error_msg = f"HTTP request failed with status {response.status}"
                    raise ReadError(error_msg)
                
                data = await response.json()
                
                # Extract sensor value from JSON response
                if isinstance(data, dict):
                    if field_name in data:
                        raw_value = data[field_name]
                    elif 'data' in data and field_name in data['data']:
                        raw_value = data['data'][field_name]
                    elif 'value' in data:
                        raw_value = data['value']
                    else:
                        raise ReadError(f"Field '{field_name}' not found in response: {data}")
                elif isinstance(data, (int, float)):
                    raw_value = data
                else:
                    raise ReadError(f"Unexpected response format: {type(data)}")
                
                # Convert to float
                try:
                    value = float(raw_value)
                except (ValueError, TypeError) as e:
                    raise ReadError(f"Cannot convert '{raw_value}' to float: {e}")
                
                logger.debug(
                    f"Read HTTP JSON sensor {register_address}: "
                    f"{field_name}={value} from {url}"
                )
                
                return value
                
        except aiohttp.ClientError as e:
            error_msg = f"HTTP request error for sensor {register_address}: {e}"
            self._log_error("read_sensor", e)
            raise ReadError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error reading sensor {register_address}: {e}"
            self._log_error("read_sensor", e)
            raise ReadError(error_msg) from e
    
    async def read_multiple_sensors(
        self, 
        sensor_configs: List[Dict[str, Any]]
    ) -> List[SensorReading]:
        """Read multiple sensors via HTTP JSON API.
        
        Can use individual sensor endpoints or a bulk endpoint if available.
        
        Args:
            sensor_configs: List of sensor configuration dictionaries
            
        Returns:
            List[SensorReading]: List of sensor readings
        """
        readings = []
        read_start = time.time()
        
        # Check if bulk endpoint is configured
        bulk_endpoint = self.protocol_config.get('bulk_endpoint')
        
        if bulk_endpoint:
            # Use bulk endpoint for efficiency
            readings = await self._read_bulk_sensors(sensor_configs, bulk_endpoint)
        else:
            # Read sensors individually
            for sensor_config in sensor_configs:
                timestamp = datetime.now()
                
                try:
                    raw_value = await self.read_sensor(
                        register_address=sensor_config['register_address'],
                        data_type=sensor_config['data_type'],
                        endpoint=sensor_config.get('endpoint'),
                        field_name=sensor_config.get('field_name', 'value')
                    )
                    
                    reading = SensorReading(
                        sensor_config_id=sensor_config['id'],
                        timestamp=timestamp,
                        raw_value=raw_value,
                        quality='good',
                        collection_latency_ms=int((time.time() - read_start) * 1000)
                    )
                    
                except Exception as e:
                    logger.warning(
                        f"Failed to read HTTP sensor {sensor_config['id']} "
                        f"at address {sensor_config['register_address']}: {e}"
                    )
                    
                    reading = SensorReading(
                        sensor_config_id=sensor_config['id'],
                        timestamp=timestamp,
                        raw_value=0.0,
                        quality='bad',
                        error_message=str(e),
                        collection_latency_ms=int((time.time() - read_start) * 1000)
                    )
                
                readings.append(reading)
        
        return readings
    
    async def _read_bulk_sensors(
        self, 
        sensor_configs: List[Dict[str, Any]], 
        bulk_endpoint: str
    ) -> List[SensorReading]:
        """Read multiple sensors using a bulk endpoint."""
        readings = []
        timestamp = datetime.now()
        read_start = time.time()
        
        # Prepare sensor IDs for bulk request
        sensor_ids = [config['register_address'] for config in sensor_configs]
        
        url = f"{self._get_base_url()}{bulk_endpoint}"
        
        try:
            # POST request with sensor IDs
            async with self.session.post(url, json={'sensor_ids': sensor_ids}) as response:
                if response.status != 200:
                    raise ReadError(f"Bulk HTTP request failed with status {response.status}")
                
                data = await response.json()
                
                # Process bulk response
                for sensor_config in sensor_configs:
                    sensor_id = str(sensor_config['register_address'])
                    
                    if sensor_id in data:
                        try:
                            raw_value = float(data[sensor_id]['value'])
                            quality = 'good'
                            error_message = None
                        except (KeyError, ValueError, TypeError) as e:
                            raw_value = 0.0
                            quality = 'bad'
                            error_message = f"Failed to parse bulk response for sensor {sensor_id}: {e}"
                    else:
                        raw_value = 0.0
                        quality = 'bad'
                        error_message = f"Sensor {sensor_id} not found in bulk response"
                    
                    reading = SensorReading(
                        sensor_config_id=sensor_config['id'],
                        timestamp=timestamp,
                        raw_value=raw_value,
                        quality=quality,
                        error_message=error_message,
                        collection_latency_ms=int((time.time() - read_start) * 1000)
                    )
                    
                    readings.append(reading)
        
        except Exception as e:
            logger.error(f"Bulk HTTP read failed: {e}")
            # Create error readings for all sensors
            for sensor_config in sensor_configs:
                reading = SensorReading(
                    sensor_config_id=sensor_config['id'],
                    timestamp=timestamp,
                    raw_value=0.0,
                    quality='bad',
                    error_message=f"Bulk read error: {e}",
                    collection_latency_ms=int((time.time() - read_start) * 1000)
                )
                readings.append(reading)
        
        return readings
    
    async def _protocol_health_check(self) -> None:
        """HTTP-specific health check."""
        if not self.session:
            raise ConnectionError("No active HTTP session")
        
        # Try to access the base endpoint
        try:
            async with self.session.get(self._get_base_url()) as response:
                if response.status >= 500:
                    raise ConnectionError(f"HTTP server error: {response.status}")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"HTTP health check failed: {e}")


__all__ = [
    'HTTPJSONHandler',
]
