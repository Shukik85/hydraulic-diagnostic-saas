"""
Base Protocol Handler Interface.

Defines the standard interface for all industrial protocol handlers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ProtocolError(Exception):
    """Base exception for protocol communication errors."""
    pass


class ConnectionError(ProtocolError):
    """Raised when connection to device fails."""
    pass


class ReadError(ProtocolError):
    """Raised when reading data from device fails."""
    pass


class ValidationError(ProtocolError):
    """Raised when data validation fails."""
    pass


class SensorReading:
    """Standard sensor reading data structure."""
    
    def __init__(
        self,
        sensor_config_id: str,
        timestamp: datetime,
        raw_value: float,
        quality: str = 'good',
        error_message: Optional[str] = None,
        collection_latency_ms: Optional[int] = None
    ):
        self.sensor_config_id = sensor_config_id
        self.timestamp = timestamp
        self.raw_value = raw_value
        self.quality = quality
        self.error_message = error_message
        self.collection_latency_ms = collection_latency_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sensor_config_id': self.sensor_config_id,
            'timestamp': self.timestamp.isoformat(),
            'raw_value': self.raw_value,
            'quality': self.quality,
            'error_message': self.error_message,
            'collection_latency_ms': self.collection_latency_ms,
        }
    
    def __repr__(self) -> str:
        return f"SensorReading(sensor={self.sensor_config_id}, value={self.raw_value}, quality={self.quality})"


class BaseProtocolHandler(ABC):
    """Abstract base class for all protocol handlers.
    
    Each protocol handler implements this interface to provide
    standardized communication with industrial devices.
    """
    
    def __init__(
        self, 
        host: str, 
        port: int, 
        protocol_config: Optional[Dict[str, Any]] = None
    ):
        self.host = host
        self.port = port
        self.protocol_config = protocol_config or {}
        self.is_connected = False
        self.last_error: Optional[str] = None
        
        # Connection parameters with defaults
        self.connection_timeout = self.protocol_config.get('connection_timeout', 5.0)
        self.read_timeout = self.protocol_config.get('read_timeout', 3.0)
        self.retry_attempts = self.protocol_config.get('retry_attempts', 3)
        self.retry_delay = self.protocol_config.get('retry_delay', 1.0)
    
    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Return the protocol name (e.g., 'modbus_tcp', 'opcua')."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the device.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the device."""
        pass
    
    @abstractmethod
    async def read_sensor(
        self, 
        register_address: int, 
        data_type: str,
        **kwargs
    ) -> float:
        """Read a single sensor value.
        
        Args:
            register_address: Protocol-specific address (e.g., Modbus register)
            data_type: Expected data type ('float32', 'int16', etc.)
            **kwargs: Protocol-specific parameters
            
        Returns:
            float: Raw sensor value
            
        Raises:
            ReadError: If reading fails
        """
        pass
    
    @abstractmethod
    async def read_multiple_sensors(
        self, 
        sensor_configs: List[Dict[str, Any]]
    ) -> List[SensorReading]:
        """Read multiple sensors in one operation (if supported).
        
        Args:
            sensor_configs: List of sensor configuration dictionaries
            
        Returns:
            List[SensorReading]: List of sensor readings
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check protocol handler health and device connectivity.
        
        Returns:
            Dict containing health status information.
        """
        try:
            if not self.is_connected:
                connected = await self.connect()
                if not connected:
                    return {
                        'status': 'error',
                        'message': f'Failed to connect to {self.host}:{self.port}',
                        'last_error': self.last_error
                    }
            
            # Protocol-specific health check can be overridden
            await self._protocol_health_check()
            
            return {
                'status': 'healthy',
                'protocol': self.protocol_name,
                'host': self.host,
                'port': self.port,
                'connected': self.is_connected,
                'config': self.protocol_config
            }
            
        except Exception as e:
            logger.error(f"Health check failed for {self.protocol_name}://{self.host}:{self.port}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'protocol': self.protocol_name,
                'host': self.host,
                'port': self.port
            }
    
    async def _protocol_health_check(self) -> None:
        """Protocol-specific health check implementation.
        
        Can be overridden by subclasses for custom health checks.
        Default implementation does nothing.
        """
        pass
    
    def _validate_data_type(self, data_type: str) -> None:
        """Validate that data type is supported.
        
        Args:
            data_type: Data type to validate
            
        Raises:
            ValidationError: If data type is not supported
        """
        valid_types = ['float32', 'float64', 'int16', 'int32', 'uint16', 'uint32', 'bool']
        if data_type not in valid_types:
            raise ValidationError(f"Unsupported data type: {data_type}. Valid types: {valid_types}")
    
    def _log_error(self, operation: str, error: Exception) -> None:
        """Log and store error information.
        
        Args:
            operation: Description of the operation that failed
            error: The exception that occurred
        """
        error_message = f"{self.protocol_name} {operation} failed: {str(error)}"
        self.last_error = error_message
        logger.error(error_message)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.host}:{self.port})"


__all__ = [
    'BaseProtocolHandler',
    'SensorReading',
    'ProtocolError',
    'ConnectionError',
    'ReadError',
    'ValidationError',
]
