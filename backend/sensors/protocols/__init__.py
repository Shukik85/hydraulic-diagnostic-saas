"""
Industrial Protocol Handlers for Sensor Data Collection.

Supported protocols:
- Modbus TCP/RTU
- OPC UA
- HTTP JSON
- MQTT (future)
"""

from typing import Dict, Type
from .base import BaseProtocolHandler
from .modbus_handler import ModbusTCPHandler, ModbusRTUHandler
from .http_handler import HTTPJSONHandler

# Protocol registry
_PROTOCOL_REGISTRY: Dict[str, Type[BaseProtocolHandler]] = {}


def register_protocol(protocol_name: str, handler_class: Type[BaseProtocolHandler]) -> None:
    """Register a protocol handler."""
    _PROTOCOL_REGISTRY[protocol_name] = handler_class


def get_protocol_handler(protocol_name: str) -> Type[BaseProtocolHandler] | None:
    """Get protocol handler class by name."""
    return _PROTOCOL_REGISTRY.get(protocol_name)


def initialize_protocol_registry() -> None:
    """Initialize and register all available protocol handlers."""
    # Register Modbus handlers
    register_protocol('modbus_tcp', ModbusTCPHandler)
    register_protocol('modbus_rtu', ModbusRTUHandler)
    
    # Register HTTP handler
    register_protocol('http_json', HTTPJSONHandler)
    
    # OPC UA handler will be registered when available
    try:
        from .opcua_handler import OPCUAHandler
        register_protocol('opcua', OPCUAHandler)
    except ImportError:
        # OPC UA dependencies not installed
        pass


def list_available_protocols() -> list[str]:
    """List all registered protocol names."""
    return list(_PROTOCOL_REGISTRY.keys())


__all__ = [
    'BaseProtocolHandler',
    'ModbusTCPHandler', 
    'ModbusRTUHandler',
    'HTTPJSONHandler',
    'register_protocol',
    'get_protocol_handler',
    'initialize_protocol_registry',
    'list_available_protocols',
]
