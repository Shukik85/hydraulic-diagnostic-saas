"""
Modbus TCP/RTU Protocol Handler for Industrial Sensors.

Implements Modbus communication for hydraulic system sensor data collection.
Supports both TCP and RTU variants with proper error handling and retry logic.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import asyncio
import struct
import time
import logging

try:
    from pymodbus.client import ModbusTcpClient, ModbusSerialClient
    from pymodbus.constants import Endian
    from pymodbus.payload import BinaryPayloadDecoder
    from pymodbus.exceptions import ModbusException, ConnectionException
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False
    ModbusTcpClient = None
    ModbusSerialClient = None
    ModbusException = Exception
    ConnectionException = Exception

from .base import (
    BaseProtocolHandler, 
    SensorReading, 
    ProtocolError, 
    ConnectionError, 
    ReadError, 
    ValidationError
)

logger = logging.getLogger(__name__)


class ModbusHandler(BaseProtocolHandler):
    """Base Modbus handler with common functionality."""
    
    def __init__(self, host: str, port: int, protocol_config: Optional[Dict[str, Any]] = None):
        if not MODBUS_AVAILABLE:
            raise ImportError(
                "pymodbus is required for Modbus support. "
                "Install with: pip install pymodbus>=3.0.0"
            )
        
        super().__init__(host, port, protocol_config)
        
        # Modbus-specific configuration
        self.unit_id = self.protocol_config.get('unit_id', 1)
        self.byte_order = self.protocol_config.get('byte_order', 'big')  # 'big' or 'little'
        self.word_order = self.protocol_config.get('word_order', 'big')  # 'big' or 'little'
        
        self.client: Optional[Union[ModbusTcpClient, ModbusSerialClient]] = None
    
    def _get_endian(self) -> tuple:
        """Get pymodbus Endian constants based on configuration."""
        byte_endian = Endian.BIG if self.byte_order == 'big' else Endian.LITTLE
        word_endian = Endian.BIG if self.word_order == 'big' else Endian.LITTLE
        return byte_endian, word_endian
    
    def _decode_value(self, registers: list, data_type: str) -> float:
        """Decode register values based on data type.
        
        Args:
            registers: List of register values from Modbus read
            data_type: Target data type ('float32', 'int16', etc.)
            
        Returns:
            float: Decoded value
            
        Raises:
            ValidationError: If decoding fails
        """
        try:
            byte_endian, word_endian = self._get_endian()
            decoder = BinaryPayloadDecoder.fromRegisters(
                registers, 
                byteorder=byte_endian,
                wordorder=word_endian
            )
            
            if data_type == 'float32':
                return float(decoder.decode_32bit_float())
            elif data_type == 'float64':
                return float(decoder.decode_64bit_float())
            elif data_type == 'int16':
                return float(decoder.decode_16bit_int())
            elif data_type == 'int32':
                return float(decoder.decode_32bit_int())
            elif data_type == 'uint16':
                return float(decoder.decode_16bit_uint())
            elif data_type == 'uint32':
                return float(decoder.decode_32bit_uint())
            elif data_type == 'bool':
                return float(1 if registers[0] != 0 else 0)
            else:
                raise ValidationError(f"Unsupported data type: {data_type}")
                
        except Exception as e:
            raise ValidationError(f"Failed to decode {data_type} from registers {registers}: {e}")
    
    def _get_register_count(self, data_type: str) -> int:
        """Get number of registers needed for data type."""
        if data_type in ['int16', 'uint16', 'bool']:
            return 1
        elif data_type in ['float32', 'int32', 'uint32']:
            return 2
        elif data_type == 'float64':
            return 4
        else:
            raise ValidationError(f"Unknown register count for data type: {data_type}")
    
    async def read_sensor(
        self, 
        register_address: int, 
        data_type: str,
        **kwargs
    ) -> float:
        """Read a single sensor value via Modbus.
        
        Args:
            register_address: Modbus register address
            data_type: Data type to decode ('float32', 'int16', etc.)
            **kwargs: Additional Modbus parameters (unit_id override, etc.)
            
        Returns:
            float: Sensor value
            
        Raises:
            ReadError: If reading fails
        """
        self._validate_data_type(data_type)
        
        if not self.is_connected:
            raise ReadError("Not connected to Modbus device")
        
        unit_id = kwargs.get('unit_id', self.unit_id)
        register_count = self._get_register_count(data_type)
        
        try:
            # Read holding registers
            result = self.client.read_holding_registers(
                address=register_address,
                count=register_count,
                unit=unit_id
            )
            
            if result.isError():
                raise ReadError(f"Modbus read error: {result}")
            
            # Decode the value
            decoded_value = self._decode_value(result.registers, data_type)
            
            logger.debug(
                f"Read Modbus register {register_address} (unit {unit_id}): "
                f"raw={result.registers} -> {data_type}={decoded_value}"
            )
            
            return decoded_value
            
        except ModbusException as e:
            error_msg = f"Modbus read failed for register {register_address}: {e}"
            self._log_error("read_sensor", e)
            raise ReadError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error reading register {register_address}: {e}"
            self._log_error("read_sensor", e)
            raise ReadError(error_msg) from e
    
    async def read_multiple_sensors(
        self, 
        sensor_configs: List[Dict[str, Any]]
    ) -> List[SensorReading]:
        """Read multiple sensors efficiently using batch reads where possible.
        
        Args:
            sensor_configs: List of sensor configuration dictionaries
            
        Returns:
            List[SensorReading]: List of sensor readings
        """
        readings = []
        read_start = time.time()
        
        # Group sensors by unit_id for efficient batch reading
        sensors_by_unit = {}
        for config in sensor_configs:
            unit_id = config.get('unit_id', self.unit_id)
            if unit_id not in sensors_by_unit:
                sensors_by_unit[unit_id] = []
            sensors_by_unit[unit_id].append(config)
        
        for unit_id, unit_sensors in sensors_by_unit.items():
            for sensor_config in unit_sensors:
                timestamp = datetime.now()
                
                try:
                    raw_value = await self.read_sensor(
                        register_address=sensor_config['register_address'],
                        data_type=sensor_config['data_type'],
                        unit_id=unit_id
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
                        f"Failed to read sensor {sensor_config['id']} "
                        f"at register {sensor_config['register_address']}: {e}"
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
    
    async def _protocol_health_check(self) -> None:
        """Modbus-specific health check - read a test register."""
        # Try to read register 0 to verify communication
        try:
            test_result = self.client.read_holding_registers(
                address=0, 
                count=1, 
                unit=self.unit_id
            )
            
            if test_result.isError():
                logger.debug(f"Modbus health check: register 0 read failed (expected for some devices)")
                # This is often expected - many devices don't have register 0
            else:
                logger.debug(f"Modbus health check: register 0 = {test_result.registers}")
                
        except Exception as e:
            logger.debug(f"Modbus health check error: {e}")
            # Health check failure is not critical


class ModbusTCPHandler(ModbusHandler):
    """Modbus TCP protocol handler."""
    
    @property
    def protocol_name(self) -> str:
        return 'modbus_tcp'
    
    async def connect(self) -> bool:
        """Establish Modbus TCP connection."""
        try:
            self.client = ModbusTcpClient(
                host=self.host,
                port=self.port,
                timeout=self.connection_timeout
            )
            
            # Attempt connection
            connected = self.client.connect()
            
            if connected:
                self.is_connected = True
                logger.info(f"Connected to Modbus TCP device at {self.host}:{self.port}")
                return True
            else:
                self.is_connected = False
                error_msg = f"Failed to connect to Modbus TCP device at {self.host}:{self.port}"
                self.last_error = error_msg
                logger.error(error_msg)
                return False
                
        except Exception as e:
            self.is_connected = False
            error_msg = f"Modbus TCP connection error: {e}"
            self._log_error("connect", e)
            return False
    
    async def disconnect(self) -> None:
        """Close Modbus TCP connection."""
        if self.client:
            try:
                self.client.close()
                logger.info(f"Disconnected from Modbus TCP device at {self.host}:{self.port}")
            except Exception as e:
                logger.warning(f"Error during Modbus TCP disconnect: {e}")
            finally:
                self.is_connected = False
                self.client = None


class ModbusRTUHandler(ModbusHandler):
    """Modbus RTU protocol handler (serial communication)."""
    
    @property
    def protocol_name(self) -> str:
        return 'modbus_rtu'
    
    async def connect(self) -> bool:
        """Establish Modbus RTU (serial) connection.
        
        Note: For RTU, 'host' should be the serial port device path.
        """
        try:
            # RTU-specific configuration
            baudrate = self.protocol_config.get('baudrate', 9600)
            parity = self.protocol_config.get('parity', 'N')  # N, E, O
            stopbits = self.protocol_config.get('stopbits', 1)
            bytesize = self.protocol_config.get('bytesize', 8)
            
            self.client = ModbusSerialClient(
                port=self.host,  # Serial port path (e.g., '/dev/ttyUSB0' or 'COM1')
                baudrate=baudrate,
                parity=parity,
                stopbits=stopbits,
                bytesize=bytesize,
                timeout=self.connection_timeout
            )
            
            # Attempt connection
            connected = self.client.connect()
            
            if connected:
                self.is_connected = True
                logger.info(
                    f"Connected to Modbus RTU device at {self.host} "
                    f"(baudrate={baudrate}, parity={parity})"
                )
                return True
            else:
                self.is_connected = False
                error_msg = f"Failed to connect to Modbus RTU device at {self.host}"
                self.last_error = error_msg
                logger.error(error_msg)
                return False
                
        except Exception as e:
            self.is_connected = False
            error_msg = f"Modbus RTU connection error: {e}"
            self._log_error("connect", e)
            return False
    
    async def disconnect(self) -> None:
        """Close Modbus RTU connection."""
        if self.client:
            try:
                self.client.close()
                logger.info(f"Disconnected from Modbus RTU device at {self.host}")
            except Exception as e:
                logger.warning(f"Error during Modbus RTU disconnect: {e}")
            finally:
                self.is_connected = False
                self.client = None


__all__ = [
    'ModbusTCPHandler',
    'ModbusRTUHandler',
]
