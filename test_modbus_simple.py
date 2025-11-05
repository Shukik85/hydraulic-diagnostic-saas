"""
Simple Modbus TCP Test Script - No Django Dependencies.

Tests basic Modbus TCP connection and register reading.
Run directly in your virtual environment.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from pymodbus.client import ModbusTcpClient
    from pymodbus.constants import Endian
    from pymodbus.payload import BinaryPayloadDecoder
    from pymodbus.exceptions import ModbusException
    MODBUS_AVAILABLE = True
    print("✅ pymodbus imported successfully")
except ImportError as e:
    MODBUS_AVAILABLE = False
    print(f"❌ pymodbus import failed: {e}")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleModbusTCPTester:
    """Simple Modbus TCP tester without Django dependencies."""
    
    def __init__(self, host: str, port: int = 502, unit_id: int = 1):
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.client: Optional[ModbusTcpClient] = None
        self.is_connected = False
    
    async def connect(self) -> bool:
        """Establish Modbus TCP connection."""
        try:
            self.client = ModbusTcpClient(
                host=self.host,
                port=self.port,
                timeout=5.0
            )
            
            logger.info(f"Attempting to connect to {self.host}:{self.port}...")
            connected = self.client.connect()
            
            if connected:
                self.is_connected = True
                logger.info(f"✅ Connected to Modbus TCP device at {self.host}:{self.port}")
                return True
            else:
                logger.error(f"❌ Failed to connect to {self.host}:{self.port}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False
    
    async def test_read_register(self, address: int, data_type: str = 'uint16') -> None:
        """Test reading a single register."""
        if not self.is_connected:
            logger.error("❌ Not connected to device")
            return
        
        try:
            # Determine register count based on data type
            register_count = 1
            if data_type in ['float32', 'int32', 'uint32']:
                register_count = 2
            elif data_type == 'float64':
                register_count = 4
            
            logger.info(f"Reading register {address} as {data_type} (count: {register_count})...")
            
            # Read holding registers
            result = self.client.read_holding_registers(
                address=address,
                count=register_count,
                unit=self.unit_id
            )
            
            if result.isError():
                logger.error(f"❌ Modbus read error: {result}")
                return
            
            logger.info(f"✅ Raw registers: {result.registers}")
            
            # Decode the value
            decoder = BinaryPayloadDecoder.fromRegisters(
                result.registers,
                byteorder=Endian.BIG,
                wordorder=Endian.BIG
            )
            
            if data_type == 'uint16':
                decoded = decoder.decode_16bit_uint()
            elif data_type == 'int16':
                decoded = decoder.decode_16bit_int()
            elif data_type == 'uint32':
                decoded = decoder.decode_32bit_uint()
            elif data_type == 'int32':
                decoded = decoder.decode_32bit_int()
            elif data_type == 'float32':
                decoded = decoder.decode_32bit_float()
            elif data_type == 'float64':
                decoded = decoder.decode_64bit_float()
            else:
                decoded = result.registers[0]
                logger.warning(f"⚠️  Unknown data type {data_type}, using raw value")
            
            logger.info(f"✅ Decoded value: {decoded} ({data_type})")
            
        except ModbusException as e:
            logger.error(f"❌ Modbus exception: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
    
    async def scan_registers(self, start_address: int, end_address: int, data_type: str = 'uint16') -> None:
        """Scan a range of registers to find responsive ones."""
        logger.info(f"Scanning registers {start_address} to {end_address} as {data_type}...")
        
        responsive_registers = []
        
        for address in range(start_address, end_address + 1):
            try:
                result = self.client.read_holding_registers(
                    address=address,
                    count=1,
                    unit=self.unit_id
                )
                
                if not result.isError():
                    value = result.registers[0]
                    responsive_registers.append((address, value))
                    print(f"  Register {address}: {value}")
                
            except Exception:
                pass  # Skip errors during scanning
        
        if responsive_registers:
            logger.info(f"✅ Found {len(responsive_registers)} responsive registers")
            for addr, val in responsive_registers[:10]:  # Show first 10
                logger.info(f"  Register {addr}: {val}")
        else:
            logger.warning("❌ No responsive registers found in the specified range")
    
    async def disconnect(self) -> None:
        """Close connection."""
        if self.client:
            try:
                self.client.close()
                logger.info(f"✅ Disconnected from {self.host}:{self.port}")
            except Exception as e:
                logger.warning(f"⚠️  Disconnect error: {e}")
            finally:
                self.is_connected = False
                self.client = None


async def main():
    """Main test function."""
    print("🚀 Starting Modbus TCP Test...")
    print(f"📦 pymodbus available: {MODBUS_AVAILABLE}")
    
    if not MODBUS_AVAILABLE:
        print("❌ pymodbus not available. Install with: pip install pymodbus>=3.6.6")
        return
    
    # Configuration - CHANGE THESE VALUES
    HOST = "127.0.0.1"  # Replace with your PLC/device IP
    PORT = 502           # Standard Modbus TCP port
    UNIT_ID = 1         # Modbus unit/slave ID
    
    print(f"📡 Testing connection to {HOST}:{PORT} (Unit ID: {UNIT_ID})")
    
    tester = SimpleModbusTCPTester(HOST, PORT, UNIT_ID)
    
    # Test connection
    connected = await tester.connect()
    
    if not connected:
        print("\n❌ CONNECTION FAILED")
        print("\n🔧 Troubleshooting steps:")
        print(f"   1. Check if {HOST} is reachable: ping {HOST}")
        print(f"   2. Check if port {PORT} is open: nc -zv {HOST} {PORT}")
        print(f"   3. Verify Modbus server is running on the device")
        print(f"   4. Check unit ID ({UNIT_ID}) - try 0 or 255 if 1 doesn't work")
        print(f"   5. Check firewall settings on both sides")
        return
    
    print("\n✅ CONNECTION SUCCESSFUL!")
    
    try:
        # Test reading common register addresses
        test_addresses = [
            (0, 'uint16', 'Register 0'),
            (1, 'uint16', 'Register 1'),
            (40001, 'uint16', 'Holding Register 40001'),
            (40001, 'float32', 'HR 40001 as Float32'),
        ]
        
        print("\n🔍 Testing register reads:")
        
        for address, data_type, description in test_addresses:
            print(f"\n📊 Testing {description} (address {address}, type {data_type})...")
            await tester.test_read_register(address, data_type)
        
        # Scan for responsive registers
        print("\n🔍 Scanning registers 0-10 for responsive ones...")
        await tester.scan_registers(0, 10)
        
        print("\n🎉 Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
    finally:
        await tester.disconnect()


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 MODBUS TCP CONNECTION TESTER")
    print("=" * 60)
    
    # Check if pymodbus is available
    if not MODBUS_AVAILABLE:
        print("\n❌ pymodbus not available")
        print("Install with: pip install pymodbus>=3.6.6")
        exit(1)
    
    # Run the test
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test cancelled by user")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        exit(1)
    
    print("\n✨ Test script finished")
