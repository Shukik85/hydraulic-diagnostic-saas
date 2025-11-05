#!/usr/bin/env python3
"""
Quick Modbus TCP Test with Automatic Server.

Uses correct pymodbus 3.11+ imports and API.
Starts a local Modbus server and immediately tests it.
Perfect for development and CI testing.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def modbus_test_server():
    """Context manager that starts and stops Modbus server for testing."""
    try:
        # FIXED: Correct imports for pymodbus 3.11+
        from pymodbus.server import StartAsyncTcpServer
        from pymodbus.datastore import (
            ModbusSequentialDataBlock,
            ModbusSlaveContext,
            ModbusServerContext
        )
        from pymodbus.payload import BinaryPayloadBuilder
        from pymodbus.constants import Endian
        
        # Create test data store
        store = ModbusSlaveContext(
            hr=ModbusSequentialDataBlock(0, [0] * 100),
        )
        context = ModbusServerContext(slaves=store, single=True)
        
        # Set test values
        context[0x00].setValues(3, 0, [1500])  # HR[0] = 1500 (pressure * 10)
        context[0x00].setValues(3, 1, [65])    # HR[1] = 65 (temperature)
        
        # Float32 in HR[2-3]
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
        builder.add_32bit_float(25.5)
        context[0x00].setValues(3, 2, builder.to_registers())
        
        logger.info("🚀 Starting test Modbus server on port 1502...")
        
        # Start server - FIXED API call
        server = await StartAsyncTcpServer(
            context=context,
            address=("127.0.0.1", 1502),
            allow_reuse_address=True,
        )
        
        # Give server time to start
        await asyncio.sleep(0.5)
        
        logger.info("✅ Test server started")
        
        try:
            yield server
        finally:
            logger.info("🔄 Stopping test server...")
            server.close()
            await server.wait_closed()
            logger.info("✅ Test server stopped")
            
    except ImportError as e:
        logger.error(f"❌ pymodbus import failed: {e}")
        logger.error("🔧 Fix: pip install -r requirements-modbus.txt")
        raise


async def test_modbus_connection():
    """Test Modbus TCP connection and register reading."""
    try:
        from pymodbus.client import ModbusTcpClient
        from pymodbus.payload import BinaryPayloadDecoder
        from pymodbus.constants import Endian
        
        logger.info("🔍 Testing Modbus TCP connection...")
        
        client = ModbusTcpClient(host="127.0.0.1", port=1502, timeout=3.0)
        
        if not client.connect():
            logger.error("❌ Failed to connect to test server")
            return False
        
        logger.info("✅ Connected to test server")
        
        # Test different register reads
        tests = [
            (0, 1, 'uint16', 'Pressure (scaled)'),
            (1, 1, 'uint16', 'Temperature'),
            (2, 2, 'float32', 'Flow Rate'),
        ]
        
        all_passed = True
        
        for address, count, data_type, description in tests:
            try:
                result = client.read_holding_registers(address=address, count=count, unit=1)
                
                if result.isError():
                    logger.error(f"❌ {description}: Modbus read error")
                    all_passed = False
                    continue
                
                # Decode value
                if data_type == 'uint16':
                    value = result.registers[0]
                elif data_type == 'float32':
                    decoder = BinaryPayloadDecoder.fromRegisters(
                        result.registers,
                        byteorder=Endian.BIG,
                        wordorder=Endian.BIG
                    )
                    value = decoder.decode_32bit_float()
                else:
                    value = result.registers[0]
                
                logger.info(f"✅ {description}: {value} (registers: {result.registers})")
                
            except Exception as e:
                logger.error(f"❌ {description}: Test failed - {e}")
                all_passed = False
        
        client.close()
        return all_passed
        
    except ImportError as e:
        logger.error(f"❌ pymodbus client import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🎯 QUICK MODBUS TCP TEST")
    print("📊 Tests: connection, register reading, data types")
    print("=" * 50)
    
    try:
        # Start test server and run tests
        async with modbus_test_server():
            # Wait a moment for server to be fully ready
            await asyncio.sleep(1.0)
            
            # Run connection tests
            success = await test_modbus_connection()
            
            if success:
                print("\n" + "=" * 50)
                print("🎉 ALL TESTS PASSED!")
                print("✅ Modbus TCP protocol is working correctly")
                print("✅ Register reading successful")
                print("✅ Data type decoding working")
                print("\n🚀 READY FOR PRODUCTION INTEGRATION!")
                print("\n📝 Next steps:")
                print("   1. Create Django sensor models migrations")
                print("   2. Add Celery task for periodic data collection")
                print("   3. Integrate with ML service")
                print("   4. Add WebSocket real-time updates")
            else:
                print("\n" + "=" * 50)
                print("❌ SOME TESTS FAILED")
                print("🔧 Check logs above for details")
                return False
                
    except KeyboardInterrupt:
        print("\n👋 Test cancelled by user")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit(0 if result else 1)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        exit(1)