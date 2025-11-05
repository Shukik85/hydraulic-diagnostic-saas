"""
Local Modbus TCP Server for Development Testing.

Creates a Modbus TCP server with pre-configured registers:
- Holding Register 0 (40001): uint16 = 1234 (Pressure sensor)
- Holding Register 1 (40002): uint16 = 42 (Temperature sensor)
- Holding Register 2-3 (40003-40004): float32 = 25.5 (Flow rate)

Run this script to simulate a PLC/industrial device for testing.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

try:
    from pymodbus.server import StartAsyncTcpServer
    from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
    from pymodbus.datastore import ModbusSequentialDataBlock
    from pymodbus.payload import BinaryPayloadBuilder
    from pymodbus.constants import Endian
    print("✅ pymodbus imported successfully")
except ImportError as e:
    print(f"❌ pymodbus not available: {e}")
    print("Install with: pip install pymodbus>=3.6.6")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global server reference for graceful shutdown
server_task: Optional[asyncio.Task] = None


def setup_register_values(context: ModbusServerContext) -> None:
    """Configure initial register values to simulate hydraulic sensors."""
    logger.info("🔧 Setting up simulated sensor values...")
    
    # Simple integer values
    context[0x00].setValues(3, 0, [1234])  # HR[0] = 1234 (pressure, uint16)
    context[0x00].setValues(3, 1, [42])    # HR[1] = 42 (temperature, uint16)
    
    # Float32 value (25.5) stored in HR[2-3]
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    builder.add_32bit_float(25.5)  # Flow rate
    float_registers = builder.to_registers()
    context[0x00].setValues(3, 2, float_registers)  # HR[2-3]
    
    # Additional test values
    context[0x00].setValues(3, 10, [999])   # HR[10] = 999 (vibration)
    context[0x00].setValues(3, 11, [1500])  # HR[11] = 1500 (speed)
    
    logger.info("✅ Register values configured:")
    logger.info("   HR[0] (40001) = 1234 (uint16) - Pressure")
    logger.info("   HR[1] (40002) = 42 (uint16) - Temperature")
    logger.info("   HR[2-3] (40003-40004) = 25.5 (float32) - Flow Rate")
    logger.info("   HR[10] (40011) = 999 (uint16) - Vibration")
    logger.info("   HR[11] (40012) = 1500 (uint16) - Speed")


async def run_modbus_server(host: str = "0.0.0.0", port: int = 1502) -> None:
    """Start the Modbus TCP server."""
    global server_task
    
    logger.info(f"🚀 Starting Modbus TCP server on {host}:{port}...")
    
    # Create data store with 100 registers for each type
    store = ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [0] * 100),  # Discrete inputs
        co=ModbusSequentialDataBlock(0, [0] * 100),  # Coils
        hr=ModbusSequentialDataBlock(0, [0] * 100),  # Holding registers
        ir=ModbusSequentialDataBlock(0, [0] * 100),  # Input registers
        zero_mode=True  # 0-based addressing
    )
    
    # Create server context
    context = ModbusServerContext(slaves=store, single=True)
    
    # Setup initial register values
    setup_register_values(context)
    
    try:
        # Start the server
        server_task = asyncio.create_task(
            StartAsyncTcpServer(
                context=context,
                address=(host, port),
                allow_reuse_address=True
            )
        )
        
        logger.info(f"✅ Modbus TCP server started successfully!")
        logger.info(f"📡 Listening on {host}:{port}")
        logger.info(f"🎯 Unit ID: 1 (default)")
        logger.info("")
        logger.info("🔍 To test connection, run:")
        logger.info(f"   python test_modbus_simple.py")
        logger.info("")
        logger.info("⏹️  Press Ctrl+C to stop server")
        
        # Wait for the server task
        await server_task
        
    except Exception as e:
        logger.error(f"❌ Server failed: {e}")
        raise


def signal_handler(signum, frame):
    """Handle graceful shutdown on Ctrl+C."""
    logger.info("⚠️  Received shutdown signal...")
    
    # Cancel server task if running
    if server_task and not server_task.done():
        server_task.cancel()
    
    logger.info("✅ Modbus server stopped gracefully")
    sys.exit(0)


def main():
    """Main function."""
    print("=" * 60)
    print("🏭 LOCAL MODBUS TCP SIMULATOR")
    print("=" * 60)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the server
    try:
        asyncio.run(run_modbus_server())
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
