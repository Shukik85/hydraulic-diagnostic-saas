"""
Enterprise Modbus TCP Server for Hydraulic System Simulation.

Uses correct pymodbus 3.11+ API with production-grade features:
- Dynamic sensor values with realistic drift and noise
- Automatic anomaly injection (2% chance)
- Comprehensive logging and monitoring
- Graceful shutdown handling
"""

import asyncio
import logging
import signal
import sys
import time
import random
from typing import Optional, Dict, Any
from datetime import datetime

try:
    # FIXED: Correct imports for pymodbus 3.11+
    from pymodbus.server import StartAsyncTcpServer
    from pymodbus.device import ModbusDeviceIdentification
    from pymodbus.datastore import (
        ModbusSequentialDataBlock,
        ModbusSlaveContext, 
        ModbusServerContext
    )
    from pymodbus.payload import BinaryPayloadBuilder
    from pymodbus.constants import Endian
    PYMODBUS_AVAILABLE = True
    print("✅ pymodbus 3.11+ imported successfully")
except ImportError as e:
    print(f"❌ pymodbus import failed: {e}")
    print("🔧 Fix: pip uninstall -y pymodbus && pip install 'pymodbus>=3.11.0'")
    PYMODBUS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hydraulic_sim")

# Global server handle
server_handle: Optional[Any] = None


class HydraulicSystemSimulator:
    """Enterprise-grade hydraulic system simulator with realistic behavior."""
    
    def __init__(self):
        # Baseline sensor values (typical hydraulic system)
        self.base_pressure = 150.0      # bar (typical working pressure)
        self.base_temperature = 65.0    # °C (normal operating temperature)
        self.base_flow_rate = 25.5      # L/min (pump flow rate)
        self.base_vibration = 0.5       # mm/s (acceptable vibration level)
        self.base_speed = 1500.0        # RPM (pump/motor speed)
        
        # Simulation parameters
        self.noise_amplitude = 0.05     # 5% noise around baseline
        self.drift_rate = 0.001         # slow drift over time
        self.anomaly_chance = 0.02      # 2% chance of anomaly per update
        
        self.start_time = time.time()
        self.update_count = 0
        self.last_anomaly = None
    
    def get_sensor_values(self) -> Dict[str, float]:
        """Generate realistic sensor readings with noise, drift, and anomalies."""
        self.update_count += 1
        elapsed = time.time() - self.start_time
        
        # Add time-based drift and random noise
        def add_noise(base_value):
            return base_value * (1 + random.uniform(-self.noise_amplitude, self.noise_amplitude))
        
        def add_drift(base_value):
            return base_value * (1 + self.drift_rate * elapsed * random.uniform(-1, 1))
        
        # Generate base values with noise and drift
        pressure = add_drift(add_noise(self.base_pressure))
        temperature = add_drift(add_noise(self.base_temperature))
        flow_rate = add_drift(add_noise(self.base_flow_rate))
        vibration = add_drift(add_noise(self.base_vibration))
        speed = add_drift(add_noise(self.base_speed))
        
        # Inject realistic anomalies occasionally
        if random.random() < self.anomaly_chance:
            anomaly_type = random.choice([
                'pressure_spike', 'pressure_drop', 'temp_rise', 
                'temp_drop', 'flow_blockage', 'cavitation'
            ])
            
            if anomaly_type == 'pressure_spike':
                pressure *= 1.8  # Hydraulic spike
            elif anomaly_type == 'pressure_drop':
                pressure *= 0.3  # Major leak
            elif anomaly_type == 'temp_rise':
                temperature *= 1.4  # Overheating
            elif anomaly_type == 'temp_drop':
                temperature *= 0.8  # Cooler malfunction
            elif anomaly_type == 'flow_blockage':
                flow_rate *= 0.1  # Blocked filter
            elif anomaly_type == 'cavitation':
                flow_rate *= 1.5  # Pump cavitation
                vibration *= 3.0  # Increased vibration
            
            self.last_anomaly = {
                'type': anomaly_type,
                'timestamp': datetime.now(),
                'update_count': self.update_count
            }
            
            logger.warning(f"🚨 Hydraulic anomaly injected: {anomaly_type}")
        
        return {
            'pressure': max(0, pressure),           # Pressure can't be negative
            'temperature': max(-40, temperature),   # Reasonable temperature range
            'flow_rate': max(0, flow_rate),         # Flow can't be negative
            'vibration': max(0, vibration),         # Vibration can't be negative
            'speed': max(0, speed),                 # Speed can't be negative
        }


def setup_initial_registers(context: ModbusServerContext, simulator: HydraulicSystemSimulator) -> None:
    """Setup initial register values with realistic hydraulic data."""
    logger.info("🔧 Initializing hydraulic system registers...")
    
    values = simulator.get_sensor_values()
    
    # HR[0] = Pressure (uint16, scaled: bar * 10)
    pressure_scaled = int(values['pressure'] * 10)  # 150.0 bar -> 1500
    context[0x00].setValues(3, 0, [pressure_scaled])
    
    # HR[1] = Temperature (int16, °C)
    temperature_scaled = int(values['temperature'])  # 65.0 -> 65
    context[0x00].setValues(3, 1, [temperature_scaled])
    
    # HR[2-3] = Flow Rate (float32)
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    builder.add_32bit_float(values['flow_rate'])
    context[0x00].setValues(3, 2, builder.to_registers())
    
    # HR[10] = Vibration (uint16, scaled: mm/s * 1000)
    vibration_scaled = int(values['vibration'] * 1000)  # 0.5 -> 500
    context[0x00].setValues(3, 10, [vibration_scaled])
    
    # HR[11] = Speed (uint16, RPM)
    speed_scaled = int(values['speed'])  # 1500.0 -> 1500
    context[0x00].setValues(3, 11, [speed_scaled])
    
    logger.info("✅ Hydraulic registers initialized:")
    logger.info(f"   HR[0] (40001) = {pressure_scaled} (pressure: {values['pressure']:.1f} bar)")
    logger.info(f"   HR[1] (40002) = {temperature_scaled} (temperature: {values['temperature']:.1f} °C)")
    logger.info(f"   HR[2-3] (40003-40004) = float32 {values['flow_rate']:.1f} (flow rate L/min)")
    logger.info(f"   HR[10] (40011) = {vibration_scaled} (vibration: {values['vibration']:.3f} mm/s)")
    logger.info(f"   HR[11] (40012) = {speed_scaled} (speed: {values['speed']:.0f} RPM)")


async def update_registers_periodically(context: ModbusServerContext, simulator: HydraulicSystemSimulator):
    """Update register values every 30 seconds to simulate live hydraulic data."""
    while True:
        try:
            await asyncio.sleep(30)  # Update every 30 seconds
            
            values = simulator.get_sensor_values()
            
            # Update registers with new values
            pressure_scaled = int(values['pressure'] * 10)
            temperature_scaled = int(values['temperature'])
            vibration_scaled = int(values['vibration'] * 1000)  
            speed_scaled = int(values['speed'])
            
            context[0x00].setValues(3, 0, [pressure_scaled])
            context[0x00].setValues(3, 1, [temperature_scaled])
            context[0x00].setValues(3, 10, [vibration_scaled])
            context[0x00].setValues(3, 11, [speed_scaled])
            
            # Update float32 flow rate
            builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
            builder.add_32bit_float(values['flow_rate'])
            context[0x00].setValues(3, 2, builder.to_registers())
            
            # Log updates with anomaly info
            status = f"P={values['pressure']:.1f}bar, T={values['temperature']:.1f}°C, Q={values['flow_rate']:.1f}L/min"
            if simulator.last_anomaly and simulator.update_count - simulator.last_anomaly['update_count'] < 2:
                status += f" [ANOMALY: {simulator.last_anomaly['type']}]"
            
            logger.info(f"🔄 Register update #{simulator.update_count}: {status}")
            
        except asyncio.CancelledError:
            logger.info("📊 Register update task cancelled")
            break
        except Exception as e:
            logger.error(f"❌ Register update failed: {e}")


async def run_modbus_server():
    """Start enterprise-grade Modbus TCP server."""
    global server_handle
    
    if not PYMODBUS_AVAILABLE:
        logger.error("❌ pymodbus not available")
        return
    
    logger.info("🚀 Starting Enterprise Hydraulic Modbus Server...")
    
    # Initialize hydraulic system simulator
    simulator = HydraulicSystemSimulator()
    
    # Create data store with all register types
    store = ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [0] * 100),  # Discrete inputs
        co=ModbusSequentialDataBlock(0, [0] * 100),  # Coils
        hr=ModbusSequentialDataBlock(0, [0] * 100),  # Holding registers
        ir=ModbusSequentialDataBlock(0, [0] * 100),  # Input registers
    )
    
    context = ModbusServerContext(slaves=store, single=True)
    
    # Device identification (appears in Modbus device info)
    identity = ModbusDeviceIdentification()
    identity.VendorName = 'Hydraulic Diagnostic Platform'
    identity.ProductCode = 'HDP-SIM'
    identity.VendorUrl = 'https://github.com/Shukik85/hydraulic-diagnostic-saas'
    identity.ProductName = 'Hydraulic System Simulator'
    identity.ModelName = 'HDP-SIM-v1.0'
    identity.MajorMinorRevision = '1.0.0'
    
    # Setup initial register values
    setup_initial_registers(context, simulator)
    
    # Start periodic register updates
    update_task = asyncio.create_task(
        update_registers_periodically(context, simulator)
    )
    
    try:
        # Start the Modbus TCP server
        logger.info("📡 Starting server on 0.0.0.0:1502...")
        
        server_handle = await StartAsyncTcpServer(
            context=context,
            identity=identity,
            address=("0.0.0.0", 1502),
            allow_reuse_address=True,
        )
        
        logger.info("✅ Modbus TCP server started successfully!")
        logger.info("📍 Server details:")
        logger.info("   Host: 0.0.0.0:1502")
        logger.info("   Unit ID: 1")
        logger.info("   Protocol: Modbus TCP")
        logger.info("   Data updates: Every 30 seconds")
        logger.info("   Anomaly injection: 2% chance per update")
        logger.info("")
        logger.info("🧪 Test connection:")
        logger.info("   python modbus_quick_test.py")
        logger.info("   python test_modbus_simple.py")
        logger.info("")
        logger.info("⏹️ Press Ctrl+C to stop server")
        
        # Keep server running indefinitely
        await asyncio.Future()  # Run forever until cancelled
        
    except Exception as e:
        logger.error(f"❌ Server failed: {e}")
        update_task.cancel()
        raise
    finally:
        if server_handle:
            server_handle.close()
            await server_handle.wait_closed()
        update_task.cancel()
        logger.info("✅ Server cleanup completed")


def signal_handler(signum, frame):
    """Graceful shutdown on Ctrl+C."""
    logger.info("⚠️ Received shutdown signal...")
    
    # Cancel all running tasks
    for task in asyncio.all_tasks():
        if not task.done():
            task.cancel()
    
    logger.info("✅ Modbus server stopped gracefully")
    sys.exit(0)


def main():
    """Main entry point."""
    print("=" * 70)
    print("🏭 ENTERPRISE HYDRAULIC MODBUS SIMULATOR")
    print("=" * 70)
    print("🎯 Production-grade simulation with dynamic sensor data")
    print("🔧 Supports: pressure, temperature, flow, vibration, speed")
    print("📊 Auto-updates every 30s with drift and occasional anomalies")
    print("🚨 Anomaly injection: pressure spikes, leaks, overheating, cavitation")
    print("=" * 70)
    
    if not PYMODBUS_AVAILABLE:
        print("❌ pymodbus not available")
        print("🔧 Fix: pip install -r requirements-modbus.txt")
        sys.exit(1)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the server
    try:
        asyncio.run(run_modbus_server())
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Server crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()