"""
Sensor Array - Complete hydraulic system monitoring
50+ sensors across all subsystems
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np


@dataclass
class SensorReading:
    """Single sensor reading with metadata"""
    timestamp: datetime
    sensor_id: str
    sensor_type: str  # pressure, temperature, flow, position, vibration, etc.
    value: float
    unit: str
    location: str
    is_critical: bool = False
    quality: float = 100.0  # 0-100%


@dataclass
class SensorArrayState:
    """Complete state of all sensors"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Pressure sensors (Pa)
    pressure_pump_outlet: float = 0.0
    pressure_boom_extend: float = 0.0
    pressure_boom_retract: float = 0.0
    pressure_stick_extend: float = 0.0
    pressure_stick_retract: float = 0.0
    pressure_bucket_extend: float = 0.0
    pressure_bucket_retract: float = 0.0
    pressure_swing_left: float = 0.0
    pressure_swing_right: float = 0.0
    pressure_track_left: float = 0.0
    pressure_track_right: float = 0.0
    pressure_pilot: float = 0.0
    pressure_tank_return: float = 0.0
    
    # Temperature sensors (°C)
    temp_oil_tank: float = 40.0
    temp_oil_return: float = 40.0
    temp_pump: float = 40.0
    temp_valve_block: float = 40.0
    temp_boom_cylinder: float = 40.0
    temp_stick_cylinder: float = 40.0
    temp_bucket_cylinder: float = 40.0
    temp_swing_motor: float = 40.0
    temp_track_motor_left: float = 40.0
    temp_track_motor_right: float = 40.0
    temp_cooler_inlet: float = 40.0
    temp_cooler_outlet: float = 40.0
    temp_ambient: float = 25.0
    
    # Flow sensors (L/min)
    flow_pump_output: float = 0.0
    flow_boom_circuit: float = 0.0
    flow_stick_circuit: float = 0.0
    flow_bucket_circuit: float = 0.0
    flow_swing_circuit: float = 0.0
    flow_track_circuit: float = 0.0
    flow_tank_return: float = 0.0
    flow_cooler: float = 0.0
    
    # Position sensors (m or rad)
    pos_boom_cylinder: float = 0.0
    pos_stick_cylinder: float = 0.0
    pos_bucket_cylinder: float = 0.0
    pos_swing_angle: float = 0.0
    
    # Velocity sensors (m/s or rad/s)
    vel_boom_cylinder: float = 0.0
    vel_stick_cylinder: float = 0.0
    vel_bucket_cylinder: float = 0.0
    vel_swing: float = 0.0
    vel_track_left: float = 0.0
    vel_track_right: float = 0.0
    
    # Vibration sensors (mm/s RMS)
    vib_pump: float = 0.0
    vib_valve_block: float = 0.0
    vib_boom_cylinder: float = 0.0
    vib_stick_cylinder: float = 0.0
    vib_swing_motor: float = 0.0
    
    # Load/Force sensors (N or N·m)
    force_boom_cylinder: float = 0.0
    force_stick_cylinder: float = 0.0
    force_bucket_cylinder: float = 0.0
    torque_swing_motor: float = 0.0
    
    # Other sensors
    oil_level_tank: float = 80.0  # %
    filter_pressure_drop: float = 0.0  # bar
    pump_speed: float = 0.0  # RPM
    engine_load: float = 0.0  # %


class SensorArray:
    """Manages all sensors and adds realistic noise"""
    
    def __init__(self, noise_enabled: bool = True, noise_std: dict[str, float] | None = None):
        self.noise_enabled = noise_enabled
        self.noise_std = noise_std or {
            'pressure': 1e5,      # 1 bar
            'temperature': 0.5,   # 0.5°C
            'flow': 0.5,          # 0.5 L/min
            'position': 0.001,    # 1 mm
            'velocity': 0.01,     # 0.01 m/s
            'vibration': 0.1      # 0.1 mm/s
        }
        
        self.state = SensorArrayState()
        self.readings_history: list[SensorArrayState] = []
    
    def add_noise(self, value: float, sensor_type: str) -> float:
        """Add realistic measurement noise"""
        if not self.noise_enabled:
            return value
        
        std = self.noise_std.get(sensor_type, 0.0)
        noise = np.random.normal(0, std)
        return value + noise
    
    def update_from_hydraulics(
        self,
        pump_pressure: float,
        boom_cyl_state: dict,
        stick_cyl_state: dict,
        bucket_cyl_state: dict,
        system_temp: float
    ) -> None:
        """Update sensor readings from hydraulic system state"""
        self.state.timestamp = datetime.now(timezone.utc)
        
        # Pressure sensors (convert to Pa and add noise)
        self.state.pressure_pump_outlet = self.add_noise(pump_pressure, 'pressure')
        self.state.pressure_boom_extend = self.add_noise(boom_cyl_state['pressure_bore'], 'pressure')
        self.state.pressure_boom_retract = self.add_noise(boom_cyl_state['pressure_rod'], 'pressure')
        self.state.pressure_stick_extend = self.add_noise(stick_cyl_state['pressure_bore'], 'pressure')
        self.state.pressure_stick_retract = self.add_noise(stick_cyl_state['pressure_rod'], 'pressure')
        self.state.pressure_bucket_extend = self.add_noise(bucket_cyl_state['pressure_bore'], 'pressure')
        self.state.pressure_bucket_retract = self.add_noise(bucket_cyl_state['pressure_rod'], 'pressure')
        
        # Temperature sensors
        base_temp = system_temp
        self.state.temp_oil_tank = self.add_noise(base_temp, 'temperature')
        self.state.temp_pump = self.add_noise(base_temp + 10, 'temperature')
        self.state.temp_valve_block = self.add_noise(base_temp + 5, 'temperature')
        
        # Position sensors
        self.state.pos_boom_cylinder = self.add_noise(boom_cyl_state['position'], 'position')
        self.state.pos_stick_cylinder = self.add_noise(stick_cyl_state['position'], 'position')
        self.state.pos_bucket_cylinder = self.add_noise(bucket_cyl_state['position'], 'position')
        
        # Velocity sensors
        self.state.vel_boom_cylinder = self.add_noise(boom_cyl_state['velocity'], 'velocity')
        self.state.vel_stick_cylinder = self.add_noise(stick_cyl_state['velocity'], 'velocity')
        self.state.vel_bucket_cylinder = self.add_noise(bucket_cyl_state['velocity'], 'velocity')
        
        # Vibration (pressure ripple induced)
        pump_vibration = 1.0 + 0.5 * (pump_pressure / 350e5)
        self.state.vib_pump = self.add_noise(pump_vibration, 'vibration')
        
        # Store history
        self.readings_history.append(SensorArrayState(**vars(self.state)))
    
    def get_all_readings(self) -> list[SensorReading]:
        """Get all current sensor readings as list"""
        readings = []
        timestamp = self.state.timestamp
        
        # Pressure sensors
        pressure_sensors = {
            'pump_outlet': (self.state.pressure_pump_outlet, 'Pa', 'Pump'),
            'boom_extend': (self.state.pressure_boom_extend, 'Pa', 'Boom Cylinder'),
            'boom_retract': (self.state.pressure_boom_retract, 'Pa', 'Boom Cylinder'),
            'stick_extend': (self.state.pressure_stick_extend, 'Pa', 'Stick Cylinder'),
            'stick_retract': (self.state.pressure_stick_retract, 'Pa', 'Stick Cylinder'),
            'bucket_extend': (self.state.pressure_bucket_extend, 'Pa', 'Bucket Cylinder'),
            'bucket_retract': (self.state.pressure_bucket_retract, 'Pa', 'Bucket Cylinder'),
        }
        
        for sensor_id, (value, _unit, location) in pressure_sensors.items():
            readings.append(SensorReading(
                timestamp=timestamp,
                sensor_id=f'P_{sensor_id}',
                sensor_type='pressure',
                value=value / 1e5,  # Convert to bar for logging
                unit='bar',
                location=location,
                is_critical=(value > 320e5)  # Above relief pressure
            ))
        
        # Temperature sensors
        temp_sensors = {
            'oil_tank': (self.state.temp_oil_tank, 'Tank'),
            'pump': (self.state.temp_pump, 'Pump'),
            'valve_block': (self.state.temp_valve_block, 'Valve Block'),
        }
        
        for sensor_id, (value, location) in temp_sensors.items():
            readings.append(SensorReading(
                timestamp=timestamp,
                sensor_id=f'T_{sensor_id}',
                sensor_type='temperature',
                value=value,
                unit='°C',
                location=location,
                is_critical=(value > 90)  # Overheating threshold
            ))
        
        # Position sensors
        position_sensors = {
            'boom': (self.state.pos_boom_cylinder, 'Boom Cylinder'),
            'stick': (self.state.pos_stick_cylinder, 'Stick Cylinder'),
            'bucket': (self.state.pos_bucket_cylinder, 'Bucket Cylinder'),
        }
        
        for sensor_id, (value, location) in position_sensors.items():
            readings.append(SensorReading(
                timestamp=timestamp,
                sensor_id=f'X_{sensor_id}',
                sensor_type='position',
                value=value,
                unit='m',
                location=location
            ))
        
        return readings
    
    def export_to_dict(self) -> dict:
        """Export current state as dictionary for logging"""
        return {
            'timestamp': self.state.timestamp.isoformat(),
            **{k: v for k, v in vars(self.state).items() if k != 'timestamp'}
        }
