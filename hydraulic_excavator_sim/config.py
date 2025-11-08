"""
Excavator Configuration - Physical Parameters
Based on typical 20-ton hydraulic excavator
"""

from dataclasses import dataclass, field


@dataclass
class HydraulicSystemConfig:
    """Hydraulic system parameters"""

    # Pump
    pump_displacement: float = 200e-6  # m³/rev (200 cc/rev)
    pump_max_speed: float = 2200  # RPM
    pump_max_pressure: float = 350e5  # Pa (350 bar)
    pump_efficiency: float = 0.92

    # Fluid properties
    fluid_density: float = 860  # kg/m³ (ISO VG 46)
    fluid_viscosity_40C: float = 46e-6  # m²/s
    fluid_bulk_modulus: float = 1.7e9  # Pa

    # System
    tank_volume: float = 300  # liters
    system_pressure_relief: float = 320e5  # Pa (320 bar)
    pilot_pressure: float = 30e5  # Pa (30 bar)


@dataclass
class CylinderConfig:
    """Hydraulic cylinder specifications"""

    bore_diameter: float  # m
    rod_diameter: float  # m
    stroke_length: float  # m
    max_force: float  # N
    max_speed: float = 0.5  # m/s


@dataclass
class MotorConfig:
    """Hydraulic motor specifications"""

    displacement: float  # m³/rev
    max_speed: float  # RPM
    max_torque: float  # N·m
    efficiency: float = 0.88


@dataclass
class ExcavatorConfig:
    """Complete excavator configuration"""

    # Mechanical dimensions
    boom_length: float = 5.5  # m
    stick_length: float = 2.8  # m
    bucket_capacity: float = 1.0  # m³

    # Masses
    boom_mass: float = 2500  # kg
    stick_mass: float = 800  # kg
    bucket_mass: float = 600  # kg
    platform_mass: float = 15000  # kg

    # Moments of inertia (simplified)
    boom_inertia: float = 5000  # kg·m²
    stick_inertia: float = 1500  # kg·m²
    bucket_inertia: float = 300  # kg·m²
    platform_inertia: float = 50000  # kg·m²

    # Hydraulic components
    boom_cylinder: CylinderConfig = field(
        default_factory=lambda: CylinderConfig(
            bore_diameter=0.20,  # 200 mm
            rod_diameter=0.14,  # 140 mm
            stroke_length=2.5,  # 2.5 m
            max_force=400000,  # 400 kN
        )
    )

    stick_cylinder: CylinderConfig = field(
        default_factory=lambda: CylinderConfig(
            bore_diameter=0.16,  # 160 mm
            rod_diameter=0.11,  # 110 mm
            stroke_length=1.8,  # 1.8 m
            max_force=280000,  # 280 kN
        )
    )

    bucket_cylinder: CylinderConfig = field(
        default_factory=lambda: CylinderConfig(
            bore_diameter=0.14,  # 140 mm
            rod_diameter=0.10,  # 100 mm
            stroke_length=1.2,  # 1.2 m
            max_force=200000,  # 200 kN
        )
    )

    swing_motor: MotorConfig = field(
        default_factory=lambda: MotorConfig(
            displacement=250e-6,  # 250 cc/rev
            max_speed=15,  # RPM (swing speed)
            max_torque=80000,  # 80 kN·m
        )
    )

    track_motor: MotorConfig = field(
        default_factory=lambda: MotorConfig(
            displacement=150e-6,  # 150 cc/rev
            max_speed=100,  # RPM
            max_torque=15000,  # 15 kN·m
        )
    )

    # Control parameters
    valve_response_time: float = 0.05  # seconds
    control_deadband: float = 0.02  # 2% joystick deadband

    # Sensor sampling
    sensor_sample_rate: float = 100  # Hz
    sensor_noise_std: dict[str, float] = field(
        default_factory=lambda: {
            "pressure": 1e5,  # 1 bar
            "temperature": 0.5,  # 0.5°C
            "flow": 0.5,  # 0.5 L/min
            "position": 0.001,  # 1 mm
            "velocity": 0.01,  # 0.01 m/s
            "vibration": 0.1,  # 0.1 mm/s
        }
    )


# Global config instance
config = ExcavatorConfig()
hydraulic_config = HydraulicSystemConfig()
