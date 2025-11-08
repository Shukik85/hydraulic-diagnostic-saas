"""
Main Excavator Simulator
Integrates hydraulics, mechanics, sensors, and control
"""

import numpy as np
from config import ExcavatorConfig, HydraulicSystemConfig

from core.hydraulic_components import HydraulicCylinder, HydraulicMotor
from core.mechanical_system import ExcavatorDynamics, ExcavatorKinematics, PlatformSwing
from core.sensor_array import SensorArray


class ExcavatorSimulator:
    """Complete excavator simulator with realistic physics"""

    def __init__(self, config: ExcavatorConfig | None = None):
        self.config = config or ExcavatorConfig()
        self.hyd_config = HydraulicSystemConfig()

        # Time
        self.time = 0.0
        self.dt = 0.01  # 10ms time step (100 Hz)

        # Hydraulic components
        self.boom_cylinder = HydraulicCylinder(
            bore_diameter=self.config.boom_cylinder.bore_diameter,
            rod_diameter=self.config.boom_cylinder.rod_diameter,
            stroke_length=self.config.boom_cylinder.stroke_length,
            max_force=self.config.boom_cylinder.max_force,
        )

        self.stick_cylinder = HydraulicCylinder(
            bore_diameter=self.config.stick_cylinder.bore_diameter,
            rod_diameter=self.config.stick_cylinder.rod_diameter,
            stroke_length=self.config.stick_cylinder.stroke_length,
            max_force=self.config.stick_cylinder.max_force,
        )

        self.bucket_cylinder = HydraulicCylinder(
            bore_diameter=self.config.bucket_cylinder.bore_diameter,
            rod_diameter=self.config.bucket_cylinder.rod_diameter,
            stroke_length=self.config.bucket_cylinder.stroke_length,
            max_force=self.config.bucket_cylinder.max_force,
        )

        self.swing_motor = HydraulicMotor(
            displacement=self.config.swing_motor.displacement,
            max_speed=self.config.swing_motor.max_speed,
            max_torque=self.config.swing_motor.max_torque,
        )

        # Mechanical system
        self.kinematics = ExcavatorKinematics(
            boom_length=self.config.boom_length, stick_length=self.config.stick_length
        )

        self.dynamics = ExcavatorDynamics(
            boom_mass=self.config.boom_mass,
            stick_mass=self.config.stick_mass,
            bucket_mass=self.config.bucket_mass,
            boom_inertia=self.config.boom_inertia,
            stick_inertia=self.config.stick_inertia,
            bucket_inertia=self.config.bucket_inertia,
            boom_length=self.config.boom_length,
            stick_length=self.config.stick_length,
        )

        self.swing = PlatformSwing(
            platform_mass=self.config.platform_mass,
            platform_inertia=self.config.platform_inertia,
        )

        # Sensors
        self.sensors = SensorArray(noise_enabled=True)

        # System state
        self.pump_pressure = 0.0  # Pa
        self.pump_speed = 0.0  # RPM
        self.system_temperature = 40.0  # °C

        # Load state
        self.load_mass = 0.0  # kg (bucket load)
        self.soil_type = "medium"

    def update(
        self, boom_cmd: float, stick_cmd: float, bucket_cmd: float, swing_cmd: float
    ) -> None:
        """
        Update simulator state for one time step

        Args:
            boom_cmd: Boom control (-1.0 to 1.0)
            stick_cmd: Stick control (-1.0 to 1.0)
            bucket_cmd: Bucket control (-1.0 to 1.0)
            swing_cmd: Swing control (-1.0 to 1.0)
        """
        # Pump dynamics (simplified)
        total_demand = abs(boom_cmd) + abs(stick_cmd) + abs(bucket_cmd) + abs(swing_cmd)
        self.pump_speed = min(self.hyd_config.pump_max_speed, 1500 + 500 * total_demand)

        # Pump pressure based on load
        base_pressure = self.hyd_config.pump_max_pressure * 0.5
        load_pressure = self.hyd_config.pump_max_pressure * 0.3 * total_demand
        self.pump_pressure = min(
            self.hyd_config.pump_max_pressure, base_pressure + load_pressure
        )

        # Calculate gravity torques
        torque_boom, torque_stick, torque_bucket = (
            self.dynamics.calculate_gravity_torques(
                boom_angle=self.kinematics.boom.angle,
                stick_angle=self.kinematics.stick.angle,
                bucket_angle=self.kinematics.bucket.angle,
                load_mass=self.load_mass,
            )
        )

        # Update cylinders
        self.boom_cylinder.update(
            dt=self.dt,
            pressure_supply=self.pump_pressure,
            pressure_return=5e5,  # 5 bar return
            valve_position=boom_cmd,
            external_load=torque_boom
            / (self.config.boom_length * 0.8),  # Convert to force
        )

        self.stick_cylinder.update(
            dt=self.dt,
            pressure_supply=self.pump_pressure,
            pressure_return=5e5,
            valve_position=stick_cmd,
            external_load=torque_stick / (self.config.stick_length * 0.8),
        )

        self.bucket_cylinder.update(
            dt=self.dt,
            pressure_supply=self.pump_pressure,
            pressure_return=5e5,
            valve_position=bucket_cmd,
            external_load=torque_bucket / 0.6,
        )

        # Update swing motor
        self.swing_motor.update(
            dt=self.dt,
            pressure_supply=self.pump_pressure,
            pressure_return=5e5,
            valve_position=swing_cmd,
            external_torque=0,
        )

        self.swing.update(dt=self.dt, motor_torque=self.swing_motor.state.torque)

        # Update kinematics (cylinder position → joint angles)
        # Simplified: assume linear mapping for now
        self.kinematics.boom.angle = np.deg2rad(30) + (
            self.boom_cylinder.get_extension_ratio() * np.deg2rad(60)
        )
        self.kinematics.stick.angle = np.deg2rad(-120) + (
            self.stick_cylinder.get_extension_ratio() * np.deg2rad(90)
        )
        self.kinematics.bucket.angle = np.deg2rad(-60) + (
            self.bucket_cylinder.get_extension_ratio() * np.deg2rad(60)
        )
        self.kinematics.swing_angle = self.swing.angle

        self.kinematics.update_forward_kinematics()

        # Update system temperature (simplified)
        power_dissipated = (
            abs(self.boom_cylinder.state.force * self.boom_cylinder.state.velocity)
            + abs(self.stick_cylinder.state.force * self.stick_cylinder.state.velocity)
            + abs(
                self.bucket_cylinder.state.force * self.bucket_cylinder.state.velocity
            )
        )

        heat_generation = power_dissipated * 0.15  # 15% inefficiency
        cooling = 50 * (self.system_temperature - 25)  # Cooling rate

        self.system_temperature += (heat_generation - cooling) * self.dt / 50000
        self.system_temperature = max(25, min(120, self.system_temperature))

        # Update sensors
        self.sensors.update_from_hydraulics(
            pump_pressure=self.pump_pressure,
            boom_cyl_state={
                "position": self.boom_cylinder.state.position,
                "velocity": self.boom_cylinder.state.velocity,
                "pressure_bore": self.boom_cylinder.state.pressure_bore,
                "pressure_rod": self.boom_cylinder.state.pressure_rod,
            },
            stick_cyl_state={
                "position": self.stick_cylinder.state.position,
                "velocity": self.stick_cylinder.state.velocity,
                "pressure_bore": self.stick_cylinder.state.pressure_bore,
                "pressure_rod": self.stick_cylinder.state.pressure_rod,
            },
            bucket_cyl_state={
                "position": self.bucket_cylinder.state.position,
                "velocity": self.bucket_cylinder.state.velocity,
                "pressure_bore": self.bucket_cylinder.state.pressure_bore,
                "pressure_rod": self.bucket_cylinder.state.pressure_rod,
            },
            system_temp=self.system_temperature,
        )

        self.time += self.dt

    def get_state_summary(self) -> dict:
        """Get current state as dictionary"""
        bucket_pos = self.kinematics.get_bucket_position()
        return {
            "time": self.time,
            "bucket_position": {
                "x": bucket_pos[0],
                "y": bucket_pos[1],
                "z": bucket_pos[2],
            },
            "boom_extension": self.boom_cylinder.get_extension_ratio(),
            "stick_extension": self.stick_cylinder.get_extension_ratio(),
            "bucket_extension": self.bucket_cylinder.get_extension_ratio(),
            "swing_angle_deg": self.swing.get_angle_degrees(),
            "pump_pressure_bar": self.pump_pressure / 1e5,
            "system_temp_C": self.system_temperature,
            "load_kg": self.load_mass,
        }
