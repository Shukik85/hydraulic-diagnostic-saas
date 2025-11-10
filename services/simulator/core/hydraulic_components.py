"""
Hydraulic Components - Cylinders, Motors, Valves
Physics-based implementation with realistic dynamics
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class HydraulicCylinderState:
    """Current state of hydraulic cylinder"""

    position: float = 0.0  # m (0 = fully retracted)
    velocity: float = 0.0  # m/s
    acceleration: float = 0.0  # m/s²
    pressure_bore: float = 0.0  # Pa (bore side pressure)
    pressure_rod: float = 0.0  # Pa (rod side pressure)
    force: float = 0.0  # N (output force)
    flow_in: float = 0.0  # m³/s
    flow_out: float = 0.0  # m³/s
    temperature: float = 40.0  # °C


class HydraulicCylinder:
    """
    Hydraulic cylinder with realistic dynamics
    Based on ISO 3320 and ISO 3321 standards
    """

    def __init__(
        self,
        bore_diameter: float,
        rod_diameter: float,
        stroke_length: float,
        max_force: float,
        friction_coefficient: float = 0.05,
        leakage_coefficient: float = 1e-12,
    ):
        self.bore_diameter = bore_diameter
        self.rod_diameter = rod_diameter
        self.stroke_length = stroke_length
        self.max_force = max_force
        self.friction_coef = friction_coefficient
        self.leakage_coef = leakage_coefficient

        # Calculate areas
        self.area_bore = np.pi * (bore_diameter / 2) ** 2
        self.area_rod = np.pi * (rod_diameter / 2) ** 2
        self.area_annulus = self.area_bore - self.area_rod

        # State
        self.state = HydraulicCylinderState()

        # Moving mass (piston + rod + load)
        self.moving_mass = 50.0  # kg (default, will be updated)

    def update(
        self,
        dt: float,
        pressure_supply: float,
        pressure_return: float,
        valve_position: float,  # -1.0 to 1.0 (retract to extend)
        external_load: float = 0.0,
    ) -> None:
        """
        Update cylinder state based on control input

        Args:
            dt: Time step (s)
            pressure_supply: Supply pressure (Pa)
            pressure_return: Return pressure (Pa)
            valve_position: Valve spool position (-1 to 1)
            external_load: External force on cylinder (N, positive = resist extension)
        """
        # Determine flow direction and pressures
        if valve_position > 0:  # Extending
            self.state.pressure_bore = pressure_supply * abs(valve_position)
            self.state.pressure_rod = pressure_return
            flow_coefficient = valve_position
        elif valve_position < 0:  # Retracting
            self.state.pressure_bore = pressure_return
            self.state.pressure_rod = pressure_supply * abs(valve_position)
            flow_coefficient = -valve_position
        else:  # Hold position
            flow_coefficient = 0  # noqa: F841
            # Pressure decay due to leakage
            self.state.pressure_bore *= 0.99
            self.state.pressure_rod *= 0.99

        # Calculate forces
        force_bore = self.state.pressure_bore * self.area_bore
        force_rod = self.state.pressure_rod * self.area_annulus
        force_net = force_bore - force_rod

        # Friction force (Coulomb + viscous)
        friction_static = self.friction_coef * self.max_force * 0.1
        friction_viscous = (
            self.friction_coef * self.moving_mass * abs(self.state.velocity)
        )
        friction_total = friction_static + friction_viscous

        if self.state.velocity > 0:
            friction_direction = -1
        elif self.state.velocity < 0:
            friction_direction = 1
        else:
            friction_direction = (
                -np.sign(force_net) if abs(force_net) > friction_static else 0
            )

        force_friction = friction_direction * friction_total

        # Total force and acceleration
        force_total = force_net + force_friction - external_load
        self.state.acceleration = force_total / self.moving_mass

        # Update velocity and position
        self.state.velocity += self.state.acceleration * dt
        self.state.position += self.state.velocity * dt

        # Position limits
        self.state.position = np.clip(self.state.position, 0, self.stroke_length)

        # Stop at limits
        if self.state.position <= 0 and self.state.velocity < 0:
            self.state.velocity = 0
            self.state.position = 0
        elif self.state.position >= self.stroke_length and self.state.velocity > 0:
            self.state.velocity = 0
            self.state.position = self.stroke_length

        # Calculate flows
        if valve_position > 0:
            self.state.flow_in = self.area_bore * self.state.velocity
            self.state.flow_out = self.area_annulus * self.state.velocity
        else:
            self.state.flow_in = self.area_annulus * abs(self.state.velocity)
            self.state.flow_out = self.area_bore * abs(self.state.velocity)

        # Internal leakage
        pressure_diff = abs(self.state.pressure_bore - self.state.pressure_rod)
        leakage_flow = self.leakage_coef * pressure_diff
        self.state.flow_out += leakage_flow

        self.state.force = force_net

    def get_extension_ratio(self) -> float:
        """Get cylinder extension as ratio (0.0 = retracted, 1.0 = extended)"""
        return self.state.position / self.stroke_length


@dataclass
class HydraulicMotorState:
    """Current state of hydraulic motor"""

    angle: float = 0.0  # rad
    angular_velocity: float = 0.0  # rad/s
    angular_acceleration: float = 0.0  # rad/s²
    pressure_in: float = 0.0  # Pa
    pressure_out: float = 0.0  # Pa
    torque: float = 0.0  # N·m
    flow: float = 0.0  # m³/s
    temperature: float = 40.0  # °C


class HydraulicMotor:
    """
    Hydraulic motor with realistic dynamics
    """

    def __init__(
        self,
        displacement: float,
        max_speed: float,
        max_torque: float,
        efficiency: float = 0.88,
        friction_coefficient: float = 0.02,
    ):
        self.displacement = displacement  # m³/rev
        self.max_speed = max_speed  # RPM
        self.max_torque = max_torque
        self.efficiency = efficiency
        self.friction_coef = friction_coefficient

        self.state = HydraulicMotorState()
        self.inertia = 10.0  # kg·m² (will be updated based on load)

    def update(
        self,
        dt: float,
        pressure_supply: float,
        pressure_return: float,
        valve_position: float,
        external_torque: float = 0.0,
    ) -> None:
        """
        Update motor state

        Args:
            dt: Time step (s)
            pressure_supply: Supply pressure (Pa)
            pressure_return: Return pressure (Pa)
            valve_position: -1.0 to 1.0 (reverse to forward)
            external_torque: Load torque (N·m, positive = resist rotation)
        """
        direction = np.sign(valve_position) if abs(valve_position) > 0.01 else 0
        valve_opening = abs(valve_position)

        # Pressures
        if valve_opening > 0:
            self.state.pressure_in = pressure_supply * valve_opening
            self.state.pressure_out = pressure_return
        else:
            self.state.pressure_in = 0
            self.state.pressure_out = 0

        # Torque from pressure
        pressure_diff = self.state.pressure_in - self.state.pressure_out
        theoretical_torque = (pressure_diff * self.displacement) / (2 * np.pi)
        actual_torque = theoretical_torque * self.efficiency * direction

        # Friction torque
        friction_torque = (
            self.friction_coef * self.max_torque * np.sign(self.state.angular_velocity)
        )

        # Total torque and acceleration
        torque_net = actual_torque - friction_torque - external_torque
        self.state.angular_acceleration = torque_net / self.inertia

        # Update velocity and angle
        self.state.angular_velocity += self.state.angular_acceleration * dt
        self.state.angle += self.state.angular_velocity * dt

        # Normalize angle
        self.state.angle = self.state.angle % (2 * np.pi)

        # Flow rate
        self.state.flow = (
            abs(self.state.angular_velocity) * self.displacement / (2 * np.pi)
        )

        self.state.torque = actual_torque

    def get_rpm(self) -> float:
        """Get motor speed in RPM"""
        return self.state.angular_velocity * 60 / (2 * np.pi)
