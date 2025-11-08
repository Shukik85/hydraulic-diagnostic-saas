"""
Mechanical System - Excavator Kinematics and Dynamics
Boom, Stick, Bucket with realistic physics
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LinkState:
    """State of a mechanical link (boom/stick/bucket)"""

    angle: float = 0.0  # rad (relative to previous link)
    angular_velocity: float = 0.0  # rad/s
    angular_acceleration: float = 0.0  # rad/s²

    # Cartesian position of end point
    x: float = 0.0  # m
    y: float = 0.0  # m
    z: float = 0.0  # m


class ExcavatorKinematics:
    """
    3-link excavator arm kinematics
    Coordinate system: X=forward, Y=left, Z=up
    Origin at swing bearing center
    """

    def __init__(
        self, boom_length: float, stick_length: float, bucket_length: float = 0.8
    ):
        self.L_boom = boom_length
        self.L_stick = stick_length
        self.L_bucket = bucket_length

        # Initial angles (rad)
        self.boom = LinkState(angle=np.deg2rad(45))  # 45° up from horizontal
        self.stick = LinkState(angle=np.deg2rad(-90))  # -90° relative to boom
        self.bucket = LinkState(angle=np.deg2rad(-45))  # -45° relative to stick

        self.swing_angle = 0.0  # rad (platform rotation)

        self.update_forward_kinematics()

    def update_forward_kinematics(self) -> None:
        """Calculate end-effector position from joint angles"""
        # Boom end position (in vertical plane, before swing)
        x_boom = self.L_boom * np.cos(self.boom.angle)
        z_boom = self.L_boom * np.sin(self.boom.angle)

        # Stick end (relative angle)
        stick_abs_angle = self.boom.angle + self.stick.angle
        x_stick = x_boom + self.L_stick * np.cos(stick_abs_angle)
        z_stick = z_boom + self.L_stick * np.sin(stick_abs_angle)

        # Bucket end
        bucket_abs_angle = stick_abs_angle + self.bucket.angle
        x_bucket = x_stick + self.L_bucket * np.cos(bucket_abs_angle)
        z_bucket = z_stick + self.L_bucket * np.sin(bucket_abs_angle)

        # Apply swing rotation
        self.boom.x = x_boom * np.cos(self.swing_angle)
        self.boom.y = x_boom * np.sin(self.swing_angle)
        self.boom.z = z_boom

        self.stick.x = x_stick * np.cos(self.swing_angle)
        self.stick.y = x_stick * np.sin(self.swing_angle)
        self.stick.z = z_stick

        self.bucket.x = x_bucket * np.cos(self.swing_angle)
        self.bucket.y = x_bucket * np.sin(self.swing_angle)
        self.bucket.z = z_bucket

    def get_bucket_position(self) -> tuple[float, float, float]:
        """Get bucket tip position (x, y, z) in meters"""
        return (self.bucket.x, self.bucket.y, self.bucket.z)

    def get_reach(self) -> float:
        """Get horizontal reach from swing center"""
        return np.sqrt(self.bucket.x**2 + self.bucket.y**2)


class ExcavatorDynamics:
    """
    Dynamic model of excavator arm
    Calculates torques, forces, and power requirements
    """

    def __init__(
        self,
        boom_mass: float,
        stick_mass: float,
        bucket_mass: float,
        boom_inertia: float,
        stick_inertia: float,
        bucket_inertia: float,
        boom_length: float,
        stick_length: float,
    ):
        self.m_boom = boom_mass
        self.m_stick = stick_mass
        self.m_bucket = bucket_mass

        self.I_boom = boom_inertia
        self.I_stick = stick_inertia
        self.I_bucket = bucket_inertia

        self.L_boom = boom_length
        self.L_stick = stick_length

        # Center of mass positions (as fraction of link length)
        self.com_boom = 0.4  # 40% from joint
        self.com_stick = 0.45
        self.com_bucket = 0.5

        self.g = 9.81  # m/s²

    def calculate_gravity_torques(
        self,
        boom_angle: float,
        stick_angle: float,
        bucket_angle: float,
        load_mass: float = 0.0,
    ) -> tuple[float, float, float]:
        """
        Calculate gravity torques on each joint

        Returns:
            (torque_boom, torque_stick, torque_bucket) in N·m
            Positive torque = needs force to resist gravity
        """
        # Boom torque (from boom weight + stick + bucket + load)
        r_boom_com = self.L_boom * self.com_boom
        torque_boom_self = self.m_boom * self.g * r_boom_com * np.cos(boom_angle)

        # Stick weight acts through boom
        stick_abs_angle = boom_angle + stick_angle
        r_stick = self.L_boom + self.L_stick * self.com_stick
        torque_boom_stick = self.m_stick * self.g * r_stick * np.cos(boom_angle)

        # Bucket + load weight
        bucket_abs_angle = stick_abs_angle + bucket_angle
        r_bucket = self.L_boom + self.L_stick
        total_bucket_mass = self.m_bucket + load_mass
        torque_boom_bucket = total_bucket_mass * self.g * r_bucket * np.cos(boom_angle)

        torque_boom = torque_boom_self + torque_boom_stick + torque_boom_bucket

        # Stick torque (from stick + bucket + load)
        r_stick_com = self.L_stick * self.com_stick
        torque_stick_self = (
            self.m_stick * self.g * r_stick_com * np.cos(stick_abs_angle)
        )
        torque_stick_bucket = (
            total_bucket_mass * self.g * self.L_stick * np.cos(stick_abs_angle)
        )

        torque_stick = torque_stick_self + torque_stick_bucket

        # Bucket torque (from bucket + load only)
        torque_bucket = total_bucket_mass * self.g * 0.3 * np.cos(bucket_abs_angle)

        return (torque_boom, torque_stick, torque_bucket)

    def calculate_inertial_torques(
        self, boom_accel: float, stick_accel: float, bucket_accel: float
    ) -> tuple[float, float, float]:
        """
        Calculate torques needed to accelerate links

        Returns:
            (torque_boom, torque_stick, torque_bucket) in N·m
        """
        # Simplified: using moment of inertia
        torque_boom = self.I_boom * boom_accel
        torque_stick = self.I_stick * stick_accel
        torque_bucket = self.I_bucket * bucket_accel

        return (torque_boom, torque_stick, torque_bucket)

    def calculate_digging_resistance(self, soil_type: str = "medium") -> float:
        """
        Estimate digging resistance force

        Args:
            soil_type: "soft", "medium", "hard", "rock"

        Returns:
            Resistance force in N
        """
        resistances = {
            "soft": 50000,  # 50 kN (sand, loose soil)
            "medium": 100000,  # 100 kN (clay, compact soil)
            "hard": 200000,  # 200 kN (dense clay, gravel)
            "rock": 400000,  # 400 kN (rock, concrete)
        }
        return resistances.get(soil_type, 100000)


class PlatformSwing:
    """Platform swing dynamics"""

    def __init__(
        self,
        platform_mass: float,
        platform_inertia: float,
        max_swing_speed: float = 15.0,  # RPM
    ):
        self.mass = platform_mass
        self.inertia = platform_inertia
        self.max_speed = max_swing_speed

        self.angle = 0.0  # rad
        self.velocity = 0.0  # rad/s
        self.acceleration = 0.0  # rad/s²

    def update(
        self, dt: float, motor_torque: float, friction_torque: float = 1000.0
    ) -> None:
        """Update swing state"""
        # Friction opposes motion
        friction = (
            friction_torque * np.sign(self.velocity) if abs(self.velocity) > 0.01 else 0
        )

        net_torque = motor_torque - friction
        self.acceleration = net_torque / self.inertia

        self.velocity += self.acceleration * dt
        self.angle += self.velocity * dt

        # Normalize angle
        self.angle = self.angle % (2 * np.pi)

    def get_angle_degrees(self) -> float:
        """Get swing angle in degrees"""
        return np.rad2deg(self.angle)
