"""
Equipment Metadata Schema
Physics-informed configuration for hydraulic systems
"""
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThresholdConfig:
    """Component threshold configuration"""
    max_value: float
    warning_value: float
    min_value: float | None = None
    
    def to_dict(self):
        return {
            "max": self.max_value,
            "warning": self.warning_value,
            "min": self.min_value
        }


@dataclass
class PumpConfig:
    """Hydraulic pump configuration"""
    model: str
    manufacturer: str
    nominal_pressure: float  # bar
    max_pressure: float      # bar
    nominal_rpm: float       # RPM
    displacement: float      # cc/rev
    flow_rate: float         # L/min
    
    # Thresholds
    pressure_threshold: ThresholdConfig
    temp_threshold: ThresholdConfig
    vibration_max: float     # mm/s
    efficiency_min: float    # ratio
    
    def to_dict(self):
        return {
            "model": self.model,
            "manufacturer": self.manufacturer,
            "nominal_pressure": self.nominal_pressure,
            "max_pressure": self.max_pressure,
            "nominal_rpm": self.nominal_rpm,
            "displacement": self.displacement,
            "flow_rate": self.flow_rate,
            "thresholds": {
                "pressure": self.pressure_threshold.to_dict(),
                "temperature": self.temp_threshold.to_dict(),
                "vibration_max": self.vibration_max,
                "efficiency_min": self.efficiency_min
            }
        }


@dataclass
class CylinderConfig:
    """Hydraulic cylinder configuration"""
    model: str
    bore_diameter: float     # mm
    rod_diameter: float      # mm
    stroke: float            # mm
    max_pressure: float      # bar
    extend_force: float      # kN
    retract_force: float     # kN
    
    # Thresholds
    pressure_extend_threshold: ThresholdConfig
    pressure_retract_threshold: ThresholdConfig
    pressure_diff_max: float
    load_ratio_max: float
    velocity_max: float      # m/s
    
    def to_dict(self):
        return {
            "model": self.model,
            "bore_diameter": self.bore_diameter,
            "rod_diameter": self.rod_diameter,
            "stroke": self.stroke,
            "max_pressure": self.max_pressure,
            "extend_force": self.extend_force,
            "retract_force": self.retract_force,
            "thresholds": {
                "pressure_extend": self.pressure_extend_threshold.to_dict(),
                "pressure_retract": self.pressure_retract_threshold.to_dict(),
                "pressure_diff_max": self.pressure_diff_max,
                "load_ratio_max": self.load_ratio_max,
                "velocity_max": self.velocity_max
            }
        }


@dataclass
class EquipmentConfig:
    """Complete equipment configuration"""
    equipment_id: str
    model: str
    manufacturer: str
    
    # Components
    pump: PumpConfig
    boom_cylinder: CylinderConfig
    stick_cylinder: CylinderConfig
    bucket_cylinder: CylinderConfig
    
    # Operational limits
    ambient_temp_min: float = -20
    ambient_temp_max: float = 50
    oil_temp_min: float = 15
    oil_temp_max: float = 90
    oil_viscosity: str = "ISO VG 46"
    
    # Maintenance
    service_interval_hours: int = 500
    operating_hours: int = 0
    
    def to_dict(self):
        return {
            "equipment_id": self.equipment_id,
            "model": self.model,
            "manufacturer": self.manufacturer,
            "components": {
                "pump": self.pump.to_dict(),
                "boom_cylinder": self.boom_cylinder.to_dict(),
                "stick_cylinder": self.stick_cylinder.to_dict(),
                "bucket_cylinder": self.bucket_cylinder.to_dict()
            },
            "operational_limits": {
                "ambient_temp_min": self.ambient_temp_min,
                "ambient_temp_max": self.ambient_temp_max,
                "oil_temp_min": self.oil_temp_min,
                "oil_temp_max": self.oil_temp_max,
                "oil_viscosity": self.oil_viscosity
            },
            "maintenance": {
                "service_interval_hours": self.service_interval_hours,
                "operating_hours": self.operating_hours
            }
        }
    
    def save_json(self, filepath: str):
        """Save configuration to JSON"""
        from pathlib import Path
        Path(filepath).write_text(json.dumps(self.to_dict(), indent=2))
    
    @classmethod
    def from_json(cls, filepath: str):
        """Load configuration from JSON"""
        from pathlib import Path
        data = json.loads(Path(filepath).read_text())
        
        # Reconstruct pump
        pump_data = data["components"]["pump"]
        pump = PumpConfig(
            model=pump_data["model"],
            manufacturer=pump_data["manufacturer"],
            nominal_pressure=pump_data["nominal_pressure"],
            max_pressure=pump_data["max_pressure"],
            nominal_rpm=pump_data["nominal_rpm"],
            displacement=pump_data["displacement"],
            flow_rate=pump_data["flow_rate"],
            pressure_threshold=ThresholdConfig(
                max_value=pump_data["thresholds"]["pressure"]["max"],
                warning_value=pump_data["thresholds"]["pressure"]["warning"]
            ),
            temp_threshold=ThresholdConfig(
                max_value=pump_data["thresholds"]["temperature"]["max"],
                warning_value=pump_data["thresholds"]["temperature"]["warning"]
            ),
            vibration_max=pump_data["thresholds"]["vibration_max"],
            efficiency_min=pump_data["thresholds"]["efficiency_min"]
        )
        
        # Helper function for cylinders
        def load_cylinder(cyl_data):
            return CylinderConfig(
                model=cyl_data["model"],
                bore_diameter=cyl_data["bore_diameter"],
                rod_diameter=cyl_data["rod_diameter"],
                stroke=cyl_data["stroke"],
                max_pressure=cyl_data["max_pressure"],
                extend_force=cyl_data["extend_force"],
                retract_force=cyl_data["retract_force"],
                pressure_extend_threshold=ThresholdConfig(
                    max_value=cyl_data["thresholds"]["pressure_extend"]["max"],
                    warning_value=cyl_data["thresholds"]["pressure_extend"]["warning"]
                ),
                pressure_retract_threshold=ThresholdConfig(
                    max_value=cyl_data["thresholds"]["pressure_retract"]["max"],
                    warning_value=cyl_data["thresholds"]["pressure_retract"]["warning"]
                ),
                pressure_diff_max=cyl_data["thresholds"]["pressure_diff_max"],
                load_ratio_max=cyl_data["thresholds"]["load_ratio_max"],
                velocity_max=cyl_data["thresholds"]["velocity_max"]
            )
        
        # Reconstruct cylinders
        boom_cyl = load_cylinder(data["components"]["boom_cylinder"])
        stick_cyl = load_cylinder(data["components"]["stick_cylinder"])
        bucket_cyl = load_cylinder(data["components"]["bucket_cylinder"])
        
        return cls(
            equipment_id=data["equipment_id"],
            model=data["model"],
            manufacturer=data["manufacturer"],
            pump=pump,
            boom_cylinder=boom_cyl,
            stick_cylinder=stick_cyl,
            bucket_cylinder=bucket_cyl,
            ambient_temp_min=data["operational_limits"]["ambient_temp_min"],
            ambient_temp_max=data["operational_limits"]["ambient_temp_max"],
            oil_temp_min=data["operational_limits"]["oil_temp_min"],
            oil_temp_max=data["operational_limits"]["oil_temp_max"],
            oil_viscosity=data["operational_limits"]["oil_viscosity"],
            service_interval_hours=data["maintenance"]["service_interval_hours"],
            operating_hours=data["maintenance"]["operating_hours"]
        )


def create_cat336_config():
    """Create default configuration for CAT 336 excavator"""
    
    config = EquipmentConfig(
        equipment_id="CAT336_SN12345",
        model="CAT 336",
        manufacturer="Caterpillar",
        
        pump=PumpConfig(
            model="A10VO_140",
            manufacturer="Rexroth",
            nominal_pressure=280,
            max_pressure=350,
            nominal_rpm=1800,
            displacement=140,
            flow_rate=252,
            pressure_threshold=ThresholdConfig(max_value=320, warning_value=300),
            temp_threshold=ThresholdConfig(max_value=90, warning_value=85),
            vibration_max=15.0,
            efficiency_min=0.85
        ),
        
        boom_cylinder=CylinderConfig(
            model="Hydraulic_Cylinder_250x140x1800",
            bore_diameter=250,
            rod_diameter=140,
            stroke=1800,
            max_pressure=350,
            extend_force=172,
            retract_force=95,
            pressure_extend_threshold=ThresholdConfig(max_value=320, warning_value=300),
            pressure_retract_threshold=ThresholdConfig(max_value=320, warning_value=300),
            pressure_diff_max=50,
            load_ratio_max=0.95,
            velocity_max=0.5
        ),
        
        stick_cylinder=CylinderConfig(
            model="Hydraulic_Cylinder_200x120x1500",
            bore_diameter=200,
            rod_diameter=120,
            stroke=1500,
            max_pressure=350,
            extend_force=110,
            retract_force=62,
            pressure_extend_threshold=ThresholdConfig(max_value=320, warning_value=300),
            pressure_retract_threshold=ThresholdConfig(max_value=320, warning_value=300),
            pressure_diff_max=50,
            load_ratio_max=0.95,
            velocity_max=0.4
        ),
        
        bucket_cylinder=CylinderConfig(
            model="Hydraulic_Cylinder_160x100x1200",
            bore_diameter=160,
            rod_diameter=100,
            stroke=1200,
            max_pressure=350,
            extend_force=70,
            retract_force=42,
            pressure_extend_threshold=ThresholdConfig(max_value=320, warning_value=300),
            pressure_retract_threshold=ThresholdConfig(max_value=320, warning_value=300),
            pressure_diff_max=45,
            load_ratio_max=0.92,
            velocity_max=0.6
        ),
        
        operating_hours=3420
    )
    
    return config


if __name__ == "__main__":
    # Create example config
    config = create_cat336_config()
    
    # Save to JSON
    config.save_json("config/cat336_equipment.json")
    print("✅ Equipment configuration saved!")
    
    # Load and verify
    loaded = EquipmentConfig.from_json("config/cat336_equipment.json")
    print(f"✅ Loaded config for: {loaded.equipment_id}")
    print(f"   Pump: {loaded.pump.model}")
    print(f"   Boom cylinder: {loaded.boom_cylinder.model}")
