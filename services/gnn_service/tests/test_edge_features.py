"""Tests for EdgeFeatureComputer module.

Tests physics-based edge feature computation including:
- Fluid properties (density, viscosity)
- Friction factor estimation
- Darcy-Weisbach flow rate estimation
- Dynamic feature computation

Author: GNN Service Team
Python: 3.14+
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.data.edge_features import (
    EdgeFeatureComputer,
    get_fluid_density,
    get_fluid_viscosity,
    get_material_roughness,
    estimate_friction_factor,
    estimate_flow_rate_darcy_weisbach,
    create_edge_feature_computer
)
from src.schemas.graph import EdgeSpec, EdgeMaterial
from src.schemas.requests import ComponentSensorReading


class TestFluidProperties:
    """Test fluid property calculations."""
    
    def test_fluid_density_at_room_temp(self):
        """Test density at standard temperature."""
        rho = get_fluid_density(20.0)
        
        # ISO VG 46 at 20°C should be ~870-880 kg/m³
        assert 860 < rho < 890
    
    def test_fluid_density_temperature_dependence(self):
        """Test density decreases with temperature."""
        rho_cold = get_fluid_density(20.0)
        rho_hot = get_fluid_density(80.0)
        
        # Density should decrease with temperature
        assert rho_hot < rho_cold
        
        # Expect ~40 kg/m³ difference over 60°C
        assert 30 < (rho_cold - rho_hot) < 50
    
    def test_fluid_density_clamping(self):
        """Test density clamping to reasonable range."""
        rho_extreme_cold = get_fluid_density(-50.0)
        rho_extreme_hot = get_fluid_density(200.0)
        
        # Should clamp to [800, 900]
        assert 800 <= rho_extreme_cold <= 900
        assert 800 <= rho_extreme_hot <= 900
    
    def test_fluid_viscosity_at_40c(self):
        """Test viscosity at standard test temperature."""
        nu = get_fluid_viscosity(40.0)
        
        # ISO VG 46 at 40°C should be ~46 cSt
        assert 44 < nu < 48
    
    def test_fluid_viscosity_at_100c(self):
        """Test viscosity at high temperature."""
        nu = get_fluid_viscosity(100.0)
        
        # ISO VG 46 at 100°C should be ~6.8 cSt
        assert 6 < nu < 8
    
    def test_fluid_viscosity_temperature_dependence(self):
        """Test viscosity decreases exponentially with temperature."""
        nu_40 = get_fluid_viscosity(40.0)
        nu_100 = get_fluid_viscosity(100.0)
        
        # Viscosity should decrease dramatically
        assert nu_100 < nu_40 / 5
    
    def test_material_roughness_steel(self):
        """Test roughness for steel pipes."""
        roughness = get_material_roughness(EdgeMaterial.STEEL)
        
        # Commercial steel: ~0.045 mm
        assert roughness == 0.045
    
    def test_material_roughness_rubber(self):
        """Test roughness for rubber hoses."""
        roughness = get_material_roughness(EdgeMaterial.RUBBER)
        
        # Rubber hose: ~0.15 mm
        assert roughness == 0.15
    
    def test_material_roughness_string(self):
        """Test roughness with string input."""
        roughness = get_material_roughness("steel")
        
        assert roughness == 0.045
    
    def test_material_roughness_default(self):
        """Test default roughness for unknown material."""
        roughness = get_material_roughness("unknown")
        
        # Should return default
        assert roughness == 0.10


class TestFrictionFactor:
    """Test friction factor estimation."""
    
    def test_friction_laminar_flow(self):
        """Test friction factor in laminar regime (Re < 2300)."""
        Re = 1000
        epsilon_d = 0.001
        
        f = estimate_friction_factor(epsilon_d, Re)
        
        # Laminar: f = 64/Re
        expected = 64.0 / Re
        assert abs(f - expected) < 0.001
    
    def test_friction_turbulent_flow(self):
        """Test friction factor in turbulent regime (Re > 4000)."""
        Re = 50000
        epsilon_d = 0.001
        
        f = estimate_friction_factor(epsilon_d, Re)
        
        # Turbulent: f should be ~0.02-0.03 for typical conditions
        assert 0.015 < f < 0.040
    
    def test_friction_transition_flow(self):
        """Test friction factor in transition regime (2300 < Re < 4000)."""
        Re = 3000
        epsilon_d = 0.001
        
        f = estimate_friction_factor(epsilon_d, Re)
        
        # Should be between laminar and turbulent values
        f_laminar = 64.0 / 2300
        f_turbulent = estimate_friction_factor(epsilon_d, 4000)
        
        assert f_turbulent < f < f_laminar
    
    def test_friction_factor_roughness_dependence(self):
        """Test friction factor increases with roughness."""
        Re = 50000
        
        f_smooth = estimate_friction_factor(0.0001, Re)  # Smooth pipe
        f_rough = estimate_friction_factor(0.01, Re)    # Rough pipe
        
        # Rougher pipe should have higher friction
        assert f_rough > f_smooth


class TestDarcyWeisbachFlow:
    """Test Darcy-Weisbach flow rate estimation."""
    
    def test_flow_estimation_typical_conditions(self):
        """Test flow estimation under typical hydraulic conditions."""
        Q = estimate_flow_rate_darcy_weisbach(
            pressure_drop_pa=200000,  # 2 bar
            diameter_m=0.025,         # 25 mm
            length_m=5.0,
            temperature_c=60.0,
            material=EdgeMaterial.STEEL
        )
        
        # Expect reasonable flow rate ~50-150 L/min
        assert 50 < Q < 150
    
    def test_flow_zero_pressure_drop(self):
        """Test zero flow for zero pressure drop."""
        Q = estimate_flow_rate_darcy_weisbach(
            pressure_drop_pa=0.0,
            diameter_m=0.025,
            length_m=5.0,
            temperature_c=60.0,
            material=EdgeMaterial.STEEL
        )
        
        assert Q == 0.0
    
    def test_flow_negative_pressure_drop(self):
        """Test zero flow for negative pressure drop (backflow)."""
        Q = estimate_flow_rate_darcy_weisbach(
            pressure_drop_pa=-100000,
            diameter_m=0.025,
            length_m=5.0,
            temperature_c=60.0,
            material=EdgeMaterial.STEEL
        )
        
        assert Q == 0.0
    
    def test_flow_increases_with_pressure(self):
        """Test flow increases with pressure drop."""
        Q1 = estimate_flow_rate_darcy_weisbach(
            pressure_drop_pa=100000,  # 1 bar
            diameter_m=0.025,
            length_m=5.0,
            temperature_c=60.0,
            material=EdgeMaterial.STEEL
        )
        
        Q2 = estimate_flow_rate_darcy_weisbach(
            pressure_drop_pa=200000,  # 2 bar
            diameter_m=0.025,
            length_m=5.0,
            temperature_c=60.0,
            material=EdgeMaterial.STEEL
        )
        
        assert Q2 > Q1
    
    def test_flow_increases_with_diameter(self):
        """Test flow increases with pipe diameter."""
        Q_small = estimate_flow_rate_darcy_weisbach(
            pressure_drop_pa=200000,
            diameter_m=0.016,  # 16 mm
            length_m=5.0,
            temperature_c=60.0,
            material=EdgeMaterial.STEEL
        )
        
        Q_large = estimate_flow_rate_darcy_weisbach(
            pressure_drop_pa=200000,
            diameter_m=0.025,  # 25 mm
            length_m=5.0,
            temperature_c=60.0,
            material=EdgeMaterial.STEEL
        )
        
        assert Q_large > Q_small


class TestEdgeFeatureComputer:
    """Test EdgeFeatureComputer class."""
    
    @pytest.fixture
    def computer(self):
        """Create EdgeFeatureComputer instance."""
        return EdgeFeatureComputer()
    
    @pytest.fixture
    def edge_spec(self):
        """Create sample EdgeSpec."""
        return EdgeSpec(
            source_id="pump_1",
            target_id="valve_1",
            edge_type="pipe",
            diameter_mm=25.0,
            length_m=5.0,
            material=EdgeMaterial.STEEL,
            pressure_rating_bar=350.0,
            install_date=datetime.now() - timedelta(days=365)
        )
    
    @pytest.fixture
    def sensor_readings(self):
        """Create sample sensor readings."""
        return {
            "pump_1": ComponentSensorReading(
                pressure_bar=150.0,
                temperature_c=65.0,
                vibration_g=0.8,
                rpm=1450
            ),
            "valve_1": ComponentSensorReading(
                pressure_bar=148.0,
                temperature_c=64.0,
                vibration_g=0.3
            )
        }
    
    def test_compute_pressure_drop(self, computer, sensor_readings):
        """Test pressure drop calculation."""
        src = sensor_readings["pump_1"]
        tgt = sensor_readings["valve_1"]
        
        dp = computer._compute_pressure_drop(src, tgt)
        
        # Expected: 150 - 148 = 2 bar
        assert abs(dp - 2.0) < 0.01
    
    def test_compute_temperature_delta(self, computer, sensor_readings):
        """Test temperature delta calculation."""
        src = sensor_readings["pump_1"]
        tgt = sensor_readings["valve_1"]
        
        dt = computer._compute_temperature_delta(src, tgt)
        
        # Expected: 65 - 64 = 1°C
        assert abs(dt - 1.0) < 0.01
    
    def test_compute_vibration_level_both_sensors(self, computer, sensor_readings):
        """Test vibration level with both sensors available."""
        src = sensor_readings["pump_1"]
        tgt = sensor_readings["valve_1"]
        
        vib = computer._compute_vibration_level(src, tgt)
        
        # Expected average: (0.8 + 0.3) / 2 = 0.55
        assert abs(vib - 0.55) < 0.01
    
    def test_compute_vibration_level_one_sensor(self, computer):
        """Test vibration level with only one sensor."""
        src = ComponentSensorReading(
            pressure_bar=150.0,
            temperature_c=65.0,
            vibration_g=0.8
        )
        tgt = ComponentSensorReading(
            pressure_bar=148.0,
            temperature_c=64.0,
            vibration_g=None
        )
        
        vib = computer._compute_vibration_level(src, tgt)
        
        # Expected: 0.8 (only source)
        assert abs(vib - 0.8) < 0.01
    
    def test_compute_vibration_level_no_sensors(self, computer):
        """Test vibration level with no sensors."""
        src = ComponentSensorReading(
            pressure_bar=150.0,
            temperature_c=65.0
        )
        tgt = ComponentSensorReading(
            pressure_bar=148.0,
            temperature_c=64.0
        )
        
        vib = computer._compute_vibration_level(src, tgt)
        
        # Expected: 0.0 (no sensors)
        assert vib == 0.0
    
    def test_estimate_flow_rate(self, computer, edge_spec):
        """Test flow rate estimation."""
        flow = computer._estimate_flow_rate(
            edge=edge_spec,
            pressure_drop_bar=2.0,
            temperature_c=64.5
        )
        
        # Should return positive flow rate
        assert flow > 0
        
        # Typical range for these conditions: 50-150 L/min
        assert 50 < flow < 150
    
    def test_estimate_flow_rate_zero_pressure(self, computer, edge_spec):
        """Test flow rate with zero pressure drop."""
        flow = computer._estimate_flow_rate(
            edge=edge_spec,
            pressure_drop_bar=0.0,
            temperature_c=64.5
        )
        
        assert flow == 0.0
    
    def test_compute_edge_features_complete(
        self, computer, edge_spec, sensor_readings
    ):
        """Test complete edge feature computation."""
        current_time = datetime.now()
        
        features = computer.compute_edge_features(
            edge=edge_spec,
            sensor_readings=sensor_readings,
            current_time=current_time
        )
        
        # Check all 6 features present
        assert "flow_rate_lpm" in features
        assert "pressure_drop_bar" in features
        assert "temperature_delta_c" in features
        assert "vibration_level_g" in features
        assert "age_hours" in features
        assert "maintenance_score" in features
        
        # Check reasonable values
        assert features["flow_rate_lpm"] > 0
        assert features["pressure_drop_bar"] > 0
        assert features["temperature_delta_c"] > 0
        assert features["vibration_level_g"] >= 0
        assert features["age_hours"] > 0
        assert 0 <= features["maintenance_score"] <= 1
    
    def test_compute_edge_features_missing_component(
        self, computer, edge_spec
    ):
        """Test error handling for missing component."""
        sensor_readings = {
            "pump_1": ComponentSensorReading(
                pressure_bar=150.0,
                temperature_c=65.0
            )
            # valve_1 missing
        }
        
        current_time = datetime.now()
        
        with pytest.raises(KeyError):
            computer.compute_edge_features(
                edge=edge_spec,
                sensor_readings=sensor_readings,
                current_time=current_time
            )
    
    def test_compute_all_edges(self, computer, edge_spec, sensor_readings):
        """Test computing features for multiple edges."""
        edges = [edge_spec]
        current_time = datetime.now()
        
        all_features = computer.compute_all_edges(
            edges=edges,
            sensor_readings=sensor_readings,
            current_time=current_time
        )
        
        # Check result structure
        assert "pump_1->valve_1" in all_features
        
        # Check features
        features = all_features["pump_1->valve_1"]
        assert len(features) == 6
    
    def test_compute_all_edges_with_error(self, computer, edge_spec):
        """Test error handling in batch computation."""
        # Missing sensor readings
        edges = [edge_spec]
        sensor_readings = {}
        current_time = datetime.now()
        
        all_features = computer.compute_all_edges(
            edges=edges,
            sensor_readings=sensor_readings,
            current_time=current_time
        )
        
        # Should return default features instead of failing
        assert "pump_1->valve_1" in all_features
        features = all_features["pump_1->valve_1"]
        
        # All features should be defaults (mostly zeros)
        assert features["flow_rate_lpm"] == 0.0
        assert features["pressure_drop_bar"] == 0.0
    
    def test_default_features(self, computer):
        """Test default feature values."""
        defaults = computer._get_default_features()
        
        assert defaults["flow_rate_lpm"] == 0.0
        assert defaults["pressure_drop_bar"] == 0.0
        assert defaults["temperature_delta_c"] == 0.0
        assert defaults["vibration_level_g"] == 0.0
        assert defaults["age_hours"] == 0.0
        assert defaults["maintenance_score"] == 0.5  # Neutral


class TestFactoryFunction:
    """Test factory function."""
    
    def test_create_edge_feature_computer(self):
        """Test factory function."""
        computer = create_edge_feature_computer()
        
        assert isinstance(computer, EdgeFeatureComputer)
