"""
Excavator Simulation Runner
Demonstrates all scenarios and generates training data
"""
from pathlib import Path

from logger import SimulationLogger
from scenarios.digging import DiggingScenario, EmergencyStopScenario, LoadingScenario
from simulator import ExcavatorSimulator


def run_digging_scenario(duration: float = 60.0) -> None:
    """Run normal digging operations"""
    print("\n" + "="*70)
    print("SCENARIO 1: NORMAL DIGGING OPERATIONS")
    print("="*70)
    
    sim = ExcavatorSimulator()
    scenario = DiggingScenario()
    
    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)
    
    with SimulationLogger(output_dir="logs") as logger:
        steps = int(duration / sim.dt)
        
        for step in range(steps):
            controls = scenario.get_controls(sim.time)
            
            # Update load
            sim.load_mass = scenario.get_load_mass(sim.time)
            sim.soil_type = scenario.get_soil_resistance(sim.time)
            
            # Simulate
            sim.update(
                boom_cmd=controls['boom'],
                stick_cmd=controls['stick'],
                bucket_cmd=controls['bucket'],
                swing_cmd=controls['swing']
            )
            
            # Detect faults
            faults = {
                'pressure_surge': sim.pump_pressure > 330e5,  # Above relief
                'overheat': sim.system_temperature > 85,
                'overload': sim.load_mass > 1800
            }
            
            # Log every 10ms
            logger.log(
                sim_time=sim.time,
                control_inputs=controls,
                sensor_state=sim.sensors.export_to_dict(),
                bucket_position=sim.kinematics.get_bucket_position(),
                system_state={
                    'pump_speed': sim.pump_speed,
                    'system_temperature': sim.system_temperature,
                    'load_mass': sim.load_mass
                },
                faults=faults
            )
            
            # Progress
            if step % 1000 == 0:
                print(f"  Progress: {sim.time:.1f}s / {duration}s "
                      f"| Phase: {scenario.phase} "
                      f"| Load: {sim.load_mass:.0f} kg "
                      f"| Temp: {sim.system_temperature:.1f}°C")
    
    print(f"\n✅ Simulation complete! {steps} samples recorded")


def run_loading_scenario(duration: float = 120.0) -> None:
    """Run fast loading cycles"""
    print("\n" + "="*70)
    print("SCENARIO 2: FAST LOADING CYCLES")
    print("="*70)
    
    sim = ExcavatorSimulator()
    scenario = LoadingScenario(cycles=5)
    
    with SimulationLogger(output_dir="logs") as logger:
        steps = int(duration / sim.dt)
        
        for step in range(steps):
            controls = scenario.get_controls(sim.time)
            sim.load_mass = scenario.get_load_mass(sim.time)
            
            sim.update(
                boom_cmd=controls['boom'],
                stick_cmd=controls['stick'],
                bucket_cmd=controls['bucket'],
                swing_cmd=controls['swing']
            )
            
            faults = {
                'pressure_surge': sim.pump_pressure > 330e5,
                'overheat': sim.system_temperature > 85,
                'overload': sim.load_mass > 1800
            }
            
            logger.log(
                sim_time=sim.time,
                control_inputs=controls,
                sensor_state=sim.sensors.export_to_dict(),
                bucket_position=sim.kinematics.get_bucket_position(),
                system_state={
                    'pump_speed': sim.pump_speed,
                    'system_temperature': sim.system_temperature,
                    'load_mass': sim.load_mass
                },
                faults=faults
            )
            
            if step % 1000 == 0:
                print(f"  Progress: {sim.time:.1f}s / {duration}s "
                      f"| Load: {sim.load_mass:.0f} kg")
    
    print("\n✅ Simulation complete!")


def run_emergency_stop_scenario(duration: float = 10.0) -> None:
    """Run emergency stop - generates pressure surge fault"""
    print("\n" + "="*70)
    print("SCENARIO 3: EMERGENCY STOP (FAULT INJECTION)")
    print("="*70)
    
    sim = ExcavatorSimulator()
    scenario = EmergencyStopScenario()
    
    with SimulationLogger(output_dir="logs") as logger:
        steps = int(duration / sim.dt)
        fault_triggered = False
        
        for _ in range(steps):
            controls = scenario.get_controls(sim.time)
            
            sim.update(
                boom_cmd=controls['boom'],
                stick_cmd=controls['stick'],
                bucket_cmd=controls['bucket'],
                swing_cmd=controls['swing']
            )
            
            # Detect pressure surge from sudden stop
            pressure_surge = sim.pump_pressure > 330e5
            
            if pressure_surge and not fault_triggered:
                print(f"  ⚠️  PRESSURE SURGE DETECTED at t={sim.time:.2f}s!")
                print(f"      Pressure: {sim.pump_pressure/1e5:.1f} bar")
                fault_triggered = True
            
            faults = {
                'pressure_surge': pressure_surge,
                'overheat': False,
                'overload': False
            }
            
            logger.log(
                sim_time=sim.time,
                control_inputs=controls,
                sensor_state=sim.sensors.export_to_dict(),
                bucket_position=sim.kinematics.get_bucket_position(),
                system_state={
                    'pump_speed': sim.pump_speed,
                    'system_temperature': sim.system_temperature,
                    'load_mass': sim.load_mass
                },
                faults=faults
            )
    
    print("\n✅ Fault scenario complete!")


def main() -> None:
    """Run all scenarios"""
    print("\n" + "="*70)
    print("HYDRAULIC EXCAVATOR SIMULATOR")
    print("Realistic physics-based BIM model")
    print("="*70)
    
    # Run scenarios
    run_digging_scenario(duration=60.0)      # 60s normal ops
    run_loading_scenario(duration=120.0)     # 120s fast cycles
    run_emergency_stop_scenario(duration=10.0)  # 10s fault
    
    print("\n" + "="*70)
    print("✅ ALL SIMULATIONS COMPLETE!")
    print("="*70)
    print("\n📊 Check 'logs/' folder for CSV data")
    print("🚀 Ready for ML training!")


if __name__ == "__main__":
    main()
