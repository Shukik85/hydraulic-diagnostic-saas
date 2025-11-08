"""
Data Logger - CSV export for ML training
"""
import csv
from datetime import datetime, timezone
from pathlib import Path


class SimulationLogger:
    """Logs all sensor data and system state to CSV"""
    
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"excavator_sim_{timestamp}.csv"
        
        self.fieldnames = [
            'timestamp',
            'time_s',
            # Control inputs
            'cmd_boom',
            'cmd_stick',
            'cmd_bucket',
            'cmd_swing',
            # Pressures (bar)
            'pressure_pump',
            'pressure_boom_extend',
            'pressure_boom_retract',
            'pressure_stick_extend',
            'pressure_stick_retract',
            'pressure_bucket_extend',
            'pressure_bucket_retract',
            # Temperatures (°C)
            'temp_oil_tank',
            'temp_pump',
            'temp_valve_block',
            # Positions (m or rad)
            'pos_boom_cylinder',
            'pos_stick_cylinder',
            'pos_bucket_cylinder',
            'pos_swing_angle',
            # Velocities (m/s or rad/s)
            'vel_boom_cylinder',
            'vel_stick_cylinder',
            'vel_bucket_cylinder',
            'vel_swing',
            # Vibrations (mm/s)
            'vib_pump',
            'vib_valve_block',
            'vib_boom_cylinder',
            'vib_stick_cylinder',
            # Forces (kN)
            'force_boom_cylinder',
            'force_stick_cylinder',
            'force_bucket_cylinder',
            # Bucket position (m)
            'bucket_x',
            'bucket_y',
            'bucket_z',
            # System state
            'pump_speed_rpm',
            'system_temp',
            'load_mass',
            # Fault labels
            'fault_pressure_surge',
            'fault_overheat',
            'fault_overload',
            'fault_any'
        ]
        
        self.writer = None
        self.file_handle = None
        self._init_csv()
    
    def _init_csv(self) -> None:
        """Initialize CSV file with headers"""
        self.file_handle = self.log_file.open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)
        self.writer.writeheader()
        print(f"📝 Logging to: {self.log_file}")
    
    def log(
        self,
        sim_time: float,
        control_inputs: dict,
        sensor_state: dict,
        bucket_position: tuple,
        system_state: dict,
        faults: dict | None = None
    ) -> None:
        """Log one time step"""
        faults = faults or {
            'pressure_surge': False,
            'overheat': False,
            'overload': False
        }
        
        row = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'time_s': f'{sim_time:.3f}',
            # Control inputs
            'cmd_boom': f'{control_inputs.get("boom", 0):.3f}',
            'cmd_stick': f'{control_inputs.get("stick", 0):.3f}',
            'cmd_bucket': f'{control_inputs.get("bucket", 0):.3f}',
            'cmd_swing': f'{control_inputs.get("swing", 0):.3f}',
            # Pressures
            'pressure_pump': f'{sensor_state["pressure_pump_outlet"]/1e5:.2f}',
            'pressure_boom_extend': f'{sensor_state["pressure_boom_extend"]/1e5:.2f}',
            'pressure_boom_retract': f'{sensor_state["pressure_boom_retract"]/1e5:.2f}',
            'pressure_stick_extend': f'{sensor_state["pressure_stick_extend"]/1e5:.2f}',
            'pressure_stick_retract': f'{sensor_state["pressure_stick_retract"]/1e5:.2f}',
            'pressure_bucket_extend': f'{sensor_state["pressure_bucket_extend"]/1e5:.2f}',
            'pressure_bucket_retract': f'{sensor_state["pressure_bucket_retract"]/1e5:.2f}',
            # Temperatures
            'temp_oil_tank': f'{sensor_state["temp_oil_tank"]:.2f}',
            'temp_pump': f'{sensor_state["temp_pump"]:.2f}',
            'temp_valve_block': f'{sensor_state["temp_valve_block"]:.2f}',
            # Positions
            'pos_boom_cylinder': f'{sensor_state["pos_boom_cylinder"]:.4f}',
            'pos_stick_cylinder': f'{sensor_state["pos_stick_cylinder"]:.4f}',
            'pos_bucket_cylinder': f'{sensor_state["pos_bucket_cylinder"]:.4f}',
            'pos_swing_angle': f'{sensor_state.get("pos_swing_angle", 0):.4f}',
            # Velocities
            'vel_boom_cylinder': f'{sensor_state["vel_boom_cylinder"]:.4f}',
            'vel_stick_cylinder': f'{sensor_state["vel_stick_cylinder"]:.4f}',
            'vel_bucket_cylinder': f'{sensor_state["vel_bucket_cylinder"]:.4f}',
            'vel_swing': f'{sensor_state.get("vel_swing", 0):.4f}',
            # Vibrations
            'vib_pump': f'{sensor_state["vib_pump"]:.3f}',
            'vib_valve_block': f'{sensor_state.get("vib_valve_block", 0):.3f}',
            'vib_boom_cylinder': f'{sensor_state.get("vib_boom_cylinder", 0):.3f}',
            'vib_stick_cylinder': f'{sensor_state.get("vib_stick_cylinder", 0):.3f}',
            # Forces
            'force_boom_cylinder': f'{sensor_state.get("force_boom_cylinder", 0)/1000:.2f}',
            'force_stick_cylinder': f'{sensor_state.get("force_stick_cylinder", 0)/1000:.2f}',
            'force_bucket_cylinder': f'{sensor_state.get("force_bucket_cylinder", 0)/1000:.2f}',
            # Bucket position
            'bucket_x': f'{bucket_position[0]:.3f}',
            'bucket_y': f'{bucket_position[1]:.3f}',
            'bucket_z': f'{bucket_position[2]:.3f}',
            # System state
            'pump_speed_rpm': f'{system_state.get("pump_speed", 0):.0f}',
            'system_temp': f'{system_state.get("system_temperature", 40):.2f}',
            'load_mass': f'{system_state.get("load_mass", 0):.1f}',
            # Faults
            'fault_pressure_surge': '1' if faults['pressure_surge'] else '0',
            'fault_overheat': '1' if faults['overheat'] else '0',
            'fault_overload': '1' if faults['overload'] else '0',
            'fault_any': '1' if any(faults.values()) else '0'
        }
        
        self.writer.writerow(row)
    
    def flush(self) -> None:
        """Flush buffer to disk"""
        if self.file_handle:
            self.file_handle.flush()
    
    def close(self) -> None:
        """Close log file"""
        if self.file_handle:
            self.file_handle.close()
            print(f"✅ Log saved: {self.log_file}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
