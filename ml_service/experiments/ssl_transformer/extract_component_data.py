"""
Extract Component-Level Data from BIM Excavator Logs
Creates universal component datasets for transfer learning
"""
import pandas as pd
from pathlib import Path
from typing import List


def extract_cylinder_data(excavator_df: pd.DataFrame, cylinder_name: str) -> pd.DataFrame:
    """
    Extract cylinder-specific features from excavator data
    
    Args:
        excavator_df: Full excavator dataset
        cylinder_name: 'boom', 'stick', or 'bucket'
    
    Returns:
        DataFrame with cylinder features
    """
    
    # Define cylinder features mapping
    cylinder_features = {
        'boom': {
            'pressure_extend': 'pressure_boom_extend',
            'pressure_retract': 'pressure_boom_retract',
            'position': 'pos_boom_cylinder',
            'velocity': 'vel_boom_cylinder',
            'force': 'force_boom_cylinder',
        },
        'stick': {
            'pressure_extend': 'pressure_stick_extend',
            'pressure_retract': 'pressure_stick_retract',
            'position': 'pos_stick_cylinder',
            'velocity': 'vel_stick_cylinder',
            'force': 'force_stick_cylinder',
        },
        'bucket': {
            'pressure_extend': 'pressure_bucket_extend',
            'pressure_retract': 'pressure_bucket_retract',
            'position': 'pos_bucket_cylinder',
            'velocity': 'vel_bucket_cylinder',
            'force': 'force_bucket_cylinder',
        }
    }
    
    # Get features for this cylinder
    features = cylinder_features[cylinder_name]
    
    # Extract and rename to canonical names
    cylinder_df = pd.DataFrame()
    cylinder_df['pressure_extend'] = excavator_df[features['pressure_extend']]
    cylinder_df['pressure_retract'] = excavator_df[features['pressure_retract']]
    cylinder_df['position'] = excavator_df[features['position']]
    cylinder_df['velocity'] = excavator_df[features['velocity']]
    cylinder_df['force'] = excavator_df[features['force']]
    
    # Add derived features
    cylinder_df['pressure_diff'] = cylinder_df['pressure_extend'] - cylinder_df['pressure_retract']
    cylinder_df['load_ratio'] = cylinder_df['force'] / (cylinder_df['pressure_extend'] * 0.0314)  # Approx area
    
    # Add metadata
    cylinder_df['component_type'] = 'cylinder'
    cylinder_df['component_id'] = cylinder_name
    cylinder_df['timestamp'] = excavator_df['timestamp']
    cylinder_df['time_s'] = excavator_df['time_s']
    
    # Add labels
    cylinder_df['fault_pressure'] = excavator_df['fault_pressure_surge']
    cylinder_df['fault_overload'] = excavator_df['fault_overload']
    cylinder_df['fault_any'] = excavator_df['fault_any']
    
    return cylinder_df


def extract_pump_data(excavator_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract pump features
    
    Returns:
        DataFrame with pump features
    """
    
    pump_df = pd.DataFrame()
    pump_df['pressure_outlet'] = excavator_df['pressure_pump']
    pump_df['speed_rpm'] = excavator_df['pump_speed_rpm']
    pump_df['temperature'] = excavator_df['temp_pump']
    pump_df['vibration'] = excavator_df['vib_pump']
    
    # Add derived features
    pump_df['power'] = pump_df['pressure_outlet'] * pump_df['speed_rpm'] / 600  # Approx power (bar * rpm / 600)
    
    # Metadata
    pump_df['component_type'] = 'pump'
    pump_df['component_id'] = 'main_pump'
    pump_df['timestamp'] = excavator_df['timestamp']
    pump_df['time_s'] = excavator_df['time_s']
    
    # Labels
    pump_df['fault_pressure'] = excavator_df['fault_pressure_surge']
    pump_df['fault_overheat'] = excavator_df['fault_overheat']
    pump_df['fault_any'] = excavator_df['fault_any']
    
    return pump_df


def main():
    """Extract all component data from BIM logs"""
    
    print("🔍 Extracting component data from BIM excavator simulator...")
    
    # Path to BIM logs
    bim_logs_path = Path("../../../hydraulic_excavator_sim/logs/")
    
    if not bim_logs_path.exists():
        print(f"❌ BIM logs not found at {bim_logs_path}")
        print("   Please run excavator simulator first!")
        return
    
    # Storage
    all_cylinder_data = []
    all_pump_data = []
    
    # Process each log file
    csv_files = list(bim_logs_path.glob("*.csv"))
    print(f"📂 Found {len(csv_files)} log files")
    
    for csv_file in csv_files:
        print(f"   Processing: {csv_file.name}")
        
        df = pd.read_csv(csv_file)
        
        # Extract cylinders
        for cylinder in ['boom', 'stick', 'bucket']:
            cyl_data = extract_cylinder_data(df, cylinder)
            all_cylinder_data.append(cyl_data)
        
        # Extract pump
        pump_data = extract_pump_data(df)
        all_pump_data.append(pump_data)
    
    # Combine all data
    print("\n🔧 Combining datasets...")
    cylinder_dataset = pd.concat(all_cylinder_data, ignore_index=True)
    pump_dataset = pd.concat(all_pump_data, ignore_index=True)
    
    # Save to data/
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    cylinder_path = data_dir / "component_cylinder.csv"
    pump_path = data_dir / "component_pump.csv"
    
    cylinder_dataset.to_csv(cylinder_path, index=False)
    pump_dataset.to_csv(pump_path, index=False)
    
    # Statistics
    print("\n✅ Extraction complete!")
    print(f"\n📊 Cylinder Dataset:")
    print(f"   Samples: {len(cylinder_dataset):,}")
    print(f"   Features: {len(cylinder_dataset.columns)}")
    print(f"   Faults: {cylinder_dataset['fault_any'].sum()} ({cylinder_dataset['fault_any'].mean()*100:.2f}%)")
    print(f"   Saved to: {cylinder_path}")
    
    print(f"\n📊 Pump Dataset:")
    print(f"   Samples: {len(pump_dataset):,}")
    print(f"   Features: {len(pump_dataset.columns)}")
    print(f"   Faults: {pump_dataset['fault_any'].sum()} ({pump_dataset['fault_any'].mean()*100:.2f}%)")
    print(f"   Saved to: {pump_path}")
    
    # Show sample
    print("\n📋 Sample cylinder data:")
    print(cylinder_dataset[['pressure_extend', 'position', 'velocity', 'fault_any']].head())


if __name__ == "__main__":
    main()
