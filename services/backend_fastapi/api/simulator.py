"""
Simulator API
Level 6B Option D: Generate synthetic sensor data
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from datetime import datetime, timedelta
import asyncio
import uuid

from db.session import get_db
from models.sensor_mapping import SensorMapping, DataSource
from models.sensor_data import SensorData
from models.equipment import Equipment
from schemas.simulator import (
    SimulatorConfig,
    SimulatorStartResponse,
    SimulatorStatus
)
from middleware.auth import get_current_user

router = APIRouter(prefix="/api/simulator", tags=["Simulator"])

# Active simulations (in-memory)
active_simulations = {}


@router.post("/start", response_model=SimulatorStartResponse)
async def start_simulator(
    config: SimulatorConfig,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Start simulator for equipment
    Generates synthetic sensor data based on scenario
    """
    # Verify equipment
    equipment = await db.get(Equipment, config.equipment_id)
    if not equipment or equipment.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Equipment not found")

    # Get sensor mappings
    result = await db.execute(
        select(SensorMapping).where(SensorMapping.equipment_id == config.equipment_id)
    )
    sensor_mappings = result.scalars().all()

    if not sensor_mappings:
        raise HTTPException(
            status_code=400,
            detail="No sensors mapped. Complete Level 6A first."
        )

    # Create simulation ID
    simulation_id = uuid.uuid4()

    # Create data source
    data_source = DataSource(
        equipment_id=config.equipment_id,
        source_type='simulator',
        source_name=f'Simulator - {config.scenario}',
        config={
            'scenario': config.scenario,
            'duration': config.duration,
            'noise_level': config.noise_level,
            'sampling_rate': config.sampling_rate
        },
        is_active=True
    )
    db.add(data_source)
    await db.commit()
    await db.refresh(data_source)

    # Store simulation metadata
    active_simulations[str(simulation_id)] = {
        'equipment_id': config.equipment_id,
        'data_source_id': data_source.id,
        'status': 'running',
        'started_at': datetime.utcnow(),
        'config': config.dict()
    }

    # Start simulation in background
    background_tasks.add_task(
        run_simulation,
        simulation_id,
        config,
        sensor_mappings,
        equipment.system_id,
        db
    )

    return {
        'simulation_id': simulation_id,
        'data_source_id': data_source.id,
        'status': 'started',
        'estimated_readings': len(sensor_mappings) * config.duration * config.sampling_rate
    }


@router.get("/status/{simulation_id}", response_model=SimulatorStatus)
async def get_simulator_status(
    simulation_id: uuid.UUID
):
    """Get simulation status"""
    sim = active_simulations.get(str(simulation_id))

    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    return {
        'simulation_id': simulation_id,
        'status': sim['status'],
        'started_at': sim['started_at'],
        'config': sim['config']
    }


@router.post("/stop/{simulation_id}")
async def stop_simulator(
    simulation_id: uuid.UUID
):
    """Stop running simulation"""
    sim = active_simulations.get(str(simulation_id))

    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    sim['status'] = 'stopped'

    return {'message': 'Simulation stopped'}


async def run_simulation(
    simulation_id: uuid.UUID,
    config: SimulatorConfig,
    sensor_mappings: list,
    system_id: str,
    db: AsyncSession
):
    """
    Background task: Generate and insert sensor readings
    """
    try:
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=config.duration)
        current_time = start_time

        readings_per_second = config.sampling_rate
        interval = 1.0 / readings_per_second

        while current_time < end_time and active_simulations[str(simulation_id)]['status'] == 'running':
            # Generate reading for each sensor
            for mapping in sensor_mappings:
                value = generate_sensor_value(
                    mapping=mapping,
                    scenario=config.scenario,
                    time_elapsed=(current_time - start_time).total_seconds(),
                    noise_level=config.noise_level
                )

                reading = SensorData(
                    system_id=system_id,
                    sensor_id=mapping.sensor_id,
                    timestamp=current_time,
                    value=value,
                    unit=mapping.unit,
                    is_valid=True,
                    is_quarantined=False
                )

                db.add(reading)

            # Commit batch every second
            await db.commit()

            # Wait for next interval
            await asyncio.sleep(interval)
            current_time = datetime.utcnow()

        # Mark as completed
        active_simulations[str(simulation_id)]['status'] = 'completed'

    except Exception as e:
        active_simulations[str(simulation_id)]['status'] = 'failed'
        active_simulations[str(simulation_id)]['error'] = str(e)


def generate_sensor_value(
    mapping: SensorMapping,
    scenario: str,
    time_elapsed: float,
    noise_level: float
) -> float:
    """
    Generate synthetic sensor value based on scenario
    """
    base_value = (mapping.expected_range_min + mapping.expected_range_max) / 2
    range_span = mapping.expected_range_max - mapping.expected_range_min

    # Scenario-specific patterns
    if scenario == 'normal':
        # Stable values with small noise
        variation = np.random.normal(0, noise_level * range_span * 0.05)

    elif scenario == 'degradation':
        # Gradual drift over time
        drift = (time_elapsed / 3600) * range_span * 0.1  # 10% drift per hour
        variation = np.random.normal(drift, noise_level * range_span * 0.05)

    elif scenario == 'failure':
        # Sudden spike at random times
        if np.random.random() < 0.01:  # 1% chance of spike
            variation = range_span * 0.5 * np.random.choice([-1, 1])
        else:
            variation = np.random.normal(0, noise_level * range_span * 0.1)

    elif scenario == 'cyclic':
        # Sinusoidal pattern
        frequency = 0.1  # Hz
        amplitude = range_span * 0.2
        variation = amplitude * np.sin(2 * np.pi * frequency * time_elapsed)

    else:
        variation = 0

    # Add noise
    noise = np.random.normal(0, noise_level * range_span * 0.02)

    value = base_value + variation + noise

    # Clamp to valid range
    return np.clip(value, mapping.expected_range_min * 0.5, mapping.expected_range_max * 1.2)
