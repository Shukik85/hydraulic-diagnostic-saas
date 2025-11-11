"""
Sensor data models for TimescaleDB hypertables
High-performance time-series storage
"""
from sqlalchemy import Column, String, Float, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from ..db.session import Base


class SensorData(Base):
    """
    Raw sensor data ingestion (before TimescaleDB conversion)
    Temporary staging table with quarantine support
    """
    __tablename__ = "sensor_data_staging"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    system_id = Column(String(255), nullable=False, index=True)
    component_id = Column(String(100), nullable=False)
    sensor_name = Column(String(100), nullable=False)

    # Data
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)

    # Timestamp
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())

    # Quality
    is_valid = Column(Boolean, default=True)
    is_quarantined = Column(Boolean, default=False)
    validation_errors = Column(JSON, nullable=True)

    __table_args__ = (
        Index('idx_sensor_lookup', 'user_id', 'system_id', 'timestamp'),
    )


class SensorDataHypertable(Base):
    """
    Production sensor data in TimescaleDB hypertable
    Compressed, partitioned by time, 5-year retention
    """
    __tablename__ = "sensor_data"

    time = Column(DateTime(timezone=True), primary_key=True)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    system_id = Column(String(255), nullable=False)
    component_id = Column(String(100), nullable=False)

    # Sensor readings (denormalized for performance)
    pressure = Column(Float, nullable=True)
    flow = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    position = Column(Float, nullable=True)
    vibration = Column(Float, nullable=True)
    current = Column(Float, nullable=True)
    voltage = Column(Float, nullable=True)

    __table_args__ = (
        Index('idx_timescale_lookup', 'user_id', 'system_id', 'time'),
    )

    # Note: TimescaleDB hypertable creation handled by migration:
    # SELECT create_hypertable('sensor_data', 'time');
    # SELECT add_compression_policy('sensor_data', INTERVAL '7 days');
    # SELECT add_retention_policy('sensor_data', INTERVAL '5 years');
