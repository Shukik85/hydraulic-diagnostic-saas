"""
Sensor Mapping Model
Maps sensors to graph components for GNN inference
"""

import uuid
from datetime import datetime

from db.session import Base
from sqlalchemy import (
    JSON,
    UUID,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import relationship


class SensorMapping(Base):
    """
    Maps physical sensors to graph components
    Critical for GNN feature construction
    """

    __tablename__ = "sensor_mappings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equipment_id = Column(
        UUID(as_uuid=True), ForeignKey("equipment.id"), nullable=False, index=True
    )

    # Graph component reference
    component_index = Column(Integer, nullable=False)  # Index in adjacency_matrix
    component_name = Column(String(255), nullable=False)  # "pump_main", "valve_1"
    component_type = Column(String(100))  # "pump", "cylinder", "valve"

    # Sensor identification
    sensor_id = Column(String(255), nullable=False, unique=True, index=True)
    sensor_type = Column(
        String(100), nullable=False
    )  # "pressure", "temperature", "vibration", "flow"

    # Expected ranges (for anomaly detection)
    expected_range_min = Column(Float)
    expected_range_max = Column(Float)
    unit = Column(String(50), nullable=False)  # "bar", "°C", "L/min"

    # GNN feature configuration
    node_feature_index = Column(Integer)  # Position in feature vector
    feature_transform = Column(
        String(50), default="normalized"
    )  # "raw" | "normalized" | "log" | "standardized"

    # Metadata
    sampling_rate = Column(Integer)  # Hz (optional)
    is_critical = Column(
        Boolean, default=False
    )  # Critical sensor (affects health score)
    description = Column(String(500))

    # Auto-detection metadata
    auto_detected = Column(Boolean, default=False)
    confidence_score = Column(Float)  # 0.0-1.0 (if auto-detected)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    equipment = relationship("Equipment", back_populates="sensor_mappings")

    def __repr__(self):
        return f"<SensorMapping {self.sensor_id} → {self.component_name}>"

    @property
    def is_within_range(self, value: float) -> bool:
        """Check if value is within expected range"""
        if self.expected_range_min is None or self.expected_range_max is None:
            return True
        return self.expected_range_min <= value <= self.expected_range_max


class DataSource(Base):
    """
    Data source configuration (CSV, IoT Gateway, API, Simulator)
    """

    __tablename__ = "data_sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equipment_id = Column(
        UUID(as_uuid=True), ForeignKey("equipment.id"), nullable=False
    )

    source_type = Column(
        String(50), nullable=False
    )  # "csv_upload", "iot_gateway", "api_polling", "simulator"
    source_name = Column(String(255))

    # Configuration (JSON)
    config = Column(JSON)  # Type-specific config

    # Status
    is_active = Column(Boolean, default=True)
    last_sync = Column(DateTime)
    total_readings = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    equipment = relationship("Equipment")


# Update Equipment model to include relationships
