"""
Equipment and Component models
Dynamic topology support for any hydraulic system
"""
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Integer, Text
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from ..db.session import Base


class Equipment(Base):
    """
    Equipment metadata (press, excavator, crane, custom)
    Stores topology, components, and configuration
    """
    __tablename__ = "equipment"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Identification
    system_id = Column(String(255), nullable=False, index=True)  # user-defined ID
    system_type = Column(String(100), nullable=False)  # press, excavator, crane, custom
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Topology
    adjacency_matrix = Column(JSON, nullable=False)  # [[0,1,0], [1,0,1], ...]]

    # Metadata
    location = Column(String(255), nullable=True)
    manufacturer = Column(String(255), nullable=True)
    model = Column(String(255), nullable=True)
    serial_number = Column(String(255), nullable=True)

    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    components = relationship("Component", back_populates="equipment", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Equipment {self.system_id} ({self.system_type})>"


class Component(Base):
    """
    Individual component within equipment (pump, valve, cylinder, etc.)
    """
    __tablename__ = "components"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equipment_id = Column(UUID(as_uuid=True), ForeignKey("equipment.id"), nullable=False, index=True)

    # Identification
    component_id = Column(String(100), nullable=False)  # pump, valve_main, cylinder_1
    component_type = Column(String(100), nullable=False)  # pump, valve, cylinder, motor
    name = Column(String(255), nullable=False)

    # Sensors
    sensors = Column(ARRAY(String), nullable=False)  # [pressure, flow, temperature]
    normal_ranges = Column(JSON, nullable=False)  # {pressure: {min: 10, max: 250}}

    # Connections
    connected_to = Column(ARRAY(String), nullable=True)  # [valve_main, cylinder_1]

    # Metadata
    position = Column(Integer, nullable=True)  # визуальная позиция в UI
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    equipment = relationship("Equipment", back_populates="components")

    def __repr__(self):
        return f"<Component {self.component_id} ({self.component_type})>"
