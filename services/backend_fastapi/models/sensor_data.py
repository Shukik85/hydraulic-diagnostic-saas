from sqlalchemy import Column, String, DateTime, Float, UUID, Boolean
from db.session import Base
import uuid
from datetime import datetime

class SensorData(Base):
    __tablename__ = "sensor_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    system_id = Column(String(255), nullable=False, index=True)
    sensor_id = Column(String(255), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String(50))
    is_valid = Column(Boolean, default=True)
    is_quarantined = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
