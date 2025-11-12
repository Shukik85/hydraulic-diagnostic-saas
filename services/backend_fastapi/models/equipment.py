import uuid
from datetime import datetime

from db.session import Base
from sqlalchemy import JSON, UUID, Boolean, Column, DateTime, String


class Equipment(Base):
    __tablename__ = "equipment"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    system_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(255))
    system_type = Column(String(100), nullable=False)
    adjacency_matrix = Column(JSON)
    components = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
