"""
Support ticket model
"""
from sqlalchemy import Column, String, Text, DateTime, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import enum

from app.db.base_class import Base


class TicketStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SupportTicket(Base):
    """Support ticket model"""
    __tablename__ = "support_tickets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    subject = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    priority = Column(SQLEnum(TicketPriority), default=TicketPriority.MEDIUM)
    status = Column(SQLEnum(TicketStatus), default=TicketStatus.OPEN)
    category = Column(String(100), nullable=True)

    assigned_to = Column(UUID(as_uuid=True), nullable=True)
    response = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
