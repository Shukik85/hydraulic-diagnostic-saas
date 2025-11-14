"""
Unified enums for all status, tier, action, and kind fields in backend apps.
"""
from enum import StrEnum, auto

class SubscriptionTier(StrEnum):
    FREE = auto()
    PRO = auto()
    ENTERPRISE = auto()

    @property
    def api_limit(self) -> int | None:
        return {self.FREE: 100, self.PRO: 10000, self.ENTERPRISE: None}[self]

    @property
    def description(self) -> str:
        return {
            self.FREE: "Free - 100 API calls/month",
            self.PRO: "Pro - 10,000 API calls/month - $29/month",
            self.ENTERPRISE: "Enterprise - Unlimited - Custom pricing",
        }[self]

class SubscriptionStatus(StrEnum):
    ACTIVE = auto()
    TRIAL = auto()
    PAST_DUE = auto()
    CANCELLED = auto()
    EXPIRED = auto()

class PaymentStatus(StrEnum):
    SUCCEEDED = auto()
    PENDING = auto()
    FAILED = auto()
    REFUNDED = auto()

class EmailCampaignStatus(StrEnum):
    DRAFT = auto()
    SCHEDULED = auto()
    SENDING = auto()
    SENT = auto()
    FAILED = auto()

class NotificationType(StrEnum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    SUCCESS = auto()

class ErrorSeverity(StrEnum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class SupportTicketStatus(StrEnum):
    OPEN = auto()
    IN_PROGRESS = auto()
    RESOLVED = auto()
    CLOSED = auto()

class SupportPriority(StrEnum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()

class DataExportStatus(StrEnum):
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()

class SupportActionType(StrEnum):
    PASSWORD_RESET = auto()
    TRIAL_EXTEND = auto()
    SUBSCRIPTION_CHANGE = auto()
    ACCOUNT_UNLOCK = auto()
    DATA_RECOVERY = auto()
    OTHER = auto()
