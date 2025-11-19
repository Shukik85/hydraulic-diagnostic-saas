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

    @classmethod
    def choices(cls):
        return [(tier.value, tier.description) for tier in cls]


class SubscriptionStatus(StrEnum):
    ACTIVE = auto()
    TRIAL = auto()
    PAST_DUE = auto()
    CANCELLED = auto()
    EXPIRED = auto()

    @classmethod
    def choices(cls):
        return [(status.value, status.name.title()) for status in cls]


class PaymentStatus(StrEnum):
    PENDING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    REFUNDED = auto()

    @classmethod
    def choices(cls):
        return [(status.value, status.name.title()) for status in cls]


class NotificationType(StrEnum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    SUCCESS = auto()

    @classmethod
    def choices(cls):
        return [(type.value, type.name.title()) for type in cls]


class EmailCampaignStatus(StrEnum):
    DRAFT = auto()
    SCHEDULED = auto()
    SENT = auto()
    FAILED = auto()

    @classmethod
    def choices(cls):
        return [(status.value, status.name.title()) for status in cls]


class ErrorSeverity(StrEnum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

    @classmethod
    def choices(cls):
        return [(severity.value, severity.name.title()) for severity in cls]


class SupportTicketStatus(StrEnum):
    OPEN = auto()
    IN_PROGRESS = auto()
    RESOLVED = auto()
    CLOSED = auto()

    @classmethod
    def choices(cls):
        return [(status.value, status.name.replace("_", " ").title()) for status in cls]


class SupportPriority(StrEnum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    URGENT = auto()

    @classmethod
    def choices(cls):
        return [(priority.value, priority.name.title()) for priority in cls]


class DataExportStatus(StrEnum):
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()

    @classmethod
    def choices(cls):
        return [(status.value, status.name.title()) for status in cls]


class SupportActionType(StrEnum):
    COMMENT = auto()
    STATUS_CHANGE = auto()
    ASSIGNMENT = auto()
    PRIORITY_CHANGE = auto()

    @classmethod
    def choices(cls):
        return [(action.value, action.name.replace("_", " ").title()) for action in cls]
