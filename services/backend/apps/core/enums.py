"""Shared enums for the application.

Using StrEnum (Python 3.11+) for better type safety and auto() for values.
These enums eliminate duplication across models and provide centralized definitions.
"""

from enum import StrEnum, auto


class SubscriptionTier(StrEnum):
    """Subscription tier levels."""

    FREE = auto()
    PRO = auto()
    ENTERPRISE = auto()

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        match self:
            case SubscriptionTier.FREE:
                return "Free - 100 API calls/month"
            case SubscriptionTier.PRO:
                return "Pro - 10,000 API calls/month - $29/month"
            case SubscriptionTier.ENTERPRISE:
                return "Enterprise - Unlimited - Custom pricing"

    @property
    def api_limit(self) -> int | None:
        """API call limit per month. None means unlimited."""
        match self:
            case SubscriptionTier.FREE:
                return 100
            case SubscriptionTier.PRO:
                return 10_000
            case SubscriptionTier.ENTERPRISE:
                return None


class SubscriptionStatus(StrEnum):
    """Subscription status."""

    ACTIVE = auto()
    TRIAL = auto()
    PAST_DUE = auto()
    CANCELLED = auto()
    EXPIRED = auto()

    @property
    def is_active(self) -> bool:
        """Check if subscription allows access."""
        return self in {SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL}

    @property
    def display_color(self) -> str:
        """Color for admin UI badges."""
        match self:
            case SubscriptionStatus.ACTIVE:
                return "green"
            case SubscriptionStatus.TRIAL:
                return "orange"
            case SubscriptionStatus.PAST_DUE:
                return "yellow"
            case SubscriptionStatus.CANCELLED:
                return "red"
            case SubscriptionStatus.EXPIRED:
                return "gray"


class PaymentStatus(StrEnum):
    """Payment transaction status."""

    SUCCEEDED = auto()
    PENDING = auto()
    FAILED = auto()
    REFUNDED = auto()


class ErrorSeverity(StrEnum):
    """Error log severity levels."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class NotificationType(StrEnum):
    """Notification types."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    SUCCESS = auto()


class EmailCampaignStatus(StrEnum):
    """Email campaign status."""

    DRAFT = auto()
    SCHEDULED = auto()
    SENDING = auto()
    SENT = auto()
    FAILED = auto()
