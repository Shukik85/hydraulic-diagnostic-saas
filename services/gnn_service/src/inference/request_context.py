"""Request context tracking with X-Request-ID.

Provides:
- RequestContext: ContextVar for request ID propagation
- Helper functions for request tracking
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

# Global context variable for request ID
request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Get current request ID from context.

    Returns:
        Request ID or None if not set
    """
    return request_id_context.get()


def set_request_id(request_id: str) -> None:
    """Set request ID in context.

    Args:
        request_id: Request identifier
    """
    request_id_context.set(request_id)


def generate_request_id() -> str:
    """Generate new request ID.

    Returns:
        UUID-based request ID
    """
    return str(uuid.uuid4())


def ensure_request_id() -> str:
    """Get or generate request ID.

    Returns:
        Existing or newly generated request ID
    """
    request_id = get_request_id()
    if request_id is None:
        request_id = generate_request_id()
        set_request_id(request_id)
    return request_id
