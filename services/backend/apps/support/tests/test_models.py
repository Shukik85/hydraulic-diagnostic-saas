"""Tests for support models."""

from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone

from apps.support.models import (
    AccessRecoveryRequest,
    SupportTicket,
    TicketMessage,
)

User = get_user_model()


@pytest.fixture
def user(db):
    """Create test user."""
    return User.objects.create_user(
        email="test@example.com",
        password="testpass123",
    )


@pytest.fixture
def staff_user(db):
    """Create staff user."""
    return User.objects.create_user(
        email="staff@example.com",
        password="testpass123",
        is_staff=True,
    )


@pytest.fixture
def ticket(user):
    """Create test ticket."""
    return SupportTicket.objects.create(
        user=user,
        category=SupportTicket.Category.TECHNICAL,
        priority=SupportTicket.Priority.MEDIUM,
        subject="Test ticket",
        description="Test description",
    )


@pytest.mark.django_db
class TestSupportTicket:
    """Tests for SupportTicket model."""

    def test_create_ticket(self, user):
        """Test ticket creation with auto-generated fields."""
        ticket = SupportTicket.objects.create(
            user=user,
            category=SupportTicket.Category.BUG,
            priority=SupportTicket.Priority.HIGH,
            subject="Bug report",
            description="Found a bug",
        )

        assert ticket.ticket_number.startswith("TKT-")
        assert ticket.status == SupportTicket.Status.NEW
        assert ticket.sla_due_date is not None
        assert not ticket.sla_breached

    def test_sla_calculation(self, user):
        """Test SLA due date calculation based on priority."""
        ticket = SupportTicket.objects.create(
            user=user,
            priority=SupportTicket.Priority.CRITICAL,
            subject="Critical issue",
            description="Urgent",
        )

        # Critical = 2 hours
        expected_due = timezone.now() + timedelta(hours=2)
        assert abs((ticket.sla_due_date - expected_due).total_seconds()) < 60

    def test_assign_to_agent(self, ticket, staff_user):
        """Test assigning ticket to staff member."""
        ticket.assign_to_agent(staff_user)

        assert ticket.assigned_to == staff_user
        assert ticket.status == SupportTicket.Status.OPEN

    def test_resolve_ticket(self, ticket):
        """Test resolving ticket."""
        ticket.resolve()

        assert ticket.status == SupportTicket.Status.RESOLVED
        assert ticket.resolved_at is not None

    def test_time_until_sla(self, ticket):
        """Test time until SLA calculation."""
        time_left = ticket.time_until_sla
        assert time_left is not None
        assert time_left.total_seconds() > 0


@pytest.mark.django_db
class TestTicketMessage:
    """Tests for TicketMessage model."""

    def test_create_message(self, ticket, user):
        """Test creating ticket message."""
        message = TicketMessage.objects.create(
            ticket=ticket,
            author=user,
            message="Test message",
        )

        assert message.ticket == ticket
        assert message.author == user
        assert not message.is_internal
        assert not message.is_system

    def test_internal_message(self, ticket, staff_user):
        """Test creating internal note."""
        message = TicketMessage.objects.create(
            ticket=ticket,
            author=staff_user,
            message="Internal note",
            is_internal=True,
        )

        assert message.is_internal


@pytest.mark.django_db
class TestAccessRecoveryRequest:
    """Tests for AccessRecoveryRequest model."""

    def test_create_request(self, user):
        """Test creating access recovery request."""
        request = AccessRecoveryRequest.objects.create(
            user=user,
            request_type=AccessRecoveryRequest.RequestType.PASSWORD_RESET,
        )

        assert request.status == AccessRecoveryRequest.Status.PENDING
        assert request.processed_by is None

    def test_approve_request(self, user, staff_user):
        """Test approving recovery request."""
        request = AccessRecoveryRequest.objects.create(
            user=user,
            request_type=AccessRecoveryRequest.RequestType.TWO_FA_RESET,
        )

        request.approve(staff_user, "Verified via email")

        assert request.status == AccessRecoveryRequest.Status.APPROVED
        assert request.processed_by == staff_user
        assert request.processed_at is not None
        assert "Approved" in request.admin_notes

    def test_reject_request(self, user, staff_user):
        """Test rejecting recovery request."""
        request = AccessRecoveryRequest.objects.create(
            user=user,
            request_type=AccessRecoveryRequest.RequestType.EMAIL_CHANGE,
        )

        request.reject(staff_user, "Insufficient verification")

        assert request.status == AccessRecoveryRequest.Status.REJECTED
        assert request.processed_by == staff_user
        assert "Rejected" in request.admin_notes
