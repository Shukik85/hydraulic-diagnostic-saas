"""Support action views for admin quick actions."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from django.contrib.admin.views.decorators import staff_member_required
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_POST

from apps.users.models import User

if TYPE_CHECKING:
    from django.http import HttpRequest


@staff_member_required
@require_POST
def reset_password(request: HttpRequest, user_id: str) -> JsonResponse:  # noqa: ARG001
    """Reset user password (admin only).

    Args:
        request: HTTP request (unused, required by decorator)
        user_id: UUID of the user

    Returns:
        JsonResponse with status and temporary password
    """
    try:
        user = User.objects.get(id=user_id)
        # Generate temporary password
        temp_password = User.objects.make_random_password(length=12)
        user.set_password(temp_password)
        user.save()

        return JsonResponse(
            {
                "status": "success",
                "temporary_password": temp_password,
                "message": "Password reset successfully",
            }
        )
    except User.DoesNotExist:
        return JsonResponse({"status": "error", "message": "User not found"}, status=404)


@staff_member_required
@require_POST
def extend_trial(request: HttpRequest, user_id: str) -> JsonResponse:
    """Extend user trial period.

    Args:
        request: HTTP request with 'days' parameter
        user_id: UUID of the user

    Returns:
        JsonResponse with status and new trial end date
    """
    try:
        user = User.objects.get(id=user_id)
        days = int(request.POST.get("days", 7))

        if user.trial_end_date:
            user.trial_end_date += timedelta(days=days)
        else:
            user.trial_end_date = timezone.now() + timedelta(days=days)

        user.save()

        return JsonResponse(
            {
                "status": "success",
                "new_trial_end_date": user.trial_end_date.isoformat(),
                "message": f"Trial extended by {days} days",
            }
        )
    except User.DoesNotExist:
        return JsonResponse({"status": "error", "message": "User not found"}, status=404)
