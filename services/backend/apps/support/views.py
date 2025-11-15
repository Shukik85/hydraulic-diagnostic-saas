"""
Support action views
"""
from datetime import timedelta

from django.contrib.admin.views.decorators import staff_member_required
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_POST

from apps.users.models import User

from .models import SupportAction


@staff_member_required
@require_POST
def reset_password(request, user_id):
    """Reset user password (admin only)"""
    try:
        user = User.objects.get(id=user_id)
        # Generate temporary password
        temp_password = User.objects.make_random_password(length=12)
        user.set_password(temp_password)
        user.save()

        # Log action
        SupportAction.objects.create(
            user_id=user_id,
            action_type='password_reset',
            description=f"Password reset by {request.user.email}",
            performed_by=request.user.email,
        )

        return JsonResponse({
            'status': 'success',
            'temporary_password': temp_password,
            'message': 'Password reset successfully'
        })
    except User.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'User not found'}, status=404)


@staff_member_required
@require_POST
def extend_trial(request, user_id):
    """Extend user trial period"""
    try:
        user = User.objects.get(id=user_id)
        days = int(request.POST.get('days', 7))

        if user.trial_end_date:
            user.trial_end_date += timedelta(days=days)
        else:
            user.trial_end_date = timezone.now() + timedelta(days=days)

        user.save()

        # Log action
        SupportAction.objects.create(
            user_id=user_id,
            action_type='trial_extension',
            description=f"Trial extended by {days} days by {request.user.email}",
            performed_by=request.user.email,
        )

        return JsonResponse({
            'status': 'success',
            'new_trial_end_date': user.trial_end_date.isoformat(),
            'message': f'Trial extended by {days} days'
        })
    except User.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'User not found'}, status=404)
