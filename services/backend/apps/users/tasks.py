"""
User-related Celery tasks
"""

from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings


@shared_task
def send_password_reset_email(user_email, reset_link):
    """Send password reset email"""
    send_mail(
        subject="Password Reset Request",
        message=f"Click here to reset your password: {reset_link}",
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user_email],
        fail_silently=False,
    )


@shared_task
def send_trial_expiration_reminder(user_email, days_left):
    """Send trial expiration reminder"""
    send_mail(
        subject=f"Your trial expires in {days_left} days",
        message="Upgrade to Pro to continue using Hydraulic Diagnostics.",
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user_email],
        fail_silently=False,
    )
