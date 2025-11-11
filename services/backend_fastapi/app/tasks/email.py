"""
Celery tasks for email sending
"""
from celery import shared_task
from typing import Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import os

# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.sendgrid.net")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_HOST_USER", "apikey")
EMAIL_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD", "")
FROM_EMAIL = os.getenv("DEFAULT_FROM_EMAIL", "noreply@hydraulic-diagnostics.com")

TEMPLATE_DIR = Path(__file__).parent.parent / "templates" / "emails"


def load_template(template_name: str) -> str:
    """Load email template"""
    template_path = TEMPLATE_DIR / template_name
    if template_path.exists():
        return template_path.read_text()
    return ""


def send_email_smtp(to_email: str, subject: str, html_body: str):
    """Send email via SMTP"""
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email

    html_part = MIMEText(html_body, 'html')
    msg.attach(html_part)

    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)


@shared_task
def send_password_reset_email(to_email: str, user_name: str, reset_link: str):
    """Send password reset email"""
    template = load_template("password_reset.html")

    html_body = template.replace("{{user_name}}", user_name).replace("{{reset_link}}", reset_link)

    send_email_smtp(
        to_email,
        "Password Reset Request - Hydraulic Diagnostics",
        html_body
    )


@shared_task
def send_new_api_key_email(to_email: str, user_name: str, api_key: str):
    """Send new API key email"""
    template = load_template("new_api_key.html")

    html_body = template.replace("{{user_name}}", user_name).replace("{{api_key}}", api_key)

    send_email_smtp(
        to_email,
        "New API Key Generated - Hydraulic Diagnostics",
        html_body
    )


@shared_task
def send_support_ticket_notification(
    ticket_id: str,
    user_email: str,
    subject: str,
    priority: str
):
    """Notify support team about new ticket"""
    # Send to support team email
    support_email = os.getenv("SUPPORT_EMAIL", "support@hydraulic-diagnostics.com")

    html_body = f"""
    <h2>New Support Ticket</h2>
    <p><strong>Ticket ID:</strong> {ticket_id}</p>
    <p><strong>From:</strong> {user_email}</p>
    <p><strong>Subject:</strong> {subject}</p>
    <p><strong>Priority:</strong> {priority}</p>
    <p><a href="https://admin.hydraulic-diagnostics.com/admin/support/supportticket/{ticket_id}/">
        View in Admin Panel
    </a></p>
    """

    send_email_smtp(
        support_email,
        f"[{priority.upper()}] New Support Ticket: {subject}",
        html_body
    )

    # Send confirmation to user
    user_html = f"""
    <h2>Support Ticket Created</h2>
    <p>Hi,</p>
    <p>Your support ticket has been created successfully.</p>
    <p><strong>Ticket ID:</strong> {ticket_id}</p>
    <p><strong>Subject:</strong> {subject}</p>
    <p>Our support team will respond within 24 hours.</p>
    <p>— Hydraulic Diagnostics Team</p>
    """

    send_email_smtp(
        user_email,
        f"Support Ticket Created: {subject}",
        user_html
    )
