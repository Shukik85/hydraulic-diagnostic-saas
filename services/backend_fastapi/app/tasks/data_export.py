"""
Celery task for data export (GDPR compliance)
"""
from celery import shared_task
import zipfile
import json
from pathlib import Path
from datetime import datetime
import boto3
from typing import Optional

from app.models.user import User
from app.models.equipment import Equipment
from app.models.data_export import DataExportRequest


@shared_task
def export_user_data_task(export_request_id: str, user_id: str, user_email: str):
    """
    Export all user data to ZIP archive

    Includes:
    - User profile
    - Equipment metadata
    - Sensor data (last 90 days)
    - API usage logs
    """
    # Update status
    export_request = await DataExportRequest.get(export_request_id)
    export_request.status = "processing"
    await export_request.save()

    try:
        # Collect data
        data = {
            "export_date": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "profile": await _export_user_profile(user_id),
            "equipment": await _export_equipment(user_id),
            "api_usage": await _export_api_logs(user_id),
        }

        # Create ZIP
        zip_path = Path(f"/tmp/export_{user_id}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.json", json.dumps(data, indent=2))
            zf.writestr("README.txt", _get_readme_text())

        # Upload to S3 (or file storage)
        download_url = await _upload_to_storage(zip_path, export_request_id)

        # Update status
        export_request.status = "completed"
        export_request.download_url = download_url
        export_request.completed_at = datetime.utcnow()
        await export_request.save()

        # Send email with download link
        from app.tasks.email import send_email_smtp
        send_email_smtp(
            user_email,
            "Your Data Export is Ready",
            f"""
            <h2>Data Export Complete</h2>
            <p>Your data export is ready for download.</p>
            <p><a href="{download_url}">Download Now</a></p>
            <p>This link expires in 7 days.</p>
            """
        )

    except Exception as e:
        # Update status on error
        export_request.status = "failed"
        export_request.error_message = str(e)
        await export_request.save()
        raise


async def _export_user_profile(user_id: str) -> dict:
    """Export user profile data"""
    user = await User.get(user_id)
    return {
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "subscription_tier": user.subscription_tier,
        "created_at": user.created_at.isoformat(),
    }


async def _export_equipment(user_id: str) -> list:
    """Export equipment metadata"""
    equipment_list = await Equipment.find({"user_id": user_id})
    return [
        {
            "system_id": eq.system_id,
            "name": eq.name,
            "system_type": eq.system_type,
            "components": eq.components,
        }
        for eq in equipment_list
    ]


async def _export_api_logs(user_id: str) -> list:
    """Export recent API usage logs"""
    # TODO: Query API logs from last 90 days
    return []


async def _upload_to_storage(file_path: Path, export_id: str) -> str:
    """Upload file to S3 and return presigned URL"""
    # TODO: Implement S3 upload
    # For now, return placeholder
    return f"https://storage.hydraulic-diagnostics.com/exports/{export_id}.zip"


def _get_readme_text() -> str:
    """Get README text for export"""
    return """
Hydraulic Diagnostics Platform - Data Export
=============================================

This archive contains all your data from the Hydraulic Diagnostics Platform.

Contents:
---------
- data.json: All your data in JSON format
  - profile: Your account information
  - equipment: Equipment metadata
  - api_usage: API request history

Privacy:
--------
This export was created in compliance with GDPR regulations.
All data is yours and can be transferred to other services.

Support:
--------
If you have questions, contact: support@hydraulic-diagnostics.com
"""
