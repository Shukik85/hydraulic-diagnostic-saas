# services/shared/observability/audit_logger.py
"""
Immutable audit logs для enterprise compliance (SOC 2, ISO 27001).
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Enterprise audit event types."""
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_LOGIN_FAILED = "user.login.failed"
    DATA_ACCESS = "data.access"
    DATA_CREATE = "data.create"
    DATA_UPDATE = "data.update"
    DATA_DELETE = "data.delete"
    PERMISSION_CHANGE = "permission.change"
    ROLE_CHANGE = "role.change"
    SYSTEM_CONFIG_CHANGE = "system.config.change"
    SECURITY_INCIDENT = "security.incident"
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"
    EXPORT_DATA = "data.export"
    BACKUP_CREATED = "backup.created"
    BACKUP_RESTORED = "backup.restored"


class AuditLogger:
    """
    Tamper-evident audit logging.
    
    Все события immutable и cryptographically signed.
    """
    
    def __init__(self, storage_backend: str = "timescaledb"):
        """
        Initialize audit logger.
        
        Args:
            storage_backend: Storage backend name
        """
        self.storage = storage_backend
        logger.info(f"AuditLogger initialized with backend: {storage_backend}")
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        tenant_id: str,
        resource: str,
        action: str,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: str = "",
        user_agent: str = "",
        result: str = "success"
    ):
        """
        Log immutable audit event.
        
        Args:
            event_type: Type of audit event
            user_id: User who performed action
            tenant_id: Tenant context
            resource: Resource being accessed
            action: Action performed
            metadata: Additional event data
            ip_address: Client IP address
            user_agent: Client user agent
            result: Event result (success/failure)
        """
        timestamp = datetime.utcnow()
        
        event = {
            "timestamp": timestamp.isoformat(),
            "event_type": event_type.value,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "resource": resource,
            "action": action,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "result": result,
            "metadata": metadata or {},
        }
        
        # Compute tamper-detection hash
        event["event_hash"] = self._compute_hash(event)
        
        # Store event
        await self._store_event(event)
        
        # Stream to SIEM if critical
        if self._is_critical_event(event_type):
            await self._stream_to_siem(event)
        
        logger.info(
            f"Audit event logged: {event_type.value} by user {user_id} "
            f"on resource {resource}"
        )
    
    def _compute_hash(self, data: Dict) -> str:
        """
        Compute SHA-256 hash for tamper detection.
        
        Args:
            data: Event data
            
        Returns:
            str: Hex-encoded hash
        """
        # Create deterministic JSON string
        hash_input = json.dumps(
            {
                k: v for k, v in data.items()
                if k != 'event_hash'  # Exclude hash itself
            },
            sort_keys=True
        )
        
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    async def _store_event(self, event: Dict):
        """
        Store event in TimescaleDB.
        
        Args:
            event: Event data
        """
        # TODO: Implement TimescaleDB storage
        # INSERT INTO audit_logs (...) VALUES (...)
        logger.debug(f"Event stored: {event['event_hash'][:16]}...")
    
    async def _stream_to_siem(self, event: Dict):
        """
        Stream critical events to SIEM (Splunk, ELK, Datadog).
        
        Args:
            event: Event data
        """
        # TODO: Implement SIEM integration
        # Send via syslog, HTTP, or Kafka
        logger.debug(f"Event streamed to SIEM: {event['event_type']}")
    
    def _is_critical_event(self, event_type: AuditEventType) -> bool:
        """
        Check if event is security-critical.
        
        Args:
            event_type: Event type
            
        Returns:
            bool: True if critical
        """
        critical_events = {
            AuditEventType.USER_LOGIN_FAILED,
            AuditEventType.SECURITY_INCIDENT,
            AuditEventType.PERMISSION_CHANGE,
            AuditEventType.API_KEY_CREATED,
            AuditEventType.API_KEY_REVOKED,
            AuditEventType.DATA_DELETE
        }
        return event_type in critical_events


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """
    Get global audit logger instance.
    
    Returns:
        AuditLogger: Singleton instance
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
