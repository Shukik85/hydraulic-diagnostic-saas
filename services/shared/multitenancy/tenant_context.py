# services/shared/multitenancy/tenant_context.py
"""
Multi-tenancy context manager с data isolation.
"""
from typing import Optional
from contextvars import ContextVar
import logging

logger = logging.getLogger(__name__)

# Context variable для async-safe tenant tracking
_tenant_context: ContextVar[Optional[str]] = ContextVar('tenant_id', default=None)
_user_context: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


class TenantContext:
    """
    Row-level security для multi-tenancy.
    
    Использует contextvars для async-safe tracking текущего tenant.
    """
    
    @classmethod
    def set_current_tenant(cls, tenant_id: str):
        """
        Set tenant for current request context.
        
        Args:
            tenant_id: Tenant identifier
        """
        _tenant_context.set(tenant_id)
        logger.debug(f"Tenant context set: {tenant_id}")
    
    @classmethod
    def get_current_tenant(cls) -> Optional[str]:
        """
        Get current tenant ID.
        
        Returns:
            str: Current tenant ID or None
        """
        return _tenant_context.get()
    
    @classmethod
    def set_current_user(cls, user_id: str):
        """
        Set user for current request context.
        
        Args:
            user_id: User identifier
        """
        _user_context.set(user_id)
        logger.debug(f"User context set: {user_id}")
    
    @classmethod
    def get_current_user(cls) -> Optional[str]:
        """
        Get current user ID.
        
        Returns:
            str: Current user ID or None
        """
        return _user_context.get()
    
    @classmethod
    def clear(cls):
        """
        Clear tenant and user context.
        """
        _tenant_context.set(None)
        _user_context.set(None)
        logger.debug("Tenant context cleared")
    
    @classmethod
    def require_tenant(cls) -> str:
        """
        Get current tenant, raising error if not set.
        
        Returns:
            str: Current tenant ID
            
        Raises:
            ValueError: If no tenant context set
        """
        tenant_id = cls.get_current_tenant()
        if not tenant_id:
            logger.error("No tenant context set")
            raise ValueError(
                "No tenant context set. This is a security violation."
            )
        return tenant_id


class TenantQueryFilter:
    """
    Automatic tenant filtering для database queries.
    """
    
    @staticmethod
    def apply_tenant_filter(query, model):
        """
        Append WHERE tenant_id = :current_tenant to query.
        
        Args:
            query: SQLAlchemy query
            model: SQLAlchemy model
            
        Returns:
            Modified query with tenant filter
            
        Raises:
            ValueError: If no tenant context
        """
        tenant_id = TenantContext.require_tenant()
        
        if hasattr(model, 'tenant_id'):
            query = query.where(model.tenant_id == tenant_id)
            logger.debug(f"Applied tenant filter: tenant_id={tenant_id}")
        else:
            logger.warning(
                f"Model {model.__name__} doesn't have tenant_id column"
            )
        
        return query
    
    @staticmethod
    def validate_tenant_access(obj, tenant_id: Optional[str] = None):
        """
        Validate that object belongs to current tenant.
        
        Args:
            obj: Database object to validate
            tenant_id: Optional tenant ID to check against
            
        Raises:
            PermissionError: If tenant mismatch
        """
        if tenant_id is None:
            tenant_id = TenantContext.require_tenant()
        
        obj_tenant = getattr(obj, 'tenant_id', None)
        
        if obj_tenant != tenant_id:
            logger.error(
                f"Tenant mismatch: object belongs to {obj_tenant}, "
                f"but current tenant is {tenant_id}"
            )
            raise PermissionError(
                "Access denied: object belongs to different tenant"
            )
