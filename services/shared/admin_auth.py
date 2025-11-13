# services/shared/admin_auth.py
"""
Admin authentication dependencies для admin-only endpoints.
"""
import os
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime

security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"


class AdminUser:
    """Admin user model."""
    def __init__(self, user_id: str, email: str, role: str):
        self.user_id = user_id
        self.email = email
        self.role = role
        self.is_admin = role in ['admin', 'superadmin']


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> AdminUser:
    """
    Validate JWT token и вернуть current user.
    
    Raises:
        HTTPException: If token invalid
    """
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        role: str = payload.get("role", "user")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        return AdminUser(user_id=user_id, email=email, role=role)
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


async def get_current_admin_user(
    user: AdminUser = Security(get_current_user)
) -> AdminUser:
    """
    Require admin role.
    
    Raises:
        HTTPException: If user is not admin
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return user


async def get_current_superadmin_user(
    user: AdminUser = Security(get_current_user)
) -> AdminUser:
    """
    Require superadmin role.
    """
    if user.role != 'superadmin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadmin access required"
        )
    
    return user
