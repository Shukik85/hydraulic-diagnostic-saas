# services/shared/security/zero_trust.py
"""
Zero-Trust Security Framework с continuous authentication.
"""
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class ZeroTrustAuthenticator:
    """
    Continuous verification authenticator для enterprise security.
    
    Принципы:
    - Never trust, always verify
    - Verify every request
    - Check device fingerprint
    - Enforce IP whitelist
    - Audit all access
    """
    
    def __init__(
        self,
        secret_key: str,
        session_timeout: int = 3600,
        algorithm: str = "HS256"
    ):
        self.secret_key = secret_key
        self.session_timeout = session_timeout
        self.algorithm = algorithm
        self.revoked_tokens: set = set()
        
        logger.info("ZeroTrustAuthenticator initialized")
    
    async def verify_continuous(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials
    ) -> Dict:
        """
        Continuous authentication verification.
        
        Args:
            request: FastAPI request object
            credentials: Bearer token credentials
            
        Returns:
            dict: JWT payload with user claims
            
        Raises:
            HTTPException: If authentication fails
        """
        token = credentials.credentials
        
        # Check if token is revoked
        if token in self.revoked_tokens:
            logger.warning(f"Revoked token used: {token[:20]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        try:
            # Decode and verify JWT
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Verify session timeout
            issued_at = datetime.fromtimestamp(payload["iat"])
            if datetime.utcnow() - issued_at > timedelta(seconds=self.session_timeout):
                logger.warning(f"Session timeout for user {payload.get('sub')}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session timeout - please re-authenticate"
                )
            
            # Verify device fingerprint
            request_fingerprint = self._get_device_fingerprint(request)
            stored_fingerprint = payload.get("device_fingerprint")
            
            if stored_fingerprint and request_fingerprint != stored_fingerprint:
                logger.error(
                    f"Device mismatch for user {payload.get('sub')}: "
                    f"expected {stored_fingerprint}, got {request_fingerprint}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Device mismatch - re-authentication required"
                )
            
            # Verify IP whitelist (if configured)
            allowed_ips = payload.get("allowed_ips", [])
            if allowed_ips and not self._verify_ip_whitelist(
                request.client.host,
                allowed_ips
            ):
                logger.error(
                    f"IP not whitelisted: {request.client.host} "
                    f"for user {payload.get('sub')}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="IP address not in whitelist"
                )
            
            logger.info(f"Authentication successful for user {payload.get('sub')}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    def _get_device_fingerprint(self, request: Request) -> str:
        """
        Generate device fingerprint from request headers.
        
        Args:
            request: FastAPI request
            
        Returns:
            str: Device fingerprint hash
        """
        fingerprint_data = {
            'user_agent': request.headers.get('user-agent', ''),
            'accept_language': request.headers.get('accept-language', ''),
            'accept_encoding': request.headers.get('accept-encoding', '')
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    def _verify_ip_whitelist(self, ip: str, allowed_ips: List[str]) -> bool:
        """
        Verify IP against whitelist.
        
        Args:
            ip: Request IP address
            allowed_ips: List of allowed IPs/CIDR ranges
            
        Returns:
            bool: True if IP is allowed
        """
        if not allowed_ips:
            return True
        
        # Direct IP match
        if ip in allowed_ips:
            return True
        
        # CIDR range match
        import ipaddress
        try:
            ip_obj = ipaddress.ip_address(ip)
            for allowed in allowed_ips:
                if '/' in allowed:  # CIDR notation
                    network = ipaddress.ip_network(allowed, strict=False)
                    if ip_obj in network:
                        return True
        except ValueError:
            logger.error(f"Invalid IP address: {ip}")
            return False
        
        return False
    
    async def revoke_token(self, token: str):
        """
        Revoke token immediately (for logout/security incidents).
        
        Args:
            token: JWT token to revoke
        """
        self.revoked_tokens.add(token)
        logger.info(f"Token revoked: {token[:20]}...")
        
        # TODO: Persist to Redis for distributed revocation
    
    def generate_token(
        self,
        user_id: str,
        tenant_id: str,
        roles: List[str],
        permissions: List[str],
        request: Request,
        allowed_ips: Optional[List[str]] = None
    ) -> str:
        """
        Generate JWT token with security claims.
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            roles: User roles
            permissions: User permissions
            request: FastAPI request for fingerprinting
            allowed_ips: Optional IP whitelist
            
        Returns:
            str: Signed JWT token
        """
        now = datetime.utcnow()
        
        payload = {
            'sub': user_id,
            'tenant_id': tenant_id,
            'roles': roles,
            'permissions': permissions,
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(hours=1)).timestamp()),
            'device_fingerprint': self._get_device_fingerprint(request),
            'allowed_ips': allowed_ips or []
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Token generated for user {user_id}")
        
        return token


# Global authenticator instance
authenticator: Optional[ZeroTrustAuthenticator] = None


def get_authenticator() -> ZeroTrustAuthenticator:
    """
    Get global authenticator instance.
    
    Returns:
        ZeroTrustAuthenticator: Singleton authenticator
    """
    global authenticator
    if authenticator is None:
        import os
        secret_key = os.getenv('JWT_SECRET_KEY', 'changeme-in-production')
        authenticator = ZeroTrustAuthenticator(secret_key=secret_key)
    return authenticator
