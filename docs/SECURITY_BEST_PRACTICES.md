# Security Best Practices: Enterprise++ Architecture

## 🔐 Overview

Этот документ описывает security best practices для production deployment.

## 🛡️ Zero-Trust Principles

### 1. Never Trust, Always Verify

- **Every request authenticated**: JWT validation on each request
- **Device fingerprinting**: Track device changes
- **IP whitelisting**: Optional per-user IP restrictions
- **Session timeouts**: Max 1 hour, configurable

### 2. Least Privilege Access

```python
# Example: RBAC implementation
from functools import wraps
from fastapi import HTTPException

def require_permission(permission: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = get_current_user()
            if permission not in user.permissions:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@app.post("/systems")
@require_permission("systems:write")
async def create_system(data: SystemCreate):
    # Only users with systems:write permission can access
    pass
```

### 3. Defense in Depth

**Multiple security layers**:

1. **Network**: WAF, DDoS protection, TLS
2. **Edge**: API Gateway rate limiting, JWT validation
3. **Service Mesh**: mTLS, authorization policies
4. **Application**: Input validation, RBAC
5. **Data**: Encryption at rest, row-level security

## 🔑 Authentication & Authorization

### JWT Best Practices

```python
# Generate secure JWT
import jwt
from datetime import datetime, timedelta

def generate_token(user_id: str, tenant_id: str, roles: list) -> str:
    payload = {
        'sub': user_id,
        'tenant_id': tenant_id,
        'roles': roles,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=1),
        'device_fingerprint': get_device_fingerprint(),
        'jti': str(uuid.uuid4())  # Unique token ID for revocation
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
```

### Token Revocation

```python
# Store revoked tokens in Redis
import redis

redis_client = redis.Redis(host='redis-cluster', port=6379)

async def revoke_token(token: str):
    # Extract expiry from token
    payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    exp = payload['exp']
    ttl = exp - int(datetime.utcnow().timestamp())
    
    # Store in Redis with TTL
    redis_client.setex(f"revoked:{token}", ttl, "1")

async def is_token_revoked(token: str) -> bool:
    return redis_client.exists(f"revoked:{token}") > 0
```

## 🔒 mTLS Configuration

### Istio mTLS Setup

```yaml
# Strict mTLS for all services
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: hydraulic-prod
spec:
  mtls:
    mode: STRICT
```

### Certificate Rotation

```bash
# Auto-rotate certificates every 90 days
istioctl install --set values.pilot.env.CERT_SIGNER_ROTATION_PERIOD=90d
```

## 👁️ Audit Logging

### What to Log

**Authentication Events**:
- Login success/failure
- Token refresh
- Logout
- Password changes
- MFA enrollment/disable

**Authorization Events**:
- Permission denied (403)
- Role changes
- Permission grants/revokes

**Data Events**:
- CRUD operations
- Exports
- Deletes (with data snapshot)
- Backups/restores

**Security Events**:
- Failed auth attempts (> 5)
- IP whitelist violations
- Device changes
- API key creation/revocation
- Certificate errors

### Audit Log Format

```json
{
  "timestamp": "2025-11-13T00:00:00Z",
  "event_type": "data.delete",
  "user_id": "user_123",
  "tenant_id": "tenant_456",
  "resource": "system/sys-789",
  "action": "delete",
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "result": "success",
  "metadata": {
    "system_name": "Excavator XYZ",
    "deletion_reason": "user_request"
  },
  "event_hash": "abc123def456..."
}
```

## 🚫 Rate Limiting

### Kong Rate Limiting

```yaml
plugins:
  - name: rate-limiting
    config:
      minute: 1000      # Per tenant
      hour: 50000
      policy: redis
      redis_host: redis-cluster
      fault_tolerant: true
      hide_client_headers: false
```

### Application-Level Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/diagnosis")
@limiter.limit("100/minute")  # Per IP
async def run_diagnosis(request: Request):
    # Expensive operation
    pass
```

## 🔐 Encryption

### At Rest

```bash
# TimescaleDB encryption
postgresql.conf:
  ssl = on
  ssl_cert_file = '/path/to/cert.pem'
  ssl_key_file = '/path/to/key.pem'
  
# Enable transparent data encryption (TDE)
ALTER SYSTEM SET encryption_at_rest = on;
```

### In Transit

- TLS 1.3 for external connections
- mTLS for inter-service communication
- Certificate pinning for mobile apps

### Sensitive Data

```python
from cryptography.fernet import Fernet

# Encrypt API keys before storage
cipher = Fernet(ENCRYPTION_KEY)

def encrypt_api_key(api_key: str) -> str:
    return cipher.encrypt(api_key.encode()).decode()

def decrypt_api_key(encrypted: str) -> str:
    return cipher.decrypt(encrypted.encode()).decode()
```

## 🐛 Vulnerability Management

### Automated Scanning

```bash
# Scan Docker images
trivy image ghcr.io/shukik85/gnn-service:1.0.0-cuda12.8

# Scan dependencies
safety check --file requirements-prod.txt

# SAST
bandit -r services/
```

### Update Policy

- **Critical vulnerabilities**: Patch within 24 hours
- **High vulnerabilities**: Patch within 7 days
- **Medium vulnerabilities**: Patch within 30 days
- **Low vulnerabilities**: Next release cycle

## 🚨 Incident Response

### Security Incident Checklist

1. **Detect**: Monitor alerts, audit logs
2. **Contain**: Isolate affected systems
3. **Eradicate**: Remove threat, patch vulnerabilities
4. **Recover**: Restore from backups if needed
5. **Review**: Post-mortem, update procedures

### Emergency Contacts

- Security team: security@hydraulic-diagnostics.com
- On-call engineer: PagerDuty
- Legal: legal@hydraulic-diagnostics.com

## ✅ Security Checklist

### Pre-Production

- [ ] All secrets in external secret manager (AWS Secrets Manager)
- [ ] TLS certificates configured
- [ ] mTLS enabled for all services
- [ ] Authorization policies applied
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] Vulnerability scanning passed
- [ ] Penetration testing completed
- [ ] Security training for team

### Post-Production

- [ ] Monitor security alerts 24/7
- [ ] Review audit logs weekly
- [ ] Rotate certificates quarterly
- [ ] Update dependencies monthly
- [ ] Conduct security audits quarterly
- [ ] Run penetration tests annually
- [ ] Review access controls monthly
- [ ] Test disaster recovery quarterly

## 📚 References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [SOC 2 Compliance](https://www.aicpa.org/interestareas/frc/assuranceadvisoryservices/aicpasoc2report)
