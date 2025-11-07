# Backend Reorganization Guide

Reorganize Django backend to follow Django 5.1 best practices.

## What Changes

### Directory Structure
- `apps/users/` -> `backend/users/`
- `apps/diagnostics/` -> `backend/diagnostics/`
- `apps/sensors/` -> `backend/sensors/`
- `core/` -> `config/`

### Import Updates
```python
# Before
from apps.users.models import User
from core.settings import DEBUG

# After
from users.models import User
from config.settings import DEBUG
```

## What DOESN'T Change

**ml_service/ - COMPLETELY UNTOUCHED!**
- All ONNX optimizations intact
- All models working
- All scripts functional
- Production deployment ready

## Execution

### 1. Dry Run (Safe)
```powershell
cd scripts
.\reorganize-backend-only.ps1 -DryRun
```

### 2. Real Execution
```powershell
.\reorganize-backend-only.ps1
```

Automatic backup created in `backend_backup_TIMESTAMP/`.

### 3. Verification
```bash
cd backend
python manage.py check
python manage.py makemigrations --dry-run
python manage.py migrate
python manage.py test
```

### 4. Verify ML Service (Should Be Unchanged)
```bash
cd ml_service
make test-onnx
```

## Post-Reorganization Updates

### manage.py
```python
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
```

### wsgi.py & asgi.py
```python
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
```

### settings/base.py
```python
INSTALLED_APPS = [
    'users',         # Not 'apps.users'
    'diagnostics',   # Not 'apps.diagnostics'
    'sensors',
    'rag_assistant',
]
```

## Rollback

If issues occur:
```powershell
Remove-Item -Recurse -Force backend
Move-Item backend_backup_TIMESTAMP backend
```

## Timeline

- Dry run: 2 minutes
- Execution: 5 minutes
- Verification: 10 minutes
- Total: ~20 minutes

## Safety Guarantees

- Automatic backup before changes
- Dry-run mode available
- ML service completely untouched
- Easy rollback
- No data loss risk

---

**Status:** Safe to execute
**Risk:** Low (backend only, with backup)
**Impact on ML:** Zero (untouched)
