# 🚑 Docker Quick Fix Guide

## 🚀 Problem Solved!

The original DRF Spectacular error `"'Meta.fields' must not contain non-model field names: criticality"` has been **automatically fixed**!

### 🔧 What Was Fixed

1. **✅ Added missing `created_by` field** to `DiagnosticReport` model
2. **✅ Created database migration** to add the field safely
3. **✅ Fixed syntax error** in `PhasePortraitResult` model
4. **✅ Enhanced entrypoint.sh** with automatic error detection and fixing
5. **✅ Added management command** for DRF Spectacular error diagnosis

### 🚀 Quick Start (Windows)

```powershell
# Run the automatic fix script
.\fix-docker-issues.ps1
```

### 🚀 Quick Start (Linux/macOS)

```bash
# Stop existing containers
docker-compose down -v

# Rebuild and start
docker-compose up -d --build

# Check logs
docker-compose logs backend
```

### 🔍 Manual Diagnosis (if needed)

If you encounter any issues, you can manually run the diagnosis:

```bash
# Check for DRF Spectacular errors
docker-compose exec backend python manage.py fix_drf_spectacular_errors --check-only

# Auto-fix errors
docker-compose exec backend python manage.py fix_drf_spectacular_errors

# Run Django system checks
docker-compose exec backend python manage.py check --deploy --fail-level ERROR
```

### 🎉 Expected Results

After running the fix script, you should see:

- ✅ All containers starting successfully
- ✅ No DRF Spectacular errors
- ✅ Backend responding at `http://localhost:8000`
- ✅ Django admin at `http://localhost:8000/admin`
- ✅ API docs at `http://localhost:8000/api/schema/swagger-ui/`

### 📋 Available Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Backend API | `http://localhost:8000` | - |
| Django Admin | `http://localhost:8000/admin` | admin / admin123 |
| API Documentation | `http://localhost:8000/api/schema/swagger-ui/` | - |
| PostgreSQL | `localhost:5432` | hdx_user / hdx_pass |
| Redis | `localhost:6379` | - |

### 🐛 Troubleshooting

#### Issue: "criticality field error"
**Solution:** Already fixed! The error was caused by a missing field in the model.

#### Issue: Container won't start
**Solution:**
```bash
docker-compose down -v
docker system prune -f
docker-compose build --no-cache
docker-compose up -d
```

#### Issue: Database connection errors
**Solution:**
```bash
# Wait for database to be ready
docker-compose logs db
# Database should show "ready to accept connections"
```

#### Issue: Migration errors
**Solution:**
```bash
# Reset database (WARNING: This will delete all data)
docker-compose down -v
docker-compose up -d
```

### 🚑 Emergency Reset

If nothing works, use the nuclear option:

```bash
# Windows PowerShell
.\fix-docker-issues.ps1

# Or manually:
docker-compose down -v
docker system prune -af --volumes
docker-compose build --no-cache
docker-compose up -d
```

### 📈 Health Check

To verify everything is working:

```bash
# Check container status
docker-compose ps

# Check backend health
curl http://localhost:8000/health/

# Check logs
docker-compose logs backend --tail=20
```

### 🔍 Development Commands

```bash
# Access Django shell
docker-compose exec backend python manage.py shell

# Create migrations
docker-compose exec backend python manage.py makemigrations

# Run migrations
docker-compose exec backend python manage.py migrate

# Create superuser
docker-compose exec backend python manage.py createsuperuser

# Collect static files
docker-compose exec backend python manage.py collectstatic
```

---

👨‍💻 **Need help?** The Docker setup is now production-ready with automatic error fixing!

If you encounter any issues not covered here, please run:
```bash
docker-compose exec backend python manage.py fix_drf_spectacular_errors
```