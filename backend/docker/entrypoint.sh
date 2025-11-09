#!/bin/bash
set -e

# ==============================================================================
# Hydraulic Diagnostic SaaS - Docker Entrypoint
# ==============================================================================
# Ensures proper Django initialization with health checks and logging

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Hydraulic Diagnostic SaaS Backend..."

# Change to backend directory
cd /app/backend

# Wait for database to be ready
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Waiting for database..."
while ! pg_isready -h "${DATABASE_HOST:-db}" -p "${DATABASE_PORT:-5432}" -U "${DATABASE_USER:-hdx_user}" -q; do
    sleep 1
    echo -n "."
done
echo ""
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Database is ready!"

# Wait for Redis to be ready
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Waiting for Redis..."
redis_host=$(echo "${REDIS_URL:-redis://redis:6379/0}" | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
while ! timeout 1 bash -c "</dev/tcp/${redis_host}/6379"; do
    sleep 1
    echo -n "."
done
echo ""
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Redis is ready!"

# Run database migrations first to fix model issues
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running database migrations..."
python manage.py migrate --noinput

# Auto-fix DRF Spectacular errors
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Checking for DRF Spectacular errors..."
if python manage.py fix_drf_spectacular_errors --check-only 2>/dev/null; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ No DRF Spectacular errors detected"
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️ DRF Spectacular errors detected, attempting auto-fix..."
    python manage.py fix_drf_spectacular_errors || {
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ❌ Auto-fix failed, continuing with warnings..."
    }
fi

# Check Django settings (but continue even if there are warnings)
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Validating Django configuration..."
if python manage.py check --deploy --fail-level ERROR 2>&1 | tee /tmp/django_check.log; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ Django configuration validation passed"
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️ Django configuration has warnings, but continuing..."
    # Show the warnings but don't fail
    cat /tmp/django_check.log || true
fi

# Collect static files (for production)
if [ "${DJANGO_ENV:-dev}" = "production" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Collecting static files..."
    python manage.py collectstatic --noinput --clear
fi

# Create superuser if in development and doesn't exist
if [ "${DEBUG:-True}" = "True" ] && [ "${DJANGO_ENV:-dev}" = "dev" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Creating development superuser if needed..."
    python manage.py shell << 'EOF'
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print('Superuser created: admin/admin123')
else:
    print('Superuser already exists')
EOF
fi

# Load sample data if requested
if [ "${LOAD_SAMPLE_DATA:-False}" = "True" ] && [ "${DEBUG:-True}" = "True" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Loading sample data..."
    # Add sample data loading command here when ready
    # python manage.py loaddata fixtures/sample_data.json
fi

# Run smoke tests if requested
if [ "${RUN_SMOKE_TESTS:-False}" = "True" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running smoke tests..."
    python tests/smoke/smoke_diagnostics.py
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Backend initialization completed!"

# Execute the main command
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting: $*"
exec "$@"
