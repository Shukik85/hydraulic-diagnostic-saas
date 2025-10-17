#!/usr/bin/env bash
set -euo pipefail

# Default settings module if not provided
export DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE:-core.settings}

# Wait for Postgres
if [[ -n "${DATABASE_URL:-}" ]]; then
  echo "Waiting for database..."
  python - <<'PY'
import os, time
import urllib.parse as up
import psycopg2

url = os.environ['DATABASE_URL']
up.uses_netloc.append("postgres")
parts = up.urlparse(url)

for i in range(30):
    try:
        conn = psycopg2.connect(
            database=parts.path[1:], user=parts.username, password=parts.password,
            host=parts.hostname, port=parts.port or 5432
        )
        conn.close()
        print("Database is ready")
        break
    except Exception as e:
        print(f"DB not ready yet: {e}")
        time.sleep(2)
else:
    raise SystemExit("Database not reachable")
PY
fi

# Apply migrations
python backend/manage.py migrate --noinput

# Collect static files
python backend/manage.py collectstatic --noinput

# Start Gunicorn
exec gunicorn core.wsgi:application \
  --chdir backend \
  --bind 0.0.0.0:8000 \
  --workers ${GUNICORN_WORKERS:-4} \
  --timeout ${GUNICORN_TIMEOUT:-30} \
  --access-logfile - \
  --error-logfile - \
  --log-level ${GUNICORN_LOG_LEVEL:-info}
