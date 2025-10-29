# Создать скрипт update.sh
cat > update.sh << 'EOF'
#!/bin/bash
echo "🔄 Updating backend without container rebuild..."

# Copy updated files
docker cp ./.env $(docker ps --filter name=backend --format "{{.Names}}"):/app/.env
docker cp ./backend/requirements.txt $(docker ps --filter name=backend --format "{{.Names}}"):/app/backend/requirements.txt

# Install dependencies
echo "📦 Installing new dependencies..."
docker exec $(docker ps --filter name=backend --format "{{.Names}}") pip install -r backend/requirements.txt

# Restart backend service
echo "🔄 Restarting backend service..."
docker compose -f docker-compose.dev.yml restart backend

# Apply migrations
echo "🗃️ Applying migrations..."
make migrate

# Health check
echo "🏥 Running health check..."
docker exec $(docker ps --filter name=backend --format "{{.Names}}") python backend/manage.py health_check

echo "✅ Update completed!"
EOF

chmod +x update.sh
./update.sh
