#!/bin/bash
set -e

# =============================================================================
# Quick Test Script for Stage 0 - Base Environment & Observability
# =============================================================================
# This script validates that all Stage 0 components are working correctly

echo "🚀 Testing Hydraulic Diagnostic SaaS - Stage 0"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed"
    exit 1
fi

log_success "Prerequisites OK"

# Check .env file
log_info "Checking .env file..."
if [ ! -f ".env" ]; then
    log_info "Creating .env from .env.example..."
    cp .env.example .env
    log_success ".env file created"
else
    log_success ".env file exists"
fi

# Start services
log_info "Starting services with Docker Compose..."
docker compose down --remove-orphans || true
docker compose up --build -d

# Wait for services to be ready
log_info "Waiting for services to start..."
sleep 30

# Test database connectivity
log_info "Testing database connectivity..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker compose exec -T db pg_isready -U hdx_user -d hydraulic_diagnostics; then
        log_success "Database is ready"
        break
    fi
    attempt=$((attempt + 1))
    echo -n "."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    log_error "Database failed to start"
    docker compose logs db
    exit 1
fi

# Test Redis connectivity
log_info "Testing Redis connectivity..."
if docker compose exec -T redis redis-cli ping | grep -q PONG; then
    log_success "Redis is ready"
else
    log_error "Redis failed to start"
    docker compose logs redis
    exit 1
fi

# Test backend health
log_info "Testing backend health..."
max_attempts=20
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s -f http://localhost:8000/health/ > /dev/null; then
        log_success "Backend health check passed"
        break
    fi
    attempt=$((attempt + 1))
    echo -n "."
    sleep 3
done

if [ $attempt -eq $max_attempts ]; then
    log_error "Backend health check failed"
    docker compose logs backend
    exit 1
fi

# Test API endpoints
log_info "Testing API endpoints..."

# Health check
health_response=$(curl -s http://localhost:8000/health/)
if echo "$health_response" | grep -q '"status":"healthy"'; then
    log_success "Health endpoint working"
else
    log_error "Health endpoint failed"
    echo "Response: $health_response"
fi

# Readiness check
readiness_response=$(curl -s http://localhost:8000/readiness/)
if echo "$readiness_response" | grep -q '"status":"ready"'; then
    log_success "Readiness endpoint working"
else
    log_error "Readiness endpoint failed"
    echo "Response: $readiness_response"
fi

# API docs
if curl -s -f http://localhost:8000/api/docs/ > /dev/null; then
    log_success "API documentation accessible"
else
    log_error "API documentation failed"
fi

# Test admin access
if curl -s -f http://localhost:8000/admin/ > /dev/null; then
    log_success "Admin panel accessible"
else
    log_error "Admin panel failed"
fi

# Test smoke diagnostics
log_info "Running smoke tests..."
if docker compose exec -T backend python smoke_diagnostics.py; then
    log_success "Smoke tests passed"
else
    log_error "Smoke tests failed"
    exit 1
fi

# Show running services
echo ""
log_info "Service Status:"
docker compose ps

# Show service logs (last 10 lines)
echo ""
log_info "Recent Backend Logs:"
docker compose logs --tail=10 backend

echo ""
log_success "🎉 Stage 0 Test Completed Successfully!"
echo "================================================"
echo "🌐 Services available at:"
echo "   - Backend API: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/health/"
echo "   - API Docs: http://localhost:8000/api/docs/"
echo "   - Admin Panel: http://localhost:8000/admin/ (admin/admin123)"
echo ""
echo "📊 To monitor services:"
echo "   docker compose logs -f"
echo ""
echo "🛑 To stop services:"
echo "   docker compose down"
echo ""
echo "✅ Project is ready for Stage 1 development!"
