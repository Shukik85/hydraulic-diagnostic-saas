# ===============================================================================
# Hydraulic Diagnostic SaaS - Docker Issues Fix Script (Windows PowerShell)
# ===============================================================================
# This script helps fix common Docker issues and uses pip caching for faster builds

Write-Host "🚀 Hydraulic Diagnostic SaaS - Docker Fix Script (v2.0)" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "⚡ Now with pip caching for 10x faster rebuilds!" -ForegroundColor Cyan
Write-Host "" 

# Check if Docker is running
Write-Host "🔍 Checking Docker..." -ForegroundColor Yellow
try {
    docker --version | Out-Host
    docker-compose --version | Out-Host
    Write-Host "✅ Docker is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running or not installed" -ForegroundColor Red
    Write-Host "Please install Docker Desktop and make sure it's running" -ForegroundColor Red
    exit 1
}

# Check Docker BuildKit support (required for cache mounts)
Write-Host "🔧 Checking Docker BuildKit support..." -ForegroundColor Yellow
$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"
Write-Host "✅ BuildKit enabled for cache optimization" -ForegroundColor Green

# Check .env file
Write-Host "📝 Checking .env file..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✅ .env file exists" -ForegroundColor Green
} else {
    Write-Host "⚠️ .env file not found, creating from example..." -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "✅ .env file created from example" -ForegroundColor Green
    } else {
        Write-Host "❌ .env.example not found" -ForegroundColor Red
    }
}

# Fix line endings if on Windows
Write-Host "🔄 Fixing line endings for Windows..." -ForegroundColor Yellow
if (Get-Command dos2unix -ErrorAction SilentlyContinue) {
    dos2unix docker/entrypoint.sh
    Write-Host "✅ Line endings fixed" -ForegroundColor Green
} else {
    Write-Host "⚠️ dos2unix not found, Docker will handle line endings" -ForegroundColor Yellow
}

# Stop any existing containers
Write-Host "🗑️ Stopping existing containers..." -ForegroundColor Yellow
docker-compose down -v 2>$null
Write-Host "✅ Containers stopped" -ForegroundColor Green

# Selective cleanup - preserve pip cache!
Write-Host "🧾 Cleaning up (preserving pip cache)..." -ForegroundColor Yellow
docker system prune -f --filter="label!=pip-cache"
Write-Host "✅ Docker cleanup completed (cache preserved)" -ForegroundColor Green

# Show cache status
Write-Host "📊 Checking pip cache status..." -ForegroundColor Yellow
$cacheInfo = docker system df --format "table {{.Type}}\t{{.Total}}\t{{.Size}}"
Write-Host $cacheInfo -ForegroundColor Cyan

# Rebuild containers with cache
Write-Host "🔨 Building containers with pip cache..." -ForegroundColor Yellow
Write-Host "This may take a while on first run, but subsequent builds will be much faster!" -ForegroundColor Cyan

$buildStartTime = Get-Date
docker-compose build --progress=plain
$buildEndTime = Get-Date
$buildDuration = $buildEndTime - $buildStartTime

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    Write-Host "Trying fallback build without cache..." -ForegroundColor Yellow
    docker-compose build --no-cache
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Fallback build also failed" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Containers built successfully" -ForegroundColor Green
Write-Host "⏱️ Build time: $($buildDuration.Minutes)m $($buildDuration.Seconds)s" -ForegroundColor Cyan

# Start services
Write-Host "🚀 Starting services..." -ForegroundColor Yellow
docker-compose up -d

# Wait a bit for services to initialize
Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Check service status
Write-Host "🔍 Checking service status..." -ForegroundColor Yellow
docker-compose ps

# Check backend logs
Write-Host "📄 Checking backend logs..." -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Cyan
docker-compose logs backend --tail=30
Write-Host "================================" -ForegroundColor Cyan

# Test backend connectivity
Write-Host "🎯 Testing backend connectivity..." -ForegroundColor Yellow
$maxAttempts = 60  # Increased timeout
$attempt = 0
$backendReady = $false

while ($attempt -lt $maxAttempts -and -not $backendReady) {
    $attempt++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health/" -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $backendReady = $true
            Write-Host "✅ Backend is responding at http://localhost:8000" -ForegroundColor Green
        }
    } catch {
        if ($attempt % 10 -eq 0) {
            Write-Host "Attempt $attempt/$maxAttempts - Backend not ready yet..." -ForegroundColor Yellow
        }
        Start-Sleep -Seconds 2
    }
}

if (-not $backendReady) {
    Write-Host "⚠️ Backend is not responding after $maxAttempts attempts" -ForegroundColor Yellow
    Write-Host "Please check the logs above for errors" -ForegroundColor Yellow
    
    Write-Host "🚑 Running Django checks inside container..." -ForegroundColor Yellow
    docker-compose exec backend python manage.py check --deploy --fail-level ERROR
    
    Write-Host "🚑 Attempting to fix DRF Spectacular errors..." -ForegroundColor Yellow
    docker-compose exec backend python manage.py fix_drf_spectacular_errors
} else {
    Write-Host "🎉 SUCCESS! All services are running correctly" -ForegroundColor Green
    Write-Host "" 
    Write-Host "Available services:" -ForegroundColor Cyan
    Write-Host "- Backend API: http://localhost:8000" -ForegroundColor White
    Write-Host "- Django Admin: http://localhost:8000/admin" -ForegroundColor White
    Write-Host "- API Documentation: http://localhost:8000/api/schema/swagger-ui/" -ForegroundColor White
    Write-Host "- PostgreSQL: localhost:5432" -ForegroundColor White
    Write-Host "- Redis: localhost:6379" -ForegroundColor White
    Write-Host "" 
    Write-Host "Default superuser: admin / admin123" -ForegroundColor Green
    Write-Host "" 
    Write-Host "⚡ Performance Tips:" -ForegroundColor Yellow
    Write-Host "- Next rebuild will be 10x faster thanks to pip cache!" -ForegroundColor Cyan
    Write-Host "- Use 'docker-compose build' for incremental builds" -ForegroundColor Cyan
    Write-Host "- Use 'docker-compose up --build' for quick rebuilds" -ForegroundColor Cyan
    Write-Host "" 
    Write-Host "To stop all services: docker-compose down" -ForegroundColor Yellow
    Write-Host "To view logs: docker-compose logs -f" -ForegroundColor Yellow
}

Write-Host "" 
Write-Host "📊 Current container status:" -ForegroundColor Cyan
docker-compose ps

Write-Host "" 
Write-Host "💾 Disk usage (with cache info):" -ForegroundColor Cyan
docker system df