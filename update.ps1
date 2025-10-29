# PowerShell script for updating Docker containers without rebuild
# Hydraulic Diagnostic SaaS - Backend Update Script

param(
    [switch]$SkipDependencies,
    [switch]$SkipRestart,
    [switch]$Verbose
)

Write-Host "🔄 Updating backend without container rebuild..." -ForegroundColor Cyan

# Function to get backend container name
function Get-BackendContainer {
    $containers = docker ps --filter "name=backend" --format "{{.Names}}"
    if ($containers) {
        return $containers.Split("`n")[0].Trim()
    }
    throw "❌ Backend container not found. Make sure containers are running."
}

try {
    # Get backend container name
    $backendContainer = Get-BackendContainer
    Write-Host "📦 Found backend container: $backendContainer" -ForegroundColor Green

    # Copy updated files
    Write-Host "📁 Copying configuration files..." -ForegroundColor Yellow
    
    if (Test-Path ".\.env") {
        docker cp ".\.env" "${backendContainer}:/app/.env"
        Write-Host "  ✅ .env file copied" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  .env file not found, skipping..." -ForegroundColor Yellow
    }

    if (Test-Path ".\backend\requirements.txt") {
        docker cp ".\backend\requirements.txt" "${backendContainer}:/app/backend/requirements.txt"
        Write-Host "  ✅ requirements.txt copied" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  requirements.txt not found, skipping..." -ForegroundColor Yellow
    }

    # Copy gunicorn config if exists
    if (Test-Path ".\backend\gunicorn.conf.py") {
        docker cp ".\backend\gunicorn.conf.py" "${backendContainer}:/app/backend/gunicorn.conf.py"
        Write-Host "  ✅ gunicorn.conf.py copied" -ForegroundColor Green
    }

    # Install dependencies
    if (-not $SkipDependencies) {
        Write-Host "📦 Installing/updating Python dependencies..." -ForegroundColor Yellow
        $installResult = docker exec $backendContainer pip install -r backend/requirements.txt 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Dependencies installed successfully" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  Some dependencies might have issues:" -ForegroundColor Yellow
            if ($Verbose) { Write-Host $installResult -ForegroundColor Gray }
        }
    } else {
        Write-Host "⏭️  Skipping dependency installation" -ForegroundColor Gray
    }

    # Restart backend service
    if (-not $SkipRestart) {
        Write-Host "🔄 Restarting backend service..." -ForegroundColor Yellow
        docker-compose -f docker-compose.dev.yml restart backend
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Backend service restarted" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Failed to restart backend service" -ForegroundColor Red
        }
    } else {
        Write-Host "⏭️  Skipping service restart" -ForegroundColor Gray
    }

    # Wait for backend to be ready
    Write-Host "⏳ Waiting for backend to be ready..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3

    # Apply migrations
    Write-Host "🗃️ Applying database migrations..." -ForegroundColor Yellow
    try {
        & make migrate
        Write-Host "  ✅ Migrations applied" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠️  Migration issues (this might be normal for first run)" -ForegroundColor Yellow
    }

    # Health check
    Write-Host "🏥 Running health check..." -ForegroundColor Yellow
    $healthResult = docker exec $backendContainer python backend/manage.py health_check --json 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Health check passed" -ForegroundColor Green
        if ($Verbose) {
            Write-Host "Health Check Results:" -ForegroundColor Cyan
            Write-Host $healthResult -ForegroundColor Gray
        }
    } else {
        Write-Host "  ⚠️  Health check had some issues:" -ForegroundColor Yellow
        Write-Host $healthResult -ForegroundColor Gray
    }

    Write-Host ""
    Write-Host "🎉 Update completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Next steps:" -ForegroundColor Cyan
    Write-Host "  • Check logs: make logs" -ForegroundColor White
    Write-Host "  • Initialize data: make init-data" -ForegroundColor White
    Write-Host "  • Create superuser: make superuser" -ForegroundColor White
    Write-Host "  • Run tests: make test" -ForegroundColor White
    Write-Host "  • Swagger UI: http://localhost:8000/api/schema/swagger-ui/" -ForegroundColor White
    Write-Host ""

} catch {
    Write-Host ""
    Write-Host "❌ Update failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔍 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  • Check if containers are running: docker ps" -ForegroundColor White
    Write-Host "  • Check logs: make logs" -ForegroundColor White
    Write-Host "  • Restart all services: make down && make dev" -ForegroundColor White
    exit 1
}

# Optional: Show container status
Write-Host "📊 Container Status:" -ForegroundColor Cyan
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" --filter "name=hydraulic"