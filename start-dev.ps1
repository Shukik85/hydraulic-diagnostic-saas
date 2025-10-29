# PowerShell script for starting Hydraulic Diagnostic SaaS
# Alternative to make commands

param(
    [switch]$Build,
    [switch]$Logs,
    [switch]$Stop
)

$envFile = ".\.env"
$composeFile = "docker-compose.dev.yml"

Write-Host "🐳 Hydraulic Diagnostic SaaS - Docker Management" -ForegroundColor Cyan

# Stop containers if requested
if ($Stop) {
    Write-Host "🛑 Stopping all containers..." -ForegroundColor Yellow
    docker-compose -f $composeFile --env-file $envFile down
    exit 0
}

# Check if .env exists, create if not
if (-not (Test-Path $envFile)) {
    Write-Host "📄 .env file not found. Creating from .env.dev.example..." -ForegroundColor Yellow
    if (Test-Path ".\.env.dev.example") {
        Copy-Item ".\.env.dev.example" $envFile
        Write-Host "✅ .env file created" -ForegroundColor Green
    } else {
        Write-Host "❌ .env.dev.example not found!" -ForegroundColor Red
        exit 1
    }
}

# Start containers
Write-Host "🚀 Starting development environment..." -ForegroundColor Green

if ($Build) {
    Write-Host "🔨 Building containers..." -ForegroundColor Yellow
    docker-compose -f $composeFile --env-file $envFile up --build -d
} else {
    docker-compose -f $composeFile --env-file $envFile up -d
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Containers started successfully!" -ForegroundColor Green
    
    # Show container status
    Write-Host "`n📊 Container Status:" -ForegroundColor Cyan
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    Write-Host "`n🌐 Available Services:" -ForegroundColor Cyan
    Write-Host "  • Backend API: http://localhost:8000" -ForegroundColor White
    Write-Host "  • Swagger UI: http://localhost:8000/api/schema/swagger-ui/" -ForegroundColor White
    Write-Host "  • Admin Panel: http://localhost:8000/admin/" -ForegroundColor White
    
    Write-Host "`n📋 Next steps:" -ForegroundColor Yellow
    Write-Host "  • Run migrations: docker-compose -f docker-compose.dev.yml exec backend python manage.py migrate" -ForegroundColor White
    Write-Host "  • Create superuser: docker-compose -f docker-compose.dev.yml exec backend python manage.py createsuperuser" -ForegroundColor White
    Write-Host "  • Initialize RAG: docker-compose -f docker-compose.dev.yml exec backend python manage.py init_rag_system" -ForegroundColor White
    Write-Host "  • Health check: docker-compose -f docker-compose.dev.yml exec backend python manage.py health_check" -ForegroundColor White
    
    if ($Logs) {
        Write-Host "`n📜 Following logs..." -ForegroundColor Yellow
        docker-compose -f $composeFile --env-file $envFile logs -f --tail=50
    }
    
} else {
    Write-Host "❌ Failed to start containers" -ForegroundColor Red
    Write-Host "`n🔍 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  • Check Docker is running" -ForegroundColor White
    Write-Host "  • Check .env file content" -ForegroundColor White
    Write-Host "  • Try: .\start-dev.ps1 -Build" -ForegroundColor White
    exit 1
}