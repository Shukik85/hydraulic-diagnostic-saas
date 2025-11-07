# ===============================================================================
# Hydraulic Diagnostic SaaS - Backend Reorganization Script (PowerShell)
# ===============================================================================
# Этот скрипт упрощает структуру backend по образцу ml_service

param(
    [switch]$DryRun,
    [switch]$Force
)

Write-Host "🚀 Backend Reorganization Script" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green
Write-Host ""

if ($DryRun) {
    Write-Host "⚠️  DRY RUN MODE - никаких изменений не будет" -ForegroundColor Yellow
    Write-Host ""
}

# Check if Python is available
try {
    python --version | Out-Null
} catch {
    Write-Host "❌ Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.7+ to run this script" -ForegroundColor Red
    exit 1
}

Write-Host "📍 Current directory: $PWD" -ForegroundColor Cyan
Write-Host ""

# Confirm action
if (-not $Force -and -not $DryRun) {
    Write-Host "⚠️  WARNING: This will reorganize backend structure!" -ForegroundColor Yellow
    Write-Host "⚠️  This includes:" -ForegroundColor Yellow
    Write-Host "    - Moving apps/* to backend root" -ForegroundColor Yellow
    Write-Host "    - Renaming core/ to config/" -ForegroundColor Yellow
    Write-Host "    - Updating all imports" -ForegroundColor Yellow
    Write-Host ""
    $confirm = Read-Host "Are you sure? Type 'yes' to continue"
    if ($confirm -ne "yes") {
        Write-Host "❌ Cancelled" -ForegroundColor Red
        exit 0
    }
    Write-Host ""
}

# Stop Docker containers first
Write-Host "🛝 Stopping Docker containers..." -ForegroundColor Yellow
try {
    docker-compose down
    Write-Host "✅ Containers stopped" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not stop containers (may not be running)" -ForegroundColor Yellow
}
Write-Host ""

# Run the Python script
Write-Host "🐍 Running Python reorganization script..." -ForegroundColor Yellow
Write-Host ""

$scriptPath = "scripts/reorganize_backend.py"
$args = @()
if ($DryRun) {
    $args += "--dry-run"
}

try {
    python $scriptPath @args
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host ""
        Write-Host "✅ Reorganization completed successfully!" -ForegroundColor Green
        
        if (-not $DryRun) {
            Write-Host ""
            Write-Host "📄 Next steps:" -ForegroundColor Cyan
            Write-Host "1. Review the changes with: git status" -ForegroundColor White
            Write-Host "2. Test the application: docker-compose build" -ForegroundColor White
            Write-Host "3. Start services: docker-compose up -d" -ForegroundColor White
            Write-Host "4. If everything works, commit: git add . && git commit -m 'refactor: simplify backend structure'" -ForegroundColor White
        } else {
            Write-Host ""
            Write-Host "📝 To apply changes, run without --dry-run:" -ForegroundColor Cyan
            Write-Host ".\scripts\reorganize_backend.ps1" -ForegroundColor White
        }
    } else {
        Write-Host ""
        Write-Host "❌ Reorganization failed with exit code: $exitCode" -ForegroundColor Red
    }
} catch {
    Write-Host ""
    Write-Host "❌ Error running script: $_" -ForegroundColor Red
    exit 1
}

Write-Host "" 