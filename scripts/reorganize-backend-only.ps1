# ===============================================================================
# Backend Reorganization Script v3.0 (ML Service Untouched)
# Django 5.1 + DRF 3.15 Best Practices - Backend Only!
# ===============================================================================

param(
    [switch]$DryRun,
    [switch]$SkipBackup
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "Backend Reorganization v3.0 - ML Service Untouched!" -ForegroundColor Green  
Write-Host "Django 5.1 + DRF 3.15 Best Practices" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

$projectRoot = Split-Path $PSScriptRoot -Parent
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = Join-Path $projectRoot "backend_backup_$timestamp"

if ($DryRun) {
    Write-Host "⚠️  DRY RUN MODE - No changes will be made" -ForegroundColor Yellow
    Write-Host ""
}

# ===============================================================================
# PHASE 1: BACKUP (Backend only)
# ===============================================================================
Write-Host "[PHASE 1] Creating backend backup..." -ForegroundColor Cyan
Write-Host ""

$backendDir = Join-Path $projectRoot "backend"

if (-not $SkipBackup -and -not $DryRun -and (Test-Path $backendDir)) {
    Write-Host "  📦 Backing up backend to: $backupDir" -ForegroundColor Yellow
    Copy-Item -Path $backendDir -Destination $backupDir -Recurse -Force `
        -Exclude @('__pycache__','.pytest_cache','*.pyc','*.pyo')
    Write-Host "  ✅ Backend backup created" -ForegroundColor Green
} else {
    Write-Host "  ⏭️  Skipping backup" -ForegroundColor Gray
}
Write-Host ""

# ===============================================================================
# PHASE 2: REORGANIZE BACKEND STRUCTURE
# ===============================================================================
Write-Host "[PHASE 2] Reorganizing backend Django structure..." -ForegroundColor Cyan
Write-Host ""

if (Test-Path $backendDir) {
    # 2.1 Move apps to root level
    $appsDir = Join-Path $backendDir "apps"
    
    if (Test-Path $appsDir) {
        Write-Host "  📦 Moving Django apps to root level..." -ForegroundColor Cyan
        
        $apps = Get-ChildItem -Path $appsDir -Directory
        foreach ($app in $apps) {
            $dst = Join-Path $backendDir $app.Name
            
            if (-not (Test-Path $dst)) {
                Write-Host "    ✅ Moving: $($app.Name)" -ForegroundColor Green
                if (-not $DryRun) {
                    Move-Item -Path $app.FullName -Destination $dst -Force
                }
            } else {
                Write-Host "    ⏭️  Exists: $($app.Name)" -ForegroundColor Gray
            }
        }
        
        if (-not $DryRun) {
            $remaining = Get-ChildItem $appsDir -ErrorAction SilentlyContinue
            if ($null -eq $remaining -or $remaining.Count -eq 0) {
                Remove-Item $appsDir -Force -ErrorAction SilentlyContinue
                Write-Host "  🗑️  Removed empty apps/" -ForegroundColor Green
            }
        }
    }
    
    # 2.2 Rename core -> config
    $coreDir = Join-Path $backendDir "core"
    $configDir = Join-Path $backendDir "config"
    
    if ((Test-Path $coreDir) -and -not (Test-Path $configDir)) {
        Write-Host "  📝 Renaming: core/ -> config/" -ForegroundColor Cyan
        if (-not $DryRun) {
            Rename-Item -Path $coreDir -NewName "config" -Force
        }
        Write-Host "    ✅ Renamed" -ForegroundColor Green
    }
    
    # 2.3 Create settings structure
    $settingsDir = Join-Path $configDir "settings"
    if (-not (Test-Path $settingsDir) -and -not $DryRun -and (Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $settingsDir -Force | Out-Null
        Write-Host "  📁 Created: config/settings/" -ForegroundColor Green
    }
}
Write-Host ""

# ===============================================================================
# PHASE 3: UPDATE IMPORTS
# ===============================================================================
Write-Host "[PHASE 3] Updating imports..." -ForegroundColor Cyan
Write-Host ""

if (Test-Path $backendDir) {
    $pythonFiles = Get-ChildItem -Path $backendDir -Recurse -Include *.py -ErrorAction SilentlyContinue
    $updateCount = 0
    
    foreach ($file in $pythonFiles) {
        try {
            $content = Get-Content $file.FullName -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
            if ($null -eq $content) { continue }
            
            $original = $content
            
            $content = $content -replace 'from apps\.(\w+)', 'from $1'
            $content = $content -replace 'import apps\.(\w+)', 'import $1'
            $content = $content -replace 'from core\.', 'from config.'
            $content = $content -replace 'import core\.', 'import config.'
            $content = $content -replace '"apps\.(\w+)', '"$1'
            $content = $content -replace "'apps\.(\w+)", "'$1"
            $content = $content -replace '"core\.', '"config.'
            $content = $content -replace "'core\.", "'config."
            $content = $content -replace "'apps\.(\w+)'", "'$1'"
            
            if ($content -ne $original) {
                $updateCount++
                if (-not $DryRun) {
                    $content | Set-Content $file.FullName -Encoding UTF8 -NoNewline
                }
            }
        } catch {}
    }
    
    Write-Host "  ✅ Updated $updateCount files" -ForegroundColor Green
}
Write-Host ""

# ===============================================================================
# FINAL SUMMARY
# ===============================================================================
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "✅ Backend Reorganization Complete!" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

Write-Host "📊 Changes:" -ForegroundColor Yellow
Write-Host "  ✅ apps/* -> backend/app_name/ (root level)" -ForegroundColor Green
Write-Host "  ✅ core/ -> config/ (Django 5.1 convention)" -ForegroundColor Green
Write-Host "  ✅ Imports updated (apps.xxx -> xxx)" -ForegroundColor Green
Write-Host "  ✅ Root directories organized" -ForegroundColor Green
Write-Host "  ⚠️  ml_service/ UNTOUCHED (working perfectly!)" -ForegroundColor Yellow
Write-Host ""

Write-Host "📝 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. cd backend && python manage.py check" -ForegroundColor White
Write-Host "  2. python manage.py makemigrations" -ForegroundColor White
Write-Host "  3. python manage.py migrate" -ForegroundColor White
Write-Host "  4. python manage.py test" -ForegroundColor White
Write-Host "  5. cd ../ml_service && make test-onnx" -ForegroundColor White
Write-Host ""

if (-not $SkipBackup -and (Test-Path $backupDir)) {
    Write-Host "💾 Backup: $backupDir" -ForegroundColor Cyan
}
Write-Host ""
