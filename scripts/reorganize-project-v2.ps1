# ===============================================================================
# Complete Project Reorganization Script v2.0
# Django 5.1 + DRF 3.15 Best Practices + Microservices Architecture
# ===============================================================================

param(
    [switch]$DryRun,
    [switch]$SkipBackup
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "Hydraulic Diagnostic SaaS - Complete Project Reorganization v2.0" -ForegroundColor Green  
Write-Host "Django 5.1 + DRF 3.15 + Microservices Architecture" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

$projectRoot = Split-Path $PSScriptRoot -Parent
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = Join-Path $projectRoot "project_backup_$timestamp"

if ($DryRun) {
    Write-Host "⚠️  DRY RUN MODE - No changes will be made" -ForegroundColor Yellow
    Write-Host ""
}

# ===============================================================================
# PHASE 1: BACKUP
# ===============================================================================
Write-Host "[PHASE 1] Creating backup..." -ForegroundColor Cyan
Write-Host ""

if (-not $SkipBackup -and -not $DryRun) {
    Write-Host "  📦 Backing up to: $backupDir" -ForegroundColor Yellow
    Copy-Item -Path $projectRoot -Destination $backupDir -Recurse -Force `
        -Exclude @('.git','.venv','node_modules','__pycache__','.pytest_cache')
    Write-Host "  ✅ Backup created successfully" -ForegroundColor Green
} else {
    Write-Host "  ⏭️  Skipping backup (dry-run or skip-backup flag)" -ForegroundColor Gray
}
Write-Host ""

# ===============================================================================
# PHASE 2: CREATE NEW MICROSERVICES STRUCTURE
# ===============================================================================
Write-Host "[PHASE 2] Creating microservices directory structure..." -ForegroundColor Cyan
Write-Host ""

$newDirs = @(
    "services/backend",
    "services/ml_service",
    "services/frontend",
    "infrastructure/docker",
    "infrastructure/nginx",
    "infrastructure/monitoring",
    "docs/architecture",
    "docs/api",
    "docs/deployment"
)

foreach ($dir in $newDirs) {
    $fullPath = Join-Path $projectRoot $dir
    if (-not (Test-Path $fullPath)) {
        Write-Host "  📁 Creating: $dir" -ForegroundColor Cyan
        if (-not $DryRun) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        }
    } else {
        Write-Host "  ⏭️  Exists: $dir" -ForegroundColor Gray
    }
}
Write-Host ""

# ===============================================================================
# PHASE 3: MOVE BACKEND TO SERVICES/
# ===============================================================================
Write-Host "[PHASE 3] Moving backend to services/backend/..." -ForegroundColor Cyan
Write-Host ""

$oldBackend = Join-Path $projectRoot "backend"
$newBackend = Join-Path $projectRoot "services/backend"

if ((Test-Path $oldBackend) -and -not (Test-Path $newBackend)) {
    Write-Host "  🚚 Moving: backend/ -> services/backend/" -ForegroundColor Cyan
    if (-not $DryRun) {
        Move-Item -Path $oldBackend -Destination $newBackend -Force
    }
} elseif (Test-Path $newBackend) {
    Write-Host "  ⏭️  Already moved: services/backend/" -ForegroundColor Gray
} else {
    Write-Host "  ⚠️  Backend directory not found!" -ForegroundColor Yellow
}
Write-Host ""

# ===============================================================================
# PHASE 4: MOVE ML_SERVICE TO SERVICES/
# ===============================================================================
Write-Host "[PHASE 4] Moving ml_service to services/ml_service/..." -ForegroundColor Cyan
Write-Host ""

$oldML = Join-Path $projectRoot "ml_service"
$newML = Join-Path $projectRoot "services/ml_service"

if ((Test-Path $oldML) -and -not (Test-Path $newML)) {
    Write-Host "  🚚 Moving: ml_service/ -> services/ml_service/" -ForegroundColor Cyan
    if (-not $DryRun) {
        Move-Item -Path $oldML -Destination $newML -Force
    }
} elseif (Test-Path $newML) {
    Write-Host "  ⏭️  Already moved: services/ml_service/" -ForegroundColor Gray
} else {
    Write-Host "  ⚠️  ml_service directory not found!" -ForegroundColor Yellow
}
Write-Host ""

# ===============================================================================
# PHASE 5: REORGANIZE BACKEND (Django 5.1 structure)
# ===============================================================================
Write-Host "[PHASE 5] Reorganizing backend structure..." -ForegroundColor Cyan
Write-Host ""

if (Test-Path $newBackend) {
    # Move apps to root level
    $apps = @("users", "diagnostics", "sensors", "rag_assistant")
    $appsDir = Join-Path $newBackend "apps"
    
    if (Test-Path $appsDir) {
        foreach ($app in $apps) {
            $src = Join-Path $appsDir $app
            $dst = Join-Path $newBackend $app
            
            if ((Test-Path $src) -and -not (Test-Path $dst)) {
                Write-Host "  📦 Moving app: $app" -ForegroundColor Cyan
                if (-not $DryRun) {
                    Move-Item -Path $src -Destination $dst -Force
                }
            }
        }
        
        # Remove empty apps dir
        if (-not $DryRun) {
            $remaining = Get-ChildItem $appsDir -ErrorAction SilentlyContinue
            if ($null -eq $remaining -or $remaining.Count -eq 0) {
                Remove-Item $appsDir -Force
                Write-Host "  🗑️  Removed empty apps/ directory" -ForegroundColor Cyan
            }
        }
    }
    
    # Rename core -> config
    $coreDir = Join-Path $newBackend "core"
    $configDir = Join-Path $newBackend "config"
    
    if ((Test-Path $coreDir) -and -not (Test-Path $configDir)) {
        Write-Host "  📝 Renaming: core/ -> config/" -ForegroundColor Cyan
        if (-not $DryRun) {
            Rename-Item -Path $coreDir -NewName "config" -Force
        }
    }
    
    # Create settings directory structure
    $settingsDir = Join-Path $configDir "settings"
    if (-not (Test-Path $settingsDir) -and -not $DryRun) {
        New-Item -ItemType Directory -Path $settingsDir -Force | Out-Null
        Write-Host "  📁 Created: config/settings/" -ForegroundColor Cyan
    }
}
Write-Host ""

# ===============================================================================
# PHASE 6: MOVE DOCKER FILES TO INFRASTRUCTURE/
# ===============================================================================
Write-Host "[PHASE 6] Reorganizing Docker configuration..." -ForegroundColor Cyan
Write-Host ""

$dockerFiles = @(
    @{Src="docker-compose.yml"; Dst="docker-compose.yml"},
    @{Src="docker-compose.dev.yml"; Dst="docker-compose.dev.yml"},
    @{Src="docker-compose.prod.yml"; Dst="docker-compose.prod.yml"},
    @{Src="docker"; Dst="infrastructure/docker"},
    @{Src=".dockerignore"; Dst=".dockerignore"}
)

foreach ($file in $dockerFiles) {
    $src = Join-Path $projectRoot $file.Src
    if (Test-Path $src) {
        Write-Host "  ✅ Found: $($file.Src)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Not found: $($file.Src)" -ForegroundColor Yellow
    }
}
Write-Host ""

# ===============================================================================
# PHASE 7: UPDATE IMPORTS AND REFERENCES
# ===============================================================================
Write-Host "[PHASE 7] Updating imports and references..." -ForegroundColor Cyan
Write-Host ""

if (Test-Path $newBackend) {
    Write-Host "  🔄 Updating Python imports..." -ForegroundColor Yellow
    
    $pythonFiles = Get-ChildItem -Path $newBackend -Recurse -Include *.py -ErrorAction SilentlyContinue
    $updateCount = 0
    
    foreach ($file in $pythonFiles) {
        try {
            $content = Get-Content $file.FullName -Raw -Encoding UTF8
            $original = $content
            
            # Update imports
            $content = $content -replace 'from apps\.([\w_]+)', 'from $1'
            $content = $content -replace 'import apps\.([\w_]+)', 'import $1'
            $content = $content -replace 'from core\.', 'from config.'
            $content = $content -replace 'import core\.', 'import config.'
            $content = $content -replace '"apps\.([\w_]+)', '"$1'
            $content = $content -replace "'apps\.([\w_]+)", "'$1"
            $content = $content -replace '"core\.', '"config.'
            $content = $content -replace "'core\.", "'config."
            
            if ($content -ne $original) {
                $updateCount++
                if (-not $DryRun) {
                    $content | Set-Content $file.FullName -Encoding UTF8 -NoNewline
                }
            }
        } catch {
            Write-Host "  ⚠️  Error processing: $($file.Name)" -ForegroundColor Yellow
        }
    }
    
    Write-Host "  ✅ Updated $updateCount files" -ForegroundColor Green
}
Write-Host ""

# ===============================================================================
# PHASE 8: CLEANUP
# ===============================================================================
Write-Host "[PHASE 8] Cleaning up..." -ForegroundColor Cyan
Write-Host ""

if (Test-Path $newBackend) {
    Write-Host "  🧹 Removing __pycache__ directories..." -ForegroundColor Yellow
    
    $pycacheDirs = Get-ChildItem -Path $newBackend -Filter "__pycache__" -Directory -Recurse -ErrorAction SilentlyContinue
    
    foreach ($dir in $pycacheDirs) {
        if (-not $DryRun) {
            Remove-Item $dir.FullName -Recurse -Force
        }
    }
    
    Write-Host "  ✅ Removed $($pycacheDirs.Count) __pycache__ directories" -ForegroundColor Green
}
Write-Host ""

# ===============================================================================
# PHASE 9: SHOW NEW STRUCTURE
# ===============================================================================
Write-Host "[PHASE 9] New project structure:" -ForegroundColor Cyan
Write-Host ""

Write-Host "📁 Project Structure:" -ForegroundColor Yellow
Write-Host "  services/" -ForegroundColor Cyan
Write-Host "    ├── backend/     (Django 5.1 + DRF 3.15)" -ForegroundColor White
Write-Host "    ├── ml_service/  (FastAPI ML service)" -ForegroundColor White
Write-Host "    └── frontend/    (Nuxt/React)" -ForegroundColor White
Write-Host "  infrastructure/" -ForegroundColor Cyan
Write-Host "    ├── docker/" -ForegroundColor White
Write-Host "    ├── nginx/" -ForegroundColor White
Write-Host "    └── monitoring/" -ForegroundColor White
Write-Host "  scripts/" -ForegroundColor Cyan
Write-Host "  docs/" -ForegroundColor Cyan
Write-Host ""

# ===============================================================================
# FINAL SUMMARY
# ===============================================================================
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "✅ Reorganization Complete!" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

Write-Host "📋 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review changes:      git status" -ForegroundColor White
Write-Host "  2. Update settings:     services/backend/config/settings/" -ForegroundColor White
Write-Host "  3. Test migration:      cd services/backend && python manage.py makemigrations" -ForegroundColor White
Write-Host "  4. Build containers:    docker-compose build" -ForegroundColor White
Write-Host "  5. Start services:      docker-compose up -d" -ForegroundColor White
Write-Host "  6. Run tests:          pytest services/backend/" -ForegroundColor White
Write-Host ""

if (-not $SkipBackup) {
    Write-Host "💾 Backup location:     $backupDir" -ForegroundColor Cyan
    Write-Host "   Rollback command:   rm -rf services && mv $backupDir/* ." -ForegroundColor Gray
}
Write-Host ""

Write-Host "📚 Documentation:       docs/REORGANIZATION.md" -ForegroundColor Cyan
Write-Host "🐛 Report issues:       https://github.com/Shukik85/hydraulic-diagnostic-saas/issues" -ForegroundColor Cyan
Write-Host ""
