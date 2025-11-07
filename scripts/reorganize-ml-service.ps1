# ===============================================================================
# ML Service Structure Reorganization Script (Windows Compatible)
# Safe automated reorganization with rollback capability
# ===============================================================================

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "ML Service Reorganization Script v1.0" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green
Write-Host ""

# Navigate to ml_service directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$mlServicePath = Join-Path (Split-Path -Parent $scriptDir) "ml_service"

if (-not (Test-Path $mlServicePath)) {
    Write-Host "[ERROR] ml_service directory not found at: $mlServicePath" -ForegroundColor Red
    exit 1
}

Set-Location $mlServicePath
Write-Host "[INFO] Working directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host ""

# Create backup
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = Join-Path (Split-Path $mlServicePath -Parent) "ml_service_backup_$timestamp"
Write-Host "[BACKUP] Creating backup at: $backupDir" -ForegroundColor Yellow
Copy-Item -Path $mlServicePath -Destination $backupDir -Recurse -Force
Write-Host "[OK] Backup created successfully" -ForegroundColor Green
Write-Host ""

# ===============================================================================
# Phase 1: Create new directory structure
# ===============================================================================

Write-Host "[PHASE 1] Creating new directory structure..." -ForegroundColor Cyan

$newDirs = @(
    "src/api",
    "src/models",
    "src/services",
    "src/data",
    "src/utils",
    "scripts/train",
    "scripts/test",
    "scripts/deploy",
    "scripts/data",
    "tests/unit",
    "tests/integration",
    "docs"
)

foreach ($dir in $newDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  [OK] Created: $dir" -ForegroundColor Green
    }
    else {
        Write-Host "  [SKIP] Already exists: $dir" -ForegroundColor Gray
    }
}

Write-Host ""

# ===============================================================================
# Phase 2: Move documentation files
# ===============================================================================

Write-Host "[PHASE 2] Moving documentation files..." -ForegroundColor Cyan

$docFiles = @{
    "TRAINING.md"           = "docs/training.md"
    "TESTING.md"            = "docs/testing.md"
    "REAL_DATA_TRAINING.md" = "docs/real_data_training.md"
    "production_plan.md"    = "docs/production_plan.md"
}

foreach ($source in $docFiles.Keys) {
    $dest = $docFiles[$source]
    if (Test-Path $source) {
        Move-Item -Path $source -Destination $dest -Force
        Write-Host "  [OK] Moved: $source -> $dest" -ForegroundColor Green
    }
    else {
        Write-Host "  [WARN] Not found: $source" -ForegroundColor Yellow
    }
}

Write-Host ""

# ===============================================================================
# Phase 3: Move script files
# ===============================================================================

Write-Host "[PHASE 3] Moving script files..." -ForegroundColor Cyan

$scriptFiles = @{
    "train_real_production_models.py" = "scripts/train/train_production.py"
    "quick_test.py"                   = "scripts/test/quick_test.py"
    "test_uci_loader.py"              = "tests/integration/test_data_loader.py"
    "validate_setup.py"               = "scripts/test/validate_setup.py"
    "make_uci_dataset.py"             = "scripts/data/make_dataset.py"
    "cleanup.sh"                      = "scripts/deploy/cleanup.sh"
}

foreach ($source in $scriptFiles.Keys) {
    $dest = $scriptFiles[$source]
    if (Test-Path $source) {
        Move-Item -Path $source -Destination $dest -Force
        Write-Host "  [OK] Moved: $source -> $dest" -ForegroundColor Green
    }
    else {
        Write-Host "  [WARN] Not found: $source" -ForegroundColor Yellow
    }
}

Write-Host ""

# ===============================================================================
# Phase 4: Move source code
# ===============================================================================

Write-Host "[PHASE 4] Moving source code..." -ForegroundColor Cyan

# Move config.py to src/
if (Test-Path "config.py") {
    Move-Item -Path "config.py" -Destination "src/config.py" -Force
    Write-Host "  [OK] Moved: config.py -> src/config.py" -ForegroundColor Green
}
else {
    Write-Host "  [WARN] Not found: config.py" -ForegroundColor Yellow
}

Write-Host ""

# ===============================================================================
# Phase 5: Update imports in Python files
# ===============================================================================

Write-Host "[PHASE 5] Updating imports in Python files..." -ForegroundColor Cyan

# Files that need import updates
$filesToUpdate = @(
    "main.py",
    "simple_predict.py"
)

foreach ($file in $filesToUpdate) {
    if (Test-Path $file) {
        try {
            $content = Get-Content $file -Raw -Encoding UTF8

            # Update config import
            $content = $content -replace 'from config import', 'from src.config import'
            $content = $content -replace '(?m)^import config$', 'import src.config as config'

            Set-Content $file -Value $content -Encoding UTF8 -NoNewline
            Write-Host "  [OK] Updated imports in: $file" -ForegroundColor Green
        }
        catch {
            Write-Host "  [ERROR] Failed to update: $file - $_" -ForegroundColor Red
        }
    }
}

# Update imports in moved scripts
$movedScripts = @(
    "scripts/train/train_production.py",
    "scripts/test/quick_test.py",
    "tests/integration/test_data_loader.py"
)

foreach ($file in $movedScripts) {
    if (Test-Path $file) {
        try {
            $content = Get-Content $file -Raw -Encoding UTF8

            # Update config import
            $content = $content -replace 'from config import', 'from src.config import'

            # Update sys.path for correct import resolution
            $content = $content -replace 'sys\.path\.insert\(0, str\(Path\(__file__\)\.parent\)\)', 'sys.path.insert(0, str(Path(__file__).parent.parent.parent))'

            Set-Content $file -Value $content -Encoding UTF8 -NoNewline
            Write-Host "  [OK] Updated imports in: $file" -ForegroundColor Green
        }
        catch {
            Write-Host "  [ERROR] Failed to update: $file - $_" -ForegroundColor Red
        }
    }
}

Write-Host ""

# ===============================================================================
# Phase 6: Update Makefile
# ===============================================================================

Write-Host "[PHASE 6] Updating Makefile..." -ForegroundColor Cyan

if (Test-Path "Makefile") {
    try {
        $content = Get-Content "Makefile" -Raw -Encoding UTF8

        # Update training script path
        $content = $content -replace 'train_real_production_models\.py', 'scripts/train/train_production.py'

        Set-Content "Makefile" -Value $content -Encoding UTF8 -NoNewline
        Write-Host "  [OK] Updated Makefile" -ForegroundColor Green
    }
    catch {
        Write-Host "  [ERROR] Failed to update Makefile: $_" -ForegroundColor Red
    }
}

Write-Host ""

# ===============================================================================
# Phase 7: Update PowerShell training scripts
# ===============================================================================

Write-Host "[PHASE 7] Updating PowerShell scripts..." -ForegroundColor Cyan

$psScripts = @(
    "scripts/train_ensemble_gpu.ps1",
    "scripts/train_xgboost_gpu_safe.ps1"
)

foreach ($file in $psScripts) {
    if (Test-Path $file) {
        try {
            $content = Get-Content $file -Raw -Encoding UTF8

            # Update Python script path
            $content = $content -replace 'train_real_production_models\.py', 'scripts/train/train_production.py'

            Set-Content $file -Value $content -Encoding UTF8 -NoNewline
            Write-Host "  [OK] Updated: $file" -ForegroundColor Green
        }
        catch {
            Write-Host "  [ERROR] Failed to update: $file - $_" -ForegroundColor Red
        }
    }
}

Write-Host ""

# ===============================================================================
# Phase 8: Update README.md
# ===============================================================================

Write-Host "[PHASE 8] Updating README.md..." -ForegroundColor Cyan

if (Test-Path "README.md") {
    try {
        $content = Get-Content "README.md" -Raw -Encoding UTF8

        # Update documentation links
        $content = $content -replace 'TRAINING\.md', 'docs/training.md'
        $content = $content -replace 'TESTING\.md', 'docs/testing.md'
        $content = $content -replace 'REAL_DATA_TRAINING\.md', 'docs/real_data_training.md'
        $content = $content -replace 'production_plan\.md', 'docs/production_plan.md'

        Set-Content "README.md" -Value $content -Encoding UTF8 -NoNewline
        Write-Host "  [OK] Updated README.md with new documentation paths" -ForegroundColor Green
    }
    catch {
        Write-Host "  [ERROR] Failed to update README.md: $_" -ForegroundColor Red
    }
}

Write-Host ""

# ===============================================================================
# Phase 9: Verification
# ===============================================================================

Write-Host "[PHASE 9] Verifying new structure..." -ForegroundColor Cyan

Write-Host ""
Write-Host "New directory structure:" -ForegroundColor Yellow
Get-ChildItem -Directory -Recurse -Depth 2 | Where-Object {
    $_.FullName -notmatch '\.git|__pycache__|\.venv|node_modules|\.pytest_cache'
} | Select-Object -First 20 | ForEach-Object {
    $relativePath = $_.FullName.Replace((Get-Location).Path + '\', '')
    Write-Host "  $relativePath" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "[SUCCESS] Reorganization completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review changes: git status" -ForegroundColor White
Write-Host "  2. Test functionality: make serve (or docker compose up)" -ForegroundColor White
Write-Host "  3. Commit changes:" -ForegroundColor White
Write-Host "     git add ." -ForegroundColor Gray
Write-Host "     git commit -m 'refactor: reorganize ml_service structure'" -ForegroundColor Gray
Write-Host "  4. Push to repo: git push" -ForegroundColor White
Write-Host ""
Write-Host "Backup location: $backupDir" -ForegroundColor Cyan
Write-Host "Use this to rollback if needed!" -ForegroundColor Cyan
Write-Host ""
