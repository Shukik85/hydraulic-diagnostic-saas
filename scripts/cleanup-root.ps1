# ===============================================================================
# Root Directory Cleanup Script v1.0
# Organize documentation and scripts into proper structure
# ===============================================================================

param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "Root Directory Cleanup v1.0" -ForegroundColor Green
Write-Host "Organize docs and scripts" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

$projectRoot = Get-Location

if ($DryRun) {
    Write-Host "⚠️  DRY RUN MODE" -ForegroundColor Yellow
    Write-Host ""
}

# ===============================================================================
# PHASE 1: CREATE STRUCTURE
# ===============================================================================
Write-Host "[PHASE 1] Creating structure..." -ForegroundColor Cyan

$newDirs = @(
    "docs/deployment",
    "docs/development",
    "docs/project",
    "scripts/deployment",
    "scripts/testing"
)

foreach ($dir in $newDirs) {
    $fullPath = Join-Path $projectRoot $dir
    if (-not (Test-Path $fullPath)) {
        Write-Host "  📁 Creating: $dir" -ForegroundColor Green
        if (-not $DryRun) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        }
    }
}
Write-Host ""

# ===============================================================================
# PHASE 2: MOVE DOCS
# ===============================================================================
Write-Host "[PHASE 2] Organizing docs..." -ForegroundColor Cyan

$docMoves = @(
    @{Src="BACKEND_REORGANIZATION.md"; Dst="docs/deployment/BACKEND_REORGANIZATION.md"},
    @{Src="DOCKER_QUICK_FIX.md"; Dst="docs/deployment/DOCKER_QUICK_FIX.md"},
    @{Src="QUICK_REBUILD.md"; Dst="docs/deployment/QUICK_REBUILD.md"},
    @{Src="DEVELOPMENT_QUICKSTART.md"; Dst="docs/development/DEVELOPMENT_QUICKSTART.md"},
    @{Src="WINDOWS_SETUP.md"; Dst="docs/development/WINDOWS_SETUP.md"},
    @{Src="WINDOWS_QUICKFIX.md"; Dst="docs/development/WINDOWS_QUICKFIX.md"},
    @{Src="CURRENT_STATUS.md"; Dst="docs/project/CURRENT_STATUS.md"},
    @{Src="ROADMAP_INCREMENTAL.md"; Dst="docs/project/ROADMAP_INCREMENTAL.md"},
    @{Src="STAGE_0_COMPLETION.md"; Dst="docs/project/STAGE_0_COMPLETION.md"}
)

foreach ($move in $docMoves) {
    $srcPath = Join-Path $projectRoot $move.Src
    $dstPath = Join-Path $projectRoot $move.Dst
    
    if ((Test-Path $srcPath) -and -not (Test-Path $dstPath)) {
        Write-Host "  📄 $($move.Src) -> $($move.Dst)" -ForegroundColor Green
        if (-not $DryRun) {
            Move-Item -Path $srcPath -Destination $dstPath -Force
        }
    }
}
Write-Host ""

# ===============================================================================
# PHASE 3: MOVE SCRIPTS
# ===============================================================================
Write-Host "[PHASE 3] Organizing scripts..." -ForegroundColor Cyan

$scriptMoves = @(
    @{Src="docker-quick-start.bat"; Dst="scripts/deployment/docker-quick-start.bat"},
    @{Src="fix-docker-issues.ps1"; Dst="scripts/deployment/fix-docker-issues.ps1"},
    @{Src="quick-test.ps1"; Dst="scripts/testing/quick-test.ps1"},
    @{Src="quick-test.sh"; Dst="scripts/testing/quick-test.sh"}
)

foreach ($move in $scriptMoves) {
    $srcPath = Join-Path $projectRoot $move.Src
    $dstPath = Join-Path $projectRoot $move.Dst
    
    if ((Test-Path $srcPath) -and -not (Test-Path $dstPath)) {
        Write-Host "  📜 $($move.Src) -> $($move.Dst)" -ForegroundColor Green
        if (-not $DryRun) {
            Move-Item -Path $srcPath -Destination $dstPath -Force
        }
    }
}
Write-Host ""

# ===============================================================================
# PHASE 4: REMOVE UNNECESSARY
# ===============================================================================
Write-Host "[PHASE 4] Removing unnecessary..." -ForegroundColor Cyan

$filesToRemove = @(
    "test-bot-ops.md",
    "requirements-fixed.txt",
    ".docker-trigger",
    ".frontend.gitignore"
)

foreach ($file in $filesToRemove) {
    $filePath = Join-Path $projectRoot $file
    
    if (Test-Path $filePath) {
        Write-Host "  🗑️  $file" -ForegroundColor Yellow
        if (-not $DryRun) {
            Remove-Item -Path $filePath -Force -ErrorAction SilentlyContinue
        }
    }
}
Write-Host ""

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "✅ Cleanup Complete!" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
