# ===============================================================================
# Backend Structure Reorganization Script (PowerShell, atomic/safe style)
# Inspired by ml_service/scripts/reorganize-ml-service.ps1
# ===============================================================================

param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Backend Structure Reorganization Script v2.0" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

$backendDir = Join-Path $PSScriptRoot "..\backend" | Resolve-Path
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = Join-Path ($backendDir | Split-Path -Parent) "backend_backup_$timestamp"

# ---------------------- BACKUP ----------------------
Write-Host "[BACKUP] Creating backup at: $backupDir" -ForegroundColor Yellow
if (-not $DryRun) {
    Copy-Item $backendDir $backupDir -Recurse -Force
}
Write-Host "[OK] Backup created successfully" -ForegroundColor Green
Write-Host ""

# ------------------- NEW STRUCTURE -----------------
$newApps = @("users", "diagnostics", "sensors", "rag_assistant")
$newConfigDir = Join-Path $backendDir "config"
$oldCoreDir = Join-Path $backendDir "core"
$appsDir = Join-Path $backendDir "apps"

# Create new structure only if hasn't been moved yet
foreach ($app in $newApps) {
    $src = Join-Path $appsDir $app
    $dst = Join-Path $backendDir $app
    if (Test-Path $src) {
        Write-Host "[MOVE] $src -> $dst" -ForegroundColor Cyan
        if (-not $DryRun) {
            Move-Item -Path $src -Destination $dst -Force
        }
    } elseif (Test-Path $dst) {
        Write-Host "[SKIP] $dst already exists" -ForegroundColor Gray
    } else {
        Write-Host "[WARN] $src not found" -ForegroundColor Yellow
    }
}

# Rename core -> config
if ((Test-Path $oldCoreDir) -and (!(Test-Path $newConfigDir))) {
    Write-Host "[RENAME] $oldCoreDir -> $newConfigDir" -ForegroundColor Cyan
    if (-not $DryRun) {
        Rename-Item -Path $oldCoreDir -NewName 'config'
    }
}

# Remove empty apps dir
if (Test-Path $appsDir) {
    $items = Get-ChildItem $appsDir
    if ($items.Count -eq 0 -and -not $DryRun) {
        Write-Host "[REMOVE] apps dir (empty)" -ForegroundColor Cyan
        Remove-Item $appsDir -Force
    } elseif ($items.Count -eq 0) {
        Write-Host "[SKIP] apps dir is empty, dry run only" -ForegroundColor Gray
    }
}

# -------------- UPDATE IMPORTS and SETTINGS --------------
function Update-ImportsRecursive {
    param ($dir)
    $files = Get-ChildItem -Path $dir -Recurse -Include *.py
    foreach ($file in $files) {
        $content = Get-Content $file.FullName -Raw -Encoding UTF8
        $orig = $content
        $content = $content -replace 'from apps\\.([\w_]+)', 'from $1'
        $content = $content -replace 'import apps\\.([\w_]+)', 'import $1'
        $content = $content -replace 'from core\\.', 'from config.'
        $content = $content -replace 'import core\\.', 'import config.'
        $content = $content -replace '"apps\\.([\w_]+)', '"$1'
        $content = $content -replace "'apps\\.([\w_]+)", "'$1"
        if ($content -ne $orig) {
            Write-Host "[UPDATE] $($file.FullName)" -ForegroundColor Cyan
            if (-not $DryRun) { $content | Set-Content $file.FullName -Encoding UTF8 -NoNewline }
        }
    }
}

Update-ImportsRecursive $backendDir

# -------------- UPDATE settings.py --------------
$settingsPY = Join-Path $newConfigDir "settings.py"
if (-not (Test-Path $settingsPY)) { $settingsPY = Join-Path $oldCoreDir "settings.py" }
if (Test-Path $settingsPY) {
    $content = Get-Content $settingsPY -Raw -Encoding UTF8
    $orig = $content
    $content = $content -replace '"apps\\.([\w_]+)\\.apps\\.[\w]+Config"', '"$1"'
    $content = $content -replace "'apps\\.([\w_]+)\\.apps\\.[\w]+Config'", "'$1'"
    $content = $content -replace '"core.settings"', '"config.settings"'
    $content = $content -replace "'core.settings'", "'config.settings'"
    if ($content -ne $orig) {
        Write-Host "[UPDATE] settings.py" -ForegroundColor Cyan
        if (-not $DryRun) { $content | Set-Content $settingsPY -Encoding UTF8 -NoNewline }
    }
}

# ------------- UPDATE manage.py/asgi/wsgi -------------
$updateEntry = @('manage.py','asgi.py','wsgi.py')
foreach ($entry in $updateEntry) {
    $cands = @(
    (Join-Path $newConfigDir $entry),
    (Join-Path $oldCoreDir $entry),
    (Join-Path $backendDir $entry)
)

    foreach ($cand in $cands) {
        if (Test-Path $cand) {
            $content = Get-Content $cand -Raw -Encoding UTF8
            $orig = $content
            $content = $content -replace 'core.settings','config.settings'
            if ($content -ne $orig) {
                Write-Host "[UPDATE] $cand" -ForegroundColor Cyan
                if (-not $DryRun) { $content | Set-Content $cand -Encoding UTF8 -NoNewline }
            }
        }
    }
}

# ----------------- CLEANUP __pycache__ -----------------
Write-Host "[CLEANUP] Removing __pycache__..." -ForegroundColor Yellow
$pycacheDirs = Get-ChildItem -Path $backendDir -Filter '__pycache__' -Directory -Recurse
foreach ($dir in $pycacheDirs) {
    Write-Host "  [REMOVE] $($dir.FullName)" -ForegroundColor Cyan
    if (-not $DryRun) { Remove-Item $dir.FullName -Recurse -Force }
}

# --------------- SHOW NEW STRUCTURE --------------------
Write-Host "\n[INFO] New directory structure (top-2 levels):" -ForegroundColor Yellow
Get-ChildItem $backendDir -Recurse -Directory | Where-Object { $_.FullName -notmatch '\.git|__pycache__|\.venv|node_modules|\.pytest_cache' } |
    Select-Object -First 20 | ForEach-Object {
    $relativePath = $_.FullName.Replace("$backendDir\", "")
        Write-Host "  $relativePath" -ForegroundColor Cyan
    }

Write-Host "\n[INFO] Next steps:" -ForegroundColor Yellow
Write-Host "1. Review changes: git status" -ForegroundColor White
Write-Host "2. Test: docker-compose build && docker-compose up -d" -ForegroundColor White
Write-Host "3. Commit: git add . && git commit -m 'refactor: reorganize backend'" -ForegroundColor White
Write-Host "4. Rollback: rm -rf backend && mv backend_backup_$timestamp backend" -ForegroundColor Gray
Write-Host "\nBackup is at: $backupDir" -ForegroundColor Green
Write-Host "Done!" -ForegroundColor Green
