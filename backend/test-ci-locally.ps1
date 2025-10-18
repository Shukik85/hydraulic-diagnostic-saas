# PowerShell 7.5 скрипт для локальной проверки CI
# Использование: .\test-ci-locally.ps1

#Requires -Version 7.5

$ErrorActionPreference = "Stop"

Write-Host "🔍 Проверка CI локально..." -ForegroundColor Cyan
Write-Host ""

# Переход в директорию backend
Push-Location backend

try {
    Write-Host "1️⃣ Установка зависимостей для тестирования..." -ForegroundColor Yellow
    pip install black isort flake8 mypy pytest pytest-django pytest-cov
    
    Write-Host ""
    Write-Host "2️⃣ Проверка Black (форматирование)..." -ForegroundColor Yellow
    $blackResult = black --check . 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Black failed. Запустите: black backend/" -ForegroundColor Red
        Write-Host $blackResult -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Black passed" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "3️⃣ Проверка isort (импорты)..." -ForegroundColor Yellow
    $isortResult = isort --check-only . 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ isort failed. Запустите: isort backend/" -ForegroundColor Red
        Write-Host $isortResult -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ isort passed" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "4️⃣ Проверка flake8 (линтинг)..." -ForegroundColor Yellow
    $flake8Result = flake8 . --max-line-length=100 --exclude=.venv,.mypy_cache,migrations 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ flake8 failed" -ForegroundColor Red
        Write-Host $flake8Result -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ flake8 passed" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "5️⃣ Проверка mypy (типы)..." -ForegroundColor Yellow
    $mypyResult = mypy . 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️ mypy warnings (not blocking)" -ForegroundColor Yellow
        Write-Host $mypyResult -ForegroundColor Yellow
    } else {
        Write-Host "✅ mypy passed" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "6️⃣ Проверка миграций..." -ForegroundColor Yellow
    $migrationsResult = python manage.py makemigrations --dry-run --check 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Есть несозданные миграции" -ForegroundColor Red
        Write-Host $migrationsResult -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Migrations check passed" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "7️⃣ Запуск тестов..." -ForegroundColor Yellow
    $testResult = pytest -q --disable-warnings --maxfail=1 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Tests failed" -ForegroundColor Red
        Write-Host $testResult -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Tests passed" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "🎉 Все проверки пройдены! CI должен пройти успешно." -ForegroundColor Green
    
} catch {
    Write-Host "❌ Произошла ошибка: $_" -ForegroundColor Red
    exit 1
} finally {
    # Возврат в исходную директорию
    Pop-Location
}

Write-Host ""
Write-Host "✨ Проверка завершена." -ForegroundColor Cyan
