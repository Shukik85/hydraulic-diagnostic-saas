# Hydraulic Diagnostic SaaS - PowerShell Development Script
# Windows-compatible version of Makefile

param(
    [string]$Command = "help",
    [string]$Args = ""
)

# Colors for output
$Global:Colors = @{
    Red = "`e[31m"
    Green = "`e[32m"
    Yellow = "`e[33m"
    Blue = "`e[34m"
    Magenta = "`e[35m"
    Cyan = "`e[36m"
    White = "`e[37m"
    Reset = "`e[0m"
}

function Write-ColorText {
    param(
        [string]$Text,
        [string]$Color = "White"
    )
    Write-Host "$($Global:Colors[$Color])$Text$($Global:Colors['Reset'])"
}

function Test-CommandExists {
    param([string]$Command)
    return $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Install-Dev {
    Write-ColorText "Installing development dependencies..." "Cyan"

    if (Test-CommandExists "uv") {
        Write-ColorText "Using uv for Python dependencies" "Green"
        Set-Location "backend"
        uv pip install -r requirements.txt -r requirements-dev.txt
        Set-Location ".."
    } else {
        Write-ColorText "Using pip for Python dependencies" "Yellow"
        Set-Location "backend"
        pip install -r requirements.txt -r requirements-dev.txt
        Set-Location ".."
    }

    Write-ColorText "Installing frontend dependencies..." "Green"
    Set-Location "nuxt_frontend"
    npm ci
    Set-Location ".."

    Write-ColorText "Installing pre-commit hooks..." "Green"
    pre-commit install

    Write-ColorText "✅ Development environment ready!" "Green"
}

function Start-Dev {
    Write-ColorText "Starting development environment..." "Cyan"

    if (Test-CommandExists "docker-compose") {
        docker-compose -f docker-compose.dev.yml up -d
        Write-ColorText "✅ Services started!" "Green"
        Write-ColorText "Backend: http://localhost:8000" "Blue"
        Write-ColorText "Frontend: http://localhost:3000" "Blue"
        Write-ColorText "Admin: http://localhost:8000/admin/" "Blue"
    } else {
        Write-ColorText "❌ docker-compose not found" "Red"
        exit 1
    }
}

function Start-DevLocal {
    Write-ColorText "Starting local development..." "Cyan"

    # Start backend in background
    Start-Job -ScriptBlock {
        Set-Location "backend"
        python manage.py runserver 8000
    } -Name "Backend"

    # Start frontend in background
    Start-Job -ScriptBlock {
        Set-Location "nuxt_frontend"
        npm run dev
    } -Name "Frontend"

    Write-ColorText "✅ Local development started!" "Green"
    Write-ColorText "Use 'Get-Job' to check status" "Yellow"
    Write-ColorText "Use 'Stop-Job -Name Backend,Frontend; Remove-Job -Name Backend,Frontend' to stop" "Yellow"
}

function Stop-Dev {
    Write-ColorText "Stopping development services..." "Cyan"

    # Stop Docker services
    docker-compose -f docker-compose.dev.yml down 2>$null

    # Stop PowerShell jobs
    Get-Job -Name "Backend", "Frontend" -ErrorAction SilentlyContinue | Stop-Job
    Get-Job -Name "Backend", "Frontend" -ErrorAction SilentlyContinue | Remove-Job

    Write-ColorText "✅ All services stopped" "Green"
}

function Test-Backend {
    Write-ColorText "Running backend tests..." "Cyan"
    Set-Location "backend"
    pytest -v --tb=short --cov=apps --cov-report=term-missing
    Set-Location ".."
}

function Test-Frontend {
    Write-ColorText "Running frontend tests..." "Cyan"
    Set-Location "nuxt_frontend"
    npm run test
    Set-Location ".."
}

function Test-All {
    Test-Backend
    Test-Frontend
}

function Test-Coverage {
    Write-ColorText "Running tests with coverage..." "Cyan"
    Set-Location "backend"
    pytest --cov=apps --cov-report=html --cov-report=term-missing
    Set-Location ".."

    Set-Location "nuxt_frontend"
    npm run test:coverage
    Set-Location ".."

    Write-ColorText "✅ Coverage reports generated" "Green"
    Write-ColorText "Backend coverage: backend/htmlcov/index.html" "Blue"
    Write-ColorText "Frontend coverage: nuxt_frontend/coverage/index.html" "Blue"
}

function Lint-Backend {
    Write-ColorText "Linting backend code..." "Cyan"

    if (Test-CommandExists "ruff") {
        ruff check backend/ --output-format=github
    } else {
        Write-ColorText "Ruff not found, using flake8" "Yellow"
        Set-Location "backend"
        flake8 .
        Set-Location ".."
    }

    Write-ColorText "✅ Backend linting completed" "Green"
}

function Lint-Frontend {
    Write-ColorText "Linting frontend code..." "Cyan"
    Set-Location "nuxt_frontend"
    npm run lint
    Set-Location ".."
    Write-ColorText "✅ Frontend linting completed" "Green"
}

function Format-Backend {
    Write-ColorText "Formatting backend code..." "Cyan"

    if (Test-CommandExists "ruff") {
        ruff format backend/
        ruff check backend/ --fix
    } else {
        Write-ColorText "Ruff not found, using black + isort" "Yellow"
        Set-Location "backend"
        black .
        Set-Location ".."
    }

    Write-ColorText "✅ Backend formatting completed" "Green"
}

function Format-Frontend {
    Write-ColorText "Formatting frontend code..." "Cyan"
    Set-Location "nuxt_frontend"
    npm run format
    npm run lint:fix
    Set-Location ".."
    Write-ColorText "✅ Frontend formatting completed" "Green"
}

function Run-PreCommit {
    Write-ColorText "Running pre-commit hooks..." "Cyan"
    pre-commit run --all-files
}

function Show-Status {
    Write-ColorText "Project Status:" "Cyan"
    Write-ColorText "=== Docker Services ===" "Blue"
    docker-compose -f docker-compose.dev.yml ps 2>$null

    Write-ColorText "`n=== Database Status ===" "Blue"
    Set-Location "backend"
    python manage.py showmigrations | Select-Object -First 20
    Set-Location ".."

    Write-ColorText "`n=== Dependencies ===" "Blue"
    $pythonVersion = python --version 2>$null
    $nodeVersion = node --version 2>$null
    $dockerVersion = docker --version 2>$null
    $uvVersion = uv --version 2>$null

    Write-Host "Python: $($pythonVersion -replace 'Python ', '')"
    Write-Host "Node.js: $nodeVersion"
    Write-Host "Docker: $($dockerVersion -split ' ')[2]"
    Write-Host "uv: $($uvVersion -split ' ')[1] $(if (!$uvVersion) { '(Not found)' })"
}

function Show-URLs {
    Write-ColorText "Application URLs:" "Cyan"
    Write-ColorText "Development:" "Green"
    Write-Host "  Frontend:  http://localhost:3000"
    Write-Host "  Backend:   http://localhost:8000"
    Write-Host "  Admin:     http://localhost:8000/admin/"
    Write-Host "  API Docs:  http://localhost:8000/api/docs/"
    Write-Host "  Flower:    http://localhost:5555 (Celery monitoring)"
    Write-ColorText "Database:" "Blue"
    Write-Host "  PostgreSQL: localhost:5432"
    Write-Host "  Redis:      localhost:6379"
}

function Clean-Artifacts {
    Write-ColorText "Cleaning build artifacts..." "Cyan"

    # Python artifacts
    Get-ChildItem -Recurse -Name "*.pyc" | Remove-Item -Force
    Get-ChildItem -Recurse -Directory -Name "__pycache__" | Remove-Item -Recurse -Force
    Get-ChildItem -Recurse -Directory -Name "*.egg-info" | Remove-Item -Recurse -Force

    # Coverage and test artifacts
    Remove-Item "backend/htmlcov" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item "backend/.coverage" -Force -ErrorAction SilentlyContinue
    Remove-Item "backend/.pytest_cache" -Recurse -Force -ErrorAction SilentlyContinue

    # Frontend artifacts
    Remove-Item "nuxt_frontend/coverage" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item "nuxt_frontend/.nuxt" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item "nuxt_frontend/.output" -Recurse -Force -ErrorAction SilentlyContinue

    Write-ColorText "✅ Cleanup completed" "Green"
}

function Show-Help {
    Write-ColorText "Hydraulic Diagnostic SaaS - Development Commands" "Cyan"
    Write-Host ""
    Write-ColorText "Usage: .\make.ps1 <command>" "Yellow"
    Write-Host ""

    $commands = @(
        @{"Command" = "install-dev"; "Description" = "Install development dependencies"},
        @{"Command" = "dev"; "Description" = "Start development environment with Docker"},
        @{"Command" = "dev-local"; "Description" = "Start development locally (without Docker)"},
        @{"Command" = "stop"; "Description" = "Stop all development services"},
        @{"Command" = "test"; "Description" = "Run all tests"},
        @{"Command" = "test-backend"; "Description" = "Run backend tests"},
        @{"Command" = "test-frontend"; "Description" = "Run frontend tests"},
        @{"Command" = "test-coverage"; "Description" = "Run tests with coverage report"},
        @{"Command" = "lint-backend"; "Description" = "Lint backend code"},
        @{"Command" = "lint-frontend"; "Description" = "Lint frontend code"},
        @{"Command" = "format-backend"; "Description" = "Format backend code"},
        @{"Command" = "format-frontend"; "Description" = "Format frontend code"},
        @{"Command" = "pre-commit"; "Description" = "Run pre-commit hooks"},
        @{"Command" = "status"; "Description" = "Show project status"},
        @{"Command" = "urls"; "Description" = "Show application URLs"},
        @{"Command" = "clean"; "Description" = "Clean build artifacts"},
        @{"Command" = "help"; "Description" = "Show this help message"}
    )

    foreach ($cmd in $commands) {
        Write-Host ("  {0,-20} {1}" -f $cmd.Command, $cmd.Description)
    }
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "install-dev" { Install-Dev }
    "dev" { Start-Dev }
    "dev-local" { Start-DevLocal }
    "stop" { Stop-Dev }
    "test" { Test-All }
    "test-backend" { Test-Backend }
    "test-frontend" { Test-Frontend }
    "test-coverage" { Test-Coverage }
    "lint-backend" { Lint-Backend }
    "lint-frontend" { Lint-Frontend }
    "format-backend" { Format-Backend }
    "format-frontend" { Format-Frontend }
    "pre-commit" { Run-PreCommit }
    "status" { Show-Status }
    "urls" { Show-URLs }
    "clean" { Clean-Artifacts }
    "help" { Show-Help }
    default {
        Write-ColorText "Unknown command: $Command" "Red"
        Write-ColorText "Use '.\make.ps1 help' for available commands" "Yellow"
        exit 1
    }
}
