#!/usr/bin/env pwsh
#Requires -Version 7.0

<#
.SYNOPSIS
    Quick Test Script for Hydraulic Diagnostic SaaS - Stage 0
.DESCRIPTION
    Validates that all Stage 0 components are working correctly on Windows/PowerShell 7
.EXAMPLE
    ./quick-test.ps1
#>

[CmdletBinding()]
param()

# =============================================================================
# Quick Test Script for Stage 0 - PowerShell 7 Version
# =============================================================================

Write-Host "🚀 Testing Hydraulic Diagnostic SaaS - Stage 0" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Helper functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO]" -ForegroundColor Yellow -NoNewline
    Write-Host " $Message" -ForegroundColor White
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS]" -ForegroundColor Green -NoNewline
    Write-Host " $Message" -ForegroundColor White
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR]" -ForegroundColor Red -NoNewline
    Write-Host " $Message" -ForegroundColor White
}

function Test-CommandExists {
    param([string]$Command)
    return (Get-Command $Command -ErrorAction SilentlyContinue) -ne $null
}

function Wait-ForService {
    param(
        [string]$Url,
        [int]$MaxAttempts = 30,
        [int]$SleepSeconds = 2
    )
    
    $attempt = 0
    while ($attempt -lt $MaxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                return $true
            }
        }
        catch {
            # Service not ready yet
        }
        $attempt++
        Write-Host "." -NoNewline -ForegroundColor Yellow
        Start-Sleep -Seconds $SleepSeconds
    }
    return $false
}

try {
    # Check prerequisites
    Write-Info "Checking prerequisites..."
    
    if (-not (Test-CommandExists "docker")) {
        Write-Error "Docker is not installed or not in PATH"
        Write-Host "Please install Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
        exit 1
    }
    
    # Check if Docker is running
    try {
        docker version | Out-Null
    }
    catch {
        Write-Error "Docker is not running. Please start Docker Desktop"
        exit 1
    }
    
    # Check Docker Compose
    if (-not (Test-CommandExists "docker-compose") -and -not (docker compose version 2>$null)) {
        Write-Error "Docker Compose is not available"
        exit 1
    }
    
    Write-Success "Prerequisites OK (PowerShell $($PSVersionTable.PSVersion), Docker available)"
    
    # Check .env file
    Write-Info "Checking .env file..."
    if (-not (Test-Path ".env")) {
        Write-Info "Creating .env from .env.example..."
        Copy-Item ".env.example" ".env"
        Write-Success ".env file created"
    } else {
        Write-Success ".env file exists"
    }
    
    # Start services
    Write-Info "Starting services with Docker Compose..."
    
    # Stop any existing services
    docker compose down --remove-orphans 2>$null
    
    # Start services
    Write-Info "Building and starting containers..."
    docker compose up --build -d
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to start Docker Compose services"
        exit 1
    }
    
    # Wait for services to be ready
    Write-Info "Waiting for services to start (this may take up to 2 minutes)..."
    Start-Sleep -Seconds 30
    
    # Test database connectivity
    Write-Info "Testing database connectivity..."
    $dbReady = $false
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $result = docker compose exec -T db pg_isready -U hdx_user -d hydraulic_diagnostics 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Database is ready"
                $dbReady = $true
                break
            }
        }
        catch {}
        Write-Host "." -NoNewline -ForegroundColor Yellow
        Start-Sleep -Seconds 2
    }
    Write-Host "" # New line after dots
    
    if (-not $dbReady) {
        Write-Error "Database failed to start within timeout"
        Write-Info "Database logs:"
        docker compose logs db
        exit 1
    }
    
    # Test Redis connectivity
    Write-Info "Testing Redis connectivity..."
    try {
        $redisResult = docker compose exec -T redis redis-cli ping 2>$null
        if ($redisResult -match "PONG") {
            Write-Success "Redis is ready"
        } else {
            throw "Redis ping failed"
        }
    }
    catch {
        Write-Error "Redis failed to start"
        Write-Info "Redis logs:"
        docker compose logs redis
        exit 1
    }
    
    # Test backend health
    Write-Info "Testing backend health endpoint..."
    if (Wait-ForService -Url "http://localhost:8000/health/" -MaxAttempts 40 -SleepSeconds 3) {
        Write-Host "" # New line after dots
        Write-Success "Backend health check passed"
    } else {
        Write-Host "" # New line after dots
        Write-Error "Backend health check failed"
        Write-Info "Backend logs:"
        docker compose logs backend
        exit 1
    }
    
    # Test API endpoints
    Write-Info "Testing API endpoints..."
    
    # Health check endpoint
    try {
        $healthResponse = Invoke-RestMethod -Uri "http://localhost:8000/health/" -UseBasicParsing -TimeoutSec 10
        if ($healthResponse.status -eq "healthy") {
            Write-Success "Health endpoint working - Status: $($healthResponse.status)"
        } else {
            Write-Error "Health endpoint returned non-healthy status: $($healthResponse.status)"
        }
    }
    catch {
        Write-Error "Health endpoint failed: $($_.Exception.Message)"
    }
    
    # Readiness check
    try {
        $readinessResponse = Invoke-RestMethod -Uri "http://localhost:8000/readiness/" -UseBasicParsing -TimeoutSec 10
        if ($readinessResponse.status -eq "ready") {
            Write-Success "Readiness endpoint working - Status: $($readinessResponse.status)"
        } else {
            Write-Error "Readiness endpoint failed: $($readinessResponse.status)"
        }
    }
    catch {
        Write-Error "Readiness endpoint failed: $($_.Exception.Message)"
    }
    
    # API docs
    try {
        $docsResponse = Invoke-WebRequest -Uri "http://localhost:8000/api/docs/" -UseBasicParsing -TimeoutSec 10
        if ($docsResponse.StatusCode -eq 200) {
            Write-Success "API documentation accessible"
        }
    }
    catch {
        Write-Error "API documentation failed: $($_.Exception.Message)"
    }
    
    # Admin panel
    try {
        $adminResponse = Invoke-WebRequest -Uri "http://localhost:8000/admin/" -UseBasicParsing -TimeoutSec 10
        if ($adminResponse.StatusCode -eq 200) {
            Write-Success "Admin panel accessible"
        }
    }
    catch {
        Write-Error "Admin panel failed: $($_.Exception.Message)"
    }
    
    # Test smoke diagnostics
    Write-Info "Running smoke tests..."
    try {
        docker compose exec -T backend python smoke_diagnostics.py
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Smoke tests passed"
        } else {
            Write-Error "Smoke tests failed"
            exit 1
        }
    }
    catch {
        Write-Error "Failed to run smoke tests: $($_.Exception.Message)"
        exit 1
    }
    
    # Show running services
    Write-Host ""
    Write-Info "Service Status:"
    docker compose ps
    
    # Show recent backend logs
    Write-Host ""
    Write-Info "Recent Backend Logs:"
    docker compose logs --tail=10 backend
    
    # Success summary
    Write-Host ""
    Write-Success "🎉 Stage 0 Test Completed Successfully!"
    Write-Host "=" * 50 -ForegroundColor Green
    Write-Host "🌐 Services available at:" -ForegroundColor Cyan
    Write-Host "   - Backend API: " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:8000" -ForegroundColor Yellow
    Write-Host "   - Health Check: " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:8000/health/" -ForegroundColor Yellow
    Write-Host "   - API Docs: " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:8000/api/docs/" -ForegroundColor Yellow
    Write-Host "   - Admin Panel: " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:8000/admin/" -ForegroundColor Yellow -NoNewline
    Write-Host " (admin/admin123)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "📊 To monitor services:" -ForegroundColor Cyan
    Write-Host "   docker compose logs -f" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🛑 To stop services:" -ForegroundColor Cyan
    Write-Host "   docker compose down" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "✅ Project is ready for Stage 1 development!" -ForegroundColor Green

}
catch {
    Write-Error "An unexpected error occurred: $($_.Exception.Message)"
    Write-Host "Stack trace: $($_.ScriptStackTrace)" -ForegroundColor Red
    exit 1
}
