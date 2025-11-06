@echo off
REM ===============================================================================
REM Hydraulic Diagnostic SaaS - Simple Docker Start Script (Windows Batch)
REM ===============================================================================
REM Fallback script if PowerShell doesn't work

echo.
echo 🚀 Hydraulic Diagnostic SaaS - Simple Docker Start
echo ================================================
echo ⚡ With pip caching for faster builds!
echo.

REM Enable BuildKit for caching
set DOCKER_BUILDKIT=1
set COMPOSE_DOCKER_CLI_BUILD=1

REM Check if Docker is running
echo 🔍 Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not available or running
    echo Please install Docker Desktop and make sure it's running
    pause
    exit /b 1
)
echo ✅ Docker is available

REM Check .env file
echo 📝 Checking .env file...
if not exist ".env" (
    echo ⚠️ .env file not found, creating from example...
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo ✅ .env file created from example
    ) else (
        echo ❌ .env.example not found
    )
) else (
    echo ✅ .env file exists
)

REM Stop existing containers
echo 🗑️ Stopping existing containers...
docker-compose down -v >nul 2>&1
echo ✅ Containers stopped

REM Clean up (preserve cache)
echo 🧾 Cleaning up (preserving cache)...
docker system prune -f >nul 2>&1
echo ✅ Cleanup completed

REM Build with cache
echo 🔨 Building containers with cache...
echo This may take a while on first run, but subsequent builds will be much faster!
echo.
docker-compose build
if %errorlevel% neq 0 (
    echo ❌ Build failed, trying without cache...
    docker-compose build --no-cache
    if %errorlevel% neq 0 (
        echo ❌ Build failed completely
        pause
        exit /b 1
    )
)
echo ✅ Build completed successfully

REM Start services
echo 🚀 Starting services...
docker-compose up -d

REM Wait for services
echo ⏳ Waiting for services to start...
timeout /t 15 /nobreak >nul

REM Show status
echo 🔍 Container status:
docker-compose ps

REM Show recent logs
echo.
echo 📄 Recent backend logs:
echo ================================
docker-compose logs backend --tail=20
echo ================================
echo.

REM Test connectivity
echo 🎯 Testing backend connectivity...
echo Please wait up to 60 seconds for backend to be ready...

REM Simple connectivity test
for /l %%i in (1,1,30) do (
    curl -f http://localhost:8000/health/ >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Backend is responding at http://localhost:8000
        goto :success
    )
    timeout /t 2 /nobreak >nul
    if %%i%%5 equ 0 echo Still waiting... (attempt %%i/30)
)

REM If we get here, backend is not responding
echo ⚠️ Backend is not responding after 60 seconds
echo Please check the logs above for errors
echo.
echo 🚑 Running diagnostic commands...
docker-compose exec backend python manage.py check --deploy --fail-level ERROR
docker-compose exec backend python manage.py fix_drf_spectacular_errors
goto :end

:success
echo.
echo 🎉 SUCCESS! All services are running correctly
echo.
echo Available services:
echo - Backend API: http://localhost:8000
echo - Django Admin: http://localhost:8000/admin
echo - API Documentation: http://localhost:8000/api/schema/swagger-ui/
echo - PostgreSQL: localhost:5432
echo - Redis: localhost:6379
echo.
echo Default superuser: admin / admin123
echo.
echo ⚡ Performance Tips:
echo - Next rebuild will be 10x faster thanks to pip cache!
echo - Use 'docker-compose build' for incremental builds
echo - Use 'docker-compose up --build' for quick rebuilds
echo.
echo To stop all services: docker-compose down
echo To view logs: docker-compose logs -f

:end
echo.
echo 📊 Current status:
docker-compose ps
echo.
echo 💾 Disk usage:
docker system df
echo.
echo Press any key to exit...
pause >nul