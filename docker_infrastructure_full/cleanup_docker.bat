@echo off
REM Скрипт очистки для Windows
REM Использование: cleanup_docker.bat

echo ============================================
echo   ОЧИСТКА DOCKER КОНФИГУРАЦИЙ
echo ============================================
echo.

echo [1/7] Останавливаем контейнеры...
docker-compose down -v 2>nul
docker stop $(docker ps -aq) 2>nul
docker rm $(docker ps -aq) 2>nul

echo [2/7] Удаляем старые docker-compose файлы...
for /r %%i in (docker-compose*.yml docker-compose*.yaml) do del "%%i" 2>nul

echo [3/7] Удаляем старые Dockerfile...
for /r .\services %%i in (Dockerfile*) do del "%%i" 2>nul
for /r .\backend %%i in (Dockerfile*) do del "%%i" 2>nul
for /r .\ml_service %%i in (Dockerfile*) do del "%%i" 2>nul
for /r .\rag_service %%i in (Dockerfile*) do del "%%i" 2>nul
del Dockerfile* 2>nul

echo [4/7] Удаляем .dockerignore...
for /r %%i in (.dockerignore) do del "%%i" 2>nul

echo.
set /p cleanup_cache="[5/7] Очистить Docker build cache? (Y/N): "
if /i "%cleanup_cache%"=="Y" (
    echo Очищаем cache...
    docker builder prune -af
)

echo.
set /p cleanup_images="[6/7] Удалить неиспользуемые образы? (Y/N): "
if /i "%cleanup_images%"=="Y" (
    echo Удаляем образы...
    docker image prune -af
)

echo.
set /p cleanup_volumes="[7/7] УДАЛИТЬ VOLUMES (данные БД)? (Y/N): "
if /i "%cleanup_volumes%"=="Y" (
    echo ВНИМАНИЕ! Удаляем volumes...
    docker volume prune -f
)

echo.
echo ============================================
echo   ОЧИСТКА ЗАВЕРШЕНА
echo ============================================
echo.
echo Следующие шаги:
echo   1. Распакуйте docker_infrastructure.zip
echo   2. Запустите: docker-compose up --build
echo.
pause
