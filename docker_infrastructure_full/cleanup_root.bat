@echo off
REM Скрипт очистки корня для Windows
REM Использование: cleanup_root.bat

echo ============================================
echo   ОЧИСТКА КОРНЯ ПРОЕКТА
echo ============================================
echo.

set /p confirm="Удалить старые Docker файлы? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Отменено
    exit /b
)

echo.
echo [1/3] Удаляем старые Docker файлы...

del docker-compose.yml 2>nul
del docker-compose.dev.yml 2>nul
del docker-compose.prod.yml 2>nul
del docker-compose.override.yml 2>nul
del Dockerfile 2>nul
del Dockerfile.light 2>nul
del .dockerignore 2>nul
del Makefile.docker 2>nul
del .env.dev.example 2>nul
del .env.prod.example 2>nul

echo [2/3] Удаляем пустые директории...

rmdir /q docker 2>nul
rmdir /q deploy 2>nul
rmdir /q certs 2>nul
rmdir /q logs 2>nul
rmdir /q models 2>nul

echo [3/3] Удаляем устаревшие конфиги...

del .bandit 2>nul
del .editorconfig 2>nul
del .eslintrc.json 2>nul
del .prettierrc 2>nul
del package-lock.json 2>nul

echo.
echo ============================================
echo   ОЧИСТКА ЗАВЕРШЕНА
echo ============================================
echo.
echo Следующие шаги:
echo   1. Распакуйте docker_infrastructure.zip
echo   2. Настройте .env
echo   3. Запустите: make build ^&^& make up
echo.
pause
