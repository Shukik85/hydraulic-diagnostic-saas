@echo off
echo ========================================
echo   OVERNIGHT TRAINING PIPELINE
echo ========================================
echo.

cd /d C:\Users\ShukikPK\hydraulic-diagnostic-saas

echo Activating virtual environment...
call .venv\Scripts\activate.bat

cd ml_service\experiments\ssl_transformer

echo Starting training...
python overnight_training.py

echo.
echo ========================================
echo   Training complete!
echo ========================================
pause
