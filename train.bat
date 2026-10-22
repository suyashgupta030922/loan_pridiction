@echo off
echo ========================================================
echo   Smart Loan Prediction System - Training ML Pipeline
echo ========================================================
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)
python train_model.py
pause
