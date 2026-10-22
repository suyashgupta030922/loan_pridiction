@echo off
echo ========================================================
echo   Smart Loan Prediction System - Running Automated Tests
echo ========================================================
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)
pytest tests\test_app.py -v
pause
