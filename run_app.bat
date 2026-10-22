@echo off
echo ========================================================
echo   Smart Loan Prediction System - Launching Web Application
echo ========================================================
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)
echo Starting Flask server on http://127.0.0.1:5000
python app.py
pause
