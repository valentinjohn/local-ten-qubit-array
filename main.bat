@echo off
cd /d "%~dp0"

REM Check for Python Launcher
py --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python Launcher not found. Please install Python from https://www.python.org/downloads/
    pause
    exit /b
)

REM Check for required Python version
py -3.10--version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python 3.10 not found. Please install it from https://www.python.org/downloads/
    pause
    exit /b
)

REM Choose Python version explicitly using the py launcher
echo Creating virtual environment with Python 3.10...
py -3.10 -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip and install dependencies using the venv's Python
echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Run your main script
echo Running main script to plot all main figures...
python main.py
echo Main figures plotted and saved.

echo All done!
pause
