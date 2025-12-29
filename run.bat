@echo off
chcp 65001 >nul

echo ======================================================================
echo FunGen Live Screen - Using Original FunGen Code
echo ======================================================================
echo.
echo This tool uses FunGen's original tracking and detection modules
echo for real-time screen capture analysis.
echo.
echo Features:
echo   [*] TrackerManager (FunGen's real-time tracking engine)
echo   [*] DualAxisFunscript (FunGen's script generator)
echo   [*] YOLO Detection (FunGen's detection models)
echo   [*] Multiple tracking modes
echo   [*] Real-time funscript generation
echo.
echo ======================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    pause
    exit /b 1
)

REM Check FunGen directory
if not exist "C:\Users\17798\Desktop\porntest\FunGen-AI-Powered-Funscript-Generator" (
    echo Error: FunGen directory not found
    echo Expected: C:\Users\17798\Desktop\porntest\FunGen-AI-Powered-Funscript-Generator
    pause
    exit /b 1
)

REM Check dependencies
echo Checking dependencies...
python -c "import cv2, numpy, mss, ultralytics" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install opencv-python numpy mss ultralytics
)

echo.
echo Starting FunGen Live Screen...
echo.

python fungen_live.py

pause
