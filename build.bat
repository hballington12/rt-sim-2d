@echo off
REM Build script for Ray Tracing App (Windows)
REM Creates a standalone executable using PyInstaller

echo Building Ray Tracing App...
echo ================================

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build with PyInstaller
echo Building executable...
pyinstaller ray_tracing_app.spec

REM Check if build was successful
if exist "dist\RayTracingApp.exe" (
    echo.
    echo Build successful!
    echo ================================
    echo Executable created at:
    echo    dist\RayTracingApp.exe
    echo.
    echo To run:
    echo    dist\RayTracingApp.exe
    echo.
    echo To distribute:
    echo    1. Copy the entire dist folder
    echo    2. Share with users
    echo    3. Users can run RayTracingApp.exe
    echo ================================
) else (
    echo.
    echo Build failed! Check the output above for errors.
    exit /b 1
)

pause
