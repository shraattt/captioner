@echo off
REM Double-click to launch the Captioner dashboard (batch captioning UI).
REM Keeps this console open so any startup error is visible; the app window
REM is the separate Qt window that appears on top.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [!] No virtualenv found at .venv
    echo     Create it first:
    echo       python -m venv .venv
    echo       .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "ui\dashboard.py" %*
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
    echo.
    echo [!] The dashboard exited with code %EXITCODE%
)
pause
