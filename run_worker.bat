@echo off
REM Run the captioning worker in the foreground: processes queued jobs, one
REM after another, without opening the UI. Close the window (or Ctrl+C) to stop.
REM Use the dashboard's "Start worker" button for the in-app equivalent.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [!] No virtualenv found at .venv
    echo     Create it first:  python -m venv .venv
    echo     Then install:     .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Worker starting — Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" cli.py work
echo.
echo Worker stopped.
pause
