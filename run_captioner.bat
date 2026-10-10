@echo off
REM Captioner - one-click launcher.
REM First run: creates the virtualenv, installs everything and checks the GPU.
REM Later runs start straight into the app.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] creating virtual environment .venv ...
    python -m venv .venv || goto :fail
    set CAPTIONER_FIRST_RUN=1
)

if defined CAPTIONER_FIRST_RUN (
    echo [setup] installing dependencies - a few minutes, once ever ...
    ".venv\Scripts\python" -m pip install --upgrade pip >nul
    ".venv\Scripts\python" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || goto :fail
    ".venv\Scripts\python" -m pip install -r requirements.txt || goto :fail
    echo [setup] checking the GPU stack ...
    ".venv\Scripts\python" scripts\check_env.py
    echo.
    echo [setup] done. Starting Captioner - the first captioning job also
    echo [setup] downloads model weights, so give it a minute.
    timeout /t 4 >nul
)

".venv\Scripts\python" ui\dashboard.py %*
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
    echo.
    echo [!] Captioner exited with code %EXITCODE%
    pause
)
exit /b %EXITCODE%

:fail
echo.
echo [!] Setup failed - see the messages above.
echo     Most common cause: "python" not on PATH. Install Python 3.11+
echo     from python.org and tick "Add python.exe to PATH".
pause
exit /b 1
