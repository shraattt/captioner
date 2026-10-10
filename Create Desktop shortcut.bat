@echo off

REM Creates a "Captioner" shortcut on your Desktop pointing at this folder.

cd /d "%~dp0"

".venv\Scripts\python" scripts\make_shortcut.py

pause

