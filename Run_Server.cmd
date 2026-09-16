@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run_App.cmd opens the website without Python.
  echo For the optional Python server or retraining, run Setup.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app\html_server.py --port 8503 --open
if errorlevel 1 pause
