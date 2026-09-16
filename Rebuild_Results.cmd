@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run Setup.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_pipeline.py
pause
