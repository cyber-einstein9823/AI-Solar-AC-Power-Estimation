@echo off
where code >nul 2>&1
if errorlevel 1 (
  echo Open VS Code, choose File then Open Folder, and select this folder.
  pause
  exit /b 1
)
call code --new-window "%~dp0AML_Assignment_1.code-workspace"
