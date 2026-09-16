param([string]$Python = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    if ($Python) { & $Python -m venv .venv }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv .venv }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { & python -m venv .venv }
    else { throw 'Install Python 3.12 or newer, then run Setup.cmd again.' }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project environment.' }
}
& $projectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your internet connection.' }
& $projectPython -c "import numpy, pandas, matplotlib, requests, streamlit; print('Project environment is ready.')"
if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed.' }
Write-Host 'Open this folder in VS Code, or double-click Run_App.cmd.'
