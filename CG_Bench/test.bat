@echo off
rem CalibGuard Bench acceptance tests (TH-8). Artifact-level tests skip until v2.5 products exist.
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo [ERROR] .venv not found. Run setup.bat first.
    exit /b 1
)
set "PYTHONPATH=."
.venv\Scripts\python.exe -m pytest tests/ -v
exit /b %ERRORLEVEL%
