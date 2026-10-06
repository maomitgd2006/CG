@echo off
rem CalibGuard Bench one-command benchmark: speed bench for every trained experiment, then report.
rem Offline HF cache (patch 7): HF_HOME + HF_HUB_OFFLINE must be set together.
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo [ERROR] .venv not found. Run setup.bat first.
    exit /b 1
)
if exist .hf_cache (
    set "HF_HOME=%~dp0.hf_cache"
    set "HF_HUB_OFFLINE=1"
)
.venv\Scripts\python.exe harness.py speed --all && .venv\Scripts\python.exe harness.py report
exit /b %ERRORLEVEL%
