@echo off
rem CalibGuard Bench one-command training queue (experiments run strictly one at a time).
rem Offline HF cache (patch 7): HF_HOME + HF_HUB_OFFLINE must be set together -- HF_HOME alone
rem still lets hf_hub probe huggingface.co on every load (minutes of wasted timeout).
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
.venv\Scripts\python.exe harness.py train --all %*
exit /b %ERRORLEVEL%
