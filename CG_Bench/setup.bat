@echo off
rem CalibGuard Bench environment setup: create .venv and install dependencies.
rem GPU tier selection follows main doc 16.6 step 4 (nvidia-smi CUDA version cap).
rem RULE: never install project dependencies into the global Python environment.
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist requirements.txt (
    echo [ERROR] Run this script from the project root directory.
    exit /b 1
)

rem ---- Step 1: create .venv (skip if present) ----
rem 2026-10-02: prefer py launcher 3.10/3.12 (cloud parity) over bare "python" (may resolve to 3.14)
if not exist .venv\Scripts\python.exe (
    echo [INFO] Creating virtual environment .venv ...
    set "VENV_PY="
    py -3.10 --version >nul 2>&1 && set "VENV_PY=py -3.10"
    if not defined VENV_PY py -3.12 --version >nul 2>&1 && set "VENV_PY=py -3.12"
    if not defined VENV_PY py -3.11 --version >nul 2>&1 && set "VENV_PY=py -3.11"
    if defined VENV_PY (
        echo [INFO] Using !VENV_PY! for the virtual environment.
        !VENV_PY! -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv. Check that Python 3.10/3.12 is installed.
        exit /b 1
    )
) else (
    echo [INFO] .venv already exists, skipping creation.
)

rem ---- Step 2: upgrade pip (mirror retry on failure) ----
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARN] pip upgrade failed, retrying with Tsinghua mirror ...
    .venv\Scripts\python.exe -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
)

rem ---- Step 3: GPU tier selection (main doc 16.6 step 4) ----
rem Read the "CUDA Version" cap from nvidia-smi: >=12.8 -> cu128; 12.4-12.7 -> cu124; lower or no GPU -> CPU wheel.
set "CUDAVER="
set "TORCH_INDEX="
for /f "tokens=3 delims=:" %%a in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do set "CUDAVER=%%a"
if defined CUDAVER for /f "tokens=1 delims= " %%b in ("!CUDAVER!") do set "CUDAVER=%%b"
if defined CUDAVER (
    for /f "tokens=1,2 delims=." %%x in ("!CUDAVER!") do (
        if %%x GEQ 13 set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
        if %%x EQU 12 if %%y GEQ 8 set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
        if %%x EQU 12 if %%y LSS 8 set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
    )
)

rem ---- Step 3b: legacy GPU guard (2026-10-02): pre-Volta GPUs (compute cap < 7.0, e.g. GTX 1060)
rem are dropped from cu12x wheels; this machine is data-prep only (training moved to cloud) -> CPU wheel.
set "COMPCAP="
for /f "tokens=1 delims=. " %%c in ('nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2^>nul') do set "COMPCAP=%%c"
if not "!COMPCAP!"=="" (
    if !COMPCAP! LSS 7 (
        echo [INFO] GPU compute capability !COMPCAP!.x ^< 7.0 - pre-Volta GPU, using CPU wheel. Local machine is data-prep only (training moved to cloud, 2026-10-02).
        set "TORCH_INDEX="
    )
)

rem ---- Step 4: install torch (CUDA wheel first; plain "pip install torch" on Windows gets the CPU wheel) ----
set "TORCH_OK="
if defined TORCH_INDEX (
    echo [INFO] NVIDIA GPU detected, CUDA cap !CUDAVER! -^> installing CUDA torch from !TORCH_INDEX!
    .venv\Scripts\python.exe -m pip install --force-reinstall torch --index-url !TORCH_INDEX!
    if not errorlevel 1 set "TORCH_OK=1"
    if not defined TORCH_OK (
        echo [WARN] CUDA torch install failed, retrying once ...
        .venv\Scripts\python.exe -m pip install --force-reinstall torch --index-url !TORCH_INDEX!
        if not errorlevel 1 set "TORCH_OK=1"
    )
    if not defined TORCH_OK (
        echo [WARN] CUDA torch still failing: falling back to the CPU wheel installed by the next step.
    )
) else (
    echo [INFO] No NVIDIA CUDA cap ^>= 12.4 detected, installing the CPU torch wheel.
)

rem ---- Step 5: install the remaining dependencies (mirror retry on failure) ----
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [WARN] Dependency install failed, retrying with Tsinghua mirror ...
    .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Check network and retry.
        exit /b 1
    )
)

rem ---- Step 6: self check ----
.venv\Scripts\python.exe -c "import torch; print('[INFO] torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
echo [OK] Environment ready. Usage: train-bench.bat / bench.bat / report.bat / test.bat
exit /b 0
