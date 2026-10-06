#!/usr/bin/env bash
# CalibGuard Bench 环境配置（Linux 云端版，对应 setup.bat）
#
# 行为：
#   1. 创建 .venv —— 若 .venv 存在但缺少 bin/python（例如从 Windows 拷贝过来的
#      Scripts\python.exe 布局，或损坏），自动删除并重建（--force 可强制重建）；
#   2. 升级 pip（失败回退清华镜像，与 setup.bat 一致）；
#   3. 按 nvidia-smi 的 CUDA 版本选 torch wheel（>=12.8 → cu128；12.4–12.7 → cu124；
#      无 GPU 或 compute capability < 7 的老卡 → CPU wheel）；
#   4. 安装 requirements.txt（失败回退清华镜像）；
#   5. 自检 import torch。
#
# 用法：
#   bash setup.sh           正常安装（.venv 可用则跳过创建）
#   bash setup.sh --force   删除并重建 .venv
#   可选环境变量：
#     PYTHON_BIN              指定解释器，如 python3.12
#     CG_REUSE_SYSTEM_TORCH=1 云端模式：不装 torch，改为 --system-site-packages 建 venv，
#                             复用系统环境（如 conda）里已装的 CUDA torch。适用于
#                             AutoDL 等「无卡模式」下装环境（此时 nvidia-smi 不可用，
#                             正常流程会误装 CPU torch；复用系统 torch 可避免）。
#
# 注意：本脚本必须保存为 LF 换行，否则 bash 会因 \r 报「bad interpreter」。
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$PWD"

FORCE=0
if [ "${1:-}" = "--force" ]; then
    FORCE=1
fi
REUSE_SYS_TORCH=0
if [ "${CG_REUSE_SYSTEM_TORCH:-0}" = "1" ]; then
    REUSE_SYS_TORCH=1
fi

if [ ! -f requirements.txt ]; then
    echo "[ERROR] 未找到 requirements.txt：请在项目根目录运行本脚本。" >&2
    exit 1
fi

PIP=(.venv/bin/python -m pip)
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# ---- Step 1: 创建（或重建）venv ----
if [ "$FORCE" = "1" ] && [ -e .venv ]; then
    echo "[INFO] --force：删除已有 .venv 并重建 ..."
    rm -rf .venv
fi

if [ -x .venv/bin/python ]; then
    echo "[INFO] .venv 已存在且可用，跳过创建（如需重建：bash setup.sh --force）"
else
    if [ -e .venv ]; then
        echo "[WARN] .venv 存在但缺少 bin/python（Windows 布局或损坏）→ 删除并重建"
        rm -rf .venv
    fi
    PY_BIN="${PYTHON_BIN:-}"
    if [ -z "$PY_BIN" ]; then
        # 云端为 Python 3.12；依次回退 3.10 / 3.11 / 通用 python3
        for candidate in python3.12 python3.10 python3.11 python3; do
            if command -v "$candidate" >/dev/null 2>&1; then
                PY_BIN="$candidate"
                break
            fi
        done
    fi
    if [ -z "$PY_BIN" ]; then
        echo "[ERROR] 未找到 python3.12/3.10/3.11/python3，请安装 Python 3.10+ 或用 PYTHON_BIN 指定。" >&2
        exit 1
    fi
    if [ "$REUSE_SYS_TORCH" = "1" ]; then
        echo "[INFO] 云端复用模式：使用 $PY_BIN 创建 .venv（--system-site-packages，复用系统 torch）..."
        "$PY_BIN" -m venv .venv --system-site-packages
    else
        echo "[INFO] 使用 $PY_BIN 创建虚拟环境 .venv ..."
        "$PY_BIN" -m venv .venv
    fi
    if [ ! -x .venv/bin/python ]; then
        echo "[ERROR] .venv 创建失败：请检查 venv 模块（Debian/Ubuntu 需 apt install python3-venv）。" >&2
        exit 1
    fi
fi

# ---- Step 2: 升级 pip ----
"${PIP[@]}" install --upgrade pip || {
    echo "[WARN] pip 升级失败，改用清华镜像重试 ..."
    "${PIP[@]}" install --upgrade pip -i "$MIRROR"
}

# ---- Step 3: 按 CUDA 版本选 torch wheel ----
TORCH_INDEX=""
if [ "$REUSE_SYS_TORCH" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDAVER="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: *[0-9.]*' | head -n1 | awk '{print $3}')"
    if [ -n "${CUDAVER:-}" ]; then
        MAJOR="${CUDAVER%%.*}"
        REST="${CUDAVER#*.}"
        MINOR="${REST%%.*}"
        if [ "$MAJOR" -ge 13 ] 2>/dev/null; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        elif [ "$MAJOR" -eq 12 ] 2>/dev/null; then
            if [ "$MINOR" -ge 8 ] 2>/dev/null; then
                TORCH_INDEX="https://download.pytorch.org/whl/cu128"
            else
                TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            fi
        fi
    fi
    # 老卡守卫（与 setup.bat 一致）：compute capability < 7 的 pre-Volta GPU
    # 已从 cu12x wheel 移除 → 退回 CPU wheel
    COMPCAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
    if [ -n "${COMPCAP:-}" ]; then
        CC_MAJOR="${COMPCAP%%.*}"
        if [ "$CC_MAJOR" -lt 7 ] 2>/dev/null; then
            echo "[INFO] GPU compute capability ${COMPCAP} < 7.0（pre-Volta）→ 使用 CPU wheel。"
            TORCH_INDEX=""
        fi
    fi
fi

# ---- Step 4: 安装 torch ----
if [ "$REUSE_SYS_TORCH" = "1" ]; then
    echo "[INFO] 云端复用模式：跳过 torch 安装，使用系统环境（conda）已装的 CUDA torch。"
elif [ -n "$TORCH_INDEX" ]; then
    echo "[INFO] 检测到 NVIDIA GPU（CUDA ${CUDAVER:-未知}）→ 从 $TORCH_INDEX 安装 CUDA torch"
    if ! "${PIP[@]}" install --force-reinstall torch --index-url "$TORCH_INDEX"; then
        echo "[WARN] CUDA torch 安装失败，重试一次 ..."
        if ! "${PIP[@]}" install --force-reinstall torch --index-url "$TORCH_INDEX"; then
            echo "[WARN] CUDA torch 仍失败：将由下一步安装 PyPI 默认 torch。"
        fi
    fi
else
    echo "[INFO] 未检测到可用 NVIDIA GPU（或为 pre-Volta 老卡）→ 安装 CPU torch wheel。"
fi

# ---- Step 5: 安装其余依赖 ----
if ! "${PIP[@]}" install -r requirements.txt; then
    echo "[WARN] 依赖安装失败，改用清华镜像重试 ..."
    if ! "${PIP[@]}" install -r requirements.txt -i "$MIRROR"; then
        echo "[ERROR] 依赖安装失败：请检查网络后重试。" >&2
        exit 1
    fi
fi

# ---- Step 6: 自检 ----
.venv/bin/python -c "import torch; print('[INFO] torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
echo "[OK] 环境就绪。用法：bash train-bench.sh（训练队列）/ python harness.py report（报告）"
