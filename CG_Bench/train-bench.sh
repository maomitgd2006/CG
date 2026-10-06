#!/usr/bin/env bash
# CalibGuard Bench 训练队列启动（Linux 云端版，对应 train-bench.bat）
#
# 严格按 experiments.yaml 的顺序串行训练所有 pending 实验（harness 自动区分
# 「训练 + 评测」与「仅评测」：*_ft/*_shared/hybrid 走完整 ft 流程；
#  v0.2.2_baseline 仅补跑步骤 4-6；llm_*/rules_baseline 仅评测）。
#
# 离线 HF 缓存（补丁 7）：HF_HOME 与 HF_HUB_OFFLINE 必须同时设置，
# 只设 HF_HOME 时 hf_hub 仍会探测 huggingface.co（每次加载浪费数分钟超时）。
#
# 用法：
#   bash train-bench.sh            训练全部 pending 实验
#   bash train-bench.sh --dry-run  只打印命令序列，不执行
#   bash train-bench.sh --only codebert_ft  只跑指定实验
#   bash train-bench.sh --epochs 1 --batch-size 4   覆盖超参（冒烟用）
#
# 注意：本脚本必须保存为 LF 换行。
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x .venv/bin/python ]; then
    echo "[ERROR] 未找到 .venv/bin/python：请先运行 bash setup.sh" >&2
    exit 1
fi

if [ -d .hf_cache ]; then
    export HF_HOME="$PWD/.hf_cache"
    export HF_HUB_OFFLINE=1
fi

exec .venv/bin/python harness.py train --all "$@"
