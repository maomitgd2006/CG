#!/usr/bin/env bash
# CalibGuard Bench 训练进程守卫（云端长时间训练用，2026-10-02）
#
# 作用：把训练队列放进守卫循环里跑——
#   * 进程崩溃 / 被 OOM 杀掉 → 自动清理硬崩溃残留的暂存区 models/，重启训练（最多 N 次）
#     （harness 要求「每实验开始前 models/ 干净」，硬崩溃不会走归档逻辑，必须守卫兜底）
#   * 训练日志长时间不增长 → 写告警（默认只告警；CG_GUARD_KILL_ON_STALL=1 时强杀重启）
#   * 已完成实验由 harness 依据 experiments.yaml 的 status 自动跳过，重启不会白跑
#
# 用法：
#   nohup bash guard_train.sh > /dev/null 2>&1 &     # 后台守护（推荐）
#   tail -f guard_train.log train_queue.log          # 看进度
#   bash guard_train.sh                             # 前台运行（Ctrl+C 只停守卫）
#
# 可选环境变量：
#   CG_GUARD_MAX_RESTART     最大重启次数（默认 8）
#   CG_GUARD_STALL_SECONDS   日志无增长多久算卡死（默认 3600 秒）
#   CG_GUARD_POLL_SECONDS    巡检间隔（默认 60 秒）
#   CG_GUARD_KILL_ON_STALL   1 = 卡死时杀掉训练进程并重启（默认 0，只告警）
#
# 注意：本脚本必须保存为 LF 换行。
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG="guard_train.log"
TRAIN_LOG="train_queue.log"
MAX_RESTART="${CG_GUARD_MAX_RESTART:-8}"
STALL_LIMIT="${CG_GUARD_STALL_SECONDS:-3600}"
POLL="${CG_GUARD_POLL_SECONDS:-60}"
KILL_ON_STALL="${CG_GUARD_KILL_ON_STALL:-0}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 已 trained 实验数（用于判断「某实验刚做完」）
trained_count() { grep -c '^ *status: trained' experiments.yaml 2>/dev/null || echo 0; }

# 实验间清理：harness 严格串行，某实验落 status=trained 后其 _tmp 检查点即可回收
cleanup_tmp() {
    local n
    n=$(find results -maxdepth 2 -type d -name '*_tmp' 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        find results -maxdepth 2 -type d -name '*_tmp' -exec rm -rf {} + 2>/dev/null || true
        log "[guard] 已回收 $n 个训练中间检查点目录（*_tmp）"
    fi
}

if [ ! -x .venv/bin/python ]; then
    log "[ERROR] 未找到 .venv/bin/python：请先运行 bash setup.sh"
    exit 1
fi

log "[guard] 启动：pid=$$，最大重启 $MAX_RESTART 次，卡死阈值 $((STALL_LIMIT / 60)) 分钟"

restarts=0
while :; do
    # 0) 清理上一轮残留进程（2026-10-02 教训：kill 父 harness 后，hposearch/train.py
    #    子进程会被孤儿化继续跑，抢占 GPU 并写同一个 models/hpo/ 目录）
    if pgrep -f '[h]arness.py' >/dev/null 2>&1 || pgrep -f '[t]rain/train.py' >/dev/null 2>&1; then
        log "[guard] 检测到上一轮残留训练进程 → 清理（防孤儿抢 GPU/写坏产物）"
        pkill -9 -f '[h]arness.py' 2>/dev/null || true
        pkill -9 -f '[t]rain/train.py' 2>/dev/null || true
        sleep 3
    fi

    # 1) 兜底清理硬崩溃残留的暂存区（正常失败时 harness 会自行归档，这里只处理 SIGKILL 类）
    if [ -d models ] && [ -n "$(ls -A models 2>/dev/null)" ]; then
        dest="results/_staging_orphan_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$dest" && mv models/* "$dest"/ 2>/dev/null || true
        log "[guard] 暂存区 models/ 有残留 → 已归档到 $dest（硬崩溃恢复）"
    fi

    # 2) 启动训练队列（输出追加到 train_queue.log）
    log "[guard] 启动训练队列（第 $((restarts + 1)) 次）"
    bash train-bench.sh >> "$TRAIN_LOG" 2>&1 &
    pid=$!
    log "[guard] 训练 PID=$pid"

    # 3) 看护循环
    last_size=0
    stall=0
    trained_prev=$(trained_count)
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL"
        size=$(wc -c < "$TRAIN_LOG" 2>/dev/null || echo 0)
        if [ "$size" = "$last_size" ]; then
            stall=$((stall + POLL))
            if [ "$stall" -ge "$STALL_LIMIT" ]; then
                log "[guard][WARN] 训练日志已 $((stall / 60)) 分钟无增长（可能卡住，也可能是长实验无输出）"
                if [ "$KILL_ON_STALL" = "1" ]; then
                    log "[guard] KILL_ON_STALL=1 → 终止训练进程准备重启"
                    kill -TERM "$pid" 2>/dev/null || true
                    sleep 20
                    kill -KILL "$pid" 2>/dev/null || true
                fi
                stall=0
            fi
        else
            stall=0
            last_size=$size
        fi
        # 有实验刚完成 → 回收其 _tmp 检查点（harness 串行，此刻无人写）
        trained_now=$(trained_count)
        if [ "$trained_now" -gt "$trained_prev" ]; then
            trained_prev=$trained_now
            cleanup_tmp
        fi
        # 磁盘守卫：低于 8GB 告警（避免训练中途写满）
        free_gb=$(df -Pk . | awk 'NR==2 {printf "%d", $4/1024/1024}')
        if [ "$free_gb" -lt 8 ]; then
            log "[guard][WARN] 磁盘剩余 ${free_gb}GB（< 8GB），请关注 results/ 体积"
        fi
    done

    wait "$pid"
    code=$?
    log "[guard] 训练进程退出，退出码 $code"
    if [ "$code" -eq 0 ]; then
        log "[guard] 训练队列正常结束，守卫退出。"
        exit 0
    fi

    restarts=$((restarts + 1))
    if [ "$restarts" -ge "$MAX_RESTART" ]; then
        log "[guard] 重启次数达上限 $MAX_RESTART，守卫停止（请查看 results/*/error.log 与 train_queue.log）"
        exit 1
    fi
    log "[guard] 30 秒后重启（已 trained 的实验会自动跳过）..."
    sleep 30
done
