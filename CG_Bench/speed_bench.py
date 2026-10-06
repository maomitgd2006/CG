"""CalibGuard Bench 速度基准（TH-6，开发指导文档第 5 节协议逐字实现）。

用法：python speed_bench.py --exp {name} [--threads N]
读 results/_splits/split_test.jsonl（唯一划分源）抽样，输出 results/{name}/speed.json。

协议要点（禁止省略任何要素，铁律 5）：
  1. torch.set_num_threads(args.threads)——默认 os.cpu_count()，对比时必须同值，记入环境戳；
  2. 加载模型 → warmup：跑 3 个 medium 文件，不计时；
  3. 逐档逐文件 time.perf_counter() 计时，记录 (档, chars, windows, sec)——
     测量对象是【全链路】：preprocess_file + 滑窗推理 + 校准；
  4. 分档（清洗后文本长度）：short < 4,000 / medium 4,000–64,000 / long > 64,000，
     每档 20 条（不足取全）；LLM 方案每档 5 条（网络成本），输出 network=true；
  5. 中位数与 p90 用 statistics.median / sorted[int(0.9*(n-1))]，禁止用均值做主指标。
"""
import argparse
import datetime
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time

import torch
import yaml

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from calibguard.calibrator import PlattCalibrator        # noqa: E402
from calibguard.config import load_config                # noqa: E402
from calibguard.model import CodeBERTClassifier          # noqa: E402
from calibguard.preprocessor import preprocess_file      # noqa: E402

SPLIT_FILE = os.path.join("results", "_splits", "split_test.jsonl")
SAMPLE_SEED = 42                       # 固定抽样种子（协议 5.2）
SAMPLES_PER_BUCKET = 20                # ft / rules / hybrid
SAMPLES_PER_BUCKET_LLM = 5             # LLM 方案（网络成本）
WARMUP_N = 3                           # 协议 5.3：跑 3 个 medium 文件，不计时
SHORT_MAX = 4000                       # 分档阈值（清洗后字符数）
LONG_MIN = 64000
TEXT_MAX_CHARS = 12000                 # LLM 方案的 12,000 字符截断（禁止事项 4）
BUCKET_ORDER = ("short", "medium", "long")


def iso_now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 环境戳
def cpu_name() -> str:
    """<wmic cpu get name 取值>；wmic 不可用时回退 CIM / platform。"""
    try:
        out = subprocess.run(["wmic", "cpu", "get", "name"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            lines = [line.strip() for line in out.stdout.splitlines()
                     if line.strip() and line.strip().lower() != "name"]
            if lines:
                return lines[0]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def gpu_name():
    """<nvidia-smi name 或 null>。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def env_stamp(threads: int) -> dict:
    return {"cpu": cpu_name(), "threads": threads, "gpu": gpu_name(),
            "torch": torch.__version__, "date": iso_now()}


# ---------------------------------------------------------------- 实验配置
def load_experiment(name: str) -> dict:
    with open(os.path.join(_ROOT, "experiments.yaml"), "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    for exp in data.get("experiments") or []:
        if exp.get("name") == name:
            return exp
    raise SystemExit(f"[错误] experiments.yaml 中不存在实验：{name}")


def _paths_of(exp: dict) -> tuple:
    name = exp["name"]
    model_dir = exp.get("model_dir") or f"results/{name}/codebert_finetuned"
    params = exp.get("params") or f"results/{name}/calibration_params.json"
    return model_dir, params


# ---------------------------------------------------------------- 抽样
def read_split(path: str) -> list:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bucket_of(chars: int) -> str:
    if chars < SHORT_MAX:
        return "short"
    if chars > LONG_MIN:
        return "long"
    return "medium"


def sample_buckets(records: list, per_bucket: int) -> dict:
    """按清洗后文本长度分档并抽样（seed 42；同一条数据在任何实验里抽到的样本一致）。"""
    buckets = {name: [] for name in BUCKET_ORDER}
    for record in records:
        try:
            text = preprocess_file(record["path"]).text
        except Exception as exc:                    # noqa: BLE001 —— 单条失败只跳过
            print(f"跳过无法预处理的样本: {record['path']}（{type(exc).__name__}: {exc}）")
            continue
        bucket = bucket_of(len(text))
        buckets[bucket].append((record["path"], len(text)))
    rng = random.Random(SAMPLE_SEED)
    picked = {}
    for name in BUCKET_ORDER:
        items = sorted(buckets[name], key=lambda item: item[0])
        rng.shuffle(items)
        picked[name] = items[:per_bucket]
        print(f"[speed_bench] 档 {name}: 候选 {len(buckets[name])} 条 → 抽 {len(picked[name])} 条")
    return picked


# ---------------------------------------------------------------- 测量
def _latency_stats(seconds: list) -> dict:
    """中位数 + p90（sorted[int(0.9*(n-1))]，禁止用均值做主指标）。"""
    if not seconds:
        return {"median": None, "p90": None}
    ordered = sorted(seconds)
    return {"median": round(statistics.median(ordered), 4),
            "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 4)}


def run_measurement(exp: dict, picked: dict, threads: int) -> dict:
    """按实验类型加载方案并逐档逐文件计时（全链路）。"""
    name = exp["name"]
    etype = exp.get("type")
    model_dir, params = _paths_of(exp)
    cfg = load_config(None)

    measure_fn = None
    window_fn = None
    network = False

    if etype == "rules":
        from adapters import rules_adapter
        measure_fn = lambda text: rules_adapter.classify(text)        # noqa: E731
        network = False
    elif etype == "llm":
        from adapters import llm_adapter
        variant = str(exp.get("variant", "zero"))
        measure_fn = lambda text: llm_adapter.classify(                  # noqa: E731
            text[:TEXT_MAX_CHARS], variant)
        network = True
    elif etype == "hybrid":
        from adapters import hybrid_adapter
        measure_fn = lambda text: hybrid_adapter.classify(text, model_dir, params)  # noqa: E731
        window_fn = lambda text: hybrid_adapter.window_count(text, model_dir, params)  # noqa: E731
        network = True                     # llm_review 桶会发起 API 调用
    else:                                   # ft（含已训版本对照）
        if not os.path.isdir(model_dir):
            raise SystemExit(f"[错误] 模型目录不存在：{model_dir}")
        classifier = CodeBERTClassifier(
            model_dir, cfg.max_length,
            window_stride=cfg.window_stride, window_batch_size=cfg.window_batch_size)
        calibrator = PlattCalibrator.load(params)

        def measure_fn(text, classifier=classifier, calibrator=calibrator):
            logit = classifier.predict_logit(text)
            return {"prob": calibrator.predict_proba(logit)}

        def window_fn(text, classifier=classifier):
            token_ids = classifier.tokenizer.encode(text, add_special_tokens=False)
            return len(classifier.window_starts(len(token_ids)))

    # 协议 5.3 第 2 步：warmup 跑 3 个 medium 文件，不计时
    warmup_pool = picked.get("medium") or picked.get("short") or picked.get("long") or []
    for path, _chars in warmup_pool[:WARMUP_N]:
        try:
            measure_fn(preprocess_file(path).text)
        except Exception as exc:                    # noqa: BLE001
            print(f"[speed_bench] warmup 跳过 {path}：{type(exc).__name__}: {exc}")
    print(f"[speed_bench] warmup 完成（{min(WARMUP_N, len(warmup_pool))} 个文件，未计入）")

    per_file = []
    for bucket in BUCKET_ORDER:
        for path, chars in picked.get(bucket, []):
            try:
                start = time.perf_counter()
                text = preprocess_file(path).text          # 预处理（全链路第 1 段）
                measure_fn(text)                           # 滑窗推理 + 校准
                seconds = time.perf_counter() - start
            except Exception as exc:                       # noqa: BLE001
                print(f"[speed_bench] 计时跳过 {path}：{type(exc).__name__}: {exc}")
                continue
            windows = window_fn(text) if window_fn else 0
            per_file.append({"bucket": bucket, "chars": chars, "windows": windows,
                             "sec": round(seconds, 4)})
            print(f"[speed_bench] {bucket:6s} chars={chars:7d} windows={windows:4d} "
                  f"sec={seconds:.4f}")

    total_seconds = sum(item["sec"] for item in per_file)
    total_windows = sum(item["windows"] for item in per_file)
    latency = {}
    for bucket in BUCKET_ORDER:
        latency[bucket] = _latency_stats([item["sec"] for item in per_file
                                          if item["bucket"] == bucket])
    return {
        "env": env_stamp(threads),
        "files_per_min_overall": (round(60.0 * len(per_file) / total_seconds, 4)
                                  if total_seconds else None),
        "windows_per_sec_mean": (round(total_windows / total_seconds, 4)
                                 if total_seconds and total_windows else None),
        "latency_sec": latency,
        "per_file": per_file,
        "network": network,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CalibGuard Bench 速度基准（TH-6）")
    parser.add_argument("--exp", required=True, help="实验名（experiments.yaml 的 name）")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch 线程数（默认 os.cpu_count()；对比时必须同值）")
    args = parser.parse_args(argv)

    os.chdir(_ROOT)
    threads = args.threads if args.threads is not None else os.cpu_count()
    torch.set_num_threads(threads)          # 协议 5.3 第 1 步：线程锁定
    print(f"[speed_bench] 实验 {args.exp} | threads={threads}（默认 os.cpu_count()）")

    exp = load_experiment(args.exp)
    if not os.path.isfile(SPLIT_FILE):
        print(f"[错误] 分段清单不存在：{SPLIT_FILE}（请先运行 python harness.py init）",
              file=sys.stderr)
        return 1
    per_bucket = SAMPLES_PER_BUCKET_LLM if exp.get("type") == "llm" else SAMPLES_PER_BUCKET
    records = read_split(SPLIT_FILE)
    print(f"[speed_bench] test 段 {len(records)} 条，按清洗后长度分档抽样"
          f"（seed {SAMPLE_SEED}，每档 {per_bucket} 条）")
    picked = sample_buckets(records, per_bucket)

    print("[speed_bench] 加载方案并测量（全链路：预处理 + 滑窗推理 + 校准）")
    payload = run_measurement(exp, picked, threads)

    out_dir = os.path.join("results", args.exp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "speed.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("\n===== 速度基准结果 =====")
    print(f"env: {payload['env']}")
    print(f"files/min: {payload['files_per_min_overall']} | "
          f"windows/s: {payload['windows_per_sec_mean']} | network: {payload['network']}")
    for bucket in BUCKET_ORDER:
        info = payload["latency_sec"][bucket]
        print(f"  {bucket:6s} median={info['median']} p90={info['p90']}")
    print(f"结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
