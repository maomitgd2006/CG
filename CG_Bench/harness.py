"""CalibGuard Bench 实验矩阵驱动（TH-2）。

子命令：
  init   把 models_baseline/v0.2.2/split_*.jsonl 复制到 results/_splits/（唯一划分源）
  train  遍历 experiments.yaml 执行未完成实验（--all / --only NAME / --dry-run）
  speed  对全部 trained 实验跑 speed_bench.py
  report 汇总生成 benchmark.json + benchmark_report.md

铁律相关实现要点：
  * 实验严格串行（铁律 7/10）——单进程顺序执行，绝不并行；
  * 失败不吞（铁律 8-7）——失败实验记 failed + 错误写 results/{name}/error.log，继续下一个；
  * 暂存区（补丁 10）——引擎把 split/metrics/evaluation 硬编码写进相对 cwd 的 models/，
    本文件用 models/ 作暂存区，实验七步全部完成后统一搬入 results/{name}/ 并清空；
  * 划分一致性（铁律 1）——train.py 现场重算的 split_*.jsonl 必须与统一划分源逐行一致；
  * API Key 绝不进日志/报告（禁止事项 8）——.env 只注入环境变量，值不回显。
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

import yaml

# ---------------------------------------------------------------- 路径常量
ROOT = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(ROOT, "experiments.yaml")
ENV_PATH = os.path.join(ROOT, ".env")
RESULTS = os.path.join(ROOT, "results")
SPLITS = os.path.join(RESULTS, "_splits")
STAGE = os.path.join(ROOT, "models")                     # 引擎硬编码产物目录 = 暂存区
BASELINE_DIR = os.path.join(ROOT, "models_baseline", "v0.2.2")
HF_CACHE = os.path.join(ROOT, ".hf_cache")
# venv 解释器路径：Windows 为 .venv\Scripts\python.exe，类 Unix（Linux 云端）为 .venv/bin/python
if os.name == "nt":
    PY_REL = os.path.join(".venv", "Scripts", "python.exe")
    SETUP_SCRIPT = "setup.bat"
else:
    PY_REL = os.path.join(".venv", "bin", "python")
    SETUP_SCRIPT = "setup.sh"
PY_ABS = os.path.join(ROOT, PY_REL)

# API 熔断退出码（llm_adapter 的 API 致命错误：404 / 长时间无响应，2026-10-02）
API_FATAL_EXIT_CODE = 4

SPLIT_FILES = ("split_train.jsonl", "split_val.jsonl",
               "split_calibration.jsonl", "split_test.jsonl")
# train_meta.json 的固定五步（逐字字段，见文档 TH-2）
STEP_NAMES = ("hpo", "train", "eval_raw", "calibrate", "eval_calib")
# train.py 的 argparse 默认值（hpo: none 时沿用；显式写进命令行便于追溯）
DEFAULT_LR = 2e-05
DEFAULT_EPOCHS = 3
# 2026-10-02 修订（训练改云端）：batch 策略 = codebert 系 16（与 v0.2.2 机房一致），
# 多基座 LLM（Qwen 等）保持 8；每实验可在 yaml 用 batch: 字段显式声明，此处仅兜底。
DEFAULT_BATCH = 8
API_KEY_ENV = "DEEPSEEK_API_KEY"     # 铁律 6 指定的环境变量名（LLM 方案）
# ---- 补丁 12：既成 HPO 结果的复用（数据版本一致才可复用，否则 fallback 真跑网格）----
BASELINE_DATA_VERSION = "v1"         # models_baseline/v0.2.2 是 v1 数据（15,310 三域）的产物，不随 v2 重建
BASELINE_HPO_REPORT = os.path.join(BASELINE_DIR, "hpo_report.json")
LOCAL_HPO_BATCH = 8                  # 非 codebert 基座 fallback 网格的 _BATCH 覆盖值（引擎文件字节不变）
CODEBERT_BASE_MODEL = "microsoft/codebert-base"
CODEBERT_HPO_BATCH = 16              # codebert 网格：云端与 v0.2.2 机房同 batch（2026-10-02）
DEFAULT_DATA_VERSION = "v2"


def iso_now() -> str:
    """本地时区 ISO8601 时间戳（秒精度）。"""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 环境准备
def load_dotenv() -> list:
    """补丁 9：读项目根 .env（KEY=VALUE 逐行注入 os.environ，已有环境变量优先不覆盖）。

    返回被注入的变量名列表（只返回键名，绝不回显值）。# 开头与空行跳过。
    """
    injected: list = []
    if not os.path.isfile(ENV_PATH):
        return injected
    with open(ENV_PATH, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key or os.environ.get(key):
                continue
            os.environ[key] = value
            injected.append(key)
    return injected


def build_env() -> dict:
    """子进程环境：PYTHONPATH 指项目根、UTF-8 IO；.hf_cache 存在时锁定离线 HF（补丁 7）。"""
    env = os.environ.copy()
    if os.path.isdir(HF_CACHE):
        env["HF_HOME"] = HF_CACHE
        env["HF_HUB_OFFLINE"] = "1"
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if ROOT not in paths:
        paths.insert(0, ROOT)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ---------------------------------------------------------------- 矩阵读写
def load_json(path: str):
    """容错读 JSON：不存在/损坏/内容是 null 一律返回 None。"""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_matrix() -> tuple:
    """读 experiments.yaml → (原始 dict, shared_hpo dict, experiments list)。"""
    with open(YAML_PATH, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    experiments = data.get("experiments") or []
    shared = data.get("shared_hpo") or {}
    if shared.get("lr") is None or shared.get("epochs") is None:
        # 兜底：yaml 未回写但 HPO 产物已存在时采纳产物（唯一划分源旁的 shared_hpo.json）
        # 补丁 14：HPO 档案复用须同 data_version，跨版本档案一律忽略（v2 起强制）
        cached = load_json(os.path.join(SPLITS, "shared_hpo.json"))
        if (isinstance(cached, dict) and cached.get("lr") is not None
                and str(cached.get("data_version") or "v1")
                == str(data.get("data_version") or DEFAULT_DATA_VERSION)):
            shared = {"lr": cached.get("lr"), "epochs": cached.get("epochs")}
    return data, shared, experiments


def data_version_of(matrix: dict) -> str:
    """§3.0 数据版本机制（补丁 13）：yaml 顶层必填 data_version（当前 v2 = 两域 code/prompt）。"""
    value = matrix.get("data_version")
    return str(value) if value else DEFAULT_DATA_VERSION


def read_lines(path: str) -> list:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def baseline_splits_match() -> bool:
    """baseline 划分清单与统一划分源逐行一致 ⇒ 同一数据 + 同一 seed 42 划分。

    这是补丁 12「同数据版本才可复用 HPO 档案」的可执行判据（baseline 档案本身不带版本字段）。
    """
    for file_name in SPLIT_FILES:
        canonical = os.path.join(SPLITS, file_name)
        baseline = os.path.join(BASELINE_DIR, file_name)
        if not os.path.isfile(canonical) or not os.path.isfile(baseline):
            return False
        if read_lines(canonical) != read_lines(baseline):
            return False
    return True


def reuse_baseline_hpo(data_version: str):
    """补丁 12 步骤 1：优先复用 models_baseline/v0.2.2/hpo_report.json 的 best。

    复用前置（全部满足才复用）：① 顶层 data_version == BASELINE_DATA_VERSION（baseline 是 v1 产物）；
    ② 档案存在且 best 含 lr/epochs；③ 档案若自带 data_version 则须一致；
    ④ baseline 划分与统一划分源逐行一致。任一不满足 → 返回 None，走 fallback 真跑网格。
    """
    if data_version != BASELINE_DATA_VERSION:
        return None
    report = load_json(BASELINE_HPO_REPORT)
    if not isinstance(report, dict):
        return None
    declared = report.get("data_version")
    if declared is not None and str(declared) != data_version:
        return None
    best = report.get("best") or {}
    if best.get("lr") is None or best.get("epochs") is None:
        return None
    if not baseline_splits_match():
        return None
    return {"lr": best["lr"], "epochs": int(best["epochs"]), "report": report,
            "source": "models_baseline/v0.2.2/hpo_report.json（baseline 复用：同数据/同划分/同网格，"
                      "机房 batch 16 环境）"}


def _read_yaml_lines() -> list:
    with open(YAML_PATH, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def _write_yaml_lines(lines: list) -> None:
    with open(YAML_PATH, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def _fmt_num(value) -> str:
    """数值 → 命令行/yaml 文本。注意 YAML 1.1 会把 `3e-05` 解析成**字符串**（需带小数点才是
    float），所以从 yaml 读回的 shared_hpo.lr 可能已是 str——必须原样透传，不能再走 %g 格式化。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def update_experiment_status(name: str, status: str) -> bool:
    """行内改写指定实验的 status 字段（保留注释与其余内容）。

    2026-10-02 修 bug：实验名后常带行内注释（如 `- name: codebert_ft  # 说明`），
    原先 `split(":", 1)[1].strip() == name` 会因注释匹配失败 → status 写不回 yaml
    → 重启队列时已完成实验被当成 pending 重跑。改为先剥掉 `#` 注释再比较。
    """
    lines = _read_yaml_lines()
    in_block = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("- name:"):
            value = stripped.split(":", 1)[1].split("#", 1)[0].strip()
            in_block = value == name
        elif in_block and stripped.startswith("status:"):
            indent = line[:len(line) - len(line.lstrip())]
            lines[index] = f"{indent}status: {status}"
            _write_yaml_lines(lines)
            return True
    return False


def update_shared_hpo(lr, epochs) -> bool:
    """把 HPO 最优超参回写 experiments.yaml 的 shared_hpo（保留注释与其余内容）。"""
    lines = _read_yaml_lines()
    in_block = False
    changed = False
    for index, line in enumerate(lines):
        if line.startswith("shared_hpo:"):
            in_block = True
            continue
        if in_block:
            if line.strip() and not line.startswith((" ", "\t")):
                break
            match = re.match(r"^(\s+)lr:\s*(.*)$", line)
            if match:
                lines[index] = f"{match.group(1)}lr: {_fmt_num(lr)}"
                changed = True
                continue
            match = re.match(r"^(\s+)epochs:\s*(.*)$", line)
            if match:
                lines[index] = f"{match.group(1)}epochs: {_fmt_num(epochs)}"
                changed = True
    if changed:
        _write_yaml_lines(lines)
    return changed


# ---------------------------------------------------------------- 命令行构造
def _cmd(*args) -> tuple:
    """返回 (绝对路径命令行, 展示用相对路径命令行)。

    Windows 用 list2cmdline（反斜杠路径 + 双引号）；类 Unix 用 shlex.join，
    否则展示命令会把路径反斜杠/引号写错（Linux 云端可读性问题，2026-10-02）。
    """
    parts = [str(a) for a in args]
    display_parts = [PY_REL] + parts
    if os.name == "nt":
        display = subprocess.list2cmdline(display_parts)
    else:
        import shlex
        display = shlex.join(display_parts)
    return [PY_ABS] + parts, display


def hpo_batch_for(base_model: str = None) -> int:
    """HPO 网格 batch（2026-10-02）：codebert=16（云端与 v0.2.2 机房一致），其余基座=8。"""
    if (base_model or CODEBERT_BASE_MODEL) == CODEBERT_BASE_MODEL:
        return CODEBERT_HPO_BATCH
    return LOCAL_HPO_BATCH


def resolve_batch(exp: dict, overrides: dict = None) -> int:
    """实验训练 batch（2026-10-02）：CLI 覆盖 > yaml 每实验 batch 字段 > DEFAULT_BATCH 兜底。"""
    if overrides and overrides.get("batch_size"):
        return int(overrides["batch_size"])
    if exp.get("batch"):
        return int(exp["batch"])
    return DEFAULT_BATCH


def build_hposearch_cmd(batch: int = None, base_model: str = None) -> tuple:
    """fallback 网格命令（补丁 12）：由本文件的 `hpo` 子命令用 importlib 加载引擎
    train/hposearch.py，仅覆盖模块常量 _BATCH 后调其 main()（引擎文件字节不变）。

    之所以不直接跑 `train/hposearch.py`：其 _BATCH 硬编码 16，小显存机器会因 WDDM
    显存换页剧慢，故统一经运行时覆盖（引擎文件字节不变，可校 sha256）。
    补丁 17：base_model 非默认 codebert 时一并运行时覆盖 _BASE（多基座 per_model HPO）。
    2026-10-02：网格 batch 按基座定档——codebert 16（云端标准），其余基座 8。
    """
    parts = ["harness.py", "hpo", "--batch", int(batch or hpo_batch_for(base_model))]
    if base_model and base_model != CODEBERT_BASE_MODEL:
        parts += ["--base-model", base_model]
    return _cmd(*parts)


def build_train_cmd(exp: dict, lr, epochs, batch: int, placeholders: tuple = None) -> tuple:
    """训练命令；lr/epochs 为 None 时用 placeholders（dry-run 展示用）。"""
    name = exp["name"]
    lr_text = _fmt_num(lr) if lr is not None else (placeholders or ("<shared_hpo.lr>",))[0]
    epochs_text = (str(epochs) if epochs is not None
                   else (placeholders or (None, "<shared_hpo.epochs>"))[1])
    return _cmd("train/train.py",
                "--data-dir", "data/train",
                "--base-model", exp.get("base_model", "microsoft/codebert-base"),
                "--output-dir", f"results/{name}/codebert_finetuned",
                "--epochs", epochs_text,
                "--batch-size", batch,
                "--lr", lr_text,
                "--seed", 42)


def build_eval_cmd(model_dir: str, params: str) -> tuple:
    return _cmd("train/evaluate.py",
                "--split", "test",
                "--model-dir", model_dir,
                "--params", params)


def build_calibrate_cmd(model_dir: str, output: str) -> tuple:
    return _cmd("train/calibrate.py",
                "--model-dir", model_dir,
                "--calib-file", f"models/{SPLIT_FILES[2]}",
                "--output", output)


def build_adapter_cmd(exp: dict) -> tuple:
    name = exp["name"]
    if exp.get("type") == "llm":
        return _cmd("adapters/llm_adapter.py",
                    "--exp", name,
                    "--variant", str(exp.get("variant", "zero")),
                    "--out-dir", f"results/{name}",
                    "--splits-dir", "results/_splits")
    if exp.get("type") == "local_zeroshot":
        # v2.5 新增（2026-10-04）：本地基座零样本（不训练，提示词直接判定）
        return _cmd("adapters/local_zeroshot_adapter.py",
                    "--exp", name,
                    "--base-model", exp["base_model"],
                    "--out-dir", f"results/{name}",
                    "--splits-dir", "results/_splits",
                    "--template", str(exp.get("template", "v_feature")))
    return _cmd("adapters/rules_adapter.py",
                "--exp", name,
                "--out-dir", f"results/{name}",
                "--splits-dir", "results/_splits")


def resolve_hparams(exp: dict, shared: dict) -> tuple:
    """返回 (lr, epochs, 来源说明)；None 表示需由本次 HPO 产生。

    2026-10-02 增：实验块显式写 `lr:`/`epochs:` 时优先采用（配合 `hpo: none` 使用）——
    用于「不跑网格、直接按官方/文献推荐超参单跑一次」的对照实验，避免 199h 级网格开销。
    """
    if exp.get("lr") is not None and exp.get("epochs") is not None:
        return (float(exp["lr"]), int(exp["epochs"]),
                "yaml 显式指定（官方/文献推荐超参，不跑网格）")
    hpo = exp.get("hpo")
    if hpo == "per_model":
        lr, epochs = shared.get("lr"), shared.get("epochs")
        if lr is not None and epochs is not None:
            return lr, epochs, "shared_hpo（per_model 展示用；实跑以网格 best 为准）"
        return None, None, "per_model HPO 产出"
    if hpo == "shared":
        lr, epochs = shared.get("lr"), shared.get("epochs")
        if lr is None or epochs is None:
            return None, None, "shared HPO 产出"
        return lr, epochs, "shared_hpo"
    return DEFAULT_LR, DEFAULT_EPOCHS, "train.py 默认值"


# ---------------------------------------------------------------- 实验执行器
class StepError(RuntimeError):
    """单步失败（子进程非 0 退出 / 产物缺失 / 划分不一致）。"""


class ExperimentRunner:
    """单个实验的执行器：日志（追加）、分步计时、暂存区管理。"""

    def __init__(self, exp: dict, env: dict, overrides: dict = None):
        self.exp = exp
        self.name = exp["name"]
        self.env = env
        self.overrides = overrides or {}
        self.out_dir = os.path.join(RESULTS, self.name)
        self.rel_out = f"results/{self.name}"
        self.log_path = os.path.join(self.out_dir, "train.log")
        self.steps = {}
        self.started_at = iso_now()
        self._clock = time.perf_counter()
        self._handle = None
        self.last_returncode = None        # 最近一次 run_step 的退出码（API 熔断判定用）

    # ---- 日志
    def open_log(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        self._handle = open(self.log_path, "a", encoding="utf-8", newline="\n")

    def close_log(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def _emit(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        if self._handle is not None:
            self._handle.write(text)
            self._handle.flush()

    def log(self, text: str) -> None:
        self._emit(text if text.endswith("\n") else text + "\n")

    def note(self, text: str) -> None:
        self.log(f"[harness] {text}\n")

    # ---- 分步执行
    def run_step(self, step_no: int, step_name: str, cmd: tuple,
                 allowed_codes: tuple = (0,)) -> float:
        """执行一步 subprocess：写分隔头 → stdout+stderr 原样留档 → 记 wall time。

        allowed_codes：允许的退出码集合（默认仅 0）。LLM 适配器的 API 熔断退出码 4
        需要被调用方识别为「跳过」而非失败，故由调用方显式放开（2026-10-02）。
        """
        abs_cmd, display = cmd
        self.log(f"[STEP {step_no}/7] {step_name} {display} {iso_now()}\n")
        start = time.perf_counter()
        process = subprocess.Popen(abs_cmd, cwd=ROOT, env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   bufsize=1)
        for line in process.stdout:
            self._emit(line)
        process.wait()
        seconds = round(time.perf_counter() - start, 1)
        self.steps[step_name] = {"seconds": seconds}
        self.last_returncode = process.returncode
        self.log(f"[harness] 步骤 {step_name} 结束：退出码 {process.returncode}，"
                 f"耗时 {seconds}s\n")
        if process.returncode not in allowed_codes:
            raise StepError(f"步骤 {step_name} 失败（退出码 {process.returncode}）：{display}")
        return seconds

    def note_step(self, step_no: int, step_name: str, text: str) -> None:
        """无 subprocess 的步骤（如复用既有校准参数）只记一行说明。"""
        self.log(f"[STEP {step_no}/7] {step_name} (无 subprocess) {text} {iso_now()}\n")

    # ---- 产物 / 暂存区
    def write_identity_params(self) -> str:
        """裸测用恒等映射参数（p = sigmoid(raw logit)）。"""
        write_json(os.path.join(self.out_dir, "identity_params.json"),
                   {"A": -1.0, "B": 0.0, "method": "platt", "n_samples": 0})
        return f"{self.rel_out}/identity_params.json"

    def verify_staging_clean(self) -> None:
        """每实验开始前校验暂存区干净（残留 = 上实验未清理，先搬走再开新实验）。"""
        leftovers = sorted(os.listdir(STAGE)) if os.path.isdir(STAGE) else []
        if leftovers:
            raise StepError(
                f"暂存区 {STAGE} 非空（残留 {leftovers}）——补丁 10 要求实验开始前暂存区干净；"
                f"请确认上一次实验是否异常中断并把残留搬进对应 results/ 目录。")

    def archive_staging(self) -> None:
        """失败时把暂存区残留归档进本实验目录，保证下一个实验能正常开始（失败不连坐）。"""
        if not os.path.isdir(STAGE):
            return
        leftovers = sorted(os.listdir(STAGE))
        if not leftovers:
            return
        dest = os.path.join(self.out_dir, "_staging_leftover")
        os.makedirs(dest, exist_ok=True)
        for entry in leftovers:
            shutil.move(os.path.join(STAGE, entry), os.path.join(dest, entry))
        self.log(f"[harness] 暂存区残留已归档：{leftovers} -> "
                 f"{self.rel_out}/_staging_leftover/\n")

    def capture_eval_outputs(self, stem: str) -> None:
        """把暂存区评估产物复制为 results/{name}/{stem}.json。

        复制而非搬移：步骤 6 仍要用同一个 models/evaluation_test.json（补丁 10 禁止中途搬 split）。
        """
        source = os.path.join(STAGE, "evaluation_test.json")
        if not os.path.isfile(source):
            raise StepError(f"评估产物缺失：{source}")
        shutil.copyfile(source, os.path.join(self.out_dir, f"{stem}.json"))
        detail = os.path.join(STAGE, "evaluation_test_detail.jsonl")
        if os.path.isfile(detail):
            shutil.copyfile(detail, os.path.join(self.out_dir, f"{stem}_detail.jsonl"))

    def verify_split_identity(self) -> None:
        """铁律 1：train.py 现场重算的划分必须与统一划分源逐行一致。"""
        for file_name in SPLIT_FILES:
            canonical = os.path.join(SPLITS, file_name)
            produced = os.path.join(STAGE, file_name)
            if not os.path.isfile(canonical):
                raise StepError(f"统一划分清单缺失：{canonical}（请先运行 python harness.py init）")
            if not os.path.isfile(produced):
                raise StepError(f"训练未产出划分清单：{produced}")
            with open(canonical, "r", encoding="utf-8") as handle:
                left = handle.read().splitlines()
            with open(produced, "r", encoding="utf-8") as handle:
                right = handle.read().splitlines()
            if left != right:
                raise StepError(
                    f"划分与统一划分源不一致（{file_name}：{len(left)} vs {len(right)} 行）——"
                    f"违反铁律 1，全部实验可比性失效，停止本实验并人工排查")

    def finish_staging(self) -> None:
        """实验末尾：暂存区产物统一搬入 results/{name}/，随后清空暂存区。"""
        mapping = {
            "split_train.jsonl": "split_train.jsonl",
            "split_val.jsonl": "split_val.jsonl",
            "split_calibration.jsonl": "split_calibration.jsonl",
            "split_test.jsonl": "split_test.jsonl",
            "metrics.json": "metrics.json",
            "evaluation_test.json": "eval_calib.json",
            "evaluation_test_detail.jsonl": "eval_calib_detail.jsonl",
        }
        for source_name, dest_name in mapping.items():
            source = os.path.join(STAGE, source_name)
            if os.path.isfile(source):
                shutil.move(source, os.path.join(self.out_dir, dest_name))
        leftovers = sorted(os.listdir(STAGE)) if os.path.isdir(STAGE) else []
        for entry in leftovers:
            path = os.path.join(STAGE, entry)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        if leftovers:
            self.log(f"[harness] 暂存区剩余中间产物已清理：{leftovers}\n")

    def write_meta(self) -> None:
        meta = {
            "started_at": self.started_at,
            "finished_at": iso_now(),
            "steps": {name: self.steps.get(name) for name in STEP_NAMES},
            "train_seconds": (self.steps.get("train") or {}).get("seconds"),
            "total_seconds": round(time.perf_counter() - self._clock, 1),
        }
        write_json(os.path.join(self.out_dir, "train_meta.json"), meta)


# ---------------------------------------------------------------- 计划（dry-run）
def describe_plan(exp: dict, shared: dict, ctx: dict) -> list:
    """返回 [(step_no, step_name, 展示命令行/说明), ...]——与实跑同源构造命令。

    ctx: {overrides, needs_shared_hpo, data_version, experiments}
    """
    name = exp["name"]
    etype = exp.get("type")
    overrides = ctx.get("overrides") or {}
    batch = resolve_batch(exp, overrides)
    plan: list = []

    if etype in ("ft", "hybrid"):
        reuse_from = exp.get("reuse_model_from")
        if reuse_from:
            # 2026-10-02 用户决策：hybrid 路由复用同基座 ft 实验的模型与校准，不二次训练
            model_dir = f"results/{name}/codebert_finetuned"
            plan.append((1, "hpo", f"(无 subprocess：无需 HPO，模型/超参同源于 {reuse_from})"))
            plan.append((3, "train", f"(无 subprocess：不训练，克隆复用 {reuse_from} 的微调模型"
                                      f"与 Platt 校准参数；其 codebert 模型即混合门控模型)"))
            plan.append((4, "eval_raw",
                         build_eval_cmd(model_dir, f"results/{name}/identity_params.json")[1]))
            plan.append((5, "calibrate",
                         f"(无 subprocess：复用 {reuse_from}/calibration_params.json)"))
            plan.append((6, "eval_calib",
                         build_eval_cmd(model_dir,
                                        f"results/{name}/calibration_params.json")[1]))
            plan.append((7, "status", "(无 subprocess：回写 experiments.yaml status=trained)"))
            return plan
        lr, epochs, _source = resolve_hparams(exp, shared)
        hpo = exp.get("hpo")
        if hpo == "shared":
            placeholders = ("<shared_hpo.lr>", "<shared_hpo.epochs>")
            if ctx.get("needs_shared_hpo"):
                if reuse_baseline_hpo(ctx.get("data_version", DEFAULT_DATA_VERSION)):
                    plan.append((1, "hpo", "(无 subprocess：复用 models_baseline/v0.2.2/"
                                           "hpo_report.json 的 best 填入 shared_hpo，补丁 12）"))
                else:
                    plan.append((1, "hpo", build_hposearch_cmd()[1]))
        elif hpo == "per_model":
            placeholders = ("<shared_hpo.lr>", "<shared_hpo.epochs>")
            # 补丁 14：per_model 各自真跑网格（2026-10-02 复核：不再有同基座退化复用分支）
            plan.append((2, "hpo", build_hposearch_cmd(
                base_model=exp.get("base_model"))[1]))
        else:
            placeholders = (None, None)
        if overrides.get("epochs") is not None:
            epochs = overrides["epochs"]
        model_dir = f"results/{name}/codebert_finetuned"
        train_display = build_train_cmd(exp, lr, epochs, batch, placeholders)[1]
        if hpo == "per_model":
            train_display += "；若 HPO best 与 shared 超参一致 → 不二次训练（2026-10-02 优化）"
        plan.append((3, "train", train_display))
        plan.append((4, "eval_raw",
                     build_eval_cmd(model_dir, f"results/{name}/identity_params.json")[1]))
        plan.append((5, "calibrate",
                     build_calibrate_cmd(model_dir, f"results/{name}/calibration_params.json")[1]))
        plan.append((6, "eval_calib",
                     build_eval_cmd(model_dir, f"results/{name}/calibration_params.json")[1]))
        plan.append((7, "status", "(无 subprocess：回写 experiments.yaml status=trained)"))
        return plan

    if etype == "reeval":
        # 2026-10-05 用户需求：训练产物已就位、但评测链未跑完（如 calibrate 中期崩溃）。
        # 与 backfill 的区别：走 pending 入队、只补 STEP 4-6（不重训），校准参数缺失时
        # **拟合**（而非复用），成功后回写 status=trained（一次性，不随重启重复补跑）。
        model_dir = exp.get("model_dir") or f"results/{name}/codebert_finetuned"
        plan.append((3, "train", f"(无 subprocess：复用既有训练产物 {model_dir}，不重训)"))
        plan.append((4, "eval_raw",
                     build_eval_cmd(model_dir, f"results/{name}/identity_params.json")[1]))
        plan.append((5, "calibrate",
                     build_calibrate_cmd(model_dir,
                                         f"results/{name}/calibration_params.json")[1]))
        plan.append((6, "eval_calib",
                     build_eval_cmd(model_dir, f"results/{name}/calibration_params.json")[1]))
        plan.append((7, "status", "(无 subprocess：回写 experiments.yaml status=trained)"))
        return plan

    if etype in ("llm", "rules", "local_zeroshot"):
        plan.append((4, "eval_raw", build_adapter_cmd(exp)[1]))
        plan.append((7, "status", "(无 subprocess：回写 experiments.yaml status=trained)"))
        return plan

    return plan


def describe_backfill(exp: dict) -> list:
    """已训版本对照：不训练，补跑步骤 4-6（补丁 10：先把 baseline split 复制进暂存区）。"""
    name = exp["name"]
    return [
        (4, "eval_raw",
         build_eval_cmd(exp.get("model_dir", ""), f"results/{name}/identity_params.json")[1]),
        (5, "calibrate",
         f"(无 subprocess：复用既有校准参数 {exp.get('params', '')} -> "
         f"results/{name}/calibration_params.json)"),
        (6, "eval_calib",
         build_eval_cmd(exp.get("model_dir", ""), f"results/{name}/calibration_params.json")[1]),
    ]


def build_plan(exp: dict, shared: dict, has_key: bool, ctx: dict) -> tuple:
    """决定某实验是否入队，返回 (动作, 计划)；动作 ∈ {run, backfill, skip_key, None}。"""
    status = exp.get("status")
    if status == "pending" or (status == "skipped_api_key" and has_key):
        plan = describe_plan(exp, shared, ctx)
        if exp.get("type") == "llm" and not has_key:
            # 无 Key：实跑只标记 skipped_api_key（铁律 6），dry-run 仍打印完整命令序列
            return "skip_key", plan
        return "run", plan
    if status == "trained" and exp.get("model_dir"):
        return "backfill", describe_backfill(exp)
    return None, []


# ---------------------------------------------------------------- 实跑
def _train_hparams_of(out_dir: str):
    """从 train.log 的 [STEP 3/7] train 命令行解析实际训练超参 (epochs, batch, lr)。

    取最后一次出现的值（重跑以最后一次为准）；解析失败返回 None。
    """
    log_path = os.path.join(out_dir, "train.log")
    if not os.path.isfile(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    matches = re.findall(
        r"^\[STEP 3/7\] train .*?--epochs (\S+).*?--batch-size (\S+).*?--lr (\S+)",
        text, re.M)
    if not matches:
        return None
    epochs_text, batch_text, lr_text = matches[-1]
    try:
        return int(epochs_text), int(batch_text), float(lr_text)
    except ValueError:
        return None


def find_shared_sibling(exp: dict, experiments: list):
    """2026-10-02 优化：找可复用的同基座 shared 训练。

    条件：同 base_model、type ∈ ft/hybrid、hpo=shared、status=trained、关键产物齐全，
    且其 test 划分与统一划分源逐行一致（保证同数据轮，防跨版本误复用）。
    返回实验名或 None；实际超参是否一致由调用方解析 train.log 后再比对。
    """
    for other in experiments:
        if other.get("name") == exp.get("name"):
            continue
        if (other.get("type") in ("ft", "hybrid")
                and other.get("hpo") == "shared"
                and other.get("base_model") == exp.get("base_model")
                and other.get("status") == "trained"):
            out_dir = os.path.join(RESULTS, other["name"])
            if (os.path.isdir(os.path.join(out_dir, "codebert_finetuned"))
                    and os.path.isfile(os.path.join(out_dir, "eval_calib.json"))
                    and os.path.isfile(os.path.join(out_dir, "calibration_params.json"))):
                sibling_split = os.path.join(out_dir, "split_test.jsonl")
                if (os.path.isfile(sibling_split)
                        and read_lines(sibling_split)
                        == read_lines(os.path.join(SPLITS, "split_test.jsonl"))):
                    return other["name"]
    return None


def _clone_file_or_tree(source: str, dest: str) -> None:
    """克隆产物：硬链接优先（同卷零拷贝、省空间），失败退回复制（跨卷/权限/不支持）。"""
    if os.path.isfile(source) or os.path.islink(source):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
        return
    if os.path.isdir(source):
        os.makedirs(dest, exist_ok=True)
        for entry in sorted(os.listdir(source)):
            _clone_file_or_tree(os.path.join(source, entry), os.path.join(dest, entry))


_REUSE_ARTIFACTS = (
    "codebert_finetuned", "eval_raw.json", "eval_raw_detail.jsonl",
    "calibration_params.json", "eval_calib.json", "eval_calib_detail.jsonl",
    "metrics.json",
)


def _reuse_shared_training(runner: ExperimentRunner, exp: dict, sibling: str,
                           lr, epochs, batch: int) -> None:
    """HPO best 与同基座 shared 训练完全一致 → 不二次训练（2026-10-02 用户优化）。

    克隆 sibling 的模型与评测产物到本实验目录（铁律 4：实验目录隔离，产物各自持有），
    并在 hpo.json / train.log 注明复用关系；同数据/同超参/同 seed=42 再训结果必然相同。
    """
    sibling_dir = os.path.join(RESULTS, sibling)
    for entry in _REUSE_ARTIFACTS:
        source = os.path.join(sibling_dir, entry)
        if os.path.exists(source):
            _clone_file_or_tree(source, os.path.join(runner.out_dir, entry))
    runner.note_step(3, "train",
                     f"HPO best（lr={_fmt_num(lr)} epochs={epochs} batch={batch}）与 shared "
                     f"超参完全一致 → 不二次训练，复用 {sibling} 的模型与评测产物"
                     f"（2026-10-02 用户优化：同数据/同超参/同 seed=42，再训结果必然相同）")
    runner.note_step(4, "eval_raw", f"复用 {sibling}/eval_raw.json")
    runner.note_step(5, "calibrate", f"复用 {sibling}/calibration_params.json")
    runner.note_step(6, "eval_calib", f"复用 {sibling}/eval_calib.json")
    hpo_path = os.path.join(runner.out_dir, "hpo.json")
    hpo_payload = load_json(hpo_path)
    if not isinstance(hpo_payload, dict):
        hpo_payload = {}
    hpo_payload.update({
        "skip_retrain": True,
        "reuse_from": sibling,
        "reuse_reason": "HPO best 与 shared_hpo 完全一致（lr/epochs/batch），"
                        "同数据同 seed 再训结果必然相同，不二次训练（2026-10-02）",
        "matched_hparams": {"lr": lr, "epochs": epochs, "batch": batch},
        "produced_at": iso_now(),
    })
    write_json(hpo_path, hpo_payload)


def run_reuse_from_experiment(runner: ExperimentRunner, exp: dict, source_name: str) -> bool:
    """hybrid 路由复用（2026-10-02 用户决策）：复用同基座 ft 实验的模型与校准，不二次训练。

    codebert_ft 的微调模型即混合门控的模型（同基座/同数据/同 seed=42），无需二次训练；
    克隆模型目录与 Platt 校准参数到本实验目录（铁律 4：实验目录隔离、产物各自持有），
    再补跑一轮 test 评测（步骤 4/6）留档。返回 False 表示源产物缺失，调用方回退正常训练。
    """
    name = exp["name"]
    src_dir = os.path.join(RESULTS, source_name)
    model_src = os.path.join(src_dir, "codebert_finetuned")
    params_src = os.path.join(src_dir, "calibration_params.json")
    if not (os.path.isdir(model_src) and os.path.isfile(params_src)):
        runner.note(f"reuse_model_from={source_name} 的产物缺失（{model_src} / {params_src}）"
                    f"→ 回退正常 ft 训练")
        return False

    runner.verify_staging_clean()
    runner.note(f"hybrid 路由复用（2026-10-02 用户决策）：{source_name} 的微调模型即混合门控"
                f"的模型（同基座/同数据/同 seed=42）→ 不二次训练，克隆模型与校准参数，"
                f"只补一轮 test 评测留档")
    runner.note_step(1, "hpo", f"无需 HPO：模型与超参同源于 {source_name}")
    runner.note_step(3, "train", f"不训练：克隆复用 {source_name} 的微调模型与 Platt 校准参数")
    _clone_file_or_tree(model_src, os.path.join(runner.out_dir, "codebert_finetuned"))
    _clone_file_or_tree(params_src, os.path.join(runner.out_dir, "calibration_params.json"))

    # evaluate.py 硬编码读 models/split_{split}.jsonl，复用流程无训练步骤不会生成 → 先复制
    os.makedirs(STAGE, exist_ok=True)
    for file_name in SPLIT_FILES:
        source = os.path.join(SPLITS, file_name)
        if not os.path.isfile(source):
            raise StepError(f"统一划分清单缺失：{source}（先跑 init 或完成本轮划分重建）")
        shutil.copyfile(source, os.path.join(STAGE, file_name))

    model_dir = f"results/{name}/codebert_finetuned"
    identity = runner.write_identity_params()
    runner.run_step(4, "eval_raw", build_eval_cmd(model_dir, identity))
    runner.capture_eval_outputs("eval_raw")
    runner.note_step(5, "calibrate", f"复用 {source_name}/calibration_params.json（已克隆）")
    runner.run_step(6, "eval_calib",
                    build_eval_cmd(model_dir, f"results/{name}/calibration_params.json"))
    runner.capture_eval_outputs("eval_calib")

    runner.finish_staging()
    runner.note_step(7, "status", "回写 experiments.yaml status=trained")
    update_experiment_status(name, "trained")
    return True


def run_ft_experiment(runner: ExperimentRunner, exp: dict, shared: dict, ctx: dict) -> None:
    """ft / hybrid：HPO(如需) → 训练 → 裸测 → 校准 → 校准后测 → status。

    HPO（补丁 12+14）：shared 优先复用 baseline 既成档案（仅同数据版本）；
    per_model 各自真跑网格（补丁 14：无退化复用）。
    2026-10-02 优化：per_model 网格的 best 若与同基座 shared 训练的超参完全一致
    （lr/epochs/batch），跳过二次训练，直接复用该 shared 实验的产物并注明。
    """
    name = exp["name"]
    overrides = runner.overrides
    batch = resolve_batch(exp, overrides)
    lr, epochs, source = resolve_hparams(exp, shared)
    # 注意：--epochs 覆盖必须在 HPO 解析**之后**应用（HPO 复用/网格都会改写 epochs；
    # 冒烟实测过一次覆盖被 HPO 分支撤销导致跑满 5 epochs 的 bug）

    # 2026-10-02 用户决策：hybrid 路由复用——直接复用同基座 ft 实验的模型与校准，
    # 不二次训练、不跑 HPO，只补一轮 test 评测留档（产物缺失时回退正常 ft 流程）
    reuse_from = exp.get("reuse_model_from")
    if reuse_from and run_reuse_from_experiment(runner, exp, reuse_from):
        return

    runner.verify_staging_clean()
    runner.note(f"data_version={ctx.get('data_version', DEFAULT_DATA_VERSION)}"
                f"（§3.0 数据版本机制）")

    # 步骤 1/2：HPO
    hpo = exp.get("hpo")
    if hpo == "shared" and (shared.get("lr") is None or shared.get("epochs") is None):
        reused = reuse_baseline_hpo(ctx.get("data_version", DEFAULT_DATA_VERSION))
        if reused is not None:
            lr, epochs = reused["lr"], reused["epochs"]
            source = "baseline 复用"
            runner.note_step(1, "hpo", f"复用既成 HPO 档案 {reused['source']}："
                                       f"lr={_fmt_num(lr)} epochs={epochs}")
            update_shared_hpo(lr, epochs)
            write_json(os.path.join(SPLITS, "shared_hpo.json"), {
                "lr": lr, "epochs": epochs, "data_version": ctx.get("data_version"),
                "grid": (reused["report"] or {}).get("grid"),
                "best": (reused["report"] or {}).get("best"),
                "results": (reused["report"] or {}).get("results"),
                "source": "baseline_reuse", "source_detail": reused["source"],
                "experiment": name, "produced_at": iso_now(),
            })
            write_json(os.path.join(runner.out_dir, "hpo.json"), {
                "source": "baseline_reuse", "source_detail": reused["source"],
                "data_version": ctx.get("data_version"),
                "lr": lr, "epochs": epochs,
                "report": reused["report"], "produced_at": iso_now(),
            })
            runner.note(f"shared_hpo 已回写 yaml 与 results/_splits/shared_hpo.json："
                        f"lr={_fmt_num(lr)} epochs={epochs}")
        else:
            runner.run_step(1, "hpo", build_hposearch_cmd())
            report = load_json(os.path.join(STAGE, "hpo_report.json"))
            best = (report or {}).get("best") or {}
            if best.get("lr") is None or best.get("epochs") is None:
                raise StepError("HPO 未产出最优超参（models/hpo_report.json 缺失或格式异常）")
            lr, epochs, source = best["lr"], int(best["epochs"]), "本机网格 HPO 产出"
            update_shared_hpo(lr, epochs)
            write_json(os.path.join(SPLITS, "shared_hpo.json"), {
                "lr": lr, "epochs": epochs, "data_version": ctx.get("data_version"),
                "grid": (report or {}).get("grid"), "best": best,
                "results": (report or {}).get("results"),
                "source": "local_grid", "experiment": name, "produced_at": iso_now(),
            })
            write_json(os.path.join(runner.out_dir, "hpo.json"), report)
            runner.note(f"shared_hpo 已产生并回写 yaml 与 results/_splits/shared_hpo.json："
                        f"lr={_fmt_num(lr)} epochs={epochs}")
    elif hpo == "per_model":
        # 补丁 14：per_model 各自真跑网格（回滚补丁 12 的同基座退化复用，2026-10-02 复核移除）
        runner.run_step(2, "hpo", build_hposearch_cmd(
            base_model=exp.get("base_model")))
        report = load_json(os.path.join(STAGE, "hpo_report.json"))
        best = (report or {}).get("best") or {}
        if best.get("lr") is None or best.get("epochs") is None:
            raise StepError("HPO 未产出最优超参（models/hpo_report.json 缺失或格式异常）")
        lr, epochs, source = best["lr"], int(best["epochs"]), "per_model HPO 产出"
        write_json(os.path.join(runner.out_dir, "hpo.json"), report)
        runner.note(f"per_model HPO 最优：lr={_fmt_num(lr)} epochs={epochs}")
    else:
        runner.note(f"跳过 HPO（hpo={hpo}），超参来源：{source}")

    if lr is None or epochs is None:
        raise StepError("超参未解析出来（shared_hpo 为空且本次未跑 HPO）")

    # --epochs 覆盖（冒烟/调试）在 HPO 之后生效，确保不被 HPO 解析撤销
    if overrides.get("epochs") is not None:
        runner.note(f"--epochs 覆盖生效：{epochs} -> {overrides['epochs']}"
                    f"（仅本次运行，不改 experiments.yaml）")
        epochs = overrides["epochs"]

    # 步骤 3：训练（2026-10-02 优化：per_model 的 HPO best 与同基座 shared 训练
    # 的实际超参（epochs/batch/lr）完全一致 → 不二次训练，复用产物并注明）
    model_dir = f"results/{name}/codebert_finetuned"
    if hpo == "per_model":
        sibling = find_shared_sibling(exp, ctx.get("experiments") or [])
        if sibling:
            sibling_hparams = _train_hparams_of(os.path.join(RESULTS, sibling))
            if sibling_hparams == (int(epochs), int(batch), float(lr)):
                _reuse_shared_training(runner, exp, sibling, lr, epochs, batch)
                runner.note_step(7, "status", "回写 experiments.yaml status=trained")
                update_experiment_status(name, "trained")
                return
            runner.note(f"同基座 shared 实验 {sibling} 的实际超参 epochs/batch/lr="
                        f"{sibling_hparams} 与本次 HPO best（epochs={int(epochs)} "
                        f"batch={int(batch)} lr={float(lr)}）不一致 → 正常执行二次训练")
    runner.run_step(3, "train", build_train_cmd(exp, lr, epochs, batch))
    runner.verify_split_identity()

    # 步骤 4：裸测（恒等映射替代 Platt）
    identity = runner.write_identity_params()
    runner.run_step(4, "eval_raw", build_eval_cmd(model_dir, identity))
    runner.capture_eval_outputs("eval_raw")

    # 步骤 5：校准（--output 直指实验目录）
    params = f"results/{name}/calibration_params.json"
    runner.run_step(5, "calibrate", build_calibrate_cmd(model_dir, params))

    # 步骤 6：校准后测
    runner.run_step(6, "eval_calib", build_eval_cmd(model_dir, params))
    runner.capture_eval_outputs("eval_calib")

    # 步骤 7：收尾
    runner.finish_staging()
    runner.note_step(7, "status", "回写 experiments.yaml status=trained")
    update_experiment_status(name, "trained")


def run_adapter_experiment(runner: ExperimentRunner, exp: dict, ctx: dict = None) -> None:
    """llm / rules：适配器自己产出 eval_raw.json（LLM 无校准，eval_calib.json = null 占位）。

    2026-10-02 用户要求的保护机制：LLM 适配器遇 API 致命错误（404 / 长时间无响应）会写
    api_error_skip.json 并以退出码 4 退出——本函数据此把实验标 skipped_api_error 并置
    ctx 熔断标志，跳过后续 API-LLM 实验（记录在案，不当作失败）。
    """
    runner.verify_staging_clean()
    is_llm = exp.get("type") == "llm"
    if is_llm:
        runner.note("llm 流程：3 模板 × val 150 条挑模板（TH-4 规则 3）→ test 全量评测，"
                    "由 adapters/llm_adapter.py 单步完成")
    runner.run_step(4, "eval_raw", build_adapter_cmd(exp),
                    allowed_codes=(0, API_FATAL_EXIT_CODE) if is_llm else (0,))
    marker = os.path.join(runner.out_dir, "api_error_skip.json")
    if is_llm and (runner.last_returncode == API_FATAL_EXIT_CODE or os.path.isfile(marker)):
        runner.note("API 熔断（404 / 长时间无响应）→ 标记 skipped_api_error 并跳过本实验；"
                    "后续 API-LLM 实验一并跳过；记录见 "
                    f"results/{exp['name']}/api_error_skip.json；"
                    "API 恢复后把 yaml 里该实验 status 改回 pending 可重跑")
        runner.note_step(7, "status", "回写 experiments.yaml status=skipped_api_error")
        update_experiment_status(exp["name"], "skipped_api_error")
        if ctx is not None:
            ctx["api_circuit_open"] = True
        return
    if not os.path.isfile(os.path.join(runner.out_dir, "eval_raw.json")):
        raise StepError(f"适配器未产出 results/{exp['name']}/eval_raw.json")
    runner.note_step(7, "status", "回写 experiments.yaml status=trained")
    update_experiment_status(exp["name"], "trained")


def run_backfill_experiment(runner: ExperimentRunner, exp: dict) -> None:
    """已训版本对照（如 v0.2.2_baseline）：跳过训练，补跑步骤 4-6。

    2026-10-02 修订：补跑改在「当前统一划分源」的 test 段上评（跨数据版本对照口径：
    旧模型遇新威胁的泛化视图）——不再复制 models_baseline 的旧划分。
    """
    name = exp["name"]
    model_dir = exp.get("model_dir")
    if not model_dir or not os.path.isdir(os.path.join(ROOT, model_dir)):
        raise StepError(f"model_dir 不存在：{model_dir}")

    runner.verify_staging_clean()
    # 补丁 10：evaluate.py 硬编码读 models/split_{split}.jsonl，baseline 无训练步骤不会自动生成
    os.makedirs(STAGE, exist_ok=True)
    for file_name in SPLIT_FILES:
        source = os.path.join(SPLITS, file_name)
        if not os.path.isfile(source):
            raise StepError(f"统一划分清单缺失：{source}（先跑 init 或完成本轮划分重建）")
        shutil.copyfile(source, os.path.join(STAGE, file_name))
    runner.note("已把统一划分源 results/_splits/split_*.jsonl 复制进暂存区——"
                "v0.2.2 将在当前 data_version 的 test 段上评（跨版本对照口径，2026-10-02 修订）")

    # 步骤 4：裸测（恒等映射）
    identity = runner.write_identity_params()
    runner.run_step(4, "eval_raw", build_eval_cmd(model_dir, identity))
    runner.capture_eval_outputs("eval_raw")

    # 步骤 5：校准参数已存在 → 直接复用（不重跑 calibrate）
    params_src = exp.get("params", "")
    if params_src and os.path.isfile(os.path.join(ROOT, params_src)):
        shutil.copyfile(os.path.join(ROOT, params_src),
                        os.path.join(runner.out_dir, "calibration_params.json"))
        runner.note_step(5, "calibrate", f"复用既有校准参数 {params_src}")
    else:
        runner.run_step(5, "calibrate",
                        build_calibrate_cmd(model_dir, f"results/{name}/calibration_params.json"))

    # 步骤 6：校准后测
    runner.run_step(6, "eval_calib",
                    build_eval_cmd(model_dir, f"results/{name}/calibration_params.json"))
    runner.capture_eval_outputs("eval_calib")

    runner.finish_staging()
    runner.note("对照实验补跑完成（status 保持 trained：不训练、不改写 yaml）")


def run_reeval_experiment(runner: ExperimentRunner, exp: dict) -> None:
    """评测补齐（type=reeval，2026-10-05 用户需求）：训练产物已就位但评测链未跑完
    （如 calibrate 中途崩溃）→ 只补跑步骤 4-6，**不重训**。

    与 run_backfill_experiment 的差异：校准参数缺失时**拟合**（而非复用既有），
    并在成功后回写 status=trained —— 故只补跑一次，不随守卫重启重复执行。
    """
    name = exp["name"]
    model_dir = exp.get("model_dir") or f"results/{name}/codebert_finetuned"
    if not os.path.isdir(os.path.join(ROOT, model_dir)):
        raise StepError(f"model_dir 不存在：{model_dir}（补评要求训练产物已就位）")

    runner.verify_staging_clean()
    os.makedirs(STAGE, exist_ok=True)
    for file_name in SPLIT_FILES:
        source = os.path.join(SPLITS, file_name)
        if not os.path.isfile(source):
            raise StepError(f"统一划分清单缺失：{source}（先跑 init 或完成本轮划分重建）")
        shutil.copyfile(source, os.path.join(STAGE, file_name))
    runner.note("已把统一划分源 results/_splits/split_*.jsonl 复制进暂存区（补评：复用既有模型）")

    # 步骤 4：裸测（恒等映射）
    identity = runner.write_identity_params()
    runner.run_step(4, "eval_raw", build_eval_cmd(model_dir, identity))
    runner.capture_eval_outputs("eval_raw")

    # 步骤 5：校准（无既有参数可复用时拟合；有则复用）
    params = f"results/{name}/calibration_params.json"
    params_src = exp.get("params", "")
    if params_src and os.path.isfile(os.path.join(ROOT, params_src)):
        shutil.copyfile(os.path.join(ROOT, params_src),
                        os.path.join(runner.out_dir, "calibration_params.json"))
        runner.note_step(5, "calibrate", f"复用既有校准参数 {params_src}")
    else:
        runner.run_step(5, "calibrate", build_calibrate_cmd(model_dir, params))

    # 步骤 6：校准后测
    runner.run_step(6, "eval_calib", build_eval_cmd(model_dir, params))
    runner.capture_eval_outputs("eval_calib")

    # 步骤 7：收尾（补评成功即视为 trained）
    runner.finish_staging()
    runner.note_step(7, "status", "回写 experiments.yaml status=trained")
    update_experiment_status(name, "trained")


def skip_llm_by_circuit(exp: dict) -> None:
    """API 熔断后跳过后续 API-LLM 实验并记录（2026-10-02 用户要求的保护机制）。

    不发起任何 subprocess；只在本实验目录留 train.log 记录 + 回写 yaml status。
    """
    name = exp["name"]
    runner = ExperimentRunner(exp, env={}, overrides={})
    runner.open_log()
    runner.note("API 熔断已触发（前置 API-LLM 实验返回 404 / 长时间无响应）→ "
                "跳过本实验并标记 skipped_api_error（2026-10-02 用户要求的保护机制）；"
                "API 恢复后把 yaml 里该实验 status 改回 pending 可重跑")
    runner.note_step(4, "eval_raw", "跳过（API 熔断，未调用）")
    runner.note_step(7, "status", "回写 experiments.yaml status=skipped_api_error")
    runner.write_meta()
    runner.close_log()
    update_experiment_status(name, "skipped_api_error")
    print(f"[harness] 实验 {name} 因 API 熔断跳过（skipped_api_error）")


def execute_experiment(action: str, exp: dict, shared: dict, env: dict,
                       ctx: dict) -> None:
    """单实验执行入口：失败记 failed + error.log（不吞异常，继续下一个）。"""
    name = exp["name"]
    overrides = ctx.get("overrides") or {}
    runner = ExperimentRunner(exp, env, overrides=overrides)
    print(f"\n{'=' * 78}\n[harness] 实验 {name}（type={exp.get('type')}）{iso_now()} "
          f"运行中，请勿并行启动其他计算\n{'=' * 78}")
    runner.open_log()
    if action == "skip_key":
        runner.note(f"预检：环境未设 {API_KEY_ENV} → 标记 skipped_api_key"
                    f"（铁律 6，不阻塞其他实验）")
        runner.write_meta()
        runner.close_log()
        update_experiment_status(name, "skipped_api_key")
        print(f"[harness] 实验 {name} 标记 skipped_api_key（无 {API_KEY_ENV}）")
        return
    try:
        if action == "backfill":
            run_backfill_experiment(runner, exp)
        elif exp.get("type") == "reeval":
            run_reeval_experiment(runner, exp)
        elif exp.get("type") in ("ft", "hybrid"):
            run_ft_experiment(runner, exp, shared, ctx)
        else:
            run_adapter_experiment(runner, exp, ctx)
    except BaseException as exc:                      # noqa: BLE001 —— 失败必须留痕
        message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        runner.log(f"[harness] 实验失败：{type(exc).__name__}: {exc}\n")
        runner.archive_staging()
        with open(os.path.join(runner.out_dir, "error.log"), "w",
                  encoding="utf-8", newline="\n") as handle:
            handle.write(f"experiment: {name}\nfinished_at: {iso_now()}\n"
                         f"error: {type(exc).__name__}: {exc}\n\n{message}")
        update_experiment_status(name, "failed")
        runner.write_meta()
        runner.close_log()
        print(f"[harness] 实验 {name} 失败，已记 failed 与 error.log，继续下一个")
        return
    runner.write_meta()
    runner.close_log()
    print(f"[harness] 实验 {name} 完成 {iso_now()}")


# ---------------------------------------------------------------- 子命令
def cmd_init(args) -> int:
    """TH-1：把 models_baseline/v0.2.2/split_*.jsonl 复制到 results/_splits/（唯一划分源）。"""
    os.makedirs(SPLITS, exist_ok=True)
    for file_name in SPLIT_FILES:
        source = os.path.join(BASELINE_DIR, file_name)
        if not os.path.isfile(source):
            print(f"[错误] baseline 划分清单缺失：{source}", file=sys.stderr)
            return 1
        dest = os.path.join(SPLITS, file_name)
        shutil.copyfile(source, dest)
        with open(dest, "r", encoding="utf-8") as handle:
            lines = [line for line in handle.read().splitlines() if line.strip()]
        print(f"[init] {file_name} -> results/_splits/{file_name}（{len(lines)} 行）")
    print(f"[init] 统一划分源就绪：{SPLITS}")
    return 0


def cmd_hpo(args) -> int:
    """fallback 网格入口（补丁 12 步骤 1 / 步骤 2 的多基座分支）。

    用 importlib 加载引擎 train/hposearch.py（**引擎文件字节不变，可校 sha256**），
    仅覆盖其模块常量 `_BATCH` 后调用 `main()`——网格（16.7 六组）、划分（复用 train.py
    的 group-aware 实现）、选优指标（val_f1_malicious）与引擎完全一致，只改 batch 以适配
    本机 6GB 显存。产物仍是 models/hpo_report.json（由 harness 搬进实验目录）。
    """
    import importlib.util

    os.chdir(ROOT)
    train_dir = os.path.join(ROOT, "train")
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)      # hposearch 内部 `import train as T` 需要
    engine_path = os.path.join(train_dir, "hposearch.py")
    with open(engine_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    spec = importlib.util.spec_from_file_location("bench_hposearch_entry", engine_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_hposearch_entry"] = module
    spec.loader.exec_module(module)
    original = module._BATCH
    module._BATCH = args.batch or hpo_batch_for(getattr(args, "base_model", None))
    base_model = getattr(args, "base_model", None)
    if base_model and base_model != CODEBERT_BASE_MODEL:
        original_base = module._BASE
        module._BASE = base_model
        print(f"[hpo] 运行时覆盖模块常量 _BASE: {original_base} -> {base_model}"
              f"（多基座 per_model HPO，补丁 17；引擎文件字节不变）")
    print(f"[hpo] 引擎 train/hposearch.py sha256={digest}（未改动）")
    print(f"[hpo] 运行时覆盖模块常量 _BATCH: {original} -> {module._BATCH}"
          f"（2026-10-02 按基座定档：codebert 16 云端标准 / 其余基座 8）")
    module.main()
    return 0


def cmd_train(args) -> int:
    matrix, shared, experiments = load_matrix()
    data_version = data_version_of(matrix)
    if not args.all and not args.only:
        print("[错误] 需要 --all 或 --only NAME[,NAME...]", file=sys.stderr)
        return 2
    selected = None
    if args.only:
        selected = {part.strip() for part in args.only.split(",") if part.strip()}
        unknown = selected - {exp["name"] for exp in experiments}
        if unknown:
            print(f"[错误] --only 指定的实验不存在：{sorted(unknown)}", file=sys.stderr)
            return 2
    has_key = bool(os.environ.get(API_KEY_ENV))
    ctx = {"overrides": {"epochs": args.epochs, "batch_size": args.batch_size},
           "needs_shared_hpo": shared.get("lr") is None or shared.get("epochs") is None,
           "data_version": data_version,
           "experiments": experiments,
           "api_circuit_open": False}

    queued = []
    for exp in experiments:
        if selected is not None and exp["name"] not in selected:
            continue
        action, plan = build_plan(exp, shared, has_key, ctx)
        if action:
            queued.append((action, exp, plan))
            # 队列语义模拟：第一个 pending 的 hpo=shared 实验会产出 shared_hpo，后续实验复用
            if (action == "run" and ctx["needs_shared_hpo"]
                    and exp.get("hpo") == "shared" and exp.get("type") in ("ft", "hybrid")):
                ctx["needs_shared_hpo"] = False

    if args.dry_run:
        print(f"[dry-run] experiments.yaml 共 {len(experiments)} 个实验"
              f"（data_version={data_version}），本次将处理 {len(queued)} 个"
              f"（只打印命令序列，不执行）")
        for action, exp, plan in queued:
            name = exp["name"]
            if action == "backfill":
                print(f"[dry-run] === experiment: {name} (type={exp.get('type')}, "
                      f"status={exp.get('status')}, 已训对照：补跑步骤 4-6) ===")
            else:
                print(f"[dry-run] === experiment: {name} (type={exp.get('type')}, "
                      f"hpo={exp.get('hpo')}, status={exp.get('status')}) ===")
            if action == "skip_key":
                print(f"[dry-run] {name} | (预检) 当前环境未设 {API_KEY_ENV} → 实跑时标记 "
                      f"skipped_api_key（不进入逐条调用）；以下为完整命令序列")
            for step_no, step_name, display in plan:
                print(f"[dry-run] {name} | STEP {step_no}/7 | {step_name} | {display}")
        return 0

    env = build_env()
    if not os.path.isfile(PY_ABS):
        print(f"[错误] 未找到 {PY_REL}：请先运行 {SETUP_SCRIPT}", file=sys.stderr)
        return 1
    if not os.path.isfile(os.path.join(SPLITS, "split_test.jsonl")):
        print("[错误] 统一划分源缺失：请先运行 python harness.py init", file=sys.stderr)
        return 1
    if not queued:
        print("[harness] 没有需要处理的实验（pending / 可用 Key 的 skipped_api_key / "
              "已训对照补跑）")
        return 0

    print(f"[harness] 训练队列启动 {iso_now()}：{len(queued)} 个实验，严格串行执行；"
          f"data_version={data_version}")
    print("[harness] batch 策略（2026-10-02）：codebert 系 16（云端，与 v0.2.2 一致）、"
          "多基座 LLM 8；逐实验以 yaml batch 字段为准")
    print("[harness] 运行中，请勿并行启动其他计算（铁律 7/10：实验串行、独占机器）")
    for action, exp, _plan in queued:
        _, fresh_shared, fresh_experiments = load_matrix()  # HPO 回写后必须重读
        ctx["experiments"] = fresh_experiments             # 复用判定须看最新 status
        # API 熔断（2026-10-02 用户要求）：前置 API-LLM 实验 404 / 长时间无响应后，
        # 后续 API-LLM 实验直接跳过并记录，不再逐条空转
        if ctx.get("api_circuit_open") and exp.get("type") == "llm":
            skip_llm_by_circuit(exp)
            continue
        execute_experiment(action, exp, fresh_shared, env, ctx)
    print(f"[harness] 训练队列结束 {iso_now()}")
    return 0


def _list_python_pids() -> list:
    """同机其他 python 进程 PID（Windows 用 tasklist，类 Unix/Linux 用 ps；排除自身）。"""
    pids = []
    own = str(os.getpid())
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe",
                                  "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=20)
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    if "python.exe" in line.lower():
                        pid = line.split(",")[1].strip('"')
                        if pid and pid != own:
                            pids.append(pid)
        else:
            out = subprocess.run(["ps", "-eo", "pid=,comm="],
                                 capture_output=True, text=True, timeout=20)
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    parts = line.split(None, 1)
                    if len(parts) == 2 and os.path.basename(parts[1].strip()).startswith("python"):
                        if parts[0] != own:
                            pids.append(parts[0])
    except (OSError, subprocess.SubprocessError):
        pass
    return pids


def machine_preflight() -> list:
    """速度基准前置检查（铁律 5/10）：把同机占用情况摆出来，有异动先给 10 秒中止窗口。"""
    lines = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,"
                              "memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            lines.append(f"GPU: {out.stdout.strip().splitlines()[0]}")
    except (OSError, subprocess.SubprocessError):
        lines.append("GPU: 未检测到 nvidia-smi")
    foreign = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            foreign = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    if foreign:
        lines.append(f"GPU 计算进程：{foreign}")
    others = _list_python_pids()
    lines.append(f"其他 python 进程 PID：{others if others else '无'}")
    for line in lines:
        print(f"[preflight] {line}")
    if foreign or others:
        print("[preflight][WARN] 检测到同机其他计算/进程：速度数据可能被污染（铁律 5）。"
              "10 秒内可 Ctrl+C 中止；请确认独占机器后继续。")
        time.sleep(10)
    else:
        print("[preflight] 机器看起来空闲，开始速度基准。")
    return lines


def cmd_speed(args) -> int:
    if not args.all:
        print("[错误] 需要 --all", file=sys.stderr)
        return 2
    _, _shared, experiments = load_matrix()
    targets = [exp for exp in experiments if exp.get("status") == "trained"]
    if not targets:
        print("[harness] 没有 status=trained 的实验可测速")
        return 0
    env = build_env()
    if not os.path.isfile(PY_ABS):
        print(f"[错误] 未找到 {PY_REL}：请先运行 {SETUP_SCRIPT}", file=sys.stderr)
        return 1
    print(f"[harness] 速度基准 {iso_now()}：依次测 {len(targets)} 个实验，"
          f"运行中请勿并行启动其他计算（铁律 5/10）")
    machine_preflight()
    failures = []
    for exp in targets:
        name = exp["name"]
        parts = ["speed_bench.py", "--exp", name]
        if args.threads is not None:
            parts += ["--threads", str(args.threads)]
        abs_cmd, display = _cmd(*parts)
        print(f"\n[harness] speed | {name} | {display}")
        result = subprocess.run(abs_cmd, cwd=ROOT, env=env)
        if result.returncode != 0:
            failures.append(name)
            print(f"[harness] 测速失败：{name}（退出码 {result.returncode}）")
    if failures:
        print(f"[harness] 以下实验测速失败：{failures}", file=sys.stderr)
        return 1
    print(f"[harness] 速度基准结束 {iso_now()}")
    return 0


# ---------------------------------------------------------------- 报告
def _fmt_duration(seconds):
    """训练成本格式化（report 用 "1h23m" 形态）。"""
    if seconds is None:
        return None
    seconds = float(seconds)
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{seconds:.1f}s"


def _round(value, digits=4):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _batch_of(out_dir: str):
    """从 train.log 的 `[STEP 3/7] train ...` 命令行里取 --batch-size（报告须注明 batch 差异）。

    日志是追加模式，取最后一次出现的值（重跑以最后一次为准）。
    """
    log_path = os.path.join(out_dir, "train.log")
    if not os.path.isfile(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        matches = re.findall(r"^\[STEP 3/7\] train .*?--batch-size (\d+)", handle.read(), re.M)
    return int(matches[-1]) if matches else None


def collect_row(exp: dict) -> dict:
    """把一个实验的全部产物汇总成报告行（列 = 文档第 7 节）。"""
    name = exp["name"]
    out_dir = os.path.join(RESULTS, name)
    calib = load_json(os.path.join(out_dir, "eval_calib.json"))
    raw = load_json(os.path.join(out_dir, "eval_raw.json"))
    speed = load_json(os.path.join(out_dir, "speed.json"))
    meta = load_json(os.path.join(out_dir, "train_meta.json"))
    prompt_sel = load_json(os.path.join(out_dir, "prompt_sel.json"))
    hpo = load_json(os.path.join(out_dir, "hpo.json"))

    main = calib if isinstance(calib, dict) else (raw if isinstance(raw, dict) else None)
    source = ("eval_calib" if isinstance(calib, dict)
              else ("eval_raw" if isinstance(raw, dict) else None))
    metrics = (main or {}).get("metrics_calibrated_p_gt05") or {}
    gate = (main or {}).get("gate") or {}
    n_malicious = (main or {}).get("n_malicious")

    allow = gate.get("allow") or {}
    block = gate.get("block") or {}
    review = gate.get("llm_review") or {}
    silent_allow_rate = (_round(allow.get("malicious", 0) / n_malicious)
                         if n_malicious else None)
    block_purity = (_round(block.get("malicious", 0) / block.get("total"))
                    if block.get("total") else None)

    delta_f1 = None
    if isinstance(calib, dict) and isinstance(raw, dict) and exp.get("type") in ("ft", "hybrid"):
        calib_f1 = (calib.get("metrics_calibrated_p_gt05") or {}).get("f1")
        raw_f1 = (raw.get("metrics_calibrated_p_gt05") or {}).get("f1")
        if calib_f1 is not None and raw_f1 is not None:
            delta_f1 = _round(calib_f1 - raw_f1)

    per_domain = {}
    for domain, info in ((main or {}).get("per_domain") or {}).items():
        per_domain[domain] = {
            "n": info.get("n"),
            "n_malicious": info.get("n_malicious"),
            "n_benign": info.get("n_benign"),
            "recall": _round(info.get("recall_p05")),
            "silent_allow": info.get("silent_allow"),
            "benign_fp": info.get("benign_fp"),
        }
    for domain in ("code", "prompt"):
        per_domain.setdefault(domain, {"n": 0, "n_malicious": 0, "n_benign": 0,
                                       "recall": None, "silent_allow": None,
                                       "benign_fp": None})

    train_seconds = (meta or {}).get("train_seconds")
    total_seconds = (meta or {}).get("total_seconds")
    config = {key: exp.get(key) for key in
              ("name", "type", "base_model", "hpo", "variant", "model_dir", "params", "status")
              if exp.get(key) is not None}

    return {
        "name": name,
        "type": exp.get("type"),
        "status": exp.get("status"),
        "config": config,
        "metrics_source": source,
        # ---- 质量常规 ----
        "acc": _round(metrics.get("accuracy")),
        "precision": _round(metrics.get("precision")),
        "recall": _round(metrics.get("recall")),
        "f1": _round(metrics.get("f1")),
        "auc": _round((main or {}).get("auc")),
        "ece": _round((main or {}).get("ece")),
        "brier": _round((main or {}).get("brier")),
        # ---- 产品口径 ----
        "silent_allow_rate": silent_allow_rate,
        "block_purity": block_purity,
        "gate_allow": allow.get("total"),
        "gate_llm_review": review.get("total"),
        "gate_block": block.get("total"),
        "n_samples": (main or {}).get("n_samples"),
        "n_malicious": n_malicious,
        "n_benign": (main or {}).get("n_benign"),
        # ---- 校准增益（LLM/rules 为 null）----
        "delta_f1": delta_f1,
        # ---- 分域 ----
        "per_domain": per_domain,
        # ---- 速度 ----
        "speed": speed if isinstance(speed, dict) else None,
        # ---- 训练成本 ----
        "train_seconds": train_seconds,
        "total_seconds": total_seconds,
        "train_human": _fmt_duration(train_seconds),
        "total_human": _fmt_duration(total_seconds),
        # ---- 训练环境（补丁 12：本机 8 / 机房 16，报告须注明 batch 差异）----
        "batch_size": _batch_of(out_dir),
        # ---- 元 ----
        "parse_error_rate": _round((main or {}).get("parse_error_rate")),
        "api_error_rate": _round((main or {}).get("api_error_rate")),
        "prompt_template": (prompt_sel or {}).get("selected"),
        "hpo_best": (hpo or {}).get("best") if isinstance(hpo, dict) else None,
        # 2026-10-02：HPO best 与 shared 一致 → 未二次训练的复用注明
        "hpo_skip_retrain": bool((hpo or {}).get("skip_retrain")) if isinstance(hpo, dict) else False,
        "hpo_reuse_from": (hpo or {}).get("reuse_from") if isinstance(hpo, dict) else None,
        "artifacts": {
            "eval_raw": os.path.isfile(os.path.join(out_dir, "eval_raw.json")),
            "eval_calib": os.path.isfile(os.path.join(out_dir, "eval_calib.json")),
            "calibration_params": os.path.isfile(
                os.path.join(out_dir, "calibration_params.json")),
            "speed": os.path.isfile(os.path.join(out_dir, "speed.json")),
            "train_meta": os.path.isfile(os.path.join(out_dir, "train_meta.json")),
        },
    }


def _cell(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(payload: dict, rows: list, env_keys: list, baseline_env) -> str:
    out = []
    out.append(f"# CalibGuard Bench 交叉对比报告（data_version = "
               f"{payload.get('data_version', DEFAULT_DATA_VERSION)}）")
    out.append("")
    out.append(f"- 生成时间：{payload['generated_at']}")
    out.append(f"- 数据版本（§3.0）：`{payload.get('data_version', DEFAULT_DATA_VERSION)}`"
               "——跨版本结果禁止混表，历史版本归档 results_archive/{data_version}/")
    out.append(f"- 数据划分：results/_splits（四段 70/10/10/10，seed 42；"
               f"全部实验共用同一 test 段 {payload.get('test_count') or '—'} 条）")
    out.append("- 口径：acc/P/R/F1 取校准后 test 口径；LLM/rules 无 Platt 校准，"
               "F1/acc/P/R 取判定布尔口径、AUC/ECE/Brier/三桶取 confidence 概率口径（ΔF1 记 —）")
    out.append("- 训练 batch（2026-10-02，云端训练）：codebert 系 **16**（与 v0.2.2 机房一致）、"
               "多基座 LLM **8**（保持不变）；逐实验 batch 见第 5 节。")
    if payload.get("has_cross_version_baseline"):
        out.append("- **v0.2.2_baseline 为跨数据版本对照**：旧模型在当前 data_version 的 "
                   "test 段上评（旧模型遇新威胁的泛化口径，2026-10-02 修订），"
                   "其指标不与同版本训练实验直接混读。")
    out.append("- 多环境测速（补丁 11）：`speed.json` 是本报告唯一读取的速度文件；"
               "机房结果另存 `speed_gpu.json`、本机另存 `speed_local.json` 供人工对比"
               "（每份自带 env，不混口径）。")
    if payload.get("excluded_results"):
        out.append(f"- 已排除非实验集目录（不进本表）：{payload['excluded_results']}")
    out.append("")

    out.append("## 1. 质量与产品口径（test 段）")
    out.append("")
    out.append("| 实验 | 类型 | acc | precision | recall | F1 | AUC | ECE | Brier | ΔF1 | "
               "静默放行率 | block 纯度 | 三桶 allow/review/block |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        buckets = (f"{_cell(row['gate_allow'])}/{_cell(row['gate_llm_review'])}/"
                   f"{_cell(row['gate_block'])}")
        out.append(
            f"| {row['name']} | {row['type']} | {_cell(row['acc'])} | "
            f"{_cell(row['precision'])} | {_cell(row['recall'])} | {_cell(row['f1'])} | "
            f"{_cell(row['auc'])} | {_cell(row['ece'])} | {_cell(row['brier'])} | "
            f"{_cell(row['delta_f1'])} | {_cell(row['silent_allow_rate'])} | "
            f"{_cell(row['block_purity'])} | {buckets} |")
    out.append("")
    out.append("> ΔF1 = eval_calib.F1 − eval_raw.F1（校准增益）；静默放行率 = allow 桶恶意 / "
               "恶意总数；block 纯度 = block 桶恶意 / block 桶总数。")
    out.append("")

    out.append("## 2. 推理速度（全链路 = 预处理 + 滑窗推理 + 校准）")
    out.append("")
    out.append("| 实验 | files/min | windows/s | short 中位/p90 | medium 中位/p90 | "
               "long 中位/p90 | env(cpu, threads) | network |")
    out.append("|---|---|---|---|---|---|---|---|")
    for row, env_key in zip(rows, env_keys):
        speed = row.get("speed") or {}
        latency = speed.get("latency_sec") or {}
        env = speed.get("env") or {}

        def bucket(name, latency=latency):
            info = latency.get(name) or {}
            if info.get("median") is None:
                return "—"
            return f"{info.get('median')}s / {info.get('p90')}s"

        same_env = (env.get("cpu"), env.get("threads")) == baseline_env
        cells = [_cell(speed.get("files_per_min_overall"), 2),
                 _cell(speed.get("windows_per_sec_mean"), 2),
                 bucket("short"), bucket("medium"), bucket("long"),
                 f"{env.get('cpu') or '—'}, {env.get('threads') if env.get('threads') else '—'}",
                 str(speed.get("network")) if speed else "—"]
        if speed and not same_env:
            cells = [f'<span style="color:gray">{cell}</span>' for cell in cells]
        out.append(f"| {row['name']} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("> **灰色行 = env（cpu+threads）与基准行不同，速度不可直接比较（禁止事项 6）**。"
               f"基准 env：cpu={baseline_env[0] if baseline_env else '—'}, "
               f"threads={baseline_env[1] if baseline_env else '—'}。"
               "主指标为中位数与 p90（sorted[int(0.9*(n-1))]），禁止用均值替代。"
               "network=true 表示该方案每档最多 5 个文件（LLM 网络成本），延迟含 API 往返。")
    out.append("")

    out.append("## 3. 训练成本")
    out.append("")
    out.append("| 实验 | train_seconds（纯训练） | total_seconds（含 HPO 全流程） |")
    out.append("|---|---|---|")
    for row in rows:
        out.append(f"| {row['name']} | {row['train_human'] or '—'} "
                   f"({_cell(row['train_seconds'], 1)}) | {row['total_human'] or '—'} "
                   f"({_cell(row['total_seconds'], 1)}) |")
    out.append("")

    out.append("## 4. 附录：分域 recall（test 段）")
    out.append("")
    out.append("| 实验 | code recall (n恶) | prompt recall (n恶) |")
    out.append("|---|---|---|")
    for row in rows:
        cells = []
        for domain in ("code", "prompt"):
            info = row["per_domain"].get(domain) or {}
            cells.append(f"{_cell(info.get('recall'))} ({info.get('n_malicious')})")
        out.append(f"| {row['name']} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("> 两域定义见引擎 train.py 的 _DOMAIN_BY_PREFIX（code / prompt）。")
    out.append("")

    out.append("## 5. 元信息")
    out.append("")
    out.append("| 实验 | status | batch | 指标来源 | parse_error_rate | api_error_rate | "
               "prompt 模板 | HPO best | 实验配置摘要 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        config = ", ".join(f"{key}={value}" for key, value in row["config"].items())
        hpo_best = (json.dumps(row.get("hpo_best"), ensure_ascii=False)
                    if row.get("hpo_best") else "—")
        if row.get("hpo_skip_retrain"):
            hpo_best += f"（与 shared 一致，未二次训练，复用 {row.get('hpo_reuse_from')}）"
        out.append(f"| {row['name']} | {row['status']} | {_cell(row.get('batch_size'))} | "
                   f"{row['metrics_source'] or '—'} | "
                   f"{_cell(row['parse_error_rate'])} | {_cell(row['api_error_rate'])} | "
                   f"{row.get('prompt_template') or '—'} | {hpo_best} | {config} |")
    out.append("")
    return "\n".join(out) + "\n"


def cmd_report(args) -> int:
    matrix, _shared, experiments = load_matrix()
    data_version = data_version_of(matrix)
    rows = [collect_row(exp) for exp in experiments]
    env_keys = []
    for row in rows:
        env = (row.get("speed") or {}).get("env") or {}
        env_keys.append((env.get("cpu"), env.get("threads")))
    baseline_env = next((key for key in env_keys if key[0] is not None), None)
    # 非实验集目录（如临时冒烟 results/smoke_ft）不进表，但在报告里注明已排除
    known = {exp["name"] for exp in experiments}
    excluded = []
    if os.path.isdir(RESULTS):
        for entry in sorted(os.listdir(RESULTS)):
            if entry.startswith("_") or entry in known:
                continue
            if os.path.isdir(os.path.join(RESULTS, entry)):
                excluded.append(entry)
    payload = {
        "generated_at": iso_now(),
        "data_version": data_version,
        "split_source": "results/_splits（当前统一划分源）",
        "test_count": len(read_lines(os.path.join(SPLITS, "split_test.jsonl")))
                      if os.path.isfile(os.path.join(SPLITS, "split_test.jsonl")) else None,
        "has_cross_version_baseline": any(exp.get("model_dir") for exp in experiments),
        "baseline_env": {"cpu": baseline_env[0] if baseline_env else None,
                         "threads": baseline_env[1] if baseline_env else None},
        "batch_policy": "codebert 系 16（云端，与 v0.2.2 一致）；多基座 LLM 8（保持不变，2026-10-02）",
        "excluded_results": excluded,
        "experiments": rows,
    }
    write_json(os.path.join(ROOT, "benchmark.json"), payload)
    with open(os.path.join(ROOT, "benchmark_report.md"), "w",
              encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(payload, rows, env_keys, baseline_env))
    print(f"[report] 已生成 benchmark.json 与 benchmark_report.md"
          f"（data_version={data_version}，{len(rows)} 个实验"
          f"{'，已排除 ' + str(excluded) if excluded else ''}）")
    return 0


# ---------------------------------------------------------------- 入口
def main(argv=None) -> int:
    injected = load_dotenv()
    os.chdir(ROOT)
    if os.path.isdir(HF_CACHE):        # .hf_cache 存在时本进程也走离线缓存（补丁 7）
        os.environ["HF_HOME"] = HF_CACHE
        os.environ["HF_HUB_OFFLINE"] = "1"

    parser = argparse.ArgumentParser(
        prog="harness.py", description="CalibGuard Bench 实验矩阵驱动（TH-2）")
    sub = parser.add_subparsers(dest="command", required=True)

    parser_init = sub.add_parser("init", help="复制统一划分清单到 results/_splits/")
    parser_init.set_defaults(func=cmd_init)

    parser_train = sub.add_parser("train", help="执行实验队列（严格串行）")
    parser_train.add_argument("--all", action="store_true", help="遍历 experiments.yaml")
    parser_train.add_argument("--only", default=None,
                              help="只跑指定实验（逗号分隔；冒烟与单实验重跑用）")
    parser_train.add_argument("--dry-run", action="store_true", help="只打印将执行的命令序列")
    parser_train.add_argument("--epochs", type=int, default=None,
                              help="覆盖训练轮数（仅冒烟/调试用）")
    parser_train.add_argument("--batch-size", type=int, default=None,
                              help="覆盖批大小（仅冒烟/调试用）")
    parser_train.set_defaults(func=cmd_train)

    parser_speed = sub.add_parser("speed", help="对全部 trained 实验跑速度基准")
    parser_speed.add_argument("--all", action="store_true")
    parser_speed.add_argument("--threads", type=int, default=None,
                              help="torch 线程数（默认 os.cpu_count()，对比时必须同值）")
    parser_speed.set_defaults(func=cmd_speed)

    parser_report = sub.add_parser("report", help="生成 benchmark.json + benchmark_report.md")
    parser_report.set_defaults(func=cmd_report)

    parser_hpo = sub.add_parser(
        "hpo", help="fallback 网格 HPO（补丁 12：importlib 加载引擎 hposearch.py，"
                    "仅覆盖 _BATCH，引擎文件字节不变）")
    parser_hpo.add_argument("--batch", type=int, default=None,
                            help="覆盖 hposearch 的 _BATCH（默认按基座定档："
                                 "codebert 16 / 其余 8，2026-10-02）")
    parser_hpo.add_argument("--base-model", default="microsoft/codebert-base",
                            help="覆盖 hposearch 的 _BASE（多基座 per_model HPO，补丁 17；"
                                 "引擎文件字节不变，可校 sha256）")
    parser_hpo.set_defaults(func=cmd_hpo)

    args = parser.parse_args(argv)
    if injected:
        print(f"[harness] 已从 .env 注入环境变量：{injected}（值不回显）")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
