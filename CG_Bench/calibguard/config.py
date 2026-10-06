"""配置加载：config.yaml + 环境变量覆盖。"""
import os
import sys
from dataclasses import dataclass

import yaml


@dataclass
class AppConfig:
    # 模型
    model_base: str = "microsoft/codebert-base"
    model_path: str = "models/codebert_finetuned"
    max_length: int = 512
    window_stride: int = 256
    window_batch_size: int = 8
    # 校准
    calibration_params_path: str = "models/calibration_params.json"
    # 门控
    allow_threshold: float = 0.15
    block_threshold: float = 0.85
    early_stop_threshold: float = 0.9
    # LLM
    llm_api_key_env: str = "DEEPSEEK_API_KEY"   # 读取 API Key 的环境变量名（可被 yaml 覆盖）
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 500
    llm_timeout: int = 30
    evidence_max_chars: int = 3000
    # 报告
    report_output_dir: str = "reports"


def load_config(path: str | None = None) -> AppConfig:
    """加载配置。规则：默认值 < config.yaml < 环境变量（仅 API Key）。"""
    cfg = AppConfig()
    if path is not None and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # 逐键覆盖（键不存在则保留默认值，禁止因缺键报错）
        m = raw.get("model", {}) or {}
        if "base_model" in m: cfg.model_base = str(m["base_model"])
        if "finetuned_path" in m: cfg.model_path = str(m["finetuned_path"])
        if "max_length" in m: cfg.max_length = int(m["max_length"])
        if "window_stride" in m: cfg.window_stride = int(m["window_stride"])
        if "window_batch_size" in m: cfg.window_batch_size = int(m["window_batch_size"])
        c = raw.get("calibration", {}) or {}
        if "params_path" in c: cfg.calibration_params_path = str(c["params_path"])
        g = raw.get("gate", {}) or {}
        if "allow_threshold" in g: cfg.allow_threshold = float(g["allow_threshold"])
        if "block_threshold" in g: cfg.block_threshold = float(g["block_threshold"])
        if "early_stop_threshold" in g: cfg.early_stop_threshold = float(g["early_stop_threshold"])
        llm = raw.get("llm", {}) or {}
        if "api_key" in llm: cfg.llm_api_key = str(llm["api_key"])
        if "base_url" in llm: cfg.llm_base_url = str(llm["base_url"])
        if "model" in llm: cfg.llm_model = str(llm["model"])
        if "temperature" in llm: cfg.llm_temperature = float(llm["temperature"])
        if "max_tokens" in llm: cfg.llm_max_tokens = int(llm["max_tokens"])
        if "timeout_seconds" in llm: cfg.llm_timeout = int(llm["timeout_seconds"])
        if "evidence_max_chars" in llm: cfg.evidence_max_chars = int(llm["evidence_max_chars"])
        if "api_key_env" in llm: cfg.llm_api_key_env = str(llm["api_key_env"])
        r = raw.get("report", {}) or {}
        if "output_dir" in r: cfg.report_output_dir = str(r["output_dir"])
    # API Key 环境变量优先级最高——必须在 if 块之外执行：
    # 无论是否指定 --config、yaml 是否存在，环境变量都要读取
    # （未加载 yaml 时，环境变量名用字段默认值 "DEEPSEEK_API_KEY"）
    env_val = os.environ.get(cfg.llm_api_key_env, "")
    if env_val:
        cfg.llm_api_key = env_val
    # 合法性校验：失败直接退出，退出码 2
    if not (0.0 < cfg.allow_threshold < cfg.block_threshold < 1.0):
        print(f"[配置错误] 阈值必须满足 0 < allow({cfg.allow_threshold}) < "
              f"block({cfg.block_threshold}) < 1", file=sys.stderr)
        sys.exit(2)
    if cfg.max_length > 512:
        print(f"[配置错误] max_length 不得超过 512（CodeBERT 上限），当前 {cfg.max_length}",
              file=sys.stderr)
        sys.exit(2)
    if not (1 <= cfg.window_stride < cfg.max_length):
        print(f"[配置错误] window_stride 必须满足 1 ≤ stride < max_length(512)，当前 {cfg.window_stride}",
              file=sys.stderr)
        sys.exit(2)
    if cfg.window_batch_size < 1:
        print(f"[配置错误] window_batch_size 必须 ≥ 1，当前 {cfg.window_batch_size}",
              file=sys.stderr)
        sys.exit(2)
    if not (0.0 < cfg.early_stop_threshold < 1.0):
        print(f"[配置错误] early_stop_threshold 必须在 (0,1) 内，当前 {cfg.early_stop_threshold}",
              file=sys.stderr)
        sys.exit(2)
    return cfg
