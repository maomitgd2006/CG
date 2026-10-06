"""配置加载：config.yaml + 环境变量覆盖。"""
import os
import sys
from dataclasses import dataclass

import yaml

# LLM 复核内容源（封闭集合）：evidence=精炼证据 / raw=预处理后的文件原文
LLM_CONTENT_SOURCES = ("evidence", "raw")
# LLM 复核投递模式（封闭集合）：truncate=截断 / window=滑窗 / all=一次性全部上传
LLM_DELIVERY_MODES = ("truncate", "window", "all")


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
    llm_enabled: bool = True                    # 是否使用云端 LLM 复核（False 时灰区按 0.5 切）
    llm_api_key_env: str = "DEEPSEEK_API_KEY"   # 读取 API Key 的环境变量名（可被 yaml 覆盖）
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 500
    llm_timeout: int = 30
    evidence_max_chars: int = 3000
    # LLM 复核投递（2026-10-05 GUI 需求：内容源 × 投递模式）
    llm_content_source: str = "evidence"        # evidence=精炼证据 / raw=文件原文
    llm_delivery_mode: str = "truncate"         # truncate / window / all
    llm_truncate_chars: int = 3000              # 截断模式：送审前 N 字符
    llm_window_chars: int = 3000                # 滑窗模式：单窗字符数
    llm_window_overlap: int = 500               # 滑窗模式：相邻窗重叠字符数
    # 报告
    report_output_dir: str = "reports"


def validate_config(cfg: AppConfig) -> str | None:
    """校验配置合法性：返回错误文本；None 表示通过。

    抽成独立函数供两处复用：load_config（非法则退出码 2）与 GUI 设置保存前的预检
    （非法则 400，绝不把非法值写进运行中的 cfg）。
    """
    if not (0.0 < cfg.allow_threshold < cfg.block_threshold < 1.0):
        return (f"阈值必须满足 0 < allow({cfg.allow_threshold}) < "
                f"block({cfg.block_threshold}) < 1")
    if cfg.max_length > 512:
        return f"max_length 不得超过 512（CodeBERT 上限），当前 {cfg.max_length}"
    if not (1 <= cfg.window_stride < cfg.max_length):
        return (f"window_stride 必须满足 1 ≤ stride < max_length(512)，"
                f"当前 {cfg.window_stride}")
    if cfg.window_batch_size < 1:
        return f"window_batch_size 必须 ≥ 1，当前 {cfg.window_batch_size}"
    if not (0.0 < cfg.early_stop_threshold < 1.0):
        return f"early_stop_threshold 必须在 (0,1) 内，当前 {cfg.early_stop_threshold}"
    if cfg.llm_content_source not in LLM_CONTENT_SOURCES:
        return (f"llm_content_source 必须是 {LLM_CONTENT_SOURCES} 之一，"
                f"当前 {cfg.llm_content_source!r}")
    if cfg.llm_delivery_mode not in LLM_DELIVERY_MODES:
        return (f"llm_delivery_mode 必须是 {LLM_DELIVERY_MODES} 之一，"
                f"当前 {cfg.llm_delivery_mode!r}")
    if cfg.llm_truncate_chars < 1:
        return f"llm_truncate_chars 必须 ≥ 1，当前 {cfg.llm_truncate_chars}"
    if cfg.llm_window_chars < 1:
        return f"llm_window_chars 必须 ≥ 1，当前 {cfg.llm_window_chars}"
    if not (0 <= cfg.llm_window_overlap < cfg.llm_window_chars):
        return (f"llm_window_overlap 必须满足 0 ≤ overlap < window_chars"
                f"({cfg.llm_window_chars})，当前 {cfg.llm_window_overlap}")
    return None


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
        if "enabled" in llm: cfg.llm_enabled = bool(llm["enabled"])
        if "api_key" in llm: cfg.llm_api_key = str(llm["api_key"])
        if "base_url" in llm: cfg.llm_base_url = str(llm["base_url"])
        if "model" in llm: cfg.llm_model = str(llm["model"])
        if "temperature" in llm: cfg.llm_temperature = float(llm["temperature"])
        if "max_tokens" in llm: cfg.llm_max_tokens = int(llm["max_tokens"])
        if "timeout_seconds" in llm: cfg.llm_timeout = int(llm["timeout_seconds"])
        if "evidence_max_chars" in llm: cfg.evidence_max_chars = int(llm["evidence_max_chars"])
        if "api_key_env" in llm: cfg.llm_api_key_env = str(llm["api_key_env"])
        if "content_source" in llm: cfg.llm_content_source = str(llm["content_source"])
        if "delivery_mode" in llm: cfg.llm_delivery_mode = str(llm["delivery_mode"])
        if "truncate_chars" in llm: cfg.llm_truncate_chars = int(llm["truncate_chars"])
        if "window_chars" in llm: cfg.llm_window_chars = int(llm["window_chars"])
        if "window_overlap" in llm: cfg.llm_window_overlap = int(llm["window_overlap"])
        r = raw.get("report", {}) or {}
        if "output_dir" in r: cfg.report_output_dir = str(r["output_dir"])
    # API Key 环境变量优先级最高——必须在 if 块之外执行：
    # 无论是否指定 --config、yaml 是否存在，环境变量都要读取
    # （未加载 yaml 时，环境变量名用字段默认值 "DEEPSEEK_API_KEY"）
    env_val = os.environ.get(cfg.llm_api_key_env, "")
    if env_val:
        cfg.llm_api_key = env_val
    # 合法性校验：失败直接退出，退出码 2
    error = validate_config(cfg)
    if error:
        print(f"[配置错误] {error}", file=sys.stderr)
        sys.exit(2)
    return cfg
