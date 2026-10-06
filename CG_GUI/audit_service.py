"""审计任务封装（T-G2.5）：把 calibguard.pipeline 包成“任务”模型供 API 层调用。

铁律：
  - 后端审计单并发：_JOB_LOCK 一把全局锁，后来的任务排队等待，禁止多 worker 并行；
  - 逐字复用引擎入口（load_config + AuditPipeline），禁止重写审计逻辑；
  - 引擎无阶段回调钩子 → stage 只写 "model" → "report" 两档，禁止伪造细粒度进度；
  - 单文件异常记账后继续下一个文件，异常文本必须能冒泡到 UI。

设置（gui-settings.json，2026-10-05 扩展）：
  - 是否启用云端 LLM 复核、API Key、Base URL、模型名；
  - 灰区范围（allow/block 阈值，默认 0.15 / 0.85）；
  - LLM 送审内容源（精炼证据 / 文件原文）× 投递模式（截断 / 滑窗 / 一次性全部上传）
    及其随附参数（截断字符数、滑窗字符数与重叠字符数）。
  - 应用方式：直接改运行中 pipeline 的 cfg 字段 + 重建 LLM 客户端，**不重载模型**
    （模型加载耗时数十秒，而上述设置都不影响模型权重与校准参数）。
"""
import dataclasses
import json
import os
import threading
import uuid

from calibguard.config import LLM_CONTENT_SOURCES, LLM_DELIVERY_MODES, load_config, validate_config
from calibguard.llm_analyzer import LLMAnalyzer
from calibguard.pipeline import AuditPipeline

_ROOT = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(_ROOT, "gui-settings.json")

_JOB_LOCK = threading.Lock()                    # 全局单并发锁
_JOBS: dict[str, dict] = {}                     # job_id -> 状态
_PIPELINE: AuditPipeline | None = None          # 引擎流水线单例（模型只加载一次）
_PENDING_SETTINGS_REFRESH = False               # 审计运行期间被暂缓的设置应用请求

# 阶段取值封闭集合（引擎无回调钩子，实际只会出现 model / report 两档）
STAGE_MODEL = "model"
STAGE_REPORT = "report"

# ============================== 设置 ==============================

# 默认设置（gui-settings.json 缺键时用这些值；与引擎 AppConfig 默认值保持一致）
DEFAULT_SETTINGS: dict = {
    "llm_enabled": True,
    "deepseek_api_key": "",
    "llm_base_url": "https://api.deepseek.com",
    "llm_model": "deepseek-chat",
    "allow_threshold": 0.15,
    "block_threshold": 0.85,
    "llm_content_source": "evidence",
    "llm_delivery_mode": "truncate",
    "llm_truncate_chars": 3000,
    "llm_window_chars": 3000,
    "llm_window_overlap": 500,
}

# 设置键 → 引擎 AppConfig 字段（用于把设置写进运行中的 cfg）
_CFG_FIELDS = {
    "llm_enabled": "llm_enabled",
    "deepseek_api_key": "llm_api_key",
    "llm_base_url": "llm_base_url",
    "llm_model": "llm_model",
    "allow_threshold": "allow_threshold",
    "block_threshold": "block_threshold",
    "llm_content_source": "llm_content_source",
    "llm_delivery_mode": "llm_delivery_mode",
    "llm_truncate_chars": "llm_truncate_chars",
    "llm_window_chars": "llm_window_chars",
    "llm_window_overlap": "llm_window_overlap",
}

_SETTINGS: dict = dict(DEFAULT_SETTINGS)


def _coerce_settings(raw: dict) -> dict:
    """把外部输入（JSON）规范化为合法设置字典；任何非法项抛 ValueError（中文说明）。

    只接受已知键；未知键忽略（向前兼容）。
    """
    if not isinstance(raw, dict):
        raise ValueError("设置必须是 JSON 对象")
    merged = dict(DEFAULT_SETTINGS)
    for key in DEFAULT_SETTINGS:
        if key in raw:
            merged[key] = raw[key]

    if not isinstance(merged["llm_enabled"], bool):
        raise ValueError("llm_enabled 必须是布尔值")
    for key in ("deepseek_api_key", "llm_base_url", "llm_model"):
        if not isinstance(merged[key], str):
            raise ValueError(f"{key} 必须是字符串")
    if merged["llm_enabled"] and not merged["llm_base_url"].strip():
        raise ValueError("启用云端复核时 llm_base_url 不能为空")
    if merged["llm_enabled"] and not merged["llm_model"].strip():
        raise ValueError("启用云端复核时 llm_model 不能为空")

    for key in ("allow_threshold", "block_threshold"):
        try:
            merged[key] = float(merged[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须是 0~1 之间的数字") from exc
    if not (0.0 < merged["allow_threshold"] < merged["block_threshold"] < 1.0):
        raise ValueError(
            f"灰区范围必须满足 0 < 下限({merged['allow_threshold']}) < "
            f"上限({merged['block_threshold']}) < 1")

    if merged["llm_content_source"] not in LLM_CONTENT_SOURCES:
        raise ValueError(f"llm_content_source 必须是 {LLM_CONTENT_SOURCES} 之一")
    if merged["llm_delivery_mode"] not in LLM_DELIVERY_MODES:
        raise ValueError(f"llm_delivery_mode 必须是 {LLM_DELIVERY_MODES} 之一")
    for key in ("llm_truncate_chars", "llm_window_chars", "llm_window_overlap"):
        try:
            merged[key] = int(merged[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须是整数") from exc
    if merged["llm_truncate_chars"] < 1:
        raise ValueError("截断大小必须 ≥ 1 个字符")
    if merged["llm_window_chars"] < 1:
        raise ValueError("滑窗大小必须 ≥ 1 个字符")
    if not (0 <= merged["llm_window_overlap"] < merged["llm_window_chars"]):
        raise ValueError(
            f"滑窗重叠必须满足 0 ≤ 重叠 < 滑窗大小({merged['llm_window_chars']})")
    return merged


def get_settings() -> dict:
    """返回当前生效设置的副本（供 GET /api/settings 与 UI 回填）。"""
    return dict(_SETTINGS)


def _apply_settings_to_cfg(cfg, settings: dict) -> None:
    """把设置写进 AppConfig 对应字段（键名映射见 _CFG_FIELDS）。"""
    for key, field_name in _CFG_FIELDS.items():
        setattr(cfg, field_name, settings[key])


def _build_llm(cfg):
    """按 cfg 构造云端 LLM 客户端（Key/BaseURL/模型名变更后重建）。"""
    return LLMAnalyzer(
        api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, model=cfg.llm_model,
        temperature=cfg.llm_temperature, max_tokens=cfg.llm_max_tokens,
        timeout=cfg.llm_timeout, evidence_max_chars=cfg.evidence_max_chars)


def _apply_settings_locked() -> None:
    """把当前设置应用到运行中的 pipeline（调用方必须已持有 _JOB_LOCK）。

    只改 cfg 字段 + 重建 LLM 客户端，**不重载模型**：这样设置变更绝不会与正在进行的
    审计并发改写状态（补丁 11 约束），也不付出模型加载代价。
    """
    global _PENDING_SETTINGS_REFRESH
    if _PIPELINE is None:
        _PENDING_SETTINGS_REFRESH = False        # 流水线未建：下次创建自然读新设置
        return
    _apply_settings_to_cfg(_PIPELINE.cfg, _SETTINGS)
    _PIPELINE.llm = _build_llm(_PIPELINE.cfg)
    _PENDING_SETTINGS_REFRESH = False


def _refresh_settings() -> bool:
    """在 _JOB_LOCK 保护下应用设置；审计运行中则暂缓（不阻塞等待）。

    不阻塞的原因：设置请求若等锁，会在审计期间把 HTTP 请求挂住；补丁 11 要求
    “审计运行中禁止触发重建，UI 提示当前任务完成后生效”，故记待办后立即返回。
    第二次尝试用于消解与任务收尾的竞态（锁刚释放时仍可即时生效）。
    """
    global _PENDING_SETTINGS_REFRESH
    for _ in range(2):
        if _JOB_LOCK.acquire(blocking=False):
            try:
                _apply_settings_locked()
                return True
            finally:
                _JOB_LOCK.release()
        _PENDING_SETTINGS_REFRESH = True         # 有任务在跑：暂缓
    return False


def load_settings() -> None:
    """读 gui-settings.json，刷新内存设置，并把 Key 注入环境变量 DEEPSEEK_API_KEY。

    引擎的 load_config 从环境变量读取 Key（主文档 6.2），本函数是 GUI 侧唯一注入点。
    铁律：禁止读写 calibguard 的 config.yaml；禁止把 Key 写进任何日志。
    """
    global _SETTINGS
    if not os.path.isfile(SETTINGS_PATH):
        return
    with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"设置文件格式错误（应为 JSON 对象）：{SETTINGS_PATH}")
    _SETTINGS = _coerce_settings(payload)
    _inject_api_key(_SETTINGS["deepseek_api_key"])


def _inject_api_key(key: str) -> None:
    """把 Key 注入（或清除）环境变量：留空即不使用云端（不留残余环境值）。"""
    if key:
        os.environ["DEEPSEEK_API_KEY"] = key
    else:
        os.environ.pop("DEEPSEEK_API_KEY", None)


def save_settings(payload: dict) -> bool:
    """合并并落盘设置（UTF-8 无 BOM），立即让配置生效。

    支持部分更新：payload 中未出现的键保持当前值（旧版只传 deepseek_api_key 的调用
    仍然可用）。返回设置是否已即时生效：False 表示当前有审计任务在跑（_JOB_LOCK 被
    占用），应用已暂缓，等当前任务结束由 _run_job 落地（补丁 11 约束一）。
    """
    global _SETTINGS
    if not isinstance(payload, dict):
        raise ValueError("设置必须是 JSON 对象")
    merged = dict(_SETTINGS)
    merged.update(payload)
    settings = _coerce_settings(merged)

    with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)
    _SETTINGS = settings
    _inject_api_key(settings["deepseek_api_key"])
    return _refresh_settings()


def _get_pipeline() -> AuditPipeline:
    """引擎流水线单例：首次调用加载模型（可耗时数十秒），之后全程复用。"""
    global _PIPELINE
    if _PIPELINE is None:
        # 铁律 7：禁止读写 calibguard 的 config.yaml → 传 None 走引擎内置默认值，
        # 相对路径（models/、reports/）由 server 的 cwd（项目根）解析。
        cfg = load_config()
        _apply_settings_to_cfg(cfg, _SETTINGS)
        error = validate_config(cfg)
        if error:
            raise ValueError(f"设置非法，无法初始化引擎：{error}")
        _PIPELINE = AuditPipeline(cfg)
    return _PIPELINE


# ============================== 任务模型 ==============================

def create_job(paths: list[str]) -> str:
    """建 job（排队态），返回 job_id；执行交给 daemon 线程 + 全局单并发锁。"""
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "state": "queued",
        "paths": list(paths),
        "stage": "",
        "done": 0,
        "total": len(paths),
        "results": [],
    }
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return job_id


def job_status(job_id: str) -> dict:
    """查询任务状态/进度/结果；未知 id 抛 KeyError（API 层转 404）。

    返回浅拷贝：handler 线程序列化时后台线程仍在推进 job，避免“字典迭代期间改变大小”。
    """
    return dict(_JOBS[job_id])


def _report_to_dict(report) -> dict:
    """引擎报告归一化：CGv0.2.2 的 audit_file 直接返回 dict，原样透传（不增删字段）。"""
    if isinstance(report, dict):
        return report
    if hasattr(report, "to_dict"):
        return report.to_dict()
    if dataclasses.is_dataclass(report):
        return dataclasses.asdict(report)
    raise TypeError(f"引擎返回了无法序列化的报告对象：{type(report).__name__}")


def _run_job(job_id: str) -> None:
    """单并发执行任务：逐文件调用引擎，单文件异常记账后继续。"""
    job = _JOBS[job_id]
    with _JOB_LOCK:                                  # 后来的任务在此排队等待
        job["state"] = "running"
        try:
            pipeline = _get_pipeline()
        except Exception as exc:
            # 模型/校准参数/设置不可用属全局失败：显式暴露，绝不假成功
            job["error"] = f"引擎初始化失败：{type(exc).__name__}: {exc}"
            job["stage"] = STAGE_REPORT
            for path in job["paths"]:
                job["results"].append({"error": job["error"], "path": path})
            job["done"] = job["total"]
            job["state"] = "error"
        else:
            for path in job["paths"]:
                job["stage"] = STAGE_MODEL               # 引擎无回调钩子：粗粒度两档
                try:
                    report = pipeline.audit_file(path)
                    job["results"].append(_report_to_dict(report))
                except Exception as exc:
                    job["results"].append({"error": f"{type(exc).__name__}: {exc}", "path": path})
                job["stage"] = STAGE_REPORT
                job["done"] += 1
            job["state"] = "done"

    # 任务已结束（锁已释放）：落地审计运行期间被暂缓的设置更新（补丁 11 约束一）
    if _PENDING_SETTINGS_REFRESH:
        _refresh_settings()
