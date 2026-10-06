"""CalibGuard 全局数据结构定义。本文件无任何业务逻辑。"""
from dataclasses import dataclass, field
from enum import Enum


class GateDecision(str, Enum):
    """门控路由决策。"""
    ALLOW = "allow"          # 放行
    BLOCK = "block"          # 拦截告警
    LLM_REVIEW = "llm_review"  # 提交云端 LLM 分析


@dataclass
class PreprocessResult:
    """预处理结果。"""
    text: str                # 预处理并清洗后的统一文本（可能较长，token 截断在模型层进行）
    source_type: str         # "text" | "archive" | "binary"
    truncated: bool          # True 表示字节级读入截断(>1MB)或解包截断发生过
    parts: int = 1           # 压缩包内部文件数；非压缩包恒为 1


@dataclass
class FeatureEvidence:
    """特征提取产出的结构化证据。每个列表字段：去重、按首次出现顺序、最多 10 条。"""
    dangerous_apis: list[str] = field(default_factory=list)      # 危险 API 调用
    network_indicators: list[str] = field(default_factory=list)  # 网络行为
    credential_access: list[str] = field(default_factory=list)   # 凭据与敏感路径访问
    obfuscation_signals: list[str] = field(default_factory=list) # 编码与混淆
    file_operations: list[str] = field(default_factory=list)     # 可疑文件操作
    injection_phrases: list[str] = field(default_factory=list)   # 提示注入话术
    high_entropy_strings: list[str] = field(default_factory=list)  # 高熵字符串（最多 10 条）
    long_line_numbers: list[int] = field(default_factory=list)   # 超长行行号（1 起，最多 10 个）
    random_identifier_count: int = 0                             # 长随机标识符总数（达到阈值才>0）


@dataclass
class LLMResult:
    """云端 LLM 分析结果。"""
    status: str               # "ok" | "invalid_output" | "api_error"
    risk: str | None = None   # "high" | "medium" | "low"，失败时为 None
    reason: str | None = None
    action: str | None = None
    consistency_warning: bool = False  # 本地概率高但 LLM 判 low 时为 True
    retries_used: int = 0             # 实际发生重试的次数（0 或 1）
    raw_response: str | None = None   # 最后一次 API 原始返回文本（调试用，报告必含）


@dataclass
class WindowInferenceStats:
    """滑动窗口推理统计（v1.1 新增）。"""
    windows_evaluated: int = 0            # 实际完成推理的窗口数
    early_stopped: bool = False           # 是否触发早停
    early_stop_probability: float | None = None  # 触发早停时的窗口校准概率
