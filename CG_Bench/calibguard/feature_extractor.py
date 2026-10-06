"""特征提取模块：文本 → 结构化可疑证据（FeatureEvidence）+ 证据转提示词文本。"""
import math
import re

from calibguard import constants as C
from calibguard.schemas import FeatureEvidence

# 高熵候选串（最短长度取常量 HIGH_ENTROPY_MIN_LEN）
_HIGH_ENTROPY_CANDIDATE_RE = re.compile(r"[A-Za-z0-9+/=_-]{%d,}" % C.HIGH_ENTROPY_MIN_LEN)
# 长随机标识符（首字符字母或下划线 + 其余为标识符合法字符，总长 ≥ RANDOM_IDENTIFIER_MIN_LEN）
# 字符类必须包含下划线：混淆器生成的随机变量名普遍为 var_xxxxx 形态，遗漏 _ 会使该检测失效
_RANDOM_IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]{%d,}\b" % (C.RANDOM_IDENTIFIER_MIN_LEN - 1)
)

# 隐藏 Unicode 命中时写入混淆类的固定条目
_HIDDEN_UNICODE_ITEM = "隐藏Unicode字符"
# 证据文本的八个固定标题（顺序固定，禁止调整）
_SECTION_TITLES = [
    "【危险API调用】",
    "【网络行为】",
    "【凭据与敏感路径访问】",
    "【编码与混淆】",
    "【可疑文件操作】",
    "【提示注入话术】",
    "【高熵字符串】",
    "【异常结构特征】",
]
# 全部类别为空时的固定返回值
_EMPTY_EVIDENCE_TEXT = "未发现明显可疑信号。"


def extract_features(text: str) -> FeatureEvidence:
    """对全文抽取七类可疑信号 + 异常特征。"""
    evidence = FeatureEvidence()

    evidence.dangerous_apis = _collect_matches(C.DANGEROUS_API_PATTERNS, text)
    evidence.network_indicators = _collect_matches(C.NETWORK_PATTERNS, text)
    evidence.credential_access = _collect_matches(C.CREDENTIAL_PATTERNS, text)
    evidence.obfuscation_signals = _collect_matches(C.OBFUSCATION_PATTERNS, text)
    evidence.file_operations = _collect_matches(C.FILE_OPERATION_PATTERNS, text)

    # 隐藏 Unicode 检测：命中则追加固定条目（仅一条，去重）
    if re.search(C.HIDDEN_UNICODE_PATTERN, text):
        if _HIDDEN_UNICODE_ITEM not in evidence.obfuscation_signals:
            evidence.obfuscation_signals.append(_HIDDEN_UNICODE_ITEM)
        evidence.obfuscation_signals = evidence.obfuscation_signals[
            :C.EVIDENCE_PER_CATEGORY_MAX
        ]

    # 提示注入话术：不区分大小写，命中整条话术原文（不截断）
    lowered = text.lower()
    injection_hits: list[str] = []
    for phrase in C.INJECTION_PHRASES:
        if phrase.lower() in lowered and phrase not in injection_hits:
            injection_hits.append(phrase)
    evidence.injection_phrases = injection_hits[:C.EVIDENCE_PER_CATEGORY_MAX]

    # 异常特征
    evidence.high_entropy_strings = _find_high_entropy_strings(text)
    evidence.long_line_numbers = _find_long_lines(text)
    random_count = len(_RANDOM_IDENTIFIER_RE.findall(text))
    if random_count >= C.RANDOM_IDENTIFIER_THRESHOLD:
        evidence.random_identifier_count = random_count

    return evidence


def _collect_matches(patterns: list[str], text: str) -> list[str]:
    """逐个模式收集匹配串 → 去重（保持首次出现顺序）→ 截断 → 每类最多 10 条。"""
    unique_hits: list[str] = []
    for pattern in patterns:
        for matched in re.findall(pattern, text):
            if matched not in unique_hits:
                unique_hits.append(matched)
            if len(unique_hits) >= C.EVIDENCE_PER_CATEGORY_MAX:
                break
        if len(unique_hits) >= C.EVIDENCE_PER_CATEGORY_MAX:
            break
    return [_clip(item) for item in unique_hits[:C.EVIDENCE_PER_CATEGORY_MAX]]


def _find_long_lines(text: str) -> list[int]:
    """行长 ≥ LONG_LINE_THRESHOLD 的行号（1 起），最多 LONG_LINE_MAX_COUNT 个。"""
    line_numbers: list[int] = []
    for index, line in enumerate(text.split("\n"), start=1):
        if len(line) >= C.LONG_LINE_THRESHOLD:
            line_numbers.append(index)
            if len(line_numbers) >= C.LONG_LINE_MAX_COUNT:
                break
    return line_numbers


def _shannon_entropy(s: str) -> float:
    """Shannon 熵：H = -Σ p_i * log2(p_i)。"""
    if s == "":
        return 0.0
    length = len(s)
    entropy = 0.0
    for char in set(s):
        probability = s.count(char) / length
        entropy -= probability * math.log2(probability)
    return entropy


def _find_high_entropy_strings(text: str) -> list[str]:
    """高熵候选串（长度 ≥ HIGH_ENTROPY_MIN_LEN 且熵 ≥ 阈值），最多 HIGH_ENTROPY_MAX_COUNT 条。"""
    hits: list[str] = []
    for candidate in _HIGH_ENTROPY_CANDIDATE_RE.findall(text):
        if candidate in hits:
            continue
        if _shannon_entropy(candidate) >= C.HIGH_ENTROPY_THRESHOLD:
            hits.append(candidate)
            if len(hits) >= C.HIGH_ENTROPY_MAX_COUNT:
                break
    return [_clip(item) for item in hits]


def evidence_to_prompt_text(evidence: FeatureEvidence, max_chars: int) -> str:
    """证据 → 固定格式提示词文本（空类别整段跳过；全空返回固定串）。"""
    category_items: list[list[str]] = [
        evidence.dangerous_apis,
        evidence.network_indicators,
        evidence.credential_access,
        evidence.obfuscation_signals,
        evidence.file_operations,
        evidence.injection_phrases,
        evidence.high_entropy_strings,
    ]

    blocks: list[str] = []
    for title, items in zip(_SECTION_TITLES, category_items):
        if not items:
            continue
        lines = [title]
        lines.extend(f"- {item}" for item in items)
        blocks.append("\n".join(lines))

    # 异常结构特征：超长行行号与长随机标识符数量
    structural_lines: list[str] = []
    if evidence.long_line_numbers:
        structural_lines.append(f"- 超长行行号: {evidence.long_line_numbers}")
    if evidence.random_identifier_count > 0:
        structural_lines.append(f"- 长随机标识符数量: {evidence.random_identifier_count}")
    if structural_lines:
        blocks.append("\n".join([_SECTION_TITLES[7]] + structural_lines))

    if not blocks:
        return _EMPTY_EVIDENCE_TEXT

    prompt_text = "\n\n".join(blocks)
    if len(prompt_text) > max_chars:
        prompt_text = prompt_text[:max_chars] + "\n…[EVIDENCE_TRUNCATED]"
    return prompt_text


def _clip(s: str, limit: int = C.EVIDENCE_ITEM_MAX_CHARS) -> str:
    """单条证据超过 limit 字符则截断并追加省略号。"""
    if len(s) <= limit:
        return s
    return s[:limit] + "…"
