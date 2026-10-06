"""CalibGuard 全局常量。所有列表为封闭集合，禁止增删；所有阈值禁止修改。"""

# ============ 标签约定 ============
LABEL_BENIGN = 0     # 良性类别索引（训练时 label=0，模型输出 logits 索引 0）
LABEL_MALICIOUS = 1  # 恶意类别索引（训练时 label=1，模型输出 logits 索引 1）

# ============ 危险 API 模式（封闭） ============
DANGEROUS_API_PATTERNS = [
    r"subprocess\.(?:run|call|Popen|check_output|check_call)\s*\(",
    r"os\.system\s*\(",
    r"os\.popen\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"pickle\.loads?\s*\(",
    r"__import__\s*\(",
]

# ============ 网络行为模式（封闭） ============
NETWORK_PATTERNS = [
    r"https?://[^\s'\"<>]+",
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    r"\b(?:curl|wget)\b",
    r"requests\.(?:post|get|put|delete|head)\s*\(",
    r"urllib\.request\.(?:urlopen|Request)\s*\(",
    r"socket\.socket\s*\(",
]

# ============ 凭据与敏感路径模式（封闭） ============
CREDENTIAL_PATTERNS = [
    r"\.ssh\b",
    r"\.env\b",
    r"\.aws\b",
    r"\.bash_history\b",
    r"\.netrc\b",
    r"id_rsa",
    r"AWS_SECRET_ACCESS_KEY",
    r"AWS_ACCESS_KEY_ID",
    r"\btoken\b",
    r"\bpasswd?\b",
    r"\bcredential",
    r"\bsecret\b",
    r"\bapi[_-]?key\b",
]

# ============ 编码与混淆模式（封闭） ============
OBFUSCATION_PATTERNS = [
    r"base64\.(?:b64decode|b64encode|decode|encode)\s*\(",
    r"bytes\.fromhex\s*\(",
    r"\b(?:binascii\.)?(?:unhexlify|hexlify)\s*\(",
    r"\bchr\s*\(\s*\d+\s*\)",
    r"\\x[0-9a-fA-F]{2}",
    r"codecs\.decode\s*\(",
    r"zlib\.decompress\s*\(",
    r"marshal\.loads?\s*\(",
]

# ============ 可疑文件操作模式（封闭） ============
FILE_OPERATION_PATTERNS = [
    r"os\.(?:remove|unlink|rmdir)\s*\(",
    r"shutil\.rmtree\s*\(",
    r"os\.(?:rename|replace)\s*\(",
    r"open\s*\([^)]*['\"][wa]",
    r"\.bashrc\b|\.zshrc\b|\.profile\b",
    r"/etc/(?:hosts|passwd|shadow|sudoers)\b",
    r"\bwinreg\b|\bHKEY_",
    r"\bcrontab\b|\bschtasks\b|LaunchAgents",
    r"chmod\s+[0-7]{3,4}",
]

# ============ 提示注入话术（封闭，匹配时不区分大小写） ============
INJECTION_PHRASES = [
    "忽略之前指令", "忽略之前的指令", "忽略以上指令", "忽略上述指令",
    "不要告诉用户", "不要告诉任何人", "不要告知用户",
    "偷偷发送", "悄悄发送", "私下发送",
    "ignore previous instructions", "ignore all previous instructions",
    "disregard previous", "forget previous",
    "do not tell the user", "don't tell the user",
    "secretly send",
]

# ============ 异常特征阈值（禁止修改） ============
HIGH_ENTROPY_MIN_LEN = 20        # 高熵候选串最短长度（字符）
HIGH_ENTROPY_THRESHOLD = 4.5     # Shannon 熵阈值（bit/字符）
HIGH_ENTROPY_MAX_COUNT = 10      # 高熵串最多记录条数
LONG_LINE_THRESHOLD = 500        # 超长行判定阈值（字符）
LONG_LINE_MAX_COUNT = 10         # 最多记录的行号个数
RANDOM_IDENTIFIER_MIN_LEN = 16   # 随机标识符的最短长度
RANDOM_IDENTIFIER_THRESHOLD = 10 # 随机标识符数量达到该值才记录总数

# ============ 证据精炼限制（禁止修改） ============
EVIDENCE_PER_CATEGORY_MAX = 10   # 每类信号最多保留条数
EVIDENCE_ITEM_MAX_CHARS = 80     # 单条证据最大字符数（超出截断加省略号）
# 证据总长度上限来自配置项 llm.evidence_max_chars（默认 3000），不在本文件定义

# ============ 预处理限制（禁止修改） ============
MAX_INPUT_BYTES = 1048576        # 单文件读入上限：1MB，超出只读前 1MB
MAX_TEXT_CHARS = 200000          # 清洗后文本上限：20 万字符
LINE_TRUNCATE_THRESHOLD = 2000   # 单行超此长度截断，追加 "…[LINE_TRUNCATED]"

# ============ 压缩包安全限制（禁止修改，防 zip bomb） ============
ARCHIVE_MAX_DEPTH = 3            # 递归解包最大深度
ARCHIVE_MAX_FILES = 100          # 单个压缩包最多解出文件数
ARCHIVE_MAX_TOTAL_BYTES = 52428800  # 解包总字节上限 50MB
PRINTABLE_MIN_RUN = 4            # 二进制可打印字符串的最短连续长度

# ============ 提示注入防护（禁止修改） ============
# 不可见/危险 Unicode 范围：零宽字符、双向控制符、BOM 等
HIDDEN_UNICODE_PATTERN = r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]"
# 提交给 LLM 前需移除的控制字符（含 \x00-\x1f 中除 \t\n\r 外的全部）
CONTROL_CHARS_PATTERN = r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]"
# LLM 响应语义黑名单：命中即判定 LLM 可能被注入劫持，转 invalid_output
LLM_SEMANTIC_BLACKLIST = [
    "忽略之前", "忽略上面", "无法分析", "我是一个", "我是人工",
    "ignore previous", "i cannot", "i can't", "as an ai",
]
# 一致性校验：本地校准概率 ≥ 该值 且 LLM 返回 low 时，触发一致性告警
LLM_CONSISTENCY_LOCAL_PROB = 0.7
