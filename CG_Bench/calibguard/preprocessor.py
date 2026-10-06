"""预处理模块：任意输入文件 → 统一文本。

只按内容判定类型（魔数 + 解码尝试），不按扩展名。压缩包一律在内存中解包，
禁止 extractall / extract 落盘（因此不需要 tarfile 的 filter 参数）。
"""
import gzip
import io
import json
import logging
import re
import tarfile
import zipfile

from calibguard import constants as C
from calibguard.schemas import PreprocessResult

logger = logging.getLogger(__name__)

# 控制字符模式（仅用于 text/binary 判定；不含 \t \n \r）
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# 控制字符占比超过该值即判定为二进制（见 7.3 步骤 4）
_CONTROL_CHAR_RATIO_MAX = 0.30
# 二进制可打印串提取模式（最短连续长度取常量 PRINTABLE_MIN_RUN）
_PRINTABLE_RE = re.compile(rb"[\x20-\x7e]{%d,}" % C.PRINTABLE_MIN_RUN)
# 连续 3 个及以上空白字符（空格/制表符）压缩模式
_BLANK_RUN_RE = re.compile(r"[ \t]{3,}")
# 解包截断标记
_ARCHIVE_TRUNCATED_MARK = "\n\n[ARCHIVE_PARTIALLY_EXTRACTED]"


def preprocess_file(path: str) -> PreprocessResult:
    """主流程：读入 → 判定压缩包 → 解包或解码 → JSON 规整 → 清洗。"""
    data, read_truncated = _read_bytes(path)

    # 步骤 2 + 步骤 3：压缩包走内存解包
    if _detect_archive(data) is not None:
        archive_text, parts, expand_truncated = _expand_archive(data, 0)
        return PreprocessResult(
            text=_clean_text(archive_text),
            source_type="archive",
            truncated=read_truncated or expand_truncated,
            parts=parts,
        )

    # 步骤 4：非压缩包 → 解码或可打印串提取
    text, source_type = _decode_text(data)

    # 步骤 5：JSON 规整（仅 text 类；解析失败原样保留）
    if source_type == "text":
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except ValueError as exc:
                logger.debug("JSON 解析失败，按原文保留: %s", exc)
            else:
                text = json.dumps(parsed, ensure_ascii=False, indent=2)

    # 步骤 6 + 步骤 7
    return PreprocessResult(
        text=_clean_text(text),
        source_type=source_type,
        truncated=read_truncated,
        parts=1,
    )


def _read_bytes(path: str) -> tuple[bytes, bool]:
    """读入文件字节。(数据, 是否被读入截断)。不存在或不可读 → FileNotFoundError。"""
    try:
        with open(path, "rb") as handle:
            data = handle.read(C.MAX_INPUT_BYTES + 1)
    except FileNotFoundError:
        raise
    except OSError as exc:
        # 不可读（权限等）按文档统一转为 FileNotFoundError
        raise FileNotFoundError(f"文件不存在或不可读: {path}") from exc
    if len(data) > C.MAX_INPUT_BYTES:
        return data[:C.MAX_INPUT_BYTES], True
    return data, False


def _detect_archive(data: bytes) -> str | None:
    """"zip" | "gzip" | "tar" | None（纯魔数判定）。"""
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    if data[257:262] == b"ustar":
        return "tar"
    return None


def _expand_archive(data: bytes, depth: int) -> tuple[str, int, bool]:
    """内存解包压缩包，返回 (拼接文本, 内部文件数, 是否被解包截断)。

    只处理常规文件成员：zip 用 zf.read(info)，tar 用 tf.extractfile(m).read()；
    目录、符号链接、硬链接、设备成员一律跳过（链接成员是 tar 的经典攻击面）。
    """
    archive_type = _detect_archive(data)
    members: list[tuple[str, bytes]] = []
    file_count = 0
    total_bytes = 0
    truncated = False

    try:
        if archive_type == "zip":
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    # 安全限制：读成员前先按其声明大小检查
                    if (file_count + 1 > C.ARCHIVE_MAX_FILES
                            or total_bytes + info.file_size > C.ARCHIVE_MAX_TOTAL_BYTES):
                        truncated = True
                        break
                    blob = zf.read(info)
                    members.append((info.filename, blob))
                    file_count += 1
                    total_bytes += len(blob)
        elif archive_type == "tar":
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
                for member in tf:
                    if not member.isreg():
                        continue
                    # 安全限制：读成员前先按其声明大小检查
                    if (file_count + 1 > C.ARCHIVE_MAX_FILES
                            or total_bytes + member.size > C.ARCHIVE_MAX_TOTAL_BYTES):
                        truncated = True
                        break
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        # 声明为常规文件却取不到内容：跳过该成员并记录
                        logger.warning("tar 成员无法读取，已跳过: %s", member.name)
                        continue
                    blob = extracted.read()
                    members.append((member.name, blob))
                    file_count += 1
                    total_bytes += len(blob)
        elif archive_type == "gzip":
            inner = gzip.decompress(data)
            if _detect_archive(inner) == "tar":
                # tar.gz 两步走：属同一压缩层级，深度不递增
                return _expand_archive(inner, depth)
            members.append(("", inner))
            file_count = 1
            total_bytes = len(inner)
        else:
            # 魔数判定为压缩包却无法识别类型：按二进制处理
            return _extract_printable(data), 1, False
    except Exception as exc:
        # 损坏的压缩包 → 视为二进制，走可打印串提取（不吞异常：记录并明确返回）
        logger.warning("压缩包解析失败，按二进制处理: %s: %s", type(exc).__name__, exc)
        return _extract_printable(data), 1, False

    # 第二遍：嵌套压缩包递归（深度 +1），其余解码为文本
    collected: list[tuple[str, str]] = []
    for name, blob in members:
        if _detect_archive(blob) is not None:
            if depth + 1 > C.ARCHIVE_MAX_DEPTH:
                continue  # 超出递归深度上限，跳过该内部文件
            inner_text, inner_parts, inner_truncated = _expand_archive(blob, depth + 1)
            file_count += inner_parts
            truncated = truncated or inner_truncated
            collected.append((name, inner_text))
        else:
            member_text, _member_type = _decode_text(blob)
            collected.append((name, member_text))

    # 全部文本按文件名排序后用 "\n\n" 连接（不加任何标记）
    collected.sort(key=lambda item: item[0])
    joined = "\n\n".join(entry_text for _name, entry_text in collected)
    if truncated:
        joined += _ARCHIVE_TRUNCATED_MARK
    return joined, file_count, truncated


def _decode_text(data: bytes) -> tuple[str, str]:
    """(文本, "text"|"binary")：先 utf-8 再 gbk，均严格模式。"""
    text: str | None = None
    for encoding in ("utf-8", "gbk"):
        try:
            text = data.decode(encoding)
            break  # 解码成功，结束尝试
        except UnicodeDecodeError as exc:
            # 该编码不适用：记录原因后继续尝试下一种编码
            logger.debug("按 %s 解码失败，尝试下一种编码: %s", encoding, exc)
    if text is None:
        # 两种解码都失败 → 二进制
        return _extract_printable(data), "binary"
    if text == "":
        return text, "text"  # 空字符串直接判 text，不做除法
    control_ratio = len(_CONTROL_CHARS_RE.findall(text)) / len(text)
    if control_ratio > _CONTROL_CHAR_RATIO_MAX:
        return _extract_printable(data), "binary"
    return text, "text"


def _extract_printable(data: bytes) -> str:
    """提取二进制中的可打印 ASCII 串，各段用 "\\n" 连接。"""
    runs = _PRINTABLE_RE.findall(data)
    return b"\n".join(runs).decode("ascii")


def _clean_text(text: str) -> str:
    """a 删控制字符 → b 压缩连续空白 → c 单行截断 → d 总长截断。"""
    text = re.sub(C.CONTROL_CHARS_PATTERN, "", text)
    text = _BLANK_RUN_RE.sub(" ", text)
    lines = text.split("\n")
    clipped_lines: list[str] = []
    for line in lines:
        if len(line) > C.LINE_TRUNCATE_THRESHOLD:
            line = line[:C.LINE_TRUNCATE_THRESHOLD] + "…[LINE_TRUNCATED]"
        clipped_lines.append(line)
    text = "\n".join(clipped_lines)
    if len(text) > C.MAX_TEXT_CHARS:
        text = text[:C.MAX_TEXT_CHARS] + "\n…[TEXT_TRUNCATED]"
    return text
