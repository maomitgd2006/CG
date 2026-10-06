"""审计报告模块：审计结果 dict → JSON 文件 + Markdown 文件。"""
import json
import os

from calibguard.schemas import FeatureEvidence

# 证据明细的七类标题（第八类"异常结构特征"由长行/随机标识符两行构成）
_EVIDENCE_TITLES = [
    ("【危险API调用】", "dangerous_apis"),
    ("【网络行为】", "network_indicators"),
    ("【凭据与敏感路径访问】", "credential_access"),
    ("【编码与混淆】", "obfuscation_signals"),
    ("【可疑文件操作】", "file_operations"),
    ("【提示注入话术】", "injection_phrases"),
    ("【高熵字符串】", "high_entropy_strings"),
]
# 证据全空时的固定文案
_EMPTY_EVIDENCE_TEXT = "未发现明显可疑信号。"


def write_reports(report: dict, output_dir: str) -> tuple[str, str]:
    """生成 <output_dir>/<report_id>.json 与 <output_dir>/<report_id>.md，返回绝对路径。"""
    os.makedirs(output_dir, exist_ok=True)
    report_id = str(report["report_id"])
    json_path = os.path.abspath(os.path.join(output_dir, report_id + ".json"))
    md_path = os.path.abspath(os.path.join(output_dir, report_id + ".md"))
    with open(json_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2))
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(_to_markdown(report))
    return json_path, md_path


def _evidence_to_dict(ev: FeatureEvidence) -> dict:
    """FeatureEvidence → 九个同名键的 dict。"""
    return {
        "dangerous_apis": list(ev.dangerous_apis),
        "network_indicators": list(ev.network_indicators),
        "credential_access": list(ev.credential_access),
        "obfuscation_signals": list(ev.obfuscation_signals),
        "file_operations": list(ev.file_operations),
        "injection_phrases": list(ev.injection_phrases),
        "high_entropy_strings": list(ev.high_entropy_strings),
        "long_line_numbers": list(ev.long_line_numbers),
        "random_identifier_count": ev.random_identifier_count,
    }


def _to_markdown(report: dict) -> str:
    """按 13.3 固定模板渲染 Markdown 报告。"""
    early_stopped = bool(report.get("early_stopped"))
    if early_stopped:
        early_stop_text = f"是，p={report.get('early_stop_probability')}"
    else:
        early_stop_text = "否"

    lines: list[str] = [
        "# CalibGuard 安全审计报告",
        "",
        f"- 报告编号：{report['report_id']}",
        f"- 生成时间：{report['timestamp']}",
        f"- 审计文件：{report['file_path']}",
        f"- 文件 SHA256：{report['file_sha256']}",
        f"- 文件大小：{report['file_size_bytes']} 字节",
        f"- 来源类型：{report['source_type']}（内部文件数：{report['archive_parts']}）",
        "",
        "## 结论",
        "",
        "| 项目 | 结果 |",
        "|---|---|",
        f"| 原始 logit | {report['raw_logit']} |",
        f"| 校准概率 | {report['calibrated_probability']} |",
        f"| 校准方法 | {report['calibration_method']} |",
        f"| 推理窗口数 | {report['windows_evaluated']} |",
        f"| 滑窗早停 | {early_stop_text} |",
        f"| 门控决策 | {report['gate_decision']} |",
        f"| 最终决策 | **{report['final_decision']}** |",
        f"| 需人工复核 | {report['manual_review_required']} |",
        f"| 一致性告警 | {report['consistency_warning']} |",
        "",
        "## LLM 分析",
        "",
    ]

    llm = report.get("llm")
    if llm is None:
        lines.append("（未提交云端分析）")
    else:
        lines.extend([
            "| 字段 | 值 |",
            "|---|---|",
            f"| status | {llm.get('status')} |",
            f"| risk | {llm.get('risk')} |",
            f"| reason | {llm.get('reason')} |",
            f"| action | {llm.get('action')} |",
            f"| 重试次数 | {llm.get('retries_used')} |",
        ])

    lines.extend(["", "## 证据明细", ""])
    lines.append(_render_evidence_section(report.get("evidence") or {}))

    lines.extend(["", "## 错误信息", ""])
    error = report.get("error")
    lines.append(str(error) if error else "（无）")

    return "\n".join(lines) + "\n"


def _render_evidence_section(evidence: dict) -> str:
    """按 8.4 的八类格式渲染证据明细；全空时返回固定文案。"""
    blocks: list[str] = []
    for title, key in _EVIDENCE_TITLES:
        items = evidence.get(key) or []
        if not items:
            continue
        block_lines = [title]
        block_lines.extend(f"- {item}" for item in items)
        blocks.append("\n".join(block_lines))

    structural_lines: list[str] = []
    long_line_numbers = evidence.get("long_line_numbers") or []
    if long_line_numbers:
        structural_lines.append(f"- 超长行行号: {long_line_numbers}")
    random_identifier_count = evidence.get("random_identifier_count") or 0
    if random_identifier_count > 0:
        structural_lines.append(f"- 长随机标识符数量: {random_identifier_count}")
    if structural_lines:
        blocks.append("\n".join(["【异常结构特征】"] + structural_lines))

    if not blocks:
        return _EMPTY_EVIDENCE_TEXT
    return "\n\n".join(blocks)
