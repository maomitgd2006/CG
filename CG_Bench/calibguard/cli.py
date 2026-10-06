"""命令行入口：argparse 解析、打印摘要、退出码管理。

退出码：0 正常 / 1 路径不存在 / 2 配置错误 / 3 校准参数缺失 / 4 模型加载失败 / 5 未知内部错误。
"""
import argparse
import os
import sys

from calibguard import __version__
from calibguard.config import load_config
from calibguard.pipeline import AuditPipeline


def main() -> None:
    """程序入口，管理退出码。"""
    parser = argparse.ArgumentParser(
        prog="calibguard",
        description="CalibGuard：本地 CodeBERT 校准门控 + 云端 LLM 按需分析的安全审计工具。")
    subparsers = parser.add_subparsers(dest="command")
    audit_parser = subparsers.add_parser("audit", help="审计指定文件或目录")
    audit_parser.add_argument("path", type=str, help="待审计的文件或目录路径")
    audit_parser.add_argument("--config", default=None, help="配置文件路径")
    audit_parser.add_argument("--output-dir", default=None, help="报告输出目录（覆盖配置）")
    subparsers.add_parser("version", help="打印版本号")

    args = parser.parse_args()

    if args.command == "version":
        print(f"CalibGuard {__version__}")
        return
    if args.command != "audit":
        parser.print_help()
        return

    # 1. 加载配置（内部已处理退出码 2）
    cfg = load_config(args.config)
    # 2. 命令行覆盖输出目录
    if args.output_dir:
        cfg.report_output_dir = args.output_dir
    # 3. 路径存在性检查
    if not os.path.exists(args.path):
        print(f"[错误] 路径不存在: {args.path}", file=sys.stderr)
        sys.exit(1)
    # 4. 构建流水线（模型 / 校准参数）
    try:
        pipeline = AuditPipeline(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(3)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(4)

    # 5-7. 审计、逐行摘要、统计
    try:
        reports = pipeline.audit_path(args.path)
        total = len(reports)
        counts = {"allow": 0, "block": 0, "suspicious": 0, "manual_review": 0}
        for index, report in enumerate(reports, start=1):
            print(f"[{index}/{total}] {report['file_path']}  "
                  f"p={report['calibrated_probability']}  "
                  f"gate={report['gate_decision']}  final={report['final_decision']}")
            decision = report.get("final_decision")
            if decision in counts:
                counts[decision] += 1
        print(f"审计完成：共 {total} 个文件 | 放行 {counts['allow']} | 拦截 {counts['block']} "
              f"| 可疑 {counts['suspicious']} | 转人工 {counts['manual_review']}")
        print(f"报告目录：{os.path.abspath(cfg.report_output_dir)}")
    except Exception as exc:
        print(f"[错误] 未知内部错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(5)

    # 8. 正常结束
    sys.exit(0)
