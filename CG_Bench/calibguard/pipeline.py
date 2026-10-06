"""流水线编排：串联 T4~T10，产出单文件完整报告 dict。"""
import datetime
import hashlib
import math
import os
import sys
import uuid

from calibguard import constants as C
from calibguard.calibrator import PlattCalibrator
from calibguard.config import AppConfig
from calibguard.feature_extractor import evidence_to_prompt_text, extract_features
from calibguard.gate import route
from calibguard.llm_analyzer import LLMAnalyzer
from calibguard.model import CodeBERTClassifier
from calibguard.preprocessor import preprocess_file
from calibguard.report import _evidence_to_dict, write_reports
from calibguard.schemas import (
    FeatureEvidence,
    GateDecision,
    LLMResult,
    WindowInferenceStats,
)


class AuditPipeline:
    """单文件/目录审计流水线。"""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        # 1. 模型（加载失败抛 RuntimeError，由 CLI 转退出码 4）
        self.classifier = CodeBERTClassifier(
            cfg.model_path, cfg.max_length,
            window_stride=cfg.window_stride, window_batch_size=cfg.window_batch_size)
        # 2. 校准器（calibration_method 一律取实例的 .method 属性）
        if cfg.calibration_params_path == "":
            self.calibrator = PlattCalibrator.identity()
        elif os.path.isfile(cfg.calibration_params_path):
            self.calibrator = PlattCalibrator.load(cfg.calibration_params_path)
        else:
            raise FileNotFoundError(
                "校准参数文件缺失，请先运行 scripts/calibrate.py，"
                "或在 config.yaml 中将 calibration.params_path 设为空字符串")
        # 3. 云端 LLM 分析器（属性名必须是 llm）
        self.llm = LLMAnalyzer(
            api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, model=cfg.llm_model,
            temperature=cfg.llm_temperature, max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout, evidence_max_chars=cfg.evidence_max_chars)

    def audit_file(self, path: str) -> dict:
        """单文件 → 完整报告 dict。任何处理异常都降级为 manual_review，不向调用方抛出。"""
        report = self._new_report(path)
        try:
            # 步骤 1：读前 1MB 字节，用于 sha256 与 size
            with open(path, "rb") as handle:
                head_bytes = handle.read(C.MAX_INPUT_BYTES)
            report["file_sha256"] = hashlib.sha256(head_bytes).hexdigest()
            report["file_size_bytes"] = len(head_bytes)

            # 步骤 2：预处理
            result = preprocess_file(path)
            report["source_type"] = result.source_type
            report["preprocess_truncated"] = result.truncated
            report["archive_parts"] = result.parts

            # 步骤 3：特征提取（全文，信息更全）
            evidence = extract_features(result.text)
            report["evidence"] = _evidence_to_dict(evidence)

            # 步骤 4：滑窗推理与批间早停
            agg_logit = -math.inf
            stats = WindowInferenceStats()
            for batch_logits in self.classifier.iter_window_batches(result.text):
                stats.windows_evaluated += len(batch_logits)
                batch_max = max(batch_logits)
                agg_logit = max(agg_logit, batch_max)
                # 批间早停：本批内任一窗口校准概率 > early_stop_threshold 即停
                for window_logit in batch_logits:
                    if (self.calibrator.predict_proba(window_logit)
                            > self.cfg.early_stop_threshold):
                        stats.early_stopped = True
                        stats.early_stop_probability = self.calibrator.predict_proba(batch_max)
                        break
                if stats.early_stopped:
                    # 生成器后续批次不再被消费，即"立即停止后续窗口"
                    break
            raw_logit = float(agg_logit)
            report["raw_logit"] = raw_logit
            report["windows_evaluated"] = stats.windows_evaluated
            report["early_stopped"] = stats.early_stopped
            report["early_stop_probability"] = stats.early_stop_probability

            # 步骤 5：校准概率（早停时 p = early_stop_probability > block_threshold）
            probability = self.calibrator.predict_proba(raw_logit)
            report["calibrated_probability"] = probability

            # 步骤 6：门控路由
            gate = route(probability, self.cfg.allow_threshold, self.cfg.block_threshold)
            report["gate_decision"] = gate.value

            # 步骤 7：仅门控为 LLM_REVIEW 时才上云（且只传精炼证据）
            llm_result: LLMResult | None = None
            if gate == GateDecision.LLM_REVIEW:
                evidence_text = evidence_to_prompt_text(evidence, self.cfg.evidence_max_chars)
                llm_result = self.llm.analyze(evidence_text, probability)
                report["llm"] = {
                    "status": llm_result.status,
                    "risk": llm_result.risk,
                    "reason": llm_result.reason,
                    "action": llm_result.action,
                    "retries_used": llm_result.retries_used,
                }
                report["consistency_warning"] = llm_result.consistency_warning

            # 步骤 8：最终决策表
            final_decision, manual_review_required = self._resolve_decision(gate, llm_result)
            report["final_decision"] = final_decision
            report["manual_review_required"] = manual_review_required
        except Exception as exc:
            # 安全降级：转人工 + 记录错误，继续处理下一个文件
            print(f"[错误] 审计失败 {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            report["final_decision"] = "manual_review"
            report["manual_review_required"] = True
            report["error"] = str(exc)

        # 步骤 9：落盘并回填路径
        try:
            json_path, md_path = write_reports(report, self.cfg.report_output_dir)
        except Exception as exc:
            print(f"[错误] 报告写入失败 {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            write_error = f"报告写入失败: {exc}"
            report["error"] = (f"{report['error']} | {write_error}"
                               if report["error"] else write_error)
        else:
            report["report_json_path"] = json_path
            report["report_md_path"] = md_path
        return report

    def audit_path(self, path: str) -> list[dict]:
        """文件或目录 → 报告列表。路径不存在抛 FileNotFoundError（CLI 转退出码 1）。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在: {path}")
        if os.path.isfile(path):
            return [self.audit_file(path)]
        reports: list[dict] = []
        for file_path in self._collect_files(path):
            try:
                reports.append(self.audit_file(file_path))
            except Exception as exc:
                # 兜底：单文件失败不得中断整体
                print(f"[错误] 审计失败 {file_path}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                fallback = self._new_report(file_path)
                fallback["final_decision"] = "manual_review"
                fallback["manual_review_required"] = True
                fallback["error"] = str(exc)
                reports.append(fallback)
        return reports

    def _collect_files(self, path: str) -> list[str]:
        """os.walk 递归收集常规文件（非符号链接），按路径排序，跳过输出目录。"""
        output_dir_abs = os.path.abspath(self.cfg.report_output_dir)
        collected: list[str] = []
        for root, _dir_names, file_names in os.walk(path):
            for file_name in file_names:
                full_path = os.path.join(root, file_name)
                if not os.path.isfile(full_path):
                    continue
                if os.path.islink(full_path):
                    continue
                file_abs = os.path.abspath(full_path)
                if file_abs == output_dir_abs or file_abs.startswith(output_dir_abs + os.sep):
                    continue
                collected.append(full_path)
        collected.sort()
        return collected

    def _resolve_decision(self, gate: GateDecision,
                          llm_result: LLMResult | None) -> tuple[str, bool]:
        """14.3 最终决策表（逐行照抄实现，禁止改动）。"""
        if gate == GateDecision.ALLOW:
            return "allow", False
        if gate == GateDecision.BLOCK:
            return "block", False
        # gate == GateDecision.LLM_REVIEW
        if llm_result is None or llm_result.status != "ok":
            # invalid_output / api_error
            return "manual_review", True
        if llm_result.risk == "high":
            return "block", False
        if llm_result.risk == "medium":
            return "suspicious", bool(llm_result.consistency_warning)
        if llm_result.risk == "low":
            if llm_result.consistency_warning:
                return "suspicious", True
            return "allow", False
        # risk 取值异常（理论上不可达）→ 铁律 5 安全降级为转人工
        print(f"[错误] LLM 返回未知 risk={llm_result.risk!r}，安全降级为转人工", file=sys.stderr)
        return "manual_review", True

    def _new_report(self, path: str) -> dict:
        """报告骨架：含 13.2 规范的全部键（错误路径同样返回完整键集）。"""
        return {
            "report_id": uuid.uuid4().hex,
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "file_path": os.path.abspath(path),
            "file_sha256": "",
            "file_size_bytes": 0,
            "source_type": "",
            "preprocess_truncated": False,
            "archive_parts": 1,
            "raw_logit": 0.0,
            "calibrated_probability": 0.0,
            "calibration_method": self.calibrator.method,
            "windows_evaluated": 0,
            "early_stopped": False,
            "early_stop_probability": None,
            "gate_decision": "",
            "final_decision": "",
            "manual_review_required": False,
            "consistency_warning": False,
            "llm": None,
            "evidence": _evidence_to_dict(FeatureEvidence()),
            "report_json_path": "",
            "report_md_path": "",
            "error": None,
        }
