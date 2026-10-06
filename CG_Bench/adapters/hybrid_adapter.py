"""CalibGuard Bench 混合门控适配器（TH-5.5）= ft 模型三桶门控 + llm_review 桶 LLM 复核。

输出契约与 llm_adapter 完全一致：{"malicious": bool, "confidence": float}
    gate=block      → malicious=True,  confidence=0.99
    gate=allow      → malicious=False, confidence=0.01
    gate=llm_review → 取 TH-4 classify（v_feature 模板）结果：malicious 与 confidence
                      均来自 LLM 返回；LLM 失败（api_error / parse_error）时
                      malicious=True, confidence=0.5（与引擎 manual_review 的保守语义一致）

门控阈值与推理参数一律取自引擎 load_config（allow 0.15 / block 0.85 / max_length 512 /
stride 256 / batch 8），不另立一套，保证与 ft 实验口径同源。
"""
import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters import llm_adapter                         # noqa: E402
from calibguard.calibrator import PlattCalibrator        # noqa: E402
from calibguard.config import load_config                # noqa: E402
from calibguard.gate import route                        # noqa: E402
from calibguard.model import CodeBERTClassifier          # noqa: E402
from calibguard.preprocessor import preprocess_file      # noqa: E402

REVIEW_TEMPLATE = "v_feature"          # TH-5.5：llm_review 桶固定用 v_feature 模板裁决
BLOCK_CONFIDENCE = 0.99
ALLOW_CONFIDENCE = 0.01
FAIL_CONFIDENCE = 0.5

_CACHE = {}                            # (model_dir, params) -> (classifier, calibrator)


def _get_runtime(model_dir: str, params: str):
    """按 (模型目录, 校准参数) 缓存推理运行时——滑窗模型加载昂贵，禁止每文件重载。"""
    key = (os.path.abspath(model_dir), os.path.abspath(params))
    if key not in _CACHE:
        cfg = load_config(None)
        classifier = CodeBERTClassifier(
            model_dir, cfg.max_length,
            window_stride=cfg.window_stride, window_batch_size=cfg.window_batch_size)
        _CACHE[key] = (classifier, PlattCalibrator.load(params))
    return _CACHE[key]


def gate_of(text: str, model_dir: str, params: str) -> tuple:
    """ft 模型三桶门控：返回 (校准概率, 桶名)。"""
    classifier, calibrator = _get_runtime(model_dir, params)
    prob = calibrator.predict_proba(classifier.predict_logit(text))
    cfg = load_config(None)
    return prob, route(prob, cfg.allow_threshold, cfg.block_threshold).value


def window_count(text: str, model_dir: str, params: str) -> int:
    """滑窗窗口数（与 ft 实验同口径，供速度基准记录 windows 字段）。"""
    classifier, _calibrator = _get_runtime(model_dir, params)
    token_ids = classifier.tokenizer.encode(text, add_special_tokens=False)
    return len(classifier.window_starts(len(token_ids)))


def _decide(text: str, prob: float, gate: str) -> dict:
    """由 (ft 校准概率, 桶名) 得到最终判定——classify 与批量评测共用的唯一实现。"""
    if gate == "allow":
        return {"malicious": False, "confidence": ALLOW_CONFIDENCE,
                "gate": gate, "prob": round(prob, 4)}
    if gate == "block":
        return {"malicious": True, "confidence": BLOCK_CONFIDENCE,
                "gate": gate, "prob": round(prob, 4)}
    review = llm_adapter.classify(text, REVIEW_TEMPLATE)
    if review.get("api_error") or review.get("parse_error"):
        return {"malicious": True, "confidence": FAIL_CONFIDENCE, "gate": gate,
                "prob": round(prob, 4),
                "llm_error": "api_error" if review.get("api_error") else "parse_error"}
    return {"malicious": bool(review["malicious"]),
            "confidence": float(review["confidence"]), "gate": gate,
            "prob": round(prob, 4), "llm_template": review.get("template")}


def classify(text: str, model_dir: str = None, params: str = None) -> dict:
    """混合门控判定（TH-5.5 逐字契约）。"""
    if not model_dir or not params:
        raise ValueError("hybrid_adapter.classify 需要 model_dir 与 params"
                         "（ft 微调模型目录与 Platt 校准参数文件）")
    prob, gate = gate_of(text, model_dir, params)
    return _decide(text, prob, gate)


def run_split_eval(exp_name: str, out_dir: str, splits_dir: str, model_dir: str,
                   params: str, split: str = "test", limit: int = None) -> dict:
    """对某分段做一次「ft 三桶门控 + llm_review 桶 LLM 复核」全量评测（TH-5.5 批量口径）。

    只有落进 llm_review 桶的样本才发起 API 调用（其余 allow/block 直接定论）——
    这正是 harness 的 ft 七步流程里缺失的一环（该流程只跑 ft 模型本身，不含 LLM）。

    产物（**不覆盖** ft 口径的 eval_raw.json / eval_calib.json）：
      {out_dir}/eval_hybrid.json         指标 + 三桶 + 分来源/分域 + LLM 调用统计
      {out_dir}/eval_hybrid_detail.jsonl 逐条：path/label/ft_prob/gate/最终判定/llm_error
    """
    split_file = os.path.join(splits_dir, f"split_{split}.jsonl")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"分段清单不存在：{split_file}（请先运行 harness.py init）")
    records = llm_adapter._read_split(split_file)
    if limit is not None:
        records = records[:limit]
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(out_dir, "eval_hybrid_detail.jsonl")

    labels, ft_probs, preds, sources = [], [], [], []
    skipped = 0
    llm_calls = api_errors = parse_errors = 0
    started = time.perf_counter()
    with open(detail_path, "w", encoding="utf-8", newline="\n") as detail_handle:
        for index, record in enumerate(records):
            try:
                text = preprocess_file(record["path"]).text
            except Exception as exc:                # noqa: BLE001 —— 单条失败只跳过
                print(f"跳过: {record['path']}（{type(exc).__name__}: {exc}）")
                skipped += 1
                continue
            prob, gate = gate_of(text, model_dir, params)
            verdict = _decide(text, prob, gate)
            if gate == "llm_review":
                llm_calls += 1
            if verdict.get("llm_error") == "api_error":
                api_errors += 1
            elif verdict.get("llm_error") == "parse_error":
                parse_errors += 1
            labels.append(int(record["label"]))
            ft_probs.append(float(prob))
            preds.append(bool(verdict["malicious"]))
            sources.append(llm_adapter._source_of(record["path"]))
            detail_handle.write(json.dumps({
                "path": record["path"], "label": int(record["label"]),
                "ft_prob": round(float(prob), 6), "gate": gate,
                "malicious": bool(verdict["malicious"]),
                "confidence": float(verdict["confidence"]),
                "llm_error": verdict.get("llm_error"),
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 25 == 0:
                print(f"  进度 {index + 1}/{len(records)}（LLM 调用 {llm_calls} / "
                      f"API 异常 {api_errors} / 解析失败 {parse_errors} / 跳过 {skipped}）")

    # 口径与引擎一致：指标用最终判定布尔，AUC/ECE/Brier/三桶用 ft 校准概率
    # （三桶本就由 ft 概率 route 而来，故与真实门控完全一致）
    summary = llm_adapter.summarize(labels, ft_probs, sources, split, preds=preds)
    summary.update({
        "adapter": "hybrid_gate_llm",
        "experiment": exp_name,
        "mode": "ft_gate + llm_review(v_feature)",
        "skipped": skipped,
        "llm_review_calls": llm_calls,
        "api_error_rate": round(api_errors / max(1, llm_calls), 4) if llm_calls else 0.0,
        "parse_error_rate": round(parse_errors / max(1, llm_calls), 4) if llm_calls else 0.0,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    })
    with open(os.path.join(out_dir, "eval_hybrid.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[hybrid_adapter] {exp_name} 混合评测完成：{summary['n_samples']} 条"
          f"（跳过 {skipped}）F1={summary['metrics_calibrated_p_gt05']['f1']:.4f}"
          f" LLM 调用={llm_calls}（api 异常 {api_errors} / 解析失败 {parse_errors}）"
          f" 用时 {summary['elapsed_seconds']}s → {out_dir}/eval_hybrid.json")
    return summary


def _pair_detail_with_split(detail_path: str, split_path: str) -> list:
    """把 evaluation_test_detail.jsonl 的行与 split_test.jsonl 的样本配对。

    evaluate.py 的明细文件只记 label/logit/prob/source（无路径），行序与分段清单一致；
    这里用 source 前缀做配对自校验，不匹配的候选直接跳过（防跳样导致错位）。
    """
    with open(detail_path, "r", encoding="utf-8") as handle:
        details = [json.loads(line) for line in handle if line.strip()]
    with open(split_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    pairs = []
    for index, detail in enumerate(details):
        if index >= len(records):
            break
        record = records[index]
        if llm_adapter._source_of(record["path"]) != detail.get("source"):
            continue
        if int(record["label"]) != int(detail.get("label", -1)):
            continue
        pairs.append((record["path"], detail))
    return pairs


def _selfcheck() -> int:
    """TH-5.5 验收：构造 1 条已知进 llm_review 的样本，输出 malicious 与 LLM 判定一致。"""
    model_dir = "models_baseline/v0.2.2/codebert_finetuned"
    params = "models_baseline/v0.2.2/calibration_params.json"
    detail_path = "models_baseline/v0.2.2/evaluation_test_detail.jsonl"
    split_path = os.path.join("results", "_splits", "split_test.jsonl")
    for path in (model_dir, params, detail_path, split_path):
        if not os.path.exists(path):
            print(f"[hybrid_adapter][selfcheck] 缺少前置产物：{path}")
            return 1

    target = None
    for sample_path, detail in _pair_detail_with_split(detail_path, split_path):
        if 0.15 <= float(detail["prob"]) < 0.85:
            target = (sample_path, detail)
            break
    if target is None:
        print("[hybrid_adapter][selfcheck] 未在 baseline 明细里找到 prob∈[0.15,0.85) 的样本")
        return 1

    sample_path, detail = target
    cfg = load_config(None)
    text = preprocess_file(sample_path).text
    prob, gate = gate_of(text, model_dir, params)
    review = llm_adapter.classify(text, REVIEW_TEMPLATE)
    final = classify(text, model_dir, params)

    print(f"[hybrid_adapter][selfcheck] 样本 {sample_path}（label={detail['label']}）")
    print(f"[hybrid_adapter][selfcheck] baseline 明细 prob={detail['prob']} "
          f"logit={detail['logit']}")
    print(f"[hybrid_adapter][selfcheck] 现场重算 prob={round(prob, 4)} gate={gate}"
          f"（阈值 allow<{cfg.allow_threshold} / block>{cfg.block_threshold}）")
    print(f"[hybrid_adapter][selfcheck] llm_review 桶 LLM 判定（{REVIEW_TEMPLATE}）：{review}")
    print(f"[hybrid_adapter][selfcheck] 混合门控最终输出：{final}")

    if gate != "llm_review":
        print("[hybrid_adapter][selfcheck] 该样本未落在 llm_review 桶，验收无效")
        return 1
    if review.get("api_error"):
        print("[hybrid_adapter][selfcheck] LLM 调用失败（api_error）→ 保守输出 "
              f"malicious={final['malicious']} confidence={final['confidence']}")
        return 0 if (final["malicious"] and final["confidence"] == FAIL_CONFIDENCE) else 1
    consistent = (bool(final["malicious"]) == bool(review["malicious"])
                  and float(final["confidence"]) == float(review["confidence"]))
    print(f"[hybrid_adapter][selfcheck] 结论："
          f"{'与 LLM 判定一致（验收通过）' if consistent else '与 LLM 判定不一致（验收失败）'}")
    return 0 if consistent else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CalibGuard Bench 混合门控适配器（TH-5.5）")
    parser.add_argument("--selfcheck", action="store_true",
                        help="跑 TH-5.5 验收（baseline 明细中 prob∈[0.15,0.85) 的样本）")
    parser.add_argument("--eval-test", action="store_true",
                        help="对分段做批量混合评测（ft 门控 + llm_review 桶 LLM 复核）")
    parser.add_argument("--exp", default="hybrid_gate_llm", help="实验名（产物命名用）")
    parser.add_argument("--out-dir", default="results/hybrid_gate_llm", help="产物目录")
    parser.add_argument("--splits-dir", default="results/_splits", help="分段清单目录")
    parser.add_argument("--model-dir", default="results/hybrid_gate_llm/codebert_finetuned",
                        help="ft 微调模型目录（默认用实验内克隆的复用模型）")
    parser.add_argument("--params", default="results/hybrid_gate_llm/calibration_params.json",
                        help="Platt 校准参数文件（默认用实验内复用的参数）")
    parser.add_argument("--split", default="test", help="评测分段（默认 test）")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条（小样验证用）")
    args = parser.parse_args(argv)

    if args.eval_test:
        if not os.environ.get(llm_adapter.API_KEY_ENV):
            print(f"[hybrid_adapter] 未配置 {llm_adapter.API_KEY_ENV}：跳过批量评测"
                  f"（避免逐条无谓调用）", file=sys.stderr)
            return 3
        llm_adapter.reset_progress()          # 熔断计时起点（最后一次成功响应时间）
        try:
            run_split_eval(args.exp, args.out_dir, args.splits_dir,
                           args.model_dir, args.params,
                           split=args.split, limit=args.limit)
        except llm_adapter.ApiFatalError as exc:
            payload = llm_adapter.write_skip_marker(args.out_dir, exc)
            print(f"[hybrid_adapter] API 熔断（{exc.reason}）：{exc.detail} → 中止本实验，"
                  f"标记 skipped_api_error（已写 {llm_adapter.SKIP_MARKER}；成功调用 "
                  f"{payload['successful_calls']} / 失败 {payload['failed_calls']}）"
                  f"；退出码 {llm_adapter.EXIT_API_FATAL}", file=sys.stderr)
            return llm_adapter.EXIT_API_FATAL
        return 0

    if not args.selfcheck:
        parser.print_help()
        return 0
    return _selfcheck()


if __name__ == "__main__":
    sys.exit(main())
