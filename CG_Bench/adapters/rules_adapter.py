"""CalibGuard Bench 规则基线适配器（TH-5，下界参照）。

不调任何模型：输入文本 → 引擎 feature_extractor.extract_features(text)
    → 命中数 = 7 类证据列表长度之和（v1.0 补丁 6 精确定义）：
        len(dangerous_apis) + len(network_indicators) + len(credential_access)
      + len(obfuscation_signals) + len(file_operations) + len(injection_phrases)
      + len(high_entropy_strings)
    → 命中数 ≥ 3 → malicious（True），否则 False
    → 概率 = min(0.99, 0.2 + 0.15 × 命中数)（伪概率仅供排序口径）

明确排除（补丁 6）：random_identifier_count（计数放大器——打包/压缩的良性文件可能有上百个
长标识符，其混淆信号已由 high_entropy_strings 部分覆盖）与超长行等结构特征（异常信号
不等于恶意证据）。

评测复用 adapters/llm_adapter.run_bench_eval（同一份引擎口径指标汇总，避免两处口径漂移）。
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.llm_adapter import run_bench_eval        # noqa: E402 —— 需先补 sys.path
from calibguard.feature_extractor import extract_features
from calibguard.preprocessor import preprocess_file

# 命中数口径：7 类证据列表（顺序固定，禁止增删）
EVIDENCE_FIELDS = ("dangerous_apis", "network_indicators", "credential_access",
                   "obfuscation_signals", "file_operations", "injection_phrases",
                   "high_entropy_strings")
HIT_THRESHOLD = 3            # 命中数 ≥ 3 → 恶意
PROB_BASE = 0.2              # 伪概率 = min(0.99, 0.2 + 0.15 × 命中数)
PROB_STEP = 0.15
PROB_MAX = 0.99


def hit_count(text: str) -> int:
    """7 类证据命中总数（不含 random_identifier_count 与超长行等结构特征）。"""
    evidence = extract_features(text)
    return sum(len(getattr(evidence, field)) for field in EVIDENCE_FIELDS)


def classify(text: str) -> dict:
    """规则基线判定：返回 {"malicious": bool, "confidence": float, "hits": int}。"""
    hits = hit_count(text)
    return {"malicious": hits >= HIT_THRESHOLD,
            "confidence": min(PROB_MAX, PROB_BASE + PROB_STEP * hits),
            "hits": hits}


def _selfcheck() -> int:
    """TH-5 验收：对已知恶意样本 classify 全 True（两域各取一条）。"""
    samples = ("data/train/malicious/mal_sb_0.txt",           # prompt 域
               "data/train/malicious/mal_py_11Cent_1.txt")    # code 域
    ok = True
    for relative in samples:
        if not os.path.isfile(relative):
            print(f"[rules_adapter][selfcheck] 样本缺失：{relative}")
            ok = False
            continue
        result = classify(preprocess_file(relative).text)
        flag = "OK" if result["malicious"] else "FAIL"
        print(f"[rules_adapter][selfcheck] {relative}: {result} -> {flag}")
        ok = ok and result["malicious"]
    print(f"[rules_adapter][selfcheck] 结论：{'全 True（验收通过）' if ok else '存在 False（验收失败）'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CalibGuard Bench 规则基线适配器（TH-5）")
    parser.add_argument("--exp", default="rules_baseline")
    parser.add_argument("--out-dir", default="results/rules_baseline")
    parser.add_argument("--splits-dir", default="results/_splits")
    parser.add_argument("--selfcheck", action="store_true", help="跑 TH-5 验收（3 个恶意样本）")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return _selfcheck()

    def _classify(text: str) -> dict:
        return classify(text)

    run_bench_eval(_classify, args.exp, args.out_dir, args.splits_dir,
                   adapter="rules_adapter", max_chars=None, split="test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
