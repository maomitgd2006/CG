"""CalibGuard 审计口径评估脚本（v1.3 新增，泄露审计修复的配套工具）。

对指定分段（默认 val）执行完整审计口径推理：
    preprocess_file → 滑动窗口全文 max logit（无早停）→ Platt 校准概率，
输出三种口径指标、混淆矩阵、门控三档统计、分来源表现，
并内置泄露自检（该分段与训练段的 sha256 原文重叠、front-matter 家族重叠）。

运行：evaluate.bat [--split val] [--model-dir models/codebert_finetuned]
                  [--params models/calibration_params.json]
输出：控制台汇总 + models/evaluation_{split}.json
退出码：0 正常 / 1 清单缺失 / 3 校准参数缺失 / 4 模型加载失败
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")  # evaluate.bat 已设 PYTHONPATH，此处兜底

from calibguard.calibrator import PlattCalibrator
from calibguard.config import load_config
from calibguard.model import CodeBERTClassifier
from calibguard.preprocessor import preprocess_file

# 与 train.py 保持一致的家族键规则（复制以保持脚本独立，修改须两处同步）
# ⚠️ 2026-10-03 修正：原先只有 front-matter 一条规则，低于 train.py 的优先级顺序，
#    且 ^name: 会命中样本**内容**里的任意 YAML/CI/issue 模板字段（实测抓到 "CI"、
#    "Bug report"、"{skill_name}" 等），把大量互不相关的样本误判为同族（裸测自检
#    曾据此误报家族重叠 414，按 train.py 口径复算实为 0）。此处补齐四段式文件名优先。
_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.M)
# 四段式文件名家族约定（train.py 同款）：mal_py_{家族名}_{序号}.txt 等
_FNAME_FAM_RE = re.compile(
    r"^(?:mal|ben)_(?:ms|sb|py|atr|twin|npm|mcp|ch|sk|mcs|ide|rb|go|jv|ps|pi|ws|skb)_(.+)_\d+\.txt$")

# 来源前缀（与 21.3.3 转存规格一致；v2 两域，mcp 已归档；rb/go/jv 为 registry 良性扩容）
_SOURCES = ("mal_ms_", "ben_ms_", "mal_sb_", "ben_sb_", "mal_py_", "ben_py_",
            "ben_atr_", "ben_twin_", "mal_npm_", "ben_npm_",
            "mal_ch_", "ben_ch_", "mal_sk_", "mal_mcs_", "ben_mcs_", "mal_ide_",
            "ben_rb_", "ben_go_", "ben_jv_")

# 来源 → 域映射（与 train.py 的 _DOMAIN_BY_PREFIX 保持一致，修改须两处同步）
_DOMAIN_OF_SOURCE = {
    "mal_ms": "prompt", "ben_ms": "prompt", "mal_sb": "prompt", "ben_sb": "prompt",
    "ben_atr": "prompt", "mal_ch": "prompt", "ben_ch": "prompt",
    "mal_sk": "prompt", "mal_mcs": "prompt", "ben_mcs": "prompt",
    "mal_py": "code", "ben_py": "code", "ben_twin": "code",
    "mal_npm": "code", "ben_npm": "code", "mal_ide": "code",
    "ben_rb": "code", "ben_go": "code", "ben_jv": "code",
}


def _source_of(path: str) -> str:
    base = os.path.basename(path)
    for prefix in _SOURCES:
        if base.startswith(prefix):
            return prefix.rstrip("_")
    return "other"


def _family_key(path: str) -> str:
    """家族键，优先级：四段式文件名 > front-matter name > 文件名自成一家族。

    与 train.py::_family_key 逐字对齐（泄露自检必须与划分同口径，否则误报，见文件头注释）。
    """
    match = _FNAME_FAM_RE.match(os.path.basename(path))
    if match:
        return match.group(1)
    try:
        with open(path, "rb") as handle:
            head = handle.read(4096)
    except OSError:
        return os.path.basename(path)
    match = _NAME_RE.search(head.decode("utf-8", "replace"))
    if match:
        return match.group(1).split("__")[0].strip()
    return os.path.basename(path)


def _read_split(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _prf(labels: list[int], preds: list[int], positive: int = 1) -> dict:
    tp = sum(1 for y, p in zip(labels, preds) if y == positive and p == positive)
    fp = sum(1 for y, p in zip(labels, preds) if y != positive and p == positive)
    fn = sum(1 for y, p in zip(labels, preds) if y == positive and p != positive)
    tn = sum(1 for y, p in zip(labels, preds) if y != positive and p != positive)
    acc = (tp + tn) / max(1, len(labels))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    parser = argparse.ArgumentParser(description="审计口径评估（滑窗 + Platt 校准 + 泄露自检）")
    parser.add_argument("--split", default="val", help="评估分段名（val 或 calibration）")
    parser.add_argument("--model-dir", default=None, help="微调模型目录（默认取 config）")
    parser.add_argument("--params", default=None, help="Platt 参数文件（默认取 config）")
    args = parser.parse_args()

    split_file = os.path.join("models", f"split_{args.split}.jsonl")
    train_file = os.path.join("models", "split_train.jsonl")
    if not os.path.isfile(split_file):
        print(f"[错误] 分段清单不存在: {split_file}（请先运行 train.bat 生成）",
              file=sys.stderr)
        sys.exit(1)

    cfg = load_config(None)
    model_dir = args.model_dir or cfg.model_path
    params_file = args.params or cfg.calibration_params_path

    try:
        classifier = CodeBERTClassifier(
            model_dir, cfg.max_length,
            window_stride=cfg.window_stride, window_batch_size=cfg.window_batch_size)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(4)
    if not os.path.isfile(params_file):
        print(f"[错误] 校准参数不存在: {params_file}（请先运行 calibrate.bat）",
              file=sys.stderr)
        sys.exit(3)
    calibrator = PlattCalibrator.load(params_file)

    records = _read_split(split_file)
    print(f"评估分段 {args.split}: {len(records)} 条 | 模型 {model_dir} | 校准 {params_file}")

    # ---- 推理（审计口径：滑窗全文 max logit，无早停）----
    labels: list[int] = []
    logits: list[float] = []
    probs: list[float] = []
    sources: list[str] = []
    skipped = 0
    for record in records:
        try:
            result = preprocess_file(record["path"])
            logit = classifier.predict_logit(result.text)
        except Exception as exc:
            print(f"跳过: {record['path']}（{type(exc).__name__}: {exc}）")
            skipped += 1
            continue
        labels.append(int(record["label"]))
        logits.append(logit)
        probs.append(calibrator.predict_proba(logit))
        sources.append(_source_of(record["path"]))
    print(f"推理完成: {len(labels)} 条（跳过 {skipped}）")

    # ---- 三口径指标 ----
    pred_logit = [1 if s > 0 else 0 for s in logits]
    pred_prob = [1 if p > 0.5 else 0 for p in probs]
    m_raw = _prf(labels, pred_logit)
    m_cal = _prf(labels, pred_prob)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, logits))
    except Exception:
        auc = None

    # ---- 门控三档统计 ----
    gates = {"allow": Counter(), "llm_review": Counter(), "block": Counter()}
    for label, prob in zip(labels, probs):
        if prob < cfg.allow_threshold:
            gates["allow"][label] += 1
        elif prob > cfg.block_threshold:
            gates["block"][label] += 1
        else:
            gates["llm_review"][label] += 1

    # ---- 分来源表现（恶意召回 / 放行漏检；良性误报）----
    per_source = {}
    for src in sorted(set(sources)):
        idx = [i for i, s in enumerate(sources) if s == src]
        sub_labels = [labels[i] for i in idx]
        sub_pred = [pred_prob[i] for i in idx]
        sub_allow = [1 if probs[i] < cfg.allow_threshold else 0 for i in idx]
        info = {"n": len(idx), "n_malicious": sum(sub_labels)}
        if sum(sub_labels):
            tp = sum(1 for y, p in zip(sub_labels, sub_pred) if y == 1 and p == 1)
            leaked = sum(1 for y, a in zip(sub_labels, sub_allow) if y == 1 and a == 1)
            info["recall_p05"] = tp / sum(sub_labels)
            info["silent_allow"] = leaked
        if sum(1 for y in sub_labels if y == 0):
            n_ben = sum(1 for y in sub_labels if y == 0)
            info["benign_fp"] = sum(1 for y, p in zip(sub_labels, sub_pred) if y == 0 and p == 1)
            info["n_benign"] = n_ben
        per_source[src] = info

    # ---- 分域汇总（code / prompt；混合训练的分域量尺与域间干扰检测）----
    per_domain = {}
    for domain in sorted({_DOMAIN_OF_SOURCE.get(s, "other") for s in sources}):
        idx = [i for i, s in enumerate(sources)
               if _DOMAIN_OF_SOURCE.get(s, "other") == domain]
        sub_labels = [labels[i] for i in idx]
        sub_pred = [pred_prob[i] for i in idx]
        sub_allow = [1 if probs[i] < cfg.allow_threshold else 0 for i in idx]
        n_mal = sum(sub_labels)
        info = {"n": len(idx), "n_malicious": n_mal, "n_benign": len(idx) - n_mal}
        if n_mal:
            tp = sum(1 for y, p in zip(sub_labels, sub_pred) if y == 1 and p == 1)
            info["recall_p05"] = tp / n_mal
            info["silent_allow"] = sum(1 for y, a in zip(sub_labels, sub_allow)
                                       if y == 1 and a == 1)
        if info["n_benign"]:
            info["benign_fp"] = sum(1 for y, p in zip(sub_labels, sub_pred)
                                    if y == 0 and p == 1)
        per_domain[domain] = info

    # ---- 泄露自检：与训练段比对 sha256 原文重叠与家族重叠 ----
    # 家族比较按 (域, 类别) 分组——与 train.py 的划分组语义对齐；
    # 同名不同组（如良恶两侧各有名为 release 的不同技能）不是泄露。
    leak = {"exact_dup_with_train": 0, "family_overlap_with_train": 0, "checked": len(records)}
    if os.path.isfile(train_file):
        train_shas = set()
        train_fams = set()

        def _group_key(path: str, label: int) -> str:
            src = _source_of(path)
            dom = _DOMAIN_OF_SOURCE.get(src, "other")
            return f"{dom}|{label}"

        for record in _read_split(train_file):
            try:
                with open(record["path"], "rb") as handle:
                    train_shas.add(hashlib.sha256(handle.read()).hexdigest())
            except OSError:
                pass
            train_fams.add((_group_key(record["path"], int(record["label"])),
                            _family_key(record["path"])))
        for record in records:
            try:
                with open(record["path"], "rb") as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
            except OSError:
                continue
            if digest in train_shas:
                leak["exact_dup_with_train"] += 1
            elif ((_group_key(record["path"], int(record["label"])),
                   _family_key(record["path"]))) in train_fams:
                leak["family_overlap_with_train"] += 1
    else:
        leak["checked"] = -1  # 训练清单缺失，自检跳过

    # ---- 汇总输出 ----
    n_mal = sum(labels)
    n_ben = len(labels) - n_mal
    summary = {
        "split": args.split,
        "n_samples": len(labels),
        "n_malicious": n_mal,
        "n_benign": n_ben,
        "skipped": skipped,
        "metrics_raw_logit_gt0": m_raw,
        "metrics_calibrated_p_gt05": m_cal,
        "auc": auc,
        "gate": {k: {"malicious": v[1], "benign": v[0], "total": v[0] + v[1]}
                 for k, v in gates.items()},
        "per_source": per_source,
        "per_domain": per_domain,
        "leakage_selfcheck": leak,
    }

    # ---- ECE（15 桶）与 Brier（校准质量量化，v2 评审 P0 补测项）----
    n_bin = 15
    ece = 0.0
    reliability = []
    for b in range(n_bin):
        lo, hi = b / n_bin, (b + 1) / n_bin
        idx = [i for i, p in enumerate(probs)
               if (lo <= p < hi) or (hi == 1.0 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc_b = sum(labels[i] for i in idx) / len(idx)
            ece += len(idx) / len(probs) * abs(conf - acc_b)
            reliability.append({"bin": [round(lo, 4), round(hi, 4)],
                                "n": len(idx), "avg_p": round(conf, 4),
                                "frac_malicious": round(acc_b, 4)})
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / max(1, len(probs))
    summary["ece"] = round(ece, 4)
    summary["brier"] = round(brier, 4)
    summary["reliability_curve"] = reliability

    # ---- 样本级明细落盘（供独立复核与后续分析）----
    detail_path = os.path.join("models", f"evaluation_{args.split}_detail.jsonl")
    with open(detail_path, "w", encoding="utf-8") as handle:
        for i in range(len(labels)):
            handle.write(json.dumps(
                {"label": labels[i], "logit": round(logits[i], 4),
                 "prob": round(probs[i], 4), "source": sources[i]},
                ensure_ascii=False) + "\n")

    out_path = os.path.join("models", f"evaluation_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n===== 审计口径评估结果 =====")
    print(f"样本: {len(labels)}（恶意 {n_mal} / 良性 {n_ben}，跳过 {skipped}）")
    print(f"AUC: {auc if auc is None else round(auc, 4)}")
    print(f"口径1 logit>0   : acc {m_raw['accuracy']:.4f} prec {m_raw['precision']:.4f} "
          f"rec {m_raw['recall']:.4f} f1 {m_raw['f1']:.4f} "
          f"(FP {m_raw['fp']} / FN {m_raw['fn']})")
    print(f"口径2 p>0.5     : acc {m_cal['accuracy']:.4f} prec {m_cal['precision']:.4f} "
          f"rec {m_cal['recall']:.4f} f1 {m_cal['f1']:.4f} "
          f"(FP {m_cal['fp']} / FN {m_cal['fn']})")
    for k in ("allow", "llm_review", "block"):
        g = gates[k]
        total = g[0] + g[1]
        pure = g[1] / total * 100 if total else 0.0
        print(f"门控 {k:10s}: {total:5d} 条（恶意 {g[1]} / 良性 {g[0]}，恶意纯度 {pure:.1f}%）")
    for src, info in per_source.items():
        extra = (f"recall {info.get('recall_p05', float('nan')):.3f} 静默放行 {info.get('silent_allow', '-')}"
                 if info.get("n_malicious") else
                 f"误报 {info.get('benign_fp', 0)}/{info.get('n_benign', 0)}")
        print(f"来源 {src:7s}: {info['n']:5d} 条（恶意 {info.get('n_malicious', 0)}）{extra}")
    for dom, info in per_domain.items():
        rec = ('recall %.3f 静默放行 %d' % (info['recall_p05'], info['silent_allow'])
               if 'recall_p05' in info else '无恶意样本')
        fp = (' 误报 %d/%d' % (info['benign_fp'], info['n_benign'])
              if 'benign_fp' in info else '')
        print(f"域   {dom:7s}: {info['n']:5d} 条（恶意 {info['n_malicious']} / 良性 {info['n_benign']}）{rec}{fp}")
    print(f"ECE(15桶): {summary['ece']:.4f} | Brier: {summary['brier']:.4f}")
    print(f"泄露自检: 原文与train重复 {leak['exact_dup_with_train']} / "
          f"家族与train重叠 {leak['family_overlap_with_train']} / 共查 {leak['checked']} 条")
    print(f"结果已写入 {out_path}（明细 {detail_path}）")


if __name__ == "__main__":
    main()
