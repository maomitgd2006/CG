"""CalibGuard 超参搜索脚本（v1.3 新增：基于验证集的网格搜索）。

方法论：超参选择【仅用 val 段】——挑 epoch 已由 load_best_model_at_end 承担，
本脚本把选择范围扩展到 lr × epochs 网格；test 段不参与任何选择，对外指标
仍以 evaluate.bat --split test 为准（val 参与过选择，其指标偏乐观，禁止对外引用）。

运行：hpo.bat（约 1 小时 GPU）。跑完后按打印的最优参数执行正式 train.bat。
产物：models/hpo_report.json（全部网格结果 + 最优组）。
"""
import itertools
import json
import os
import random
import shutil
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import train as T
from calibguard.constants import LABEL_MALICIOUS

# 网格（刻意小：val 信息量有限，大网格会过拟合 val）
GRID = {"lr": [1e-5, 2e-5, 3e-5], "epochs": [3, 5]}
_BATCH = 16
_BASE = "microsoft/codebert-base"


def run_one(lr: float, epochs: int, out_dir: str, train_texts, val_texts) -> dict:
    """训练一组超参并在 val 上评估；不保留模型（正式模型由 train.bat 重训）。"""
    tokenizer = AutoTokenizer.from_pretrained(_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(_BASE, num_labels=2)
    train_ds = T.TextDataset(train_texts, tokenizer)
    val_ds = T.TextDataset(val_texts, tokenizer)

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean()),
                "f1_malicious": float(f1_score(labels, preds, pos_label=LABEL_MALICIOUS))}

    kwargs = dict(
        output_dir=out_dir + "_tmp", learning_rate=lr,
        per_device_train_batch_size=_BATCH, per_device_eval_batch_size=_BATCH,
        num_train_epochs=epochs, weight_decay=0.01, warmup_ratio=0.1, seed=42,
        save_strategy="epoch", save_total_limit=1, logging_steps=200,
        report_to="none", load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    try:
        args = TrainingArguments(eval_strategy="epoch", **kwargs)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **kwargs)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    ev = trainer.evaluate()
    shutil.rmtree(out_dir + "_tmp", ignore_errors=True)
    return {"lr": lr, "epochs": epochs,
            "val_loss": ev.get("eval_loss"),
            "val_accuracy": ev.get("eval_accuracy"),
            "val_f1_malicious": ev.get("eval_f1_malicious")}


def main() -> None:
    # 划分与 train.py 完全一致（seed 42 确定性），保证网格各组公平比较
    mal = T._collect_files("data/train/malicious")
    ben = T._collect_files("data/train/benign")
    samples = [(p, 1) for p in mal] + [(p, 0) for p in ben]
    fams, _, _, _, _ = T._dedup_and_group(samples)
    group_families = {}
    for fam, members in fams.items():
        cls = "mal" if any(l == 1 for _, l in members) else "ben"
        dom = T._domain_of(members[0][0])
        group_families.setdefault((dom, cls), {})[fam] = members
    rng = random.Random(42)
    parts = {k: [] for k in ("train", "val", "calib", "test")}
    for key in sorted(group_families.keys()):
        t, v, c, s = T._fill_by_family(group_families[key], rng)
        parts["train"].extend(t)
        parts["val"].extend(v)
        parts["calib"].extend(c)
        parts["test"].extend(s)
    print(f"划分（与 train.py 一致）: train {len(parts['train'])} / val {len(parts['val'])}")

    train_texts = T._load_texts(parts["train"])
    val_texts = T._load_texts(parts["val"])
    print(f"预处理完成: 训练 {len(train_texts)} / 验证 {len(val_texts)}")

    os.makedirs("models/hpo", exist_ok=True)
    results = []
    combos = list(itertools.product(GRID["lr"], GRID["epochs"]))
    for i, (lr, epochs) in enumerate(combos, 1):
        print(f"\n=== [{i}/{len(combos)}] lr={lr} epochs={epochs} ===")
        r = run_one(lr, epochs, f"models/hpo/lr{lr}_e{epochs}",
                    train_texts, val_texts)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))

    results.sort(key=lambda r: -(r["val_f1_malicious"] or 0.0))
    best = results[0]
    with open("models/hpo_report.json", "w", encoding="utf-8") as f:
        json.dump({"grid": GRID, "results": results, "best": best},
                  f, ensure_ascii=False, indent=2)

    print("\n===== HPO 汇总（按 val_f1_malicious 降序） =====")
    for r in results:
        print(f"  lr={r['lr']:<8} epochs={r['epochs']}: "
              f"f1_mal={r['val_f1_malicious']:.4f} acc={r['val_accuracy']:.4f} "
              f"loss={r['val_loss']:.4f}")
    print(f"\n最优: lr={best['lr']} epochs={best['epochs']}")
    print(f"正式训练: train.bat --lr {best['lr']} --epochs {best['epochs']}")
    print("提醒: 重训 + calibrate 后，对外指标以 evaluate.bat --split test 为准；"
          "val 指标已参与超参选择，禁止对外引用")


if __name__ == "__main__":
    main()
