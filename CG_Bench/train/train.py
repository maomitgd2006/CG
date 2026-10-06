"""CalibGuard 微调训练脚本（T13）。

二分类：0=良性（LABEL_BENIGN）/ 1=恶意（LABEL_MALICIOUS）。
训练 / 校准 / 审计必须共用同一个 preprocess_file，保证特征一致性。
运行方式：train.bat [参数]，或先设 PYTHONPATH=src 再 python scripts/train.py [参数]。
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from calibguard.constants import LABEL_BENIGN, LABEL_MALICIOUS
from calibguard.preprocessor import preprocess_file

# 落盘产物目录（split_*.jsonl 与 metrics.json 的固定位置，见 16.4 第 7 条）
_ARTIFACT_DIR = "models"
# 留出段占比（v1.3 补丁：四段划分 train 70% / val 10% / calib 10% / test 10%）
# val 仅用于训练中的模型选择（load_best_model_at_end）；test 仅用于最终评估——
# 模型选择与对外报告的指标不得共用数据（否则轻度乐观偏差）。
_SPLIT_RATIOS = {"val": 0.10, "calib": 0.10, "test": 0.10}
# 划分用的固定随机种子（见 16.3，与 --seed 相互独立）
_SPLIT_SEED = 42
# 训练固定 512（与 config.yaml 无关，见 16.4 第 2 条）
_TRAIN_MAX_LENGTH = 512


class TextDataset(Dataset):
    """样本文本数据集：__getitem__ 返回 input_ids / attention_mask / labels。"""

    def __init__(self, samples: list[tuple[str, int]], tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        text, label = self.samples[index]
        encoded = self.tokenizer(text, truncation=True, max_length=_TRAIN_MAX_LENGTH,
                                 padding="max_length")
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def _collect_files(directory: str) -> list[str]:
    """递归收集目录下的全部常规文件，按路径排序。"""
    collected: list[str] = []
    for root, _dir_names, file_names in os.walk(directory):
        for file_name in file_names:
            full_path = os.path.join(root, file_name)
            if os.path.isfile(full_path):
                collected.append(full_path)
    collected.sort()
    return collected


# 家族键正则：SKILL.md front-matter 的 name 字段（仅查文件头 4096 字节）
_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.M)
# 四段式文件名家族约定（v1.3 代码域/MCP 域数据）：mal_py_{家族名}_{序号}.txt 等。
# 家族名直接编码在文件名中（恶意 PyPI 包名 / 良性技能名 / GitHub slug），同族多文件同段。
# 三段式旧命名（mal_ms_0.txt）不匹配此正则，自然回退到 front-matter 规则。
_FNAME_FAM_RE = re.compile(r"^(?:mal|ben)_(?:ms|sb|py|atr|twin|npm|mcp|ch|sk|mcs|ide|rb|go|jv|ps|pi|ws|skb)_(.+)_\d+\.txt$")

# v2.5 近重复防线（2026-10-02）：sha8 命名源（HF 扩容 ps/pi/ws/skb）的归一化家族指纹
_NORM_FAM_RE = re.compile(r"^(?:mal|ben)_(?:ps|pi|ws|skb)_[0-9a-f]+_\d+\.txt$")
_NORM_WS_TRANS = str.maketrans("", "", " \t\r\n\x0b\x0c")

# 来源前缀 → 域映射（v2 两域：code / prompt）
# prompt 域：MalSkillBench(ms) / SkillBench(sb) / ATR 良性技能(atr) / ClawHub 共识裁决(ch) /
#            DataDog ai-skills(sk) / MCPShield 文本操纵筛选集(mcs)
# code 域：PyPI(py) / npm / 良性孪生(twin) / DataDog IDE 扩展(ide) /
#          registry 良性扩容——RubyGems(rb) / Go(go) / Maven sources(jv)
# v1 的 mal_mcp/ben_mcp 已归档至 data/archive/（F/A 级仓库标签非文本层，召回 24.4% 教训）
_DOMAIN_BY_PREFIX = {
    "mal_ms": "prompt", "ben_ms": "prompt", "mal_sb": "prompt", "ben_sb": "prompt",
    "ben_atr": "prompt", "mal_ch": "prompt", "ben_ch": "prompt",
    "mal_sk": "prompt", "mal_mcs": "prompt", "ben_mcs": "prompt",
    "mal_py": "code", "ben_py": "code", "ben_twin": "code",
    "mal_npm": "code", "ben_npm": "code", "mal_ide": "code",
    "ben_rb": "code", "ben_go": "code", "ben_jv": "code",
    # v2.5（2026-10-02）HF 扩容源登记：PowerShell/Webshell → code；
    # 恶意 prompt（Necent/guychuk/ahsanayub）/MaliciousSkillBench → prompt
    "mal_ps": "code", "ben_ps": "code", "mal_ws": "code", "ben_ws": "code",
    "mal_pi": "prompt", "ben_pi": "prompt", "mal_skb": "prompt", "ben_skb": "prompt",
}


def _family_key(path: str) -> str:
    """家族键，优先级：四段式文件名 > front-matter name > 文件名自成一家族。

    MalSkillBench 等成对构造数据集中，同一良性原型派生多个恶意变体（CI/PI/MIXED），
    它们共享同一 name——家族键用于保证同族样本永不跨段（group-aware split）。
    代码域数据（mal_py_/ben_py_）无 front-matter，家族名由转存规格编码在文件名中。
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


def _domain_of(path: str) -> str:
    """样本所属域：由文件名前缀查 _DOMAIN_BY_PREFIX，未知前缀回退 prompt。

    域感知划分保证每个域内部各自 70/10/10/10——每个留出段中每个域都有代表，
    支持混合训练后的分域评估（域间干扰检测）。
    """
    base = os.path.basename(path)
    for prefix, domain in _DOMAIN_BY_PREFIX.items():
        if base.startswith(prefix + "_"):
            return domain
    return "prompt"


def _dedup_and_group(samples: list[tuple[str, int]], drop_log: list | None = None,
                     sha_cache: dict | None = None
                     ) -> tuple[dict, int, int, int, int]:
    """防御性内容去重 + 家族分组（幂等，与 data/train/CLEAN_REPORT.txt 的物理清洗规则一致）。

    规则（v2.5，2026-10-02 归一化层加固——首轮划分自检发现 576 对跨段
    "仅大小写/空白不同"近重复，v2 raw 级去重无法覆盖，故将规则 1/2 从
    raw sha 扩展到归一化指纹）：
    1) raw sha256 相同且标签不一致 → 整组丢弃（同文本亦良亦恶，标签不可信）；
    2) raw sha256 相同且标签一致 → 仅保留 basename 最小的一条；
    3) 归一化指纹（小写+去空白 sha256）相同且标签不一致 → 整组丢弃；
    4) 归一化指纹相同且标签一致 → 仅保留 basename 最小的一条（格式变体
       同一信号只留代表——文件仍在盘上，仅不参与划分）；
    5) 家族分组：sha8 命名源（ps/pi/ws/skb，无真实家族键）用 "ns:"+指纹
       作家族键，其余前缀用 _family_key(path)。
    可选参数（默认不启用，不影响行为）：drop_log 收集每次丢弃的
    (path, label, reason)；sha_cache 收集 {path: (raw_sha, norm_sha)}
    供划分自检零额外读盘复核。
    返回 (families, dropped_conflict, dropped_dup, dropped_norm_conflict,
          dropped_norm_dup)。
    """
    def _drop(path, label, reason):
        if drop_log is not None:
            drop_log.append((path, label, reason))

    sha_groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    norm_of: dict[str, str] = {}
    for path, label in sorted(samples, key=lambda s: os.path.basename(s[0])):
        with open(path, "rb") as handle:
            raw = handle.read()
        raw_sha = hashlib.sha256(raw).hexdigest()
        norm_sha = hashlib.sha256(
            raw.decode("utf-8", "replace").lower()
            .translate(_NORM_WS_TRANS).encode("utf-8")).hexdigest()
        del raw
        sha_groups[raw_sha].append((path, label))
        norm_of[path] = norm_sha
        if sha_cache is not None:
            sha_cache[path] = (raw_sha, norm_sha)
    raw_kept: list[tuple[str, int]] = []
    dropped_conflict = 0
    dropped_dup = 0
    for group in sha_groups.values():
        labels = {label for _, label in group}
        if len(labels) > 1:
            dropped_conflict += len(group)
            for path, label in group:
                _drop(path, label, "raw_conflict")
            continue
        kept_member = min(group, key=lambda m: os.path.basename(m[0]))
        raw_kept.append(kept_member)
        dropped_dup += len(group) - 1
        for path, label in group:
            if path != kept_member[0]:
                _drop(path, label, "raw_dup")
    del sha_groups
    norm_groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path, label in raw_kept:
        norm_groups[norm_of[path]].append((path, label))
    kept: list[tuple[str, int]] = []
    dropped_norm_conflict = 0
    dropped_norm_dup = 0
    for group in norm_groups.values():
        labels = {label for _, label in group}
        if len(labels) > 1:
            dropped_norm_conflict += len(group)
            for path, label in group:
                _drop(path, label, "norm_conflict")
            continue
        kept_member = min(group, key=lambda m: os.path.basename(m[0]))
        kept.append(kept_member)
        dropped_norm_dup += len(group) - 1
        for path, label in group:
            if path != kept_member[0]:
                _drop(path, label, "norm_dup")
    del norm_groups
    families: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path, label in kept:
        if _NORM_FAM_RE.match(os.path.basename(path)):
            fam_key = "ns:" + norm_of[path][:16]
        else:
            fam_key = _family_key(path)
        families[fam_key].append((path, label))
    return (families, dropped_conflict, dropped_dup,
            dropped_norm_conflict, dropped_norm_dup)


def _fill_by_family(families: dict[str, list[tuple[str, int]]], rng: random.Random
                    ) -> tuple[list, list, list, list]:
    """把一个 (域,类别) 分组的家族整体装入 70/10/10/10 四段（家族不拆分、不跨段）。

    贪心装填：按 shuffle 顺序先填训练段至配额，再依次填验证段、校准段，
    剩余进测试段；max(配额, 家族大小) 保证单一家族超过配额时也至少能放下。
    """
    keys = sorted(families.keys())
    rng.shuffle(keys)
    total = sum(len(families[k]) for k in keys)
    target_train = round(total * (1 - sum(_SPLIT_RATIOS.values())))
    target_val = round(total * _SPLIT_RATIOS["val"])
    target_calib = round(total * _SPLIT_RATIOS["calib"])
    train_part: list[tuple[str, int]] = []
    val_part: list[tuple[str, int]] = []
    calib_part: list[tuple[str, int]] = []
    test_part: list[tuple[str, int]] = []
    for key in keys:
        fam = families[key]
        if len(train_part) + len(fam) <= max(target_train, len(fam)):
            train_part.extend(fam)
        elif len(val_part) + len(fam) <= max(target_val, len(fam)):
            val_part.extend(fam)
        elif len(calib_part) + len(fam) <= max(target_calib, len(fam)):
            calib_part.extend(fam)
        else:
            test_part.extend(fam)
    return train_part, val_part, calib_part, test_part


def _load_texts(samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """预处理每个样本；失败样本打印警告后丢弃。返回 [(text, label), ...]。"""
    loaded: list[tuple[str, int]] = []
    for path, label in samples:
        try:
            result = preprocess_file(path)
        except Exception as exc:
            print(f"跳过无法预处理的样本: {path}")
            print(f"  原因: {type(exc).__name__}: {exc}")
            continue
        loaded.append((result.text, label))
    return loaded


def _write_split_jsonl(file_name: str, samples: list[tuple[str, int]]) -> None:
    """写 models/split_*.jsonl，每行 {"path": 相对项目根的正斜杠路径, "label": 0或1}。

    v1.3 起改存相对路径：旧版绝对路径在项目目录迁移/换机后全部失效
    （CGv0.2.0 审计实测教训）。读取方须在项目根目录运行
    （train.bat / calibrate.bat / evaluate.bat 均满足该约定）。
    """
    target = os.path.join(_ARTIFACT_DIR, file_name)
    with open(target, "w", encoding="utf-8") as handle:
        for path, label in samples:
            rel = os.path.relpath(path, os.getcwd()).replace("\\", "/")
            handle.write(json.dumps({"path": rel, "label": label},
                                    ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="微调 CodeBERT 二分类模型（0=良性 / 1=恶意）")
    parser.add_argument("--data-dir", default="data/train",
                        help="训练数据目录（下含 malicious/ 与 benign/）")
    parser.add_argument("--base-model", default="microsoft/codebert-base",
                        help="基座模型")
    parser.add_argument("--output-dir", default="models/codebert_finetuned",
                        help="微调模型输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    malicious_dir = os.path.join(args.data_dir, "malicious")
    benign_dir = os.path.join(args.data_dir, "benign")
    if not os.path.isdir(malicious_dir) or not os.path.isdir(benign_dir):
        print(f"[错误] 数据目录不完整，需要 {malicious_dir} 与 {benign_dir} 同时存在",
              file=sys.stderr)
        sys.exit(1)

    # 16.3(v1.3) group-aware + 域感知分层划分（泄露审计 2026-09-24 修复）：
    # 先防御性内容去重，再按 (域, 类别) 分四组、组内按家族键整体分段——
    # 同一家族的样本（含原文重复与近重复变体）永不跨段；每个域内部各自 70/15/15，
    # 验证/校准段中每个域都有代表（支持混合训练后的分域评估与域间干扰检测）。
    malicious_files = _collect_files(malicious_dir)
    benign_files = _collect_files(benign_dir)
    all_samples = ([(p, LABEL_MALICIOUS) for p in malicious_files]
                   + [(p, LABEL_BENIGN) for p in benign_files])
    families, dropped_conflict, dropped_dup, dropped_norm_conflict, dropped_norm_dup = (
        _dedup_and_group(all_samples))

    # 按 (域, 类别) 归组家族：code/prompt × mal/ben 四组，组内独立划分。
    # 归组以【家族为单位】而非成员——跨域同族（benign 技能的 SKILL 样本与其
    # 脚本样本共享家族键）与跨类同族（原型-变体对）都必须整族同组同段，
    # 否则家族会被拆到不同组而跨段（这是 v1.3 修复的泄露形态之一）。
    group_families: dict[tuple[str, str], dict[str, list[tuple[str, int]]]] = {}
    for fam, members in families.items():
        cls = "mal" if any(l == LABEL_MALICIOUS for _, l in members) else "ben"
        dom = _domain_of(members[0][0])
        group_families.setdefault((dom, cls), {})[fam] = members

    rng = random.Random(_SPLIT_SEED)
    parts: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "calib": [], "test": []}
    # 固定顺序遍历全部 (域,类别) 分组（排序保证确定性），同一 rng 顺序确定
    for key in sorted(group_families.keys()):
        fams = group_families[key]
        t, v, c, s = _fill_by_family(fams, rng)
        parts["train"].extend(t)
        parts["val"].extend(v)
        parts["calib"].extend(c)
        parts["test"].extend(s)

    train_samples = parts["train"]
    val_samples = parts["val"]
    calib_samples = parts["calib"]
    test_samples = parts["test"]

    # 留出三段（val/calib/test）必须双类齐备（calib 是 T14 拟合前提，test 是对外指标前提）
    for seg_name, seg in [("验证", val_samples), ("校准", calib_samples), ("测试", test_samples)]:
        if not any(l == LABEL_MALICIOUS for _, l in seg) or not any(l == LABEL_BENIGN for _, l in seg):
            print(f"[错误] {seg_name}段缺少双类别样本，无法继续", file=sys.stderr)
            sys.exit(1)

    # 分域统计（验证集的分域构成，供分域评估引用）
    domain_stat: dict[str, dict[str, int]] = {}
    for path, label in all_samples:
        dom = _domain_of(path)
        bucket = domain_stat.setdefault(dom, {"mal": 0, "ben": 0})
        bucket["mal" if label == LABEL_MALICIOUS else "ben"] += 1
    val_domain: dict[str, dict[str, int]] = {}
    for path, label in val_samples:
        dom = _domain_of(path)
        bucket = val_domain.setdefault(dom, {"mal": 0, "ben": 0})
        bucket["mal" if label == LABEL_MALICIOUS else "ben"] += 1

    print(f"数据划分（group-aware + 域感知，四段）：家族 {len(families)} 个 | "
          f"训练 {len(train_samples)} | 验证 {len(val_samples)} | "
          f"校准 {len(calib_samples)} | 测试 {len(test_samples)} | "
          f"丢弃标签矛盾 {dropped_conflict} / 内容重复 {dropped_dup} | "
          f"归一化矛盾 {dropped_norm_conflict} / 归一化重复 {dropped_norm_dup}")
    for dom, bucket in sorted(domain_stat.items()):
        v = val_domain.get(dom, {"mal": 0, "ben": 0})
        print(f"  域 {dom}: 全量 mal {bucket['mal']} / ben {bucket['ben']} | "
              f"验证段 mal {v['mal']} / ben {v['ben']}")

    # 16.4 第 1 条：预处理取文本（与审计/校准共用同一个 preprocess_file）
    train_texts = _load_texts(train_samples)
    val_texts = _load_texts(val_samples)

    # 16.4 第 7 条：落盘四个分段清单（按 16.3 的划分结果原样记录）
    os.makedirs(_ARTIFACT_DIR, exist_ok=True)
    _write_split_jsonl("split_train.jsonl", train_samples)
    _write_split_jsonl("split_val.jsonl", val_samples)
    _write_split_jsonl("split_calibration.jsonl", calib_samples)
    _write_split_jsonl("split_test.jsonl", test_samples)

    if not train_texts or not val_texts:
        print(f"[错误] 训练集或验证集为空（训练 {len(train_texts)}，验证 {len(val_texts)}），"
              f"请增加数据量后重试", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    # 2026-10-03 修正：Qwen 系（decoder-only）tokenizer 有 pad_token 但 **config.pad_token_id 为 None**，
    # 而 transformers 的 ...ForSequenceClassification 前向在 batch>1 时要求它非空
    # （modeling_layers.py：`if self.config.pad_token_id is None and batch_size != 1: raise`），
    # 否则训练第 1 步即崩（v2.5 三个 Qwen *_shared 均因此失败）。codebert 本就有 pad_token_id，
    # 此分支为 no-op。修正后的 config 会随 save_model() 落盘，下游 calibrate/evaluate 自然继承。
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = TextDataset(train_texts, tokenizer)
    val_dataset = TextDataset(val_texts, tokenizer)

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = float((preds == labels).mean())
        f1_malicious = float(f1_score(labels, preds, pos_label=LABEL_MALICIOUS))
        return {"accuracy": accuracy, "f1_malicious": f1_malicious}

    training_kwargs = {
        "output_dir": args.output_dir + "_tmp",
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "seed": args.seed,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "save_total_limit": 1,
        "logging_steps": 50,
        # 强制项：禁止 Trainer 接入 tensorboard 等可视化后端（避免写出额外日志目录）
        "report_to": "none",
    }
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **training_kwargs)
    except TypeError as exc:
        # 版本兼容：老版本 transformers 不认识 eval_strategy（只允许这一处替换）
        print(f"[提示] 当前 transformers 不支持 eval_strategy，改用 evaluation_strategy：{exc}")
        training_args = TrainingArguments(evaluation_strategy="epoch", **training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # 16.4 第 6 条：保存模型与 tokenizer
    trainer.save_model(args.output_dir)
    processing = getattr(trainer, "processing_class", None)
    if processing is None:
        processing = getattr(trainer, "tokenizer", None)
    if processing is None:
        # 兜底：保证输出目录一定含 tokenizer 文件，audit 才能加载
        print("[提示] trainer 未提供 processing_class/tokenizer，改用本地 tokenizer 保存")
        processing = tokenizer
    processing.save_pretrained(args.output_dir)

    eval_metrics = trainer.evaluate()
    metrics = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "calib_samples": len(calib_samples),
        "test_samples": len(test_samples),
        "eval_accuracy": eval_metrics.get("eval_accuracy"),
        "eval_f1_malicious": eval_metrics.get("eval_f1_malicious"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "base_model": args.base_model,
        # v1.3：划分元信息（group-aware split，供性能报告 v2 引用与审计追溯）
        "split": {
            "method": "group-aware",
            "families": len(families),
            "dropped_label_conflict": dropped_conflict,
            "dropped_content_dup": dropped_dup,
            "dropped_norm_conflict": dropped_norm_conflict,
            "dropped_norm_dup": dropped_norm_dup,
        },
    }
    with open(os.path.join(_ARTIFACT_DIR, "metrics.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False, indent=2))
    # 16.4 第 8 条：打印 metrics.json 内容
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
