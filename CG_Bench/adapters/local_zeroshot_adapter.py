# -*- coding: utf-8 -*-
"""CalibGuard Bench 本地基座零样本适配器（v2.5 新增，2026-10-04 用户需求）。

用途：**不经微调**，直接把文件内容喂给本地 Causal LM 基座（Qwen 系），按提示词让模型
自行判定是否恶意——即"未微调基座"的下界对照（与 `llm_zero` 对 DeepSeek 做的事同义，
只是推理后端从云端 API 换成本地 transformers 生成）。

口径刻意与 `llm_adapter` 保持一致，保证横向可比：
  * 提示词模板：复用 `llm_adapter.PROMPTS`（本文件不另立模板，遵守规则 2）；
  * 分窗长度：`llm_adapter.TEXT_MAX_CHARS`（12000 字符/窗，与规则 1 同量级）；
  * 输出契约：`{"malicious": bool, "confidence": float}`，解析复用
    `llm_adapter._parse_response`；解析失败按规则 5 用 `_STRICT_SUFFIX` 重试 1 次，
    仍失败则该窗不计入（全部窗都失败 → 回退
    `{"malicious": False, "confidence": 0.5, "parse_error": True}`，与 llm_adapter 逐字一致）。

判定口径（2026-10-05 用户决策：**test 段改滑窗**）：
  * 与微调模型评测的 `CodeBERTClassifier.iter_window_batches`（全文滑窗 + max 聚合）语义对齐；
  * 把文本按 `WINDOW_CHARS` 切成窗口逐窗判定，取 **max(confidence)** 作该样本概率。
    confidence 即 P(恶意) 且 logit 与 p 单调等价 ⇒ 与 ft 的 `max(窗口 logit)` **同义**；
  * `--max-windows N`（N>0）时在全文**均匀抽 N 个窗**并保留头窗（头窗含 setup.py /
    package.json 等信号位），N=0 为**全覆盖**（每个窗都判，代价约为单窗的 8.4 倍）。

与 llm_adapter 的差异（仅此三点）：
  * 无网调用 → 无 ApiFatalError / 无 api_error_skip.json / 无熔断；
  * 判定用贪心解码（do_sample=False）保证可复现；
  * Qwen3 系模板支持 `enable_thinking=False`（避免把预算耗在思维链上；不支持则自动回退）。
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
from calibguard.preprocessor import preprocess_file      # noqa: E402

DEFAULT_TEMPLATE = "v_feature"      # 与 hybrid 的复核模板一致；三候选模板见 llm_adapter.PROMPTS
MAX_NEW_TOKENS = 128                # 只需吐一个 JSON，留足余量防截断
MAX_INPUT_TOKENS = 8192             # 安全上限（TEXT_MAX_CHARS=12000 ≈ 4k token，不会触及）
WINDOW_CHARS = llm_adapter.TEXT_MAX_CHARS   # 每窗字符预算（12000，与规则 1 同量级）
DEFAULT_MAX_WINDOWS = 0             # 0 = 全覆盖（每个窗都判）

_WIN_STATS = {"windows": 0, "samples": 0, "windowed_samples": 0, "all_fail": 0}


def _windows(text: str, win: int, max_windows: int) -> list:
    """把文本切成不重叠的 win 字符窗；max_windows>0 时在全文均匀抽窗（始终保留头窗）。

    头窗必须保留：语料搬运时 setup.py / package.json 被前置（见
    data/raw/_prep/dd_transfer.py::extract_text），信号位在头部。
    """
    if len(text) <= win:
        return [text]
    starts = list(range(0, len(text), win))
    if max_windows and len(starts) > max_windows:
        k = max_windows
        step = (len(starts) - 1) / (k - 1) if k > 1 else 0.0
        picked = sorted({int(round(i * step)) for i in range(k)})
        starts = [starts[i] for i in picked]
    return [text[s:s + win] for s in starts]


_CACHE = {}                         # model_path -> (tokenizer, model)


def _runtime(model_path: str):
    """按模型路径缓存 (tokenizer, model)——生成式加载昂贵，禁止逐条重载。"""
    if model_path not in _CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        _CACHE[model_path] = (tokenizer, model)
        print(f"[local_zeroshot] 已加载基座 {model_path}（device={device}）", flush=True)
    return _CACHE[model_path]


def _render(tokenizer, messages: list) -> str:
    """套用基座自带 chat_template；Qwen3 系关闭 thinking（不支持时回退）。"""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)


def _generate(tokenizer, model, prompt: str) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_INPUT_TOKENS).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _judge_chunk(tokenizer, model, text: str, template: str):
    """单个（≤WINDOW_CHARS）片段判定；非合法 JSON 按规则 5 重试 1 次；仍失败返回 None。"""
    raw = _generate(tokenizer, model,
                    _render(tokenizer, llm_adapter._build_messages(text, template)))
    parsed = llm_adapter._parse_response(raw)
    if parsed is None:
        # 规则 5：追加"只输出 JSON"提醒重试 1 次（后缀同样加在用户消息末尾）
        retry = _render(tokenizer, llm_adapter._build_messages(
            text, template, suffix=llm_adapter._STRICT_SUFFIX))
        parsed = llm_adapter._parse_response(_generate(tokenizer, model, retry))
    return parsed


def classify(text: str, model_path: str = None, template: str = DEFAULT_TEMPLATE,
             max_windows: int = DEFAULT_MAX_WINDOWS,
             window_chars: int = WINDOW_CHARS) -> dict:
    """零样本判定（**全文滑窗**）：文本 → {"malicious": bool, "confidence": float}。

    逐窗判定后取 **max(confidence)**（confidence 即 P(恶意)，与 ft 评测的
    `max(窗口 logit)` 同义）；某窗解析失败则跳过该窗；全部窗都失败 → 按 llm_adapter
    口径回退 `{"malicious": False, "confidence": 0.5, "parse_error": True}`。
    """
    if not model_path:
        raise ValueError("local_zeroshot_adapter.classify 需要 model_path（本地基座目录）")
    if template not in llm_adapter.PROMPTS:
        raise ValueError(f"未知模板 {template}（可选：{sorted(llm_adapter.PROMPTS)}）")
    tokenizer, model = _runtime(model_path)

    chunks = _windows(text, window_chars, max_windows)
    best = None
    for chunk in chunks:
        parsed = _judge_chunk(tokenizer, model, chunk, template)
        if parsed is not None and (best is None or parsed["confidence"] > best["confidence"]):
            best = parsed

    _WIN_STATS["samples"] += 1
    _WIN_STATS["windows"] += len(chunks)
    if len(chunks) > 1:
        _WIN_STATS["windowed_samples"] += 1
    if best is None:
        _WIN_STATS["all_fail"] += 1
        return {"malicious": False, "confidence": 0.5, "parse_error": True,
                "template": template, "variant": "zero"}
    best.update({"template": template, "variant": "zero"})
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="CalibGuard Bench 本地基座零样本适配器（不训练，直接提示词判定）")
    parser.add_argument("--exp", required=True, help="实验名（experiments.yaml 的 name）")
    parser.add_argument("--base-model", required=True, help="本地基座目录（如 base_models/Qwen3-0.6B）")
    parser.add_argument("--out-dir", required=True, help="实验产物目录 results/{name}")
    parser.add_argument("--splits-dir", default="results/_splits")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE,
                        choices=tuple(llm_adapter.PROMPTS), help="提示词模板（默认 v_feature）")
    parser.add_argument("--window-chars", type=int, default=WINDOW_CHARS,
                        help=f"每窗字符预算（默认 {WINDOW_CHARS}）")
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS,
                        help="每样本最多判定的窗数：0=全覆盖（默认）；N>0=全文均匀抽 N 窗（保留头窗）")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.base_model):
        print(f"[local_zeroshot] 基座目录不存在：{args.base_model}", file=sys.stderr)
        return 1

    def _classify(text: str) -> dict:
        return classify(text, args.base_model, args.template,
                        max_windows=args.max_windows, window_chars=args.window_chars)

    # max_chars=None 是滑窗生效的前提：分窗在 classify 内完成，外层若再整体截断到
    # 12000 字符，滑窗就会退化成"只判头窗"（2026-10-05 修正）。
    llm_adapter.run_bench_eval(_classify, args.exp, args.out_dir, args.splits_dir,
                               adapter="local_zeroshot_adapter",
                               max_chars=None, split="test")
    print(f"[local_zeroshot] 滑窗统计：样本 {_WIN_STATS['samples']} 条 / 窗 {_WIN_STATS['windows']} 个 / "
          f"多窗样本 {_WIN_STATS['windowed_samples']} 条 / 全窗失败 {_WIN_STATS['all_fail']} 条"
          f"（window_chars={args.window_chars}, max_windows={args.max_windows}）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
