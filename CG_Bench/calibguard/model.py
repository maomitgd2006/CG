"""CodeBERT 推理模块（v1.1 滑动窗口全文推理）。

把任意长度文本切分为重叠滑动窗口（窗口 max_length token、步长 window_stride token），
按批做张量级批量推理，产出各窗口的恶意类原始 logit。全文都被窗口覆盖，不设长度硬上限。
批间早停的决策在 pipeline 层（见 14.3 步骤 4），本模块不做决策。
"""
from typing import Iterator

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from calibguard import constants as C


class CodeBERTClassifier:
    """加载微调后的 CodeBERT 分类模型，做滑动窗口批量推理。"""

    def __init__(self, model_path: str, max_length: int = 512, device: str | None = None,
                 window_stride: int = 256, window_batch_size: int = 8):
        # 1. 设备选择
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # 2. 加载 tokenizer 与模型（微调目录已含 num_labels=2 配置，不再传 num_labels）
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            # 3. 移到设备并切换到推理模式
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(f"模型加载失败: {exc}") from exc
        # 4. 保存窗口参数
        self.max_length = max_length
        self.window_stride = window_stride
        self.window_batch_size = window_batch_size

    def window_starts(self, n_tokens: int) -> list[int]:
        """纯函数：给定全文 token 数，返回全部窗口起始下标（token 空间，不含特殊 token）。"""
        W = self.max_length - 2          # 每窗有效内容长度（预留 [CLS] 与 [SEP] 两个特殊位）
        if n_tokens <= W:
            return [0]                   # 单窗口，内容不足由 padding 补齐
        starts = list(range(0, n_tokens - W + 1, self.window_stride))
        if starts[-1] + W < n_tokens:    # 尾部未覆盖时补一个贴着文末的窗口
            tail = n_tokens - W
            if starts[-1] != tail:
                starts.append(tail)
        return starts

    def iter_window_batches(self, text: str) -> Iterator[list[float]]:
        """生成器：每次 yield 一批（≤ window_batch_size 个）窗口 logit，按窗口在文中的顺序。"""
        full_ids = self.tokenizer.encode(text, add_special_tokens=False)   # 不加特殊 token、不截断
        # 2026-10-04 修正：GPT 系分词器（Qwen 等）没有 cls/sep token（tokenizer.cls_token_id /
        # sep_token_id 均为 None），原实现无条件拼接 [cls]+...+[sep] 会注入 None，
        # 导致 tokenizer.pad 抛 "type of None unknown"（v2.5 三个 Qwen 的 eval/calibrate 全因此
        # 失败；codebert 有 [CLS]/[SEP] 故行为不变）。
        head = [self.tokenizer.cls_token_id] if self.tokenizer.cls_token_id is not None else []
        tail = [self.tokenizer.sep_token_id] if self.tokenizer.sep_token_id is not None else []
        W = self.max_length - len(head) - len(tail)
        windows = [head + full_ids[s:s + W] + tail
                   for s in self.window_starts(len(full_ids))]
        for i in range(0, len(windows), self.window_batch_size):
            yield self._infer_batch(windows[i:i + self.window_batch_size])

    def predict_logit(self, text: str) -> float:
        """兼容接口：消费全部批次，返回 max(全部窗口 logit)；无早停。"""
        window_logits: list[float] = []
        for batch_logits in self.iter_window_batches(text):
            window_logits.extend(batch_logits)
        return float(max(window_logits))

    def predict_logits_batch(self, texts: list[str]) -> list[float]:
        """逐条调用 predict_logit 循环实现（训练/校准脚本用）。"""
        return [self.predict_logit(text) for text in texts]

    def _infer_batch(self, windows_ids: list[list[int]]) -> list[float]:
        """单批张量推理：输入为已含 [CLS]/[SEP] 的 token id 列表。"""
        batch = self.tokenizer.pad(
            {"input_ids": windows_ids},
            padding="max_length", max_length=self.max_length,
            return_tensors="pt").to(self.device)      # pad 会自动生成 attention_mask
        with torch.no_grad():
            logits = self.model(**batch).logits       # 形状 [批大小, 2]
        return [float(x) for x in logits[:, C.LABEL_MALICIOUS].tolist()]
