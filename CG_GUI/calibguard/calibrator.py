"""概率校准模块：Platt Scaling。

公式（禁止改写）：P(y=1 | s) = 1 / (1 + exp(A * s + B))
与温度缩放的关系（v1.0 不实现温度缩放）：P = σ(s/T) 等价于 A = -1/T, B = 0 的特例。
"""
import datetime
import json
import math

from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt 校准器：logit → 校准概率；支持拟合与参数存取。"""

    def __init__(self, A: float = 0.0, B: float = 0.0, method: str = "platt",
                 n_samples: int = 0):
        self.A = A
        self.B = B
        self.method = method
        self.n_samples = n_samples

    def fit(self, logits: list[float], labels: list[int]) -> None:
        """在（logit, label）校准集上拟合 A、B。"""
        if len(logits) != len(labels) or len(logits) == 0:
            raise ValueError("logits 与 labels 长度必须相等且非空")
        try:
            lr = LogisticRegression(C=1e10, solver="lbfgs")
            lr.fit([[s] for s in logits], labels)
        except ValueError as exc:
            # sklearn 在标签只有单一类别时抛 ValueError
            raise RuntimeError("校准集必须同时包含良性与恶意样本") from exc
        # sklearn: P = 1/(1+exp(-(w*s+b)))，与目标公式 1/(1+exp(A*s+B)) 对比得：
        self.A = -float(lr.coef_[0][0])
        self.B = -float(lr.intercept_[0])
        self.n_samples = len(labels)

    def predict_proba(self, logit: float) -> float:
        """logit → 校准概率（含溢出保护）。"""
        z = self.A * logit + self.B
        if z > 700:
            return 1e-16        # p 趋近 0
        if z < -700:
            return 1.0 - 1e-16  # p 趋近 1
        return 1.0 / (1.0 + math.exp(z))

    def save(self, path: str) -> None:
        """写 JSON 参数文件（键名固定）。"""
        payload = {
            "A": self.A,
            "B": self.B,
            "method": self.method,
            "n_samples": self.n_samples,
            "fit_at": datetime.datetime.now().astimezone().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "PlattCalibrator":
        """读 JSON 还原；文件不存在抛 FileNotFoundError，内容损坏抛 ValueError。"""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            calibrator = cls(
                A=float(payload["A"]),
                B=float(payload["B"]),
                method=str(payload.get("method", "platt")),
                n_samples=int(payload.get("n_samples", 0)),
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"校准参数文件损坏: {path}") from exc
        return calibrator

    @classmethod
    def identity(cls) -> "PlattCalibrator":
        """跳过校准时使用：A=-1, B=0（predict_proba 即 sigmoid）。"""
        return cls(A=-1.0, B=0.0, method="none", n_samples=0)
