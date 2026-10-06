"""门控路由模块：校准概率 → ALLOW / BLOCK / LLM_REVIEW。

边界归属（禁止改为 <= / >=）：严格小于放行、严格大于拦截、含边界走云端 LLM。
"""
from calibguard.schemas import GateDecision


def route(probability: float, allow_threshold: float, block_threshold: float) -> GateDecision:
    """按阈值把校准概率路由到三个决策之一。"""
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"概率必须在 [0, 1] 内，当前 {probability}")
    if probability < allow_threshold:
        return GateDecision.ALLOW
    if probability > block_threshold:
        return GateDecision.BLOCK
    # 其余（含恰好等于两阈值的情况）走云端 LLM
    return GateDecision.LLM_REVIEW
