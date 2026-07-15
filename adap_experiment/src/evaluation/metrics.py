"""
평가 지표: MAE, RMSE, Recall@K, NDCG@K
"""
import math


def mae(preds: list[float], actuals: list[float]) -> float:
    """Mean Absolute Error."""
    if not preds:
        return float("nan")
    return sum(abs(p - a) for p, a in zip(preds, actuals)) / len(preds)


def rmse(preds: list[float], actuals: list[float]) -> float:
    """Root Mean Squared Error."""
    if not preds:
        return float("nan")
    mse = sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(preds)
    return math.sqrt(mse)


def recall_at_k(ranked_ids: list[int], positive_id: int, k: int = 10) -> int:
    """positive_id가 ranked_ids 상위 k개 안에 있으면 1, 아니면 0."""
    return 1 if positive_id in ranked_ids[:k] else 0


def ndcg_at_k(ranked_ids: list[int], positive_id: int, k: int = 10) -> float:
    """
    positive_id의 순위(1-indexed)가 rank일 때:
    rank <= k이면 1/log2(rank+1), 아니면 0.
    """
    try:
        rank = ranked_ids.index(positive_id) + 1  # 1-indexed
    except ValueError:
        return 0.0
    if rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0
