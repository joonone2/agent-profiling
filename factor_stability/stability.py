# =============================================================================
# stability.py
# split-half 재현성 측정.
#
# ★ 기법마다 측정 대상이 다름 (직접 비교 불가 — 같은 표에 "무엇의 안정성"인지 명시):
#   type='matrix'     (NMF/FA) : loading 패턴의 Tucker congruence
#   type='importance' (SHAP)   : 변수 중요도 순위의 Spearman 상관
#
# 공통 함수:
#   - 헝가리안 매칭: 축 순서가 매번 다르므로 최대 일치 조합으로 먼저 짝지음
#   - 시드 N회 반복 → 평균 ± 표준편차
# =============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from dataclasses import dataclass
from typing import Callable

from config import N_SEEDS, K, BASE_SEED


# -----------------------------------------------------------------------------
# 기본 측정 함수
# -----------------------------------------------------------------------------

def tucker_congruence(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 loading 벡터의 Tucker's congruence coefficient.
    부호 무관 (절댓값) — FA/PCA는 부호가 임의로 뒤집힐 수 있음.
    범위: 0 ~ 1. 0.95↑ 거의 동일, 0.85↑ 공정한 유사, 0.85↓ 의심.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(abs(np.dot(a, b)) / denom)


def hungarian_match_congruence(H1: np.ndarray, H2: np.ndarray) -> float:
    """
    두 loading 행렬(K×피처)을 헝가리안 알고리즘으로 최적 매칭 후
    매칭된 축들의 Tucker congruence 평균 반환.

    축 순서·부호가 매번 다르므로 직접 비교 불가 → 먼저 짝지어야 함.
    """
    k = H1.shape[0]
    # 비용 행렬: (K, K) — [i,j] = 1 - tucker(H1[i], H2[j]) (최소화)
    cost = np.array([
        [1.0 - tucker_congruence(H1[i], H2[j]) for j in range(k)]
        for i in range(k)
    ])
    row_idx, col_idx = linear_sum_assignment(cost)
    matched_scores = [tucker_congruence(H1[r], H2[c]) for r, c in zip(row_idx, col_idx)]
    return float(np.mean(matched_scores))


def spearman_importance(imp1: np.ndarray, imp2: np.ndarray) -> float:
    """
    두 변수 중요도 벡터의 Spearman 순위상관.
    SHAP의 stability 측정에 사용.
    """
    corr, _ = spearmanr(imp1, imp2)
    return float(corr) if not np.isnan(corr) else 0.0


# -----------------------------------------------------------------------------
# split-half 재현성
# -----------------------------------------------------------------------------

@dataclass
class StabilityResult:
    """
    단일 기법의 안정성 측정 결과.
    """
    method:      str
    metric_name: str          # 'Tucker congruence' | 'Spearman rank corr'
    mean:        float
    std:         float
    scores:      list         # 시드별 raw 점수 (분포 확인용)
    type:        str          # 'matrix' | 'importance'
    auto_axis:   bool         # 축이 자동으로 나오는가 (SHAP은 False)
    model_rmse_a: float = 0.0 # SHAP 전용 (A 절반 모델)
    model_rmse_b: float = 0.0 # SHAP 전용 (B 절반 모델)


def _single_split_matrix(
    X,
    extractor_fn: Callable,
    seed: int,
) -> float:
    """
    단일 시드로 split-half → Tucker congruence 반환.
    extractor_fn: X_sub → ExtractResult (matrix형)

    ★ X의 타입에 따라 분할 방식이 다름:
      - tuple (X_array, mask)  → NMF. 둘 다 같은 행으로 분할.
      - pd.DataFrame           → FA. 행(유저) 기준으로 분할.
      - np.ndarray             → 일반 배열. 행 기준 분할.
    """
    if isinstance(X, tuple):
        X_arr, mask = X
        n = X_arr.shape[0]
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        half = n // 2
        res_a = extractor_fn((X_arr[idx[:half]], mask[idx[:half]]))
        res_b = extractor_fn((X_arr[idx[half:]], mask[idx[half:]]))
    elif isinstance(X, pd.DataFrame):
        n = len(X)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        half = n // 2
        res_a = extractor_fn(X.iloc[idx[:half]])
        res_b = extractor_fn(X.iloc[idx[half:]])
    else:
        n = len(X)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        half = n // 2
        res_a = extractor_fn(X[idx[:half]])
        res_b = extractor_fn(X[idx[half:]])

    return hungarian_match_congruence(res_a.loadings, res_b.loadings)


def _single_split_importance(
    X: np.ndarray,
    feature_log,
    extractor_fn: Callable,
    seed: int,
) -> tuple:
    """
    단일 시드로 split-half → (Spearman corr, rmse_a, rmse_b) 반환.
    SHAP은 feature_log도 절반으로 나눠야 모델을 독립적으로 학습 가능.
    """
    import pandas as pd
    rng = np.random.RandomState(seed)

    unique_users = feature_log["user_id"].unique()
    idx = rng.permutation(len(unique_users))
    half = len(unique_users) // 2
    users_a = unique_users[idx[:half]]
    users_b = unique_users[idx[half:]]

    fl_a = feature_log[feature_log["user_id"].isin(users_a)]
    fl_b = feature_log[feature_log["user_id"].isin(users_b)]

    # X에서 해당 유저 행만 추출 (user_ids와 X가 같은 순서라 가정)
    user_ids = feature_log["user_id"].unique()
    uid_to_row = {uid: i for i, uid in enumerate(user_ids)}
    rows_a = np.array([uid_to_row[u] for u in users_a if u in uid_to_row])
    rows_b = np.array([uid_to_row[u] for u in users_b if u in uid_to_row])

    res_a = extractor_fn(X[rows_a], fl_a, seed=seed)
    res_b = extractor_fn(X[rows_b], fl_b, seed=seed + 1)

    corr = spearman_importance(res_a.importances, res_b.importances)
    return corr, res_a.model_rmse, res_b.model_rmse


def measure_stability_matrix(
    X: np.ndarray,
    extractor_fn: Callable,
    method_name: str,
    n_seeds: int = N_SEEDS,
) -> StabilityResult:
    """
    NMF / FA 용 안정성 측정.
    extractor_fn: (X_sub: np.ndarray) → ExtractResult
    """
    scores = []
    for s in range(n_seeds):
        seed = BASE_SEED + s
        score = _single_split_matrix(X, extractor_fn, seed)
        scores.append(score)
        print(f"  [stability/{method_name}] seed={seed}: Tucker={score:.4f}")

    return StabilityResult(
        method=method_name,
        metric_name="Tucker congruence",
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        scores=scores,
        type="matrix",
        auto_axis=True,
    )


def measure_stability_importance(
    X: np.ndarray,
    feature_log,
    extractor_fn: Callable,
    method_name: str = "SHAP",
    n_seeds: int = N_SEEDS,
) -> StabilityResult:
    """
    SHAP 용 안정성 측정.
    extractor_fn: (X_sub, fl_sub, seed) → ExtractResult
    ★ SHAP은 모델이 끼어드므로 RMSE도 함께 기록 (공정성 확인용)
    """
    scores, rmse_a_list, rmse_b_list = [], [], []
    for s in range(n_seeds):
        seed = BASE_SEED + s
        corr, rmse_a, rmse_b = _single_split_importance(X, feature_log, extractor_fn, seed)
        scores.append(corr)
        rmse_a_list.append(rmse_a)
        rmse_b_list.append(rmse_b)
        print(f"  [stability/SHAP] seed={seed}: Spearman={corr:.4f}, "
              f"RMSE_A={rmse_a:.4f}, RMSE_B={rmse_b:.4f}")

    return StabilityResult(
        method=method_name,
        metric_name="Spearman rank corr",
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        scores=scores,
        type="importance",
        auto_axis=False,   # ★ SHAP은 축이 자동으로 안 나옴
        model_rmse_a=float(np.mean(rmse_a_list)),
        model_rmse_b=float(np.mean(rmse_b_list)),
    )