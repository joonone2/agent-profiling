"""
유저별 Factor 가중치 벡터(wᵢₖ) 계산
"""
import os
import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def compute_user_weights(
    user_feature_matrix: pd.DataFrame,
    loading_matrix: pd.DataFrame,
    method: str | None = None,
) -> pd.DataFrame:
    """
    각 유저 i, 각 Factor k에 대해:
      raw_wik = cosine_similarity(user_feature_matrix.loc[i], loading_matrix["factor_k"])

    ★ 수정 사항 (기존: 음수 클리핑 → 균등분배 fallback):
      NMF loading은 비음수라 문제가 없었지만, FA loading은 대부분의 피처에서
      음수가 자연스럽게 나타난다(예: "이 장르는 비선호"라는 방향 정보).
      유저 피처(장르별 평균 평점)는 항상 0 이상이므로, 코사인 유사도를
      "음수는 0으로 클리핑"하면 FA의 음수 위주 factor(factor_1, factor_2 등)는
      거의 모든 유저에게서 0이 되어버리고, 결과적으로 유일하게 양수인 factor
      하나가 정규화 후 항상 1.0을 독식하는 문제가 발생했다
      (6,040명 전원이 [1.0, 0.0, 0.0]로 동일해짐 — 유저 개인차가 완전히 사라짐).

      해결: 코사인 유사도([-1, 1] 범위)를 클리핑하지 않고
      (sim + 1) / 2 로 선형 변환하여 [0, 1] 범위로 매핑한다.
      이렇게 하면 "이 factor와 얼마나 정렬되는지(양의 유사도)"뿐 아니라
      "이 factor와 얼마나 반대되는지(음의 유사도)"도 0에 가까운 값으로 보존되어,
      유저별 차이가 사라지지 않는다. NMF처럼 원래 유사도가 대부분 양수인
      경우에도 이 변환은 상대적 크기 순서를 그대로 보존하므로 문제가 없다.

    Args:
        method: "nmf" 또는 "fa" — 저장 파일명 구분용 (None이면 method 접미사 없이 저장)

    반환: pd.DataFrame, index=userId, columns=factor_0..factor_{k-1}, 각 행의 합=1
    """
    k = loading_matrix.shape[1]
    factor_cols = list(loading_matrix.columns)
    user_ids = user_feature_matrix.index

    # NaN 방어: user_feature_matrix나 loading_matrix에 NaN이 있으면
    # cosine_similarity가 NaN을 전파하므로 0으로 대체
    user_matrix = np.nan_to_num(user_feature_matrix.values, nan=0.0)   # (n_users, 19)
    loading_vectors = np.nan_to_num(loading_matrix.values, nan=0.0)    # (19, k)

    # cosine_similarity: (n_users, k), 값 범위 [-1, 1]
    sim_matrix = cosine_similarity(user_matrix, loading_vectors.T)

    # NaN 방어 (cosine_similarity가 zero-norm 벡터에서 NaN을 줄 수 있음)
    sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)

    # ── 변경된 부분: 음수 클리핑 대신 [-1,1] → [0,1] 선형 변환 ──
    # sim=+1(완전 정렬)  -> 1.0
    # sim= 0(무관)       -> 0.5
    # sim=-1(완전 반대)  -> 0.0
    sim_matrix = (sim_matrix + 1.0) / 2.0

    # 유저별 정규화 (합이 1)
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    # 위 변환 덕분에 모든 값이 0이 되는 경우는 이론상 거의 발생하지 않지만
    # (모든 factor와 완전히 반대(-1)로 정렬되는 극단적 경우에만 발생),
    # 안전을 위해 fallback은 그대로 유지한다.
    zero_mask = (row_sums.flatten() == 0)
    n_uniform = zero_mask.sum()
    if n_uniform > 0:
        logger.info(
            "compute_user_weights: %d/%d users have all-zero similarity after transform "
            "(method=%s). Applying uniform fallback (1/%d).",
            n_uniform, len(user_ids), method or "unknown", k,
        )
    sim_matrix[zero_mask] = 1.0 / k
    row_sums[zero_mask] = 1.0
    weights = sim_matrix / row_sums

    result = pd.DataFrame(weights, index=user_ids, columns=factor_cols)

    # 진단용 로그: 가중치가 특정 factor 하나로 완전히 쏠려 있는지(예: 이전 버그 재발) 확인
    dominant_share = result.max(axis=1)
    n_degenerate = (dominant_share >= 0.999).sum()
    if n_degenerate > 0:
        logger.warning(
            "compute_user_weights: %d/%d users have a single factor with weight >= 0.999 "
            "(method=%s). Check loading_matrix for scale/sign issues if this seems unexpected.",
            n_degenerate, len(user_ids), method or "unknown",
        )

    # 저장 (method별 파일명 구분)
    factors_dir = os.path.join(OUTPUT_DIR, "factors")
    os.makedirs(factors_dir, exist_ok=True)
    suffix = f"_{method}" if method else ""
    result.to_csv(os.path.join(factors_dir, f"user_weights{suffix}.csv"))

    logger.info("User weights computed for %d users (method=%s). Shape: %s",
                len(result), method or "default", result.shape)
    return result