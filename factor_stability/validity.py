# =============================================================================
# validity.py
# 검증 ③ 외부 타당성 — "축이 실제 행동과 말이 되는가"
#
# ★ 핵심 원칙 (순환논리 방지):
#   검증 신호는 반드시 축 추출에 쓰이지 않은 변수에서 계산.
#   - popularity_bias가 피처로 들어갔으므로, 인기작 선호 검증 시
#     피처 테이블의 popularity_bias를 쓰지 않고 validation_log 원본에서 재계산.
#   - 유저 메타데이터(age/gender/occupation)는 피처 테이블에 없으므로 직접 사용 가능.
#
# 가설:
#   - 대중성 가설 축 ↔ 원본 로그 기반 인기작 선호 → 양의 상관 기대
#   - 작품성 가설 축 ↔ 감독 충성도(특정 감독 반복 시청 집중도) → 양의 상관 기대
#   - 모든 축 ↔ 유저 메타데이터(age/occupation/gender) → 패턴 탐색
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pointbiserialr
from config import VALIDITY_MIN_CORR


# -----------------------------------------------------------------------------
# 독립 신호 계산 (validation_log 원본에서 — 피처 테이블 사용 금지)
# -----------------------------------------------------------------------------

def compute_popularity_preference_raw(validation_log: pd.DataFrame) -> pd.Series:
    """
    인기작 선호도 (독립 신호 — 피처의 popularity_bias와 다른 계산).
    ★ 피처 테이블의 popularity_bias를 쓰지 않음 (순환논리 방지).

    방법: 유저별로 "인기 상위 20% 아이템에 준 평점 평균"을 계산.
    피처의 popularity_bias(상관 기반)와 측정 방식이 다르므로 독립 신호.
    """
    # 아이템별 인기도 (validation_log 내에서만 계산)
    item_pop = validation_log.groupby("item_id")["user_id"].count()
    threshold = item_pop.quantile(0.80)
    popular_items = item_pop[item_pop >= threshold].index

    result = {}
    for uid, grp in validation_log.groupby("user_id"):
        pop_ratings = grp[grp["item_id"].isin(popular_items)]["rating"]
        result[uid] = pop_ratings.mean() if len(pop_ratings) > 0 else np.nan

    return pd.Series(result, name="pop_pref_raw")


def compute_director_loyalty(validation_log: pd.DataFrame) -> pd.Series:
    """
    작품성 가설 검증용: 감독 충성도 (특정 감독 반복 시청 집중도).
    MovieLens는 감독 정보가 없으므로, 대리 지표로 "특정 장르 집중도" 사용.
    특정 장르(Drama, Sci-Fi)에 평점이 집중될수록 높은 값.

    ★ 피처 테이블의 장르 평균 평점(genre_Drama 등)과 다름:
      여기서는 "평점 비율(집중도)"을 사용 — 피처는 "평균값".
    """
    drama_scifi = {"Drama", "Sci-Fi"}
    result = {}
    for uid, grp in validation_log.groupby("user_id"):
        total = len(grp)
        if total == 0:
            result[uid] = np.nan
            continue
        # 장르 컬럼이 파이프 구분이므로 Drama 또는 Sci-Fi 포함 비율
        mask = grp["genres"].str.contains("Drama|Sci-Fi", na=False)
        result[uid] = mask.sum() / total

    return pd.Series(result, name="director_loyalty_proxy")


def compute_meta_signals(validation_log: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    유저 메타데이터 신호 (age, occupation, gender).
    이 정보는 피처 테이블에 포함되지 않으므로 순환논리 없음.
    gender는 숫자 인코딩(M=1, F=0).
    """
    meta = users[["user_id", "age", "occupation", "gender"]].copy()
    meta["gender_num"] = (meta["gender"] == "M").astype(int)
    return meta.set_index("user_id")


# -----------------------------------------------------------------------------
# 상관 계산
# -----------------------------------------------------------------------------

def correlate_axis_signal(
    axis_scores: np.ndarray,
    signal: pd.Series,
    user_ids: np.ndarray,
    signal_name: str,
    method_name: str,
    axis_idx: int,
) -> dict:
    """
    단일 축 점수 vs 단일 외부 신호의 Spearman 상관.
    공통 유저에 대해서만 계산.
    """
    common_users = np.intersect1d(user_ids, signal.dropna().index)
    if len(common_users) < 10:
        return {
            "method": method_name, "axis_idx": axis_idx,
            "signal": signal_name, "n_users": len(common_users),
            "spearman_r": None, "p_value": None, "significant": False,
        }

    uid_to_row = {uid: i for i, uid in enumerate(user_ids)}
    rows = np.array([uid_to_row[u] for u in common_users])
    scores = axis_scores[rows]
    sig_vals = signal.loc[common_users].values

    corr, p = spearmanr(scores, sig_vals)
    significant = (p < 0.05) and (abs(corr) >= VALIDITY_MIN_CORR)

    return {
        "method":      method_name,
        "axis_idx":    axis_idx,
        "signal":      signal_name,
        "n_users":     len(common_users),
        "spearman_r":  round(float(corr), 4) if not np.isnan(corr) else None,
        "p_value":     round(float(p), 4) if not np.isnan(p) else None,
        "significant": significant,
    }


# -----------------------------------------------------------------------------
# 통합 진입점
# -----------------------------------------------------------------------------

def validate_all(
    results: dict,
    user_ids: np.ndarray,
    validation_log: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """
    세 기법의 외부 타당성 검증.

    Args:
        results      : extractors.extract_all() 반환값
        user_ids     : features.get_preprocessed()['user_ids']
        validation_log: data_loader의 validation_log (축 추출에 안 쓰인 것)
        users        : 유저 메타데이터 DataFrame

    Returns:
        external_validity.csv에 저장할 DataFrame
    """
    print("\n[validity] 외부 신호 계산 중 (validation_log 원본 기반)...")
    pop_pref   = compute_popularity_preference_raw(validation_log)
    dir_loyal  = compute_director_loyalty(validation_log)
    meta       = compute_meta_signals(validation_log, users)

    rows = []
    for method, res in results.items():
        print(f"\n[validity] {method} 외부 타당성:")
        k = res.user_scores.shape[1]

        for k_ in range(k):
            axis_scores = res.user_scores[:, k_]

            # 신호 1: 인기작 선호 (대중성 가설 검증)
            r = correlate_axis_signal(
                axis_scores, pop_pref, user_ids,
                "pop_pref_raw", method, k_
            )
            rows.append(r)
            print(f"  축{k_} ↔ pop_pref_raw   : r={r['spearman_r']}, "
                  f"p={r['p_value']}, sig={r['significant']}")

            # 신호 2: 작품성 집중도 (작품성 가설 검증)
            r = correlate_axis_signal(
                axis_scores, dir_loyal, user_ids,
                "director_loyalty_proxy", method, k_
            )
            rows.append(r)
            print(f"  축{k_} ↔ dir_loyalty     : r={r['spearman_r']}, "
                  f"p={r['p_value']}, sig={r['significant']}")

            # 신호 3~5: 메타데이터 (패턴 탐색)
            for col in ["age", "occupation", "gender_num"]:
                if col in meta.columns:
                    r = correlate_axis_signal(
                        axis_scores, meta[col], user_ids,
                        col, method, k_
                    )
                    rows.append(r)

    df = pd.DataFrame(rows)

    # 요약 출력
    sig_rows = df[df["significant"] == True]
    print(f"\n[validity] 유의미한 상관(|r|≥{VALIDITY_MIN_CORR}, p<0.05): "
          f"{len(sig_rows)} / {len(df)}개")
    if len(sig_rows) > 0:
        print(sig_rows[["method","axis_idx","signal","spearman_r"]].to_string(index=False))

    return df