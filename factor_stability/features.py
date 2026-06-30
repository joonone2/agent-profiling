# =============================================================================
# features.py
# feature_log → 사용자×피처 집계 행렬 생성 + 기법별 전처리
#
# ★ 결측 처리 철학 변경:
#   평균 대체는 "정보 없음"을 "평범한 의견"으로 둔갑시켜 가짜 패턴을 만든다
#   (Documentary 63%가 동일값 → NMF/FA가 이를 진짜 구조로 오인할 위험).
#   그래서 결측을 미리 메우지 않고, 각 기법이 자기 방식대로 NaN을 직접 다룬다:
#     - NMF  → EM 기반 결측-허용 분해 (extractors.py에서 처리)
#     - FA   → 쌍별 결측 제거(pairwise deletion) 공분산 사용
#     - SHAP → XGBoost가 결측을 자체 처리 (표준화 생략, NaN 그대로 전달)
#
# ★ 옵션 A는 유지: 결측 비율이 MAX_MISSING_RATIO를 넘는 극단적 컬럼은
#   (정보가 너무 적어 EM/pairwise로도 신뢰하기 어려우므로) 여전히 제외.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import GENRES, MAX_MISSING_RATIO


# -----------------------------------------------------------------------------
# 피처 엔지니어링
# -----------------------------------------------------------------------------

def build_feature_table(feature_log: pd.DataFrame) -> pd.DataFrame:
    """
    feature_log(평점 + 장르 + item_pop) → 사용자 1명 = 1행 피처 테이블.
    ★ 결측은 여기서 채우지 않고 NaN으로 유지한다 (기법별로 다르게 처리됨).

    피처 목록:
      - 장르별 평균 평점 (결측 비율 높은 장르는 제외)
      - rating_mean / rating_std / rating_count
      - popularity_bias
    """
    rows = []
    for user_id, grp in feature_log.groupby("user_id"):
        row = {"user_id": user_id}

        for genre in GENRES:
            mask = grp["genres"].str.contains(genre, na=False)
            sub = grp.loc[mask, "rating"]
            row[f"genre_{genre}"] = sub.mean() if len(sub) > 0 else np.nan

        row["rating_mean"]  = grp["rating"].mean()
        row["rating_std"]   = grp["rating"].std()
        row["rating_count"] = len(grp)

        if "item_pop" in grp.columns and grp["item_pop"].nunique() > 1:
            corr = grp["rating"].corr(grp["item_pop"])
            row["popularity_bias"] = corr if not np.isnan(corr) else np.nan
        else:
            row["popularity_bias"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows).set_index("user_id")

    # 옵션 A: 결측 비율이 너무 높은(MAX_MISSING_RATIO 초과) 장르 컬럼 제외
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    missing_ratio = df[genre_cols].isnull().mean()
    dropped = missing_ratio[missing_ratio > MAX_MISSING_RATIO].index.tolist()

    if dropped:
        print(f"[features] ★ 옵션A: 결측비율 > {MAX_MISSING_RATIO:.0%} 인 "
              f"장르 컬럼 {len(dropped)}개 제외:")
        for c in dropped:
            print(f"    {c:25s} 결측비율={missing_ratio[c]:.1%}")
        df = df.drop(columns=dropped)
    else:
        print(f"[features] 옵션A: 결측비율 > {MAX_MISSING_RATIO:.0%} 인 컬럼 없음")

    n_missing_total = df.isnull().sum().sum()
    print(f"[features] 피처 테이블: {df.shape[0]} 유저 × {df.shape[1]} 피처 "
          f"(NaN {n_missing_total}개, 유지됨 — 기법별로 직접 처리)")
    print(f"[features] 피처 목록: {df.columns.tolist()}")
    return df


# -----------------------------------------------------------------------------
# 기법별 전처리 — ★ 셋 다 다른 방식
# -----------------------------------------------------------------------------

def preprocess_for_nmf(X: pd.DataFrame) -> tuple:
    """
    NMF용 전처리: MinMax(0~1) 스케일 + 결측 마스크 반환.
    ★ NaN을 메우지 않고 그대로 둔 채, "어디가 결측인지" 마스크를 같이 넘김.
       extractors.extract_nmf_em()이 이 마스크를 보고 EM으로 결측을 추정.

    Returns:
        (X_scaled: np.ndarray with NaN, mask: np.ndarray bool — True=관측됨)
    """
    mask = ~X.isnull().values  # True = 관측된 값, False = 결측

    # MinMax 스케일은 관측된 값만 기준으로 계산 (결측이 min/max를 왜곡 안 하게)
    scaler = MinMaxScaler()
    X_filled_for_fit = X.copy()
    col_means = X_filled_for_fit.mean()  # fit 전용 임시값 (최종 결과엔 영향 없음)
    X_filled_for_fit = X_filled_for_fit.fillna(col_means.fillna(0))
    scaler.fit(X_filled_for_fit.values)

    X_scaled = scaler.transform(X.fillna(0).values)  # 일단 0으로 두고 EM이 덮어씀
    X_scaled[~mask] = np.nan  # 결측 위치는 다시 NaN으로 표시 (EM이 채울 자리)

    return X_scaled, mask


def preprocess_for_fa(X: pd.DataFrame) -> pd.DataFrame:
    """
    FA용 전처리: 표준화하되 NaN은 유지.
    ★ 평균/표준편차는 관측된 값만으로 계산(pandas가 기본으로 그렇게 함).
       extractors.extract_fa()가 쌍별 결측 제거(pairwise deletion)로
       공분산을 직접 계산하므로, 여기서는 표준화된 DataFrame(NaN 포함)만 반환.

    Returns:
        표준화된 DataFrame (컬럼명 유지, NaN 포함)
    """
    means = X.mean()   # NaN 자동 무시하고 계산됨 (pandas 기본 동작)
    stds  = X.std()
    stds_safe = stds.replace(0, 1e-9)  # 분산 0 컬럼 방어
    X_std = (X - means) / stds_safe
    return X_std  # NaN 위치는 그대로 NaN


def preprocess_for_shap(X: pd.DataFrame) -> np.ndarray:
    """
    SHAP(XGBoost)용 전처리: 표준화 생략, NaN 그대로 전달.
    ★ XGBoost는 결측을 자체적으로 처리하는 트리 분기 로직이 있어서
       (어느 분기로 보낼지 학습 중 스스로 결정) 별도 결측 처리가 불필요.
       트리 기반 모델은 스케일에도 둔감해 표준화도 생략.
    """
    return X.values  # NaN 포함 원본 그대로 반환


def get_preprocessed(X: pd.DataFrame) -> dict:
    """
    세 기법 모두를 위한 전처리 결과를 한 번에 반환.
    ★ 반환 형태가 기법마다 다름 — extractors.py가 각자 맞게 처리.

    Returns:
        {
          'nmf'        : (X_scaled_with_nan, mask) 튜플,
          'fa'         : 표준화된 DataFrame (NaN 포함),
          'shap'       : np.ndarray (NaN 포함, 원본 스케일),
          'feature_names': list[str],
          'user_ids'     : np.ndarray,
        }
    """
    return {
        "nmf":           preprocess_for_nmf(X),
        "fa":            preprocess_for_fa(X),
        "shap":          preprocess_for_shap(X),
        "feature_names": X.columns.tolist(),
        "user_ids":      X.index.values,
    }