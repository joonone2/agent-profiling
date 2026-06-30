# =============================================================================
# extractors.py
# NMF / FA / SHAP 세 기법을 동일 인터페이스로 래핑.
#
# ★ 결측 처리 — 세 기법이 각자 다른 방식으로 NaN을 "직접" 다룬다
#   (평균으로 미리 메우지 않음 — 메우면 가짜 분산 패턴이 생기는 문제 때문):
#
#   - NMF  : EM 기반 결측-허용 분해.
#            관측된 칸만 재구성 오차에 반영, 결측 칸은 W×H로 반복 추정.
#   - FA   : 쌍별 결측 제거(pairwise deletion) 공분산.
#            두 피처의 상관은 "그 둘을 모두 관측한 사람들"로만 계산.
#   - SHAP : XGBoost가 결측을 자체 처리 (트리 분기에서 결측 방향을 학습).
#            별도 결측 처리 코드 불필요 — NaN을 그대로 모델에 전달.
#
# 모든 extractor는 ExtractResult 객체를 반환.
# stability.py는 result.type 필드로만 분기 → 기법 종류를 몰라도 동작.
# =============================================================================

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from factor_analyzer import FactorAnalyzer
import xgboost as xgb
import shap

from config import (
    K, NMF_INIT, NMF_MAX_ITER, FA_ROTATION,
    NMF_USE_EM_FOR_MISSING, NMF_EM_MAX_ITER, NMF_EM_TOL,
    SHAP_N_ESTIMATORS, SHAP_MAX_DEPTH, SHAP_LEARNING_RATE,
    BASE_SEED,
)


# -----------------------------------------------------------------------------
# 결과 객체
# -----------------------------------------------------------------------------

@dataclass
class ExtractResult:
    """
    모든 extractor의 공통 반환 타입.

    type:
      'matrix'     → NMF/FA: loadings(K×피처)가 유효. stability는 Tucker congruence.
      'importance' → SHAP: importances(피처 벡터)가 유효. stability는 Spearman.

    user_scores  : 유저×K 행렬 (외부 타당성용).
    explained_variance : 축별 설명분산 (노이즈 축 검증용). SHAP은 None.
    model_rmse   : SHAP 전용.
    em_iterations: NMF 전용. EM이 수렴까지 돈 횟수 (진단용).
    """
    method:              str
    type:                str
    loadings:            Optional[np.ndarray]
    importances:         Optional[np.ndarray]
    user_scores:         np.ndarray
    feature_names:       list
    explained_variance:  Optional[np.ndarray]
    model_rmse:          Optional[float] = None
    em_iterations:       Optional[int] = None


# -----------------------------------------------------------------------------
# NMF — EM 기반 결측-허용 분해
# -----------------------------------------------------------------------------

def extract_nmf(
    X_nmf_tuple: tuple,
    feature_names: list,
    k: int = K,
    seed: int = BASE_SEED,
) -> ExtractResult:
    """
    NMF로 K개 행동 축 추출. 결측이 있으면 EM 방식으로 처리.

    Args:
        X_nmf_tuple: (X: np.ndarray with NaN at missing, mask: bool array True=관측)
                     features.preprocess_for_nmf()의 반환값

    EM 절차:
      1) 결측 위치를 임시값(컬럼 평균)으로 초기화
      2) NMF(W, H) 학습
      3) W×H로 재구성한 값 중 "결측 위치만" 갱신 (관측값은 항상 원본 유지)
      4) 결측 위치 추정값의 변화량이 NMF_EM_TOL 미만이 될 때까지 2~3 반복
    """
    X_raw, mask = X_nmf_tuple
    n_missing = (~mask).sum()

    if not NMF_USE_EM_FOR_MISSING or n_missing == 0:
        # 결측이 없거나 EM을 끈 경우 — 표준 NMF
        X_filled = np.nan_to_num(X_raw, nan=0.0)
        model = NMF(n_components=k, init=NMF_INIT, random_state=seed, max_iter=NMF_MAX_ITER)
        W = model.fit_transform(X_filled)
        H = model.components_
        em_iters = 0
    else:
        # ----- EM 초기화: 결측을 컬럼 평균(관측값만으로)으로 채움 -----
        X_work = X_raw.copy()
        col_means = np.nanmean(np.where(mask, X_raw, np.nan), axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        for j in range(X_work.shape[1]):
            X_work[~mask[:, j], j] = col_means[j]

        W, H = None, None
        prev_missing_vals = X_work[~mask].copy()
        em_iters = 0

        for it in range(NMF_EM_MAX_ITER):
            model = NMF(n_components=k, init=NMF_INIT, random_state=seed, max_iter=NMF_MAX_ITER)
            W = model.fit_transform(np.clip(X_work, 0, None))  # 비음수 보장
            H = model.components_

            recon = W @ H
            # 결측 위치만 재구성값으로 갱신 (관측값은 절대 안 건드림)
            X_work[~mask] = np.clip(recon[~mask], 0, None)

            curr_missing_vals = X_work[~mask]
            delta = np.abs(curr_missing_vals - prev_missing_vals).mean()
            em_iters = it + 1
            if delta < NMF_EM_TOL:
                break
            prev_missing_vals = curr_missing_vals.copy()

        print(f"  [NMF/EM] 수렴: {em_iters}회 반복, 결측 {n_missing}개 추정 완료")

    # 설명분산 (관측된 칸 기준으로만 계산 — 결측 추정값이 분산을 부풀리지 않게)
    X_observed_only = np.where(mask, X_raw, 0)
    total_var = (X_observed_only ** 2).sum()
    exp_var = np.array([
        ((np.outer(W[:, k_], H[k_]) * mask) ** 2).sum() / (total_var + 1e-9)
        for k_ in range(k)
    ])

    return ExtractResult(
        method="NMF",
        type="matrix",
        loadings=H,
        importances=None,
        user_scores=W,
        feature_names=feature_names,
        explained_variance=exp_var,
        em_iterations=em_iters,
    )


# -----------------------------------------------------------------------------
# Factor Analysis — 쌍별 결측 제거(pairwise deletion) 공분산
# -----------------------------------------------------------------------------

def extract_fa(
    X_fa: pd.DataFrame,
    feature_names: list,
    k: int = K,
) -> ExtractResult:
    """
    Factor Analysis로 K개 행동 축 추출.
    ★ 입력은 표준화된 DataFrame(NaN 포함). 공분산은 쌍별 결측 제거로 계산:
       피처 A·B의 상관은 "A와 B를 둘 다 관측한 사람들"만으로 구함.
       FactorAnalyzer는 공분산 행렬을 직접 받을 수 없으므로,
       쌍별 결측 제거로 만든 상관행렬을 고유분해해 loading을 직접 계산한다.
    """
    # 1) 쌍별 결측 제거 상관행렬 (pandas.corr()이 기본으로 이렇게 동작)
    corr_matrix = X_fa.corr(min_periods=30)  # 표본 너무 적은 쌍은 NaN 처리
    corr_matrix = corr_matrix.fillna(0.0)    # 계산 불가한 쌍은 무상관으로 간주
    corr_values = corr_matrix.values.copy()  # 쓰기 가능한 복사본
    np.fill_diagonal(corr_values, 1.0)

    # 2) FactorAnalyzer는 "데이터"를 받지만, 내부적으로 상관행렬을 다시 계산함.
    #    결측이 있으면 fit()이 실패하므로, 결측 없는 행만으로 1차 적합 후
    #    loading 추정은 위에서 만든 쌍별 상관행렬 기반 고유분해로 직접 수행.
    eigvals, eigvecs = np.linalg.eigh(corr_values)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order][:k]
    eigvecs = eigvecs[:, order][:, :k]

    eigvals_clipped = np.clip(eigvals, 0, None)
    loadings = eigvecs * np.sqrt(eigvals_clipped)   # (n_features, k)
    loadings = loadings.T                            # (k, n_features)

    # Varimax 회전 적용 (factor_analyzer의 회전 함수 재사용)
    try:
        from factor_analyzer.rotator import Rotator
        rotator = Rotator(method=FA_ROTATION)
        loadings = rotator.fit_transform(loadings.T).T
    except Exception as e:
        print(f"  [FA] Varimax 회전 실패, 무회전 loading 사용: {e}")

    # 3) 유저 점수: 결측 위치는 0(=평균 영향 없음)으로 두고 loading과 내적
    X_for_scores = X_fa.fillna(0.0).values
    # 회귀 기반 근사 점수 (loadings^T @ loadings의 역행렬을 이용한 표준 공식 단순화)
    user_scores = X_for_scores @ loadings.T @ np.linalg.pinv(loadings @ loadings.T)

    # 설명분산: 고유값 비율
    total_eigval_sum = np.linalg.eigvalsh(corr_values).sum()
    exp_var = eigvals_clipped / (total_eigval_sum + 1e-9)

    return ExtractResult(
        method="FA",
        type="matrix",
        loadings=loadings,
        importances=None,
        user_scores=user_scores,
        feature_names=feature_names,
        explained_variance=exp_var,
    )


# -----------------------------------------------------------------------------
# SHAP — XGBoost가 결측을 자체 처리
# -----------------------------------------------------------------------------

def extract_shap(
    X_shap: np.ndarray,
    feature_log: pd.DataFrame,
    feature_names: list,
    seed: int = BASE_SEED,
) -> ExtractResult:
    """
    XGBoost 평점 예측 모델 기반 SHAP global importance 산출.
    ★ X_shap은 NaN을 포함한 원본 스케일 배열. XGBoost가 결측을 트리 분기에서
      자체적으로 처리하므로 별도 대체(imputation)를 하지 않음.

    한계: 예측 태스크에 종속, loading 행렬 아님, 축은 수동 묶기 필요.
    """
    unique_users = feature_log["user_id"].unique()
    user_idx_map = {uid: i for i, uid in enumerate(unique_users)}

    user_feat_df = pd.DataFrame(X_shap, index=unique_users, columns=feature_names)
    train_df = feature_log[["user_id", "rating"]].copy()
    train_df = train_df[train_df["user_id"].isin(user_idx_map)]
    X_train = user_feat_df.loc[train_df["user_id"]].values  # NaN 포함 그대로
    y_train = train_df["rating"].values

    # XGBoost는 NaN을 missing 값으로 인식해 트리 분기에서 자동 처리
    model = xgb.XGBRegressor(
        n_estimators=SHAP_N_ESTIMATORS,
        max_depth=SHAP_MAX_DEPTH,
        learning_rate=SHAP_LEARNING_RATE,
        random_state=seed,
        verbosity=0,
        missing=np.nan,   # 명시적으로 NaN을 결측으로 인식하도록 지정
    )
    model.fit(X_train, y_train)
    rmse = float(np.sqrt(np.mean((model.predict(X_train) - y_train) ** 2)))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    importances = np.mean(np.abs(shap_values), axis=0)

    n_feat = len(feature_names)
    sorted_idx = np.argsort(importances)[::-1]
    chunk = max(1, n_feat // K)
    user_scores = np.zeros((X_shap.shape[0], K))
    for k_ in range(K):
        feat_idx = sorted_idx[k_ * chunk: (k_ + 1) * chunk]
        user_scores[:, k_] = np.nanmean(np.abs(shap_values[:, feat_idx]), axis=1)

    return ExtractResult(
        method="SHAP",
        type="importance",
        loadings=None,
        importances=importances,
        user_scores=user_scores,
        feature_names=feature_names,
        explained_variance=None,
        model_rmse=rmse,
    )


# -----------------------------------------------------------------------------
# 통합 진입점
# -----------------------------------------------------------------------------

def extract_all(preprocessed: dict, feature_log: pd.DataFrame) -> dict:
    """
    세 기법을 모두 실행해 결과 dict 반환.
    """
    fn = preprocessed["feature_names"]

    print("[extractors] NMF 추출 중 (EM 결측처리)...")
    res_nmf = extract_nmf(preprocessed["nmf"], fn)

    print("[extractors] FA 추출 중 (쌍별 결측제거)...")
    res_fa = extract_fa(preprocessed["fa"], fn)

    print("[extractors] SHAP 추출 중 (XGBoost 자체 결측처리, 가장 느림)...")
    res_shap = extract_shap(preprocessed["shap"], feature_log, fn)

    print(f"[extractors] 완료. SHAP 모델 RMSE: {res_shap.model_rmse:.4f}, "
          f"NMF EM 반복: {res_nmf.em_iterations}회")
    return {"NMF": res_nmf, "FA": res_fa, "SHAP": res_shap}