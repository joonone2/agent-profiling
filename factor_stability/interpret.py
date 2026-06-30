# =============================================================================
# interpret.py
# 검증 ② 해석가능성 + 검증 ④ 노이즈 축 거르기
#
# ① 정량: 축별 loading 희소도(지니계수) — 높을수록 소수 피처 집중 = 해석 쉬움
# ② 정성: 축별 상위 N개 피처 + loading 값 → 사람이 이름 붙일 수 있나
# ③ 노이즈: 설명분산이 MIN_EXPLAINED_VAR 미만인 축에 플래그
# =============================================================================

import numpy as np
import pandas as pd
from config import MIN_EXPLAINED_VAR, K


# -----------------------------------------------------------------------------
# 희소도 (Gini coefficient)
# -----------------------------------------------------------------------------

def gini(v: np.ndarray) -> float:
    """
    벡터의 지니계수 (0=균등분포, 1=한 항목에 집중).
    loading 절댓값에 적용 → 높을수록 소수 피처에 집중 = 해석 쉬움.
    """
    v = np.abs(v)
    if v.sum() < 1e-9:
        return 0.0
    v_sorted = np.sort(v)
    n = len(v_sorted)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v_sorted) / (n * v_sorted.sum())) - (n + 1) / n)


# -----------------------------------------------------------------------------
# 해석가능성 분석
# -----------------------------------------------------------------------------

def interpret_matrix(
    loadings: np.ndarray,
    feature_names: list,
    method: str,
    explained_variance=None,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    NMF / FA: loading 행렬(K×피처)에서 해석가능성 정보 추출.

    반환 DataFrame 컬럼:
      method, axis_idx, top_features(상위 피처:loading 쌍),
      gini(희소도), explained_variance, noise_flag
    """
    rows = []
    k = loadings.shape[0]
    for k_ in range(k):
        h = loadings[k_]

        # 상위 N개 피처 (절댓값 기준, 부호 포함)
        top_idx = np.argsort(np.abs(h))[::-1][:top_n]
        top_feats = [(feature_names[i], round(float(h[i]), 4)) for i in top_idx]

        # 설명분산
        ev = float(explained_variance[k_]) if explained_variance is not None else None
        noise = (ev is not None) and (ev < MIN_EXPLAINED_VAR)

        rows.append({
            "method":            method,
            "axis_idx":          k_,
            "top_features":      str(top_feats),   # CSV 저장용 문자열
            "gini_sparsity":     round(gini(h), 4),
            "explained_variance": round(ev, 4) if ev is not None else None,
            "noise_flag":        noise,
        })

        # 콘솔 출력 (정성 확인용)
        flag_str = " ⚠️ NOISE" if noise else ""
        ev_str = f"{ev:.3f}" if ev is not None else "N/A"
        print(f"  [{method}] 축{k_}{flag_str}: "
              f"gini={gini(h):.3f}, expVar={ev_str}")

        for feat, val in top_feats:
            print(f"      {feat:30s} {val:+.4f}")

    return pd.DataFrame(rows)


def interpret_importance(
    importances: np.ndarray,
    feature_names: list,
    method: str = "SHAP",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    SHAP: 변수 중요도 벡터에서 해석가능성 정보 추출.
    ★ loading 행렬이 아니므로 축별 분석이 불가 → 전체 중요도 순위만 제공.
    """
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_feats = [(feature_names[i], round(float(importances[i]), 4)) for i in top_idx]

    print(f"  [SHAP] global importance 상위 {top_n}개:")
    for feat, val in top_feats:
        print(f"      {feat:30s} {val:.4f}")

    # SHAP은 축이 없으므로 axis_idx=-1, noise_flag=None
    return pd.DataFrame([{
        "method":            method,
        "axis_idx":          -1,
        "top_features":      str(top_feats),
        "gini_sparsity":     round(gini(importances), 4),
        "explained_variance": None,
        "noise_flag":        None,   # SHAP은 설명분산 개념 없음
    }])


# -----------------------------------------------------------------------------
# 통합 진입점
# -----------------------------------------------------------------------------

def interpret_all(results: dict) -> pd.DataFrame:
    """
    세 기법 결과를 받아 해석가능성 표 생성.

    Args:
        results: extractors.extract_all() 반환값
                 {'NMF': ExtractResult, 'FA': ExtractResult, 'SHAP': ExtractResult}
    Returns:
        axis_interpretation.csv에 저장할 DataFrame
    """
    frames = []
    for method, res in results.items():
        print(f"\n[interpret] {method} 해석가능성 분석:")
        if res.type == "matrix":
            df = interpret_matrix(
                res.loadings,
                res.feature_names,
                method,
                res.explained_variance,
            )
        else:
            df = interpret_importance(res.importances, res.feature_names, method)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # 노이즈 경고 출력
    noisy = combined[combined["noise_flag"] == True]
    if len(noisy) > 0:
        print(f"\n⚠️  [interpret] 노이즈 의심 축 {len(noisy)}개 "
              f"(explained_variance < {MIN_EXPLAINED_VAR}):")
        print(noisy[["method", "axis_idx", "explained_variance"]].to_string(index=False))

    return combined