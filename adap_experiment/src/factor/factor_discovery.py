"""
NMF / FA로 K개 Factor Loading 추출
★ 자체 NMF/FA 구현 대신 factor_stability 프로젝트의 검증된 extract_nmf / extract_fa를
   import하여 재사용한다. 전처리(스케일링, 결측 처리)까지 동일하게 적용하여 일관성을 유지.

어댑터 역할:
  - factor_stability의 입력 형식(preprocess_for_nmf/fa)으로 변환
  - factor_stability의 ExtractResult 출력을 adap_experiment 인터페이스
    (loading_matrix: DataFrame index=FEATURE_NAMES, user_scores: DataFrame index=userId)로 변환
  - loading 수치 자체는 절대 변경하지 않음 — 이름/순서 재배열만 수행
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

from config import K_FACTORS, FEATURE_NAMES, OUTPUT_DIR

logger = logging.getLogger(__name__)

import importlib.util

# ── factor_stability 프로젝트 모듈 격리 로드 ────────
# adap_experiment의 부모 디렉토리(agent_profiling)에 factor_stability가 형제 폴더로 존재
_adap_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_factor_stability_dir = os.path.join(_adap_root, "factor_stability")

def _load_fs_module(module_name: str):
    """
    모듈 이름 충돌 방지(config 등)를 위해 factor_stability의 모듈을 격리하여 로드합니다.
    """
    safe_name = f"factor_stability_{module_name}"
    if safe_name in sys.modules:
        return sys.modules[safe_name]

    target_path = os.path.join(_factor_stability_dir, f"{module_name}.py")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"{target_path} not found")

    original_sys_path = sys.path.copy()
    sys.path.insert(0, _factor_stability_dir)

    # 잠재적 충돌 모듈 백업
    conflicts = ["config", "features", "extractors", "interpret"]
    backups = {k: sys.modules.pop(k) for k in conflicts if k in sys.modules}

    try:
        spec = importlib.util.spec_from_file_location(module_name, target_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        sys.modules[safe_name] = module
        return module
    finally:
        sys.path = original_sys_path
        for k in conflicts:
            sys.modules.pop(k, None)
        for k, v in backups.items():
            sys.modules[k] = v

fs_features = _load_fs_module("features")
fs_extractors = _load_fs_module("extractors")

preprocess_for_nmf = fs_features.preprocess_for_nmf
preprocess_for_fa = fs_features.preprocess_for_fa
extract_nmf = fs_extractors.extract_nmf
extract_fa = fs_extractors.extract_fa


def run_factor_discovery(
    user_feature_matrix: pd.DataFrame,
    k: int = K_FACTORS,
    method: str = "nmf",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    method: "nmf" 또는 "fa"
    내부적으로 factor_stability의 extract_nmf / extract_fa를 호출.
    전처리도 factor_stability의 preprocess_for_nmf / preprocess_for_fa를 그대로 사용하여
    기존에 검증된 NMF/FA 결과와 동일한 스케일링 방식을 적용.

    반환:
      loading_matrix: pd.DataFrame
        index = FEATURE_NAMES (19개, config.py의 순서 그대로)
        columns = factor_0, factor_1, ..., factor_{k-1}
      user_scores: pd.DataFrame
        index = userId
        columns = factor_0 ... factor_{k-1}
    """
    user_ids = user_feature_matrix.index
    feature_names_input = list(user_feature_matrix.columns)
    factor_cols = [f"factor_{i}" for i in range(k)]

    # K값 검증 블록 삭제됨

    if method == "nmf":
        # [주석 명시] adap_experiment의 user_feature_matrix는 이미 결측치(NaN)가 
        # imputation되어 채워진 상태로 전달된다. 원래 factor_stability는 NaN이 포함된 데이터를
        # 전제로 설계되었으나, 여기서는 파이프라인 정합성을 위해 그대로 넘긴다. (의도된 동작)
        # factor_stability의 전처리: MinMax(0~1) + 결측 마스크
        X_nmf_tuple = preprocess_for_nmf(user_feature_matrix)
        # factor_stability의 NMF 추출 (EM 기반 결측 허용 분해)
        result = extract_nmf(X_nmf_tuple, feature_names_input, k=k, seed=42)

    elif method == "fa":
        # [주석 명시] 마찬가지로 NaN이 없는 데이터가 넘어가므로, FA의 pairwise deletion 로직은
        # 실질적으로 모든 관측값을 사용하게 되며, 이는 adap_experiment 파이프라인의 의도된 동작이다.
        # factor_stability의 전처리: 표준화(NaN 유지)
        X_fa = preprocess_for_fa(user_feature_matrix)
        # factor_stability의 FA 추출 (쌍별 결측제거 공분산 + Varimax 회전)
        result = extract_fa(X_fa, feature_names_input, k=k)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'nmf' or 'fa'.")

    # ── 어댑터: ExtractResult → adap_experiment 인터페이스 ──
    # result.loadings: shape=(k, n_features) numpy array
    # → loading_matrix: DataFrame, index=FEATURE_NAMES, columns=factor_0..factor_{k-1}
    # 값 자체는 변경하지 않고 이름/순서만 재배열

    raw_loadings = result.loadings          # (k, n_features)
    raw_feature_names = result.feature_names  # 입력 순서 그대로
    raw_user_scores = result.user_scores    # (n_users, k)

    # loading_matrix: 전치해서 (n_features, k) 형태로 만들고 FEATURE_NAMES 순서로 정렬
    loading_df = pd.DataFrame(
        raw_loadings.T,
        index=raw_feature_names,
        columns=factor_cols,
    )
    # FEATURE_NAMES 순서에 맞게 reindex (없는 피처는 0으로 채움 — 정상적이면 발생 안 함)
    missing_feats = [f for f in FEATURE_NAMES if f not in loading_df.index]
    if missing_feats:
        logger.warning("reindex 중 누락된 피처가 발견되어 0.0으로 채워집니다: %s", missing_feats)
    loading_matrix = loading_df.reindex(FEATURE_NAMES, fill_value=0.0)

    # user_scores: (n_users, k)
    user_scores = pd.DataFrame(
        raw_user_scores,
        index=user_ids,
        columns=factor_cols,
    )

    # 저장 (method별로 파일명 구분)
    factors_dir = os.path.join(OUTPUT_DIR, "factors")
    os.makedirs(factors_dir, exist_ok=True)
    loading_matrix.to_csv(os.path.join(factors_dir, f"factor_loadings_{method}.csv"))
    user_scores.to_csv(os.path.join(factors_dir, f"user_scores_{method}.csv"))

    logger.info(
        "Factor discovery (%s, k=%d) complete via factor_stability. Loading matrix shape: %s",
        method, k, loading_matrix.shape,
    )
    return loading_matrix, user_scores


def interpret_factors(loading_matrix: pd.DataFrame, top_n: int = 3) -> dict:
    """
    각 factor별로 loading 절댓값 상위 top_n개 피처와 그 값을 추출.
    반환: {factor_id: [(feature_name, loading_value), ...]}

    NOTE: factor_stability/interpret.py의 interpret_matrix()도 유사한 기능을 하지만,
    반환 형식이 pd.DataFrame이라 adap_experiment 파이프라인의 기대 형식(dict)과 다르므로
    이 간단한 래퍼를 유지한다.
    """
    result: dict[str, list[tuple[str, float]]] = {}
    for col in loading_matrix.columns:
        abs_loadings = loading_matrix[col].abs()
        top_idx = abs_loadings.nlargest(top_n).index
        top_features = [
            (feat, float(loading_matrix.loc[feat, col]))
            for feat in top_idx
        ]
        result[col] = top_features
    logger.info("Factor interpretation: %s", {k: [f[0] for f in v] for k, v in result.items()})
    return result
