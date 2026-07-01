# =============================================================================
# run_k_sweep.py
# K = [3, 5, 7, 10] 각각에 대해 Phase 0 핵심 검증(안정성/해석가능성/외부타당성)을
# 100% 데이터로 반복 실행하고, results/k_sweep/K{n}/ 아래에 K별로 저장한다.
#
# ★ run_phase0.py와의 차이:
#   - run_phase0.py : K=3 고정, 데이터 볼륨 곡선(10/30/60/100%) 포함 — 무거움
#   - run_k_sweep.py: 여러 K값 비교가 목적. 볼륨 곡선은 제외(K×볼륨이면 너무 무거움).
#                     데이터는 항상 100%만 사용.
#
# ★ 로직은 run_phase0.py와 완전히 동일 — extractors/stability/interpret/validity
#   모듈을 그대로 재사용하고, K값만 바꿔가며 반복하는 오케스트레이션만 다름.
#
# 사용법: python run_k_sweep.py
# 실행 전 config.py의 DATA_DIR, K_SWEEP_VALUES를 확인하세요.
# =============================================================================

import pandas as pd
from pathlib import Path

from config import (
    DATA_DIR, RESULTS_DIR, K_SWEEP_VALUES, K_SWEEP_DIR, get_k_results_dir,
    STABILITY_GO, STABILITY_WEAK,
    FNAME_STABILITY, FNAME_INTERPRET, FNAME_VALIDITY,
    FNAME_FEATURE_TABLE, FNAME_USER_SCORES,
)
from data_loader import load_all
from features import build_feature_table, get_preprocessed
from extractors import extract_nmf, extract_fa, extract_shap
from stability import measure_stability_matrix, measure_stability_importance
from interpret import interpret_all
from validity import validate_all


def go_nogo(mean: float, valid_sig: int, noisy_axes: int) -> str:
    """
    run_phase0.py의 go_nogo()와 동일한 판정 로직 (네 가지 검증 종합).
    """
    if mean >= STABILITY_GO and valid_sig > 0 and noisy_axes == 0:
        return "GO"
    elif mean >= STABILITY_GO and (valid_sig == 0 or noisy_axes > 0):
        return "CAUTION (stable but validity weak)"
    elif STABILITY_WEAK <= mean < STABILITY_GO:
        return "WEAK"
    else:
        return "REVIEW"


def run_single_k(k: int, feature_table, preprocessed, feature_log,
                  validation_log, users, user_ids) -> dict:
    """
    주어진 k값 하나에 대해 NMF/FA/SHAP 축 추출 + 4가지 검증을 전부 수행하고,
    K{k} 폴더에 결과를 저장한다. 데이터 로딩/피처 테이블은 바깥에서 한 번만
    수행한 결과를 그대로 재사용 (k마다 다시 만들 필요 없음 — 피처는 k와 무관).
    """
    k_dir = get_k_results_dir(k)
    k_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"K = {k}")
    print("=" * 60)

    fn = preprocessed["feature_names"]

    # ------------------------------------------------------------------
    # 1. 축 추출 (NMF / FA / SHAP), k를 명시적으로 전달
    # ------------------------------------------------------------------
    print(f"[K={k}] NMF 추출 중 (EM 결측처리)...")
    res_nmf = extract_nmf(preprocessed["nmf"], fn, k=k)

    print(f"[K={k}] FA 추출 중 (쌍별 결측제거)...")
    res_fa = extract_fa(preprocessed["fa"], fn, k=k)

    print(f"[K={k}] SHAP 추출 중...")
    res_shap = extract_shap(preprocessed["shap"], feature_log, fn, k=k)

    results = {"NMF": res_nmf, "FA": res_fa, "SHAP": res_shap}

    # 유저별 축 점수 저장 (시각화용 — visualize_phase0.py가 그대로 읽을 수 있는 형식)
    scores_df = pd.DataFrame({"user_id": user_ids})
    for method, res in results.items():
        for k_ in range(res.user_scores.shape[1]):
            scores_df[f"{method}_axis{k_}"] = res.user_scores[:, k_]
    scores_df.to_csv(k_dir / FNAME_USER_SCORES, index=False)
    print(f"[K={k}] 유저 축 점수 저장: {k_dir / FNAME_USER_SCORES}")

    # 피처 테이블도 K별 폴더에 동일하게 둠 (visualize_phase0.py가 폴더 하나만
    # 보고 모든 입력을 찾을 수 있게 — 매번 results/ 루트를 따로 안 봐도 됨)
    feature_table.to_csv(k_dir / FNAME_FEATURE_TABLE)

    # ------------------------------------------------------------------
    # 2. 검증 ① 안정성 (split-half)
    # ------------------------------------------------------------------
    print(f"[K={k}] 안정성(split-half) 측정...")

    def nmf_fn(X_sub):
        return extract_nmf(X_sub, fn, k=k)

    def fa_fn(X_sub):
        return extract_fa(X_sub, fn, k=k)

    def shap_fn(X_sub, fl_sub, seed=42):
        return extract_shap(X_sub, fl_sub, fn, k=k, seed=seed)

    stab_nmf = measure_stability_matrix(preprocessed["nmf"], nmf_fn, "NMF")
    stab_fa  = measure_stability_matrix(preprocessed["fa"],  fa_fn,  "FA")
    stab_shap = measure_stability_importance(
        preprocessed["shap"], feature_log, shap_fn, "SHAP"
    )
    stability_records = [stab_nmf, stab_fa, stab_shap]

    stab_df = pd.DataFrame([{
        "k":            k,
        "method":       r.method,
        "metric_name":  r.metric_name,
        "mean":         round(r.mean, 4),
        "std":          round(r.std, 4),
        "type":         r.type,
        "auto_axis":    r.auto_axis,
    } for r in stability_records])
    stab_df.to_csv(k_dir / FNAME_STABILITY, index=False)
    print(f"[K={k}] 안정성 결과 저장: {k_dir / FNAME_STABILITY}")

    # ------------------------------------------------------------------
    # 3. 검증 ② 해석가능성 + ④ 노이즈
    # ------------------------------------------------------------------
    print(f"[K={k}] 해석가능성 + 노이즈 축 검증...")
    interp_df = interpret_all(results)
    interp_df.insert(0, "k", k)
    interp_df.to_csv(k_dir / FNAME_INTERPRET, index=False)
    print(f"[K={k}] 해석가능성 결과 저장: {k_dir / FNAME_INTERPRET}")

    # ------------------------------------------------------------------
    # 4. 검증 ③ 외부 타당성
    # ------------------------------------------------------------------
    print(f"[K={k}] 외부 타당성 검증...")
    valid_df = validate_all(results, user_ids, validation_log, users)
    valid_df.insert(0, "k", k)
    valid_df.to_csv(k_dir / FNAME_VALIDITY, index=False)
    print(f"[K={k}] 외부 타당성 결과 저장: {k_dir / FNAME_VALIDITY}")

    # ------------------------------------------------------------------
    # 5. K={k} 단위 GO/NO-GO 요약 (콘솔 + 반환용)
    # ------------------------------------------------------------------
    summary_rows = []
    for r in stability_records:
        v_sig = len(valid_df[(valid_df["method"] == r.method) &
                              (valid_df["significant"] == True)])
        noisy = len(interp_df[(interp_df["method"] == r.method) &
                               (interp_df["noise_flag"] == True)])
        verdict = go_nogo(r.mean, v_sig, noisy)
        summary_rows.append({
            "k": k, "method": r.method,
            "stability_mean": round(r.mean, 4), "stability_std": round(r.std, 4),
            "valid_significant": v_sig, "noisy_axes": noisy, "verdict": verdict,
        })
        print(f"  [K={k}][{r.method}] stability={r.mean:.4f}±{r.std:.4f}, "
              f"valid_sig={v_sig}, noisy={noisy} -> {verdict}")

    return {"k": k, "summary": summary_rows}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    K_SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"K SWEEP — K values: {K_SWEEP_VALUES}")
    print("=" * 60)

    # 데이터 로딩 + 피처 테이블은 K와 무관하므로 한 번만 수행 (반복 부담 줄임)
    print("\n[k_sweep] 데이터 로드 (100%)...")
    data = load_all(DATA_DIR)
    feature_log     = data["feature_log"]
    validation_log  = data["validation_log"]
    users           = data["users"]

    print("[k_sweep] 피처 테이블 생성...")
    feature_table = build_feature_table(feature_log)
    preprocessed  = get_preprocessed(feature_table)
    user_ids      = preprocessed["user_ids"]

    # K값별로 반복
    all_summaries = []
    for k in K_SWEEP_VALUES:
        result = run_single_k(
            k, feature_table, preprocessed, feature_log,
            validation_log, users, user_ids,
        )
        all_summaries.extend(result["summary"])

    # 전체 K를 가로지르는 요약표 (K_SWEEP_DIR 바로 아래에 저장)
    summary_df = pd.DataFrame(all_summaries)
    summary_path = K_SWEEP_DIR / "k_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("K SWEEP 완료 — 종합 요약")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\n[k_sweep] 요약 저장: {summary_path}")
    print(f"[k_sweep] K별 상세 결과: {K_SWEEP_DIR}/K{{3,5,7,10}}/")


if __name__ == "__main__":
    main()