# =============================================================================
# run_phase0.py
# Phase 0 전체 실험 오케스트레이션.
# 사용법: python run_phase0.py
#
# 실행 전 config.py의 DATA_DIR을 본인 MovieLens 경로로 수정하세요.
# =============================================================================

import pandas as pd
from pathlib import Path

from config import (
    DATA_DIR, RESULTS_DIR, K,
    STABILITY_GO, STABILITY_WEAK,
    FNAME_STABILITY, FNAME_INTERPRET, FNAME_VALIDITY, FNAME_VOLUME_CSV,
    FNAME_FEATURE_TABLE, FNAME_USER_SCORES,
)
from data_loader import load_all
from features import build_feature_table, get_preprocessed
from extractors import extract_all
from stability import measure_stability_matrix, measure_stability_importance
from interpret import interpret_all
from validity import validate_all
from volume_curve import run_volume_curve
from plots import plot_volume_curve, plot_method_comparison


def go_nogo(method: str, mean: float, valid_sig: int, noisy_axes: int) -> str:
    """
    네 가지 검증을 종합한 GO/NO-GO 판정.
    ★ 안정성 하나만으로 판정하지 않는다.
    """
    if mean >= STABILITY_GO and valid_sig > 0 and noisy_axes == 0:
        return "✅ GO"
    elif mean >= STABILITY_GO and (valid_sig == 0 or noisy_axes > 0):
        return "⚠️  CAUTION: 안정적이나 유의미성 의심 (피처 재설계 권고)"
    elif STABILITY_WEAK <= mean < STABILITY_GO:
        return "🔶 WEAK: 약한 신호 (피처 재검토)"
    else:
        return "❌ REVIEW: 재검토 필요 (공통 축 존재 전제 재고)"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 데이터 로드 (100% 데이터 — 기본 실험)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 0 — 행동 축 추출 타당성 검증")
    print("="*60)
    data         = load_all(DATA_DIR)
    feature_log  = data["feature_log"]
    validation_log = data["validation_log"]
    users        = data["users"]

    # ------------------------------------------------------------------
    # 2. 피처 테이블 + 전처리
    # ------------------------------------------------------------------
    print("\n[Phase0] Step 1: 피처 테이블 생성...")
    feature_table = build_feature_table(feature_log)

    # 피처 테이블 저장 — "이 사용자가 어떤 22개 숫자를 가졌는지" 재현 가능하게
    feature_table.to_csv(RESULTS_DIR / FNAME_FEATURE_TABLE)
    print(f"[Phase0] 피처 테이블 저장: {RESULTS_DIR / FNAME_FEATURE_TABLE}")

    preprocessed  = get_preprocessed(feature_table)
    user_ids      = preprocessed["user_ids"]
    fn            = preprocessed["feature_names"]

    # ------------------------------------------------------------------
    # 3. 축 추출 (NMF / FA / SHAP)
    # ------------------------------------------------------------------
    print("\n[Phase0] Step 2: 축 추출 (K={})...".format(K))
    results = extract_all(preprocessed, feature_log)

    # 유저별 축 점수 저장 — 산점도/2D 시각화용
    # 컬럼명: {method}_axis{k} 형태로 한 표에 세 기법 다 모음
    scores_df = pd.DataFrame({"user_id": user_ids})
    for method, res in results.items():
        for k_ in range(res.user_scores.shape[1]):
            scores_df[f"{method}_axis{k_}"] = res.user_scores[:, k_]
    scores_df.to_csv(RESULTS_DIR / FNAME_USER_SCORES, index=False)
    print(f"[Phase0] 유저 축 점수 저장: {RESULTS_DIR / FNAME_USER_SCORES}")

    # ------------------------------------------------------------------
    # 4. 검증 ① 안정성
    # ------------------------------------------------------------------
    print("\n[Phase0] Step 3: 안정성(split-half) 측정...")

    def nmf_fn(X_sub):
        from extractors import extract_nmf
        return extract_nmf(X_sub, fn)

    def fa_fn(X_sub):
        from extractors import extract_fa
        return extract_fa(X_sub, fn)

    def shap_fn(X_sub, fl_sub, seed=42):
        from extractors import extract_shap
        return extract_shap(X_sub, fl_sub, fn, seed=seed)

    stab_nmf  = measure_stability_matrix(preprocessed["nmf"], nmf_fn, "NMF")
    stab_fa   = measure_stability_matrix(preprocessed["fa"],  fa_fn,  "FA")
    stab_shap = measure_stability_importance(
        preprocessed["shap"], feature_log, shap_fn, "SHAP"
    )
    stability_records = [stab_nmf, stab_fa, stab_shap]

    stab_df = pd.DataFrame([{
        "method":       r.method,
        "metric_name":  r.metric_name,
        "mean":         round(r.mean, 4),
        "std":          round(r.std, 4),
        "type":         r.type,
        "auto_axis":    r.auto_axis,
        "shap_rmse_a":  round(r.model_rmse_a, 4) if r.model_rmse_a else None,
        "shap_rmse_b":  round(r.model_rmse_b, 4) if r.model_rmse_b else None,
    } for r in stability_records])
    stab_df.to_csv(RESULTS_DIR / FNAME_STABILITY, index=False)
    print(f"\n[Phase0] 안정성 결과 저장: {RESULTS_DIR / FNAME_STABILITY}")

    # ------------------------------------------------------------------
    # 5. 검증 ② 해석가능성 + ④ 노이즈
    # ------------------------------------------------------------------
    print("\n[Phase0] Step 4: 해석가능성 + 노이즈 축 검증...")
    interp_df = interpret_all(results)
    interp_df.to_csv(RESULTS_DIR / FNAME_INTERPRET, index=False)
    print(f"[Phase0] 해석가능성 결과 저장: {RESULTS_DIR / FNAME_INTERPRET}")

    # ------------------------------------------------------------------
    # 6. 검증 ③ 외부 타당성
    # ------------------------------------------------------------------
    print("\n[Phase0] Step 5: 외부 타당성 검증...")
    valid_df = validate_all(results, user_ids, validation_log, users)
    valid_df.to_csv(RESULTS_DIR / FNAME_VALIDITY, index=False)
    print(f"[Phase0] 외부 타당성 결과 저장: {RESULTS_DIR / FNAME_VALIDITY}")

    # ------------------------------------------------------------------
    # 7. 데이터 볼륨 곡선
    # ------------------------------------------------------------------
    print("\n[Phase0] Step 6: 데이터 볼륨 곡선 생성 (가장 오래 걸림)...")
    volume_df = run_volume_curve(DATA_DIR)
    volume_df.to_csv(RESULTS_DIR / FNAME_VOLUME_CSV, index=False)

    # ------------------------------------------------------------------
    # 8. 그림
    # ------------------------------------------------------------------
    print("\n[Phase0] Step 7: 그림 생성...")
    plot_volume_curve(volume_df)
    plot_method_comparison(stability_records)

    # ------------------------------------------------------------------
    # 9. 종합 GO/NO-GO 판정
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 0 — 종합 GO/NO-GO 판정")
    print("="*60)
    print("★ 안정성 하나가 아닌 네 가지 검증을 종합합니다.\n")

    stab_map = {r.method: r for r in stability_records}

    for method in ["NMF", "FA", "SHAP"]:
        r = stab_map[method]

        # 외부 타당성: 유의미한 상관 개수
        v_rows = valid_df[
            (valid_df["method"] == method) & (valid_df["significant"] == True)
        ]
        valid_sig = len(v_rows)

        # 노이즈 축 수
        noise_rows = interp_df[
            (interp_df["method"] == method) & (interp_df["noise_flag"] == True)
        ]
        noisy_axes = len(noise_rows)

        verdict = go_nogo(method, r.mean, valid_sig, noisy_axes)

        print(f"[{method}]")
        print(f"  ① 안정성   : {r.metric_name} = {r.mean:.4f} ± {r.std:.4f}")
        print(f"  ② 해석가능성: 콘솔 출력 참조 (axis_interpretation.csv)")
        print(f"  ③ 외부타당성: 유의미한 상관 {valid_sig}개")
        print(f"  ④ 노이즈   : 의심 축 {noisy_axes}개")
        print(f"  → 판정: {verdict}\n")

    print("="*60)
    print(f"산출물 위치: {RESULTS_DIR.resolve()}")
    print("  - stability_comparison.csv")
    print("  - axis_interpretation.csv")
    print("  - external_validity.csv")
    print("  - volume_curve.csv / volume_curve.png")
    print("  - method_comparison.png")
    print("="*60)


if __name__ == "__main__":
    main()