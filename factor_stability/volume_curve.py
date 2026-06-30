# =============================================================================
# volume_curve.py
# 데이터 볼륨(10→30→60→100%) × 기법별 안정성 곡선 데이터 생성.
# "데이터가 많을수록 축이 안정적으로 수렴한다" → 핵심 주장의 첫 증거.
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path

from config import VOLUME_FRACTIONS, N_SEEDS, BASE_SEED, K
from data_loader import load_all
from features import build_feature_table, get_preprocessed
from extractors import extract_nmf, extract_fa, extract_shap
from stability import (
    measure_stability_matrix,
    measure_stability_importance,
)


def run_volume_curve(data_dir: Path) -> pd.DataFrame:
    """
    VOLUME_FRACTIONS 각각에 대해 NMF/FA/SHAP 안정성을 측정해 DataFrame 반환.

    반환 컬럼: fraction, method, metric_name, mean, std
    """
    records = []

    for frac in VOLUME_FRACTIONS:
        print(f"\n{'='*60}")
        print(f"[volume_curve] 데이터 비율: {int(frac*100)}%")
        print(f"{'='*60}")

        # 데이터 로드 (유저 서브샘플)
        data = load_all(data_dir, subsample=frac, seed=BASE_SEED)
        feature_log   = data["feature_log"]
        feature_table = build_feature_table(feature_log)
        preprocessed  = get_preprocessed(feature_table)

        X_nmf  = preprocessed["nmf"]
        X_fa   = preprocessed["fa"]
        X_shap = preprocessed["shap"]
        fn     = preprocessed["feature_names"]

        # ----- NMF -----
        def nmf_fn(X_sub):
            return extract_nmf(X_sub, fn, seed=BASE_SEED)

        print(f"\n[volume_curve/{int(frac*100)}%] NMF 안정성 측정...")
        nmf_res = measure_stability_matrix(X_nmf, nmf_fn, "NMF")
        records.append({
            "fraction":    frac,
            "method":      "NMF",
            "metric_name": nmf_res.metric_name,
            "mean":        nmf_res.mean,
            "std":         nmf_res.std,
        })

        # ----- FA -----
        def fa_fn(X_sub):
            return extract_fa(X_sub, fn)

        print(f"\n[volume_curve/{int(frac*100)}%] FA 안정성 측정...")
        fa_res = measure_stability_matrix(X_fa, fa_fn, "FA")
        records.append({
            "fraction":    frac,
            "method":      "FA",
            "metric_name": fa_res.metric_name,
            "mean":        fa_res.mean,
            "std":         fa_res.std,
        })

        # ----- SHAP -----
        def shap_fn(X_sub, fl_sub, seed=BASE_SEED):
            return extract_shap(X_sub, fl_sub, fn, seed=seed)

        print(f"\n[volume_curve/{int(frac*100)}%] SHAP 안정성 측정 (느림)...")
        shap_res = measure_stability_importance(X_shap, feature_log, shap_fn, "SHAP")
        records.append({
            "fraction":    frac,
            "method":      "SHAP",
            "metric_name": shap_res.metric_name,
            "mean":        shap_res.mean,
            "std":         shap_res.std,
        })

        print(f"\n[volume_curve/{int(frac*100)}%] 요약: "
              f"NMF={nmf_res.mean:.3f}±{nmf_res.std:.3f} | "
              f"FA={fa_res.mean:.3f}±{fa_res.std:.3f} | "
              f"SHAP={shap_res.mean:.3f}±{shap_res.std:.3f}")

    return pd.DataFrame(records)