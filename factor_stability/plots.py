# =============================================================================
# plots.py
# 산출물 그림 생성.
# ★ 축 라벨·제목은 영어 (한글 폰트 깨짐 방지)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # 화면 없는 환경에서도 동작

from config import STABILITY_GO, RESULTS_DIR, FNAME_VOLUME_PNG, FNAME_METHOD_PNG


def plot_volume_curve(volume_df: pd.DataFrame) -> None:
    """
    데이터 볼륨 곡선: x=데이터 비율(%), y=안정성, 선 3개=NMF/FA/SHAP.
    에러바 = ±표준편차.
    메인 그림 — 핵심 주장("더 많은 데이터 → 더 안정적 축")의 첫 증거.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = {"NMF": "#2196F3", "FA": "#4CAF50", "SHAP": "#FF9800"}
    markers = {"NMF": "o",       "FA": "s",        "SHAP": "^"}

    for method, grp in volume_df.groupby("method"):
        grp = grp.sort_values("fraction")
        x = grp["fraction"].values * 100
        ax.errorbar(
            x, grp["mean"], yerr=grp["std"],
            label=method, color=colors.get(method, "gray"),
            marker=markers.get(method, "o"),
            linewidth=2, capsize=4, capthick=1.5,
        )

    # GO 기준선
    ax.axhline(STABILITY_GO, ls="--", color="red", linewidth=1.2,
               label=f"GO threshold ({STABILITY_GO})")

    ax.set_xlabel("Data Volume (%)", fontsize=12)
    ax.set_ylabel("Stability (Tucker congruence / Spearman)", fontsize=12)
    ax.set_title("Stability vs Data Volume by Method\n"
                 "(Higher & faster convergence = better behavioral factor structure)",
                 fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([10, 30, 60, 100])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = RESULTS_DIR / FNAME_VOLUME_PNG
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plots] 저장: {path}")


def plot_method_comparison(stability_records: list) -> None:
    """
    100% 데이터에서 세 기법의 안정성 비교 막대 그래프.
    stability_records: [StabilityResult, ...] (100% 데이터 기준)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [r.method for r in stability_records]
    means   = [r.mean   for r in stability_records]
    stds    = [r.std    for r in stability_records]
    metrics = [r.metric_name for r in stability_records]

    colors = {"NMF": "#2196F3", "FA": "#4CAF50", "SHAP": "#FF9800"}
    bar_colors = [colors.get(m, "gray") for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, color=bar_colors,
                  capsize=5, width=0.5, alpha=0.85)

    # 기준선
    ax.axhline(STABILITY_GO, ls="--", color="red", linewidth=1.2,
               label=f"GO threshold ({STABILITY_GO})")

    # 막대 위에 수치 표시
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylabel("Stability Score", fontsize=12)
    ax.set_title("Method Comparison: Split-Half Stability (100% Data)\n"
                 "Note: NMF/FA = Tucker congruence, SHAP = Spearman rank corr",
                 fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    path = RESULTS_DIR / FNAME_METHOD_PNG
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plots] 저장: {path}")