# =============================================================================
# compare_with_axes.py
# LLM(TopicGPT)이 배정한 취향 유형을, 우리 NMF/FA 축 공간에 색칠해서 시각화.
#
# ★ 원본 피처 클러스터링(axis_space_rawclustered)과 나란히 만들어서,
#   "원본 피처로 나눈 그룹" vs "LLM이 원본 영화목록만 보고 판단한 유형"이
#   축 공간에서 각각 얼마나 잘 갈리는지 비교할 수 있게 함.
#
# ★ 한 사람이 여러 LLM 유형에 동시에 속할 수 있음(다중배정). 색칠은 한
#   사람에 한 색만 가능하므로, 배정된 유형 중 "가장 드문(=변별력 있는)"
#   유형 하나를 그 사람의 대표 유형으로 삼음. 너무 흔한 유형(예: 전체의
#   99%에게 붙는 유형)은 대표로 뽑혀도 의미가 없어 EXCLUDE_TOPICS로 아예
#   제외 가능.
#
# 사용 전 확인:
#   - ASSIGNMENT_PATH: TopicGPT 배정 결과 (llm_topic_pilot/data/output/n100/
#     assignment_corrected.jsonl)
#   - AXIS_SCORES_PATH, FEATURE_TABLE_PATH: 비교 대상 K의 axis/feature 파일
#     (factor_stability/results/ 또는 results/k_sweep/K{n}/)
#   - EXCLUDE_TOPICS: 제외할 유형 이름 (기본값: 100명 중 99명에게 붙어서
#     구분력이 없다고 확인된 "Emotional Dramas")
# =============================================================================

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# 상위 factor_stability 폴더의 시각화 유틸리티 재사용 (중복 구현 방지)
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

from visualize_phase0 import (  # noqa: E402
    infer_n_axes, _all_axis_pairs, CLUSTER_PALETTE,
    cluster_raw_feature_space, plot_axis_space_raw_clustered,
)

# -----------------------------------------------------------------------------
# 경로 설정 — 실행 전 확인/수정
# -----------------------------------------------------------------------------
ASSIGNMENT_PATH = Path(__file__).resolve().parent / "data" / "output" / "n100" / "assignment_corrected.jsonl"
AXIS_SCORES_PATH = PARENT_DIR / "results" / "user_axis_scores.csv"       # K=3 기준 예시. K sweep 결과면 results/k_sweep/K{n}/로 변경
FEATURE_TABLE_PATH = PARENT_DIR / "results" / "user_feature_table.csv"
OUT_DIR = Path(__file__).resolve().parent / "data" / "output" / "axis_comparison"

# 100명 중 99명에게 붙어 구분력이 없다고 확인된 유형. 필요시 추가/변경.
EXCLUDE_TOPICS = ["Emotional Dramas"]


# -----------------------------------------------------------------------------
# 1) assignment_corrected.jsonl 파싱 — 유저별 배정된 유형 목록(다중배정 그대로)
# -----------------------------------------------------------------------------

def parse_llm_assignments(assignment_path: Path) -> pd.DataFrame:
    """
    반환: [user_id, llm_topic, score] — 이제 배정 프롬프트가 "확정된 유형
    전부에 0~5점 밀집 채점"을 하므로, 한 유저당 모든 유형에 대해 한 행씩
    나옴 (이전의 희소 배정과 달리 결측 없음).
    """
    records = []
    with open(assignment_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            uid = int(d["id"].replace("user_", ""))
            resp = d.get("responses", "")
            # "[1] 유형이름: 점수/5 - 근거" 형식에서 유형명과 점수를 함께 추출
            matches = re.findall(r"\[1\]\s*([^:]+):\s*(\d+)\s*/\s*5", resp)
            for topic, score in matches:
                records.append({
                    "user_id": uid,
                    "llm_topic": topic.strip(),
                    "score": int(score),
                })
    df = pd.DataFrame(records)
    n_users = df["user_id"].nunique()
    n_topics = df["llm_topic"].nunique()
    avg_per_user = len(df) / n_users if n_users else 0
    print(f"[compare] LLM 배정 파싱: {n_users}명, 고유 유형 {n_topics}개, "
          f"유저당 평균 {avg_per_user:.1f}개 (밀집 채점이면 {n_topics}이어야 정상)")
    return df


def pivot_to_dense_table(assign_long_df: pd.DataFrame) -> pd.DataFrame:
    """
    [user_id, llm_topic, score] long format을, 우리 W행렬(user_axis_scores.csv)과
    같은 구조의 wide format(유저×유형, 각 칸이 0~5점)으로 변환.
    """
    wide = assign_long_df.pivot_table(
        index="user_id", columns="llm_topic", values="score", aggfunc="first"
    ).reset_index()
    n_missing = wide.drop(columns="user_id").isna().sum().sum()
    if n_missing > 0:
        print(f"[compare] 경고: 밀집 채점인데 결측 {n_missing}칸 발견 "
              f"(일부 유저 응답이 8개를 채우지 못했을 수 있음)")
    else:
        print(f"[compare] 밀집 표 생성 완료: {wide.shape[0]}명 x "
              f"{wide.shape[1]-1}개 유형, 결측 없음")
    return wide


# -----------------------------------------------------------------------------
# 2) 다중배정 -> 대표 유형 1개로 축약 (시각화는 유저당 색 1개만 가능하므로)
# -----------------------------------------------------------------------------

def assign_primary_topic(dense_wide_df: pd.DataFrame,
                          exclude_topics: list = None) -> pd.DataFrame:
    """
    밀집 점수 표(유저×유형, 0~5점)에서, exclude_topics를 제외한 나머지 유형
    중 점수가 가장 높은 유형을 그 유저의 대표 유형으로 선택.

    (이전 버전은 점수가 없어 "가장 드문 유형"으로 대표를 정하는 임시방편을
    썼는데, 이제 실제 점수가 있으므로 최댓값 기준이 더 원칙적임.)

    반환: [user_id, llm_topic, score] — 유저당 한 행.
    """
    exclude_topics = exclude_topics or []
    topic_cols = [c for c in dense_wide_df.columns
                  if c != "user_id" and c not in exclude_topics]

    sub = dense_wide_df[["user_id"] + topic_cols].set_index("user_id")
    primary_topic = sub.idxmax(axis=1)
    primary_score = sub.max(axis=1)

    result = pd.DataFrame({
        "user_id": primary_topic.index,
        "llm_topic": primary_topic.values,
        "score": primary_score.values,
    }).reset_index(drop=True)

    print(f"[compare] 대표 유형 배정 완료 (최고점 기준): {len(result)}명")
    print(result["llm_topic"].value_counts())
    print(f"[compare] 대표 유형 평균 점수: {result['score'].mean():.2f} / 5")
    return result


# -----------------------------------------------------------------------------
# 3) LLM 유형으로 축 공간 색칠 (기존 클러스터 그림과 같은 배치, 범례만 실제 유형명)
# -----------------------------------------------------------------------------

def plot_axis_space_llm_topic(scores: pd.DataFrame, method: str, out_dir: Path,
                               primary_topic_df: pd.DataFrame):
    n_axes = infer_n_axes(scores, method)
    cols = [f"{method}_axis{k}" for k in range(n_axes)]

    merged = scores.merge(primary_topic_df, on="user_id")  # inner join
    n_shown, n_total = len(merged), len(scores)
    topics_sorted = sorted(merged["llm_topic"].unique())
    topic_to_color = {t: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                       for i, t in enumerate(topics_sorted)}

    X = merged[cols].values
    axis_pairs = _all_axis_pairs(n_axes)
    n_cols = min(len(axis_pairs), 5)
    n_rows = int(np.ceil(len(axis_pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.3 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, (i, j) in enumerate(axis_pairs):
        ax = axes[idx]
        x, y = X[:, i], X[:, j]
        for t in topics_sorted:
            mask = (merged["llm_topic"] == t).values
            ax.scatter(x[mask], y[mask], alpha=0.35, s=12,
                      color=topic_to_color[t],
                      label=t if idx == 0 else None)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(f"{method} axis{i}")
        ax.set_ylabel(f"{method} axis{j}")
        ax.set_title(f"r = {r:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    for idx in range(len(axis_pairs), len(axes)):
        axes[idx].axis("off")

    axes[0].legend(loc="best", fontsize=7, markerscale=1.5)

    fig.suptitle(
        f"{method}: Axis Space Colored by LLM-Assigned Taste Topic "
        f"(n={n_shown}/{n_total} users, {len(topics_sorted)} topics shown)\n"
        f"(Excluded from color assignment: {', '.join(EXCLUDE_TOPICS) or 'none'})",
        fontsize=11)
    fig.tight_layout()
    path = out_dir / f"axis_space_llmtopic_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[compare] saved: {path}")


# -----------------------------------------------------------------------------
# 2.5) LLM 유형끼리의 독립성 확인 — 클러스터링 없이, 순수 산점도 + 상관계수
#
# ★ 우리 FA/NMF의 axis_correlation_{method}.png와 완전히 같은 로직.
#   "LLM이 만든 8개 유형이 서로 독립적인 원색인지, 아니면 NMF축0-1처럼
#   겹치는지"를 확인. 클러스터링(그룹 나누기)은 하지 않음 — 두 유형
#   점수가 같이 움직이는지만 봄.
# -----------------------------------------------------------------------------

def compute_llm_topic_correlation_table(dense_wide_df: pd.DataFrame, out_dir: Path,
                                         exclude_topics: list = None) -> pd.DataFrame:
    """
    산점도 대신 표 형태로 확인. 점수가 0~5 정수 6단계뿐이라 산점도는 점들이
    격자 모양으로만 찍혀 "독립적/겹침" 패턴을 눈으로 읽기 어려움 — 상관계수
    행렬(표)이 더 명확함.

    반환: N x N 상관계수 행렬(대각선=1.0). CSV로도 저장.
    """
    exclude_topics = exclude_topics or []
    topic_cols = [c for c in dense_wide_df.columns
                  if c != "user_id" and c not in exclude_topics]

    corr = dense_wide_df[topic_cols].corr().round(3)

    path = out_dir / "llm_topic_correlation_matrix.csv"
    corr.to_csv(path)
    print(f"[compare] saved: {path}")

    print(f"\n[compare] LLM 유형 간 상관계수 행렬 ({len(topic_cols)}개 유형):")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(corr)

    # 대각선(자기 자신과의 상관=1.0) 제외하고, 절댓값 큰 순서로 별도 요약
    pairs = []
    for i, a in enumerate(topic_cols):
        for b in topic_cols[i + 1:]:
            pairs.append((a, b, corr.loc[a, b]))
    pairs.sort(key=lambda x: -abs(x[2]))

    print(f"\n[compare] 상관계수 절댓값 큰 순서 (|r| > 0.5면 overlap 의심):")
    for a, b, r in pairs:
        flag = " (possible overlap)" if abs(r) > 0.5 else ""
        print(f"  {a} vs {b}: r={r:.3f}{flag}")

    return corr


def plot_llm_topic_correlation_heatmap(corr: pd.DataFrame, out_dir: Path):
    """
    compute_llm_topic_correlation_table()이 반환한 상관계수 행렬(DataFrame)을
    색깔 격자(heatmap)로 시각화. 값이 클수록(양의 상관) 진한 빨강,
    작을수록(음의 상관) 진한 파랑, 0 근처는 흰색.
    """
    labels = corr.columns.tolist()
    n = len(labels)

    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 2))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # 칸마다 숫자 값도 같이 표시 (히트맵 색만으로는 정확한 값을 읽기 어려우므로)
    for i in range(n):
        for j in range(n):
            r = corr.values[i, j]
            text_color = "white" if abs(r) > 0.6 else "black"
            weight = "bold" if (i != j and abs(r) > 0.5) else "normal"
            ax.text(j, i, f"{r:.2f}", ha="center", va="center",
                    color=text_color, fontsize=8, fontweight=weight)

    fig.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)
    ax.set_title(f"LLM Taste Topics: Correlation Heatmap ({n} topics)\n"
                 "(Bold = |r| > 0.5, possible overlap)", fontsize=11)
    fig.tight_layout()

    path = out_dir / "llm_topic_correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[compare] saved: {path}")


# -----------------------------------------------------------------------------
# (참고용, 더 이상 기본 실행에는 포함하지 않음) 산점도 버전 —
# 점수가 이산값(0~5)이라 격자 패턴만 보여서 표보다 해석이 어려움
# -----------------------------------------------------------------------------

def plot_llm_topic_correlation(dense_wide_df: pd.DataFrame, out_dir: Path,
                                exclude_topics: list = None):
    exclude_topics = exclude_topics or []
    topic_cols = [c for c in dense_wide_df.columns
                  if c != "user_id" and c not in exclude_topics]
    n_topics = len(topic_cols)
    axis_pairs = [(a, b) for a in range(n_topics) for b in range(a + 1, n_topics)]

    n_cols = min(len(axis_pairs), 5)
    n_rows = int(np.ceil(len(axis_pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows))
    axes = np.array(axes).reshape(-1)

    overlap_summary = []
    for idx, (a, b) in enumerate(axis_pairs):
        ax = axes[idx]
        x = dense_wide_df[topic_cols[a]].values
        y = dense_wide_df[topic_cols[b]].values

        ax.scatter(x, y, alpha=0.3, s=14, color="#8E24AA")
        r = np.corrcoef(x, y)[0, 1]
        title_color = "red" if abs(r) > 0.5 else "black"

        # 유형 이름이 길어서 축 전체를 라벨로 쓰면 겹치므로, 축약해서 표시
        ax.set_xlabel(topic_cols[a], fontsize=8)
        ax.set_ylabel(topic_cols[b], fontsize=8)
        ax.set_title(f"r = {r:.3f}" + (" (overlap)" if abs(r) > 0.5 else ""),
                    color=title_color, fontsize=10)
        ax.grid(True, alpha=0.3)
        overlap_summary.append((topic_cols[a], topic_cols[b], r))

    for idx in range(len(axis_pairs), len(axes)):
        axes[idx].axis("off")

    excl_note = f" (excluded: {', '.join(exclude_topics)})" if exclude_topics else ""
    fig.suptitle(
        f"LLM Taste Topics: Independence Check ({n_topics} topics, "
        f"{len(axis_pairs)} pairs){excl_note}\n"
        "(Round cloud = independent / Diagonal cloud = overlapping - no clustering, raw scores only)",
        fontsize=12)
    fig.tight_layout()
    path = out_dir / "llm_topic_correlation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[compare] saved: {path}")

    print(f"\n[compare] LLM 유형 간 상관관계 요약 ({n_topics}개 유형, {len(axis_pairs)}개 쌍):")
    for t1, t2, r in sorted(overlap_summary, key=lambda x: -abs(x[2])):
        flag = " (possible overlap)" if abs(r) > 0.5 else ""
        print(f"  {t1} vs {t2}: r={r:.3f}{flag}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(AXIS_SCORES_PATH)
    feats = pd.read_csv(FEATURE_TABLE_PATH)

    assign_long = parse_llm_assignments(ASSIGNMENT_PATH)
    dense_wide = pivot_to_dense_table(assign_long)
    primary_topic_df = assign_primary_topic(dense_wide, exclude_topics=EXCLUDE_TOPICS)

    print("\n[compare] === LLM 유형 간 독립성 확인 (표) ===")
    corr = compute_llm_topic_correlation_table(dense_wide, OUT_DIR, exclude_topics=EXCLUDE_TOPICS)
    plot_llm_topic_correlation_heatmap(corr, OUT_DIR)

    # LLM으로 100명만 샘플링했으므로, scores/feats(6040명)에서 그 100명만
    # 남기고 비교. raw-feature 클러스터링도 같은 100명 기준으로 다시 계산
    # (behavioral K 전체 6040명이 아니라, LLM과 동일 인원끼리 공정 비교).
    sample_user_ids = primary_topic_df["user_id"].unique()
    scores_sub = scores[scores["user_id"].isin(sample_user_ids)].reset_index(drop=True)
    feats_sub = feats[feats["user_id"].isin(sample_user_ids)].reset_index(drop=True)
    print(f"[compare] 비교 대상 표본: {len(scores_sub)}명 (LLM 파일럿과 동일 인원)")

    n_llm_topics = primary_topic_df["llm_topic"].nunique()

    for method in ["FA", "NMF", "SHAP"]:
        n_axes = infer_n_axes(scores_sub, method)
        if n_axes == 0:
            continue
        print(f"\n[compare] === {method} ===")

        # ① LLM 유형으로 색칠
        plot_axis_space_llm_topic(scores_sub, method, OUT_DIR, primary_topic_df)

        # ② 비교용: 원본 피처로 K-means 클러스터링 (기존 함수 재사용)
        #    클러스터 개수를 LLM 유형 개수와 맞춰서 "같은 개수로 나눴을 때"
        #    공정하게 비교되도록 함.
        raw_cluster_df = cluster_raw_feature_space(feats_sub, n_clusters=n_llm_topics)
        plot_axis_space_raw_clustered(scores_sub, feats_sub, method, OUT_DIR, raw_cluster_df)

    print(f"\n[compare] 완료. 결과 폴더: {OUT_DIR}")
    print("  각 기법(FA/NMF/SHAP)마다 2장씩:")
    print("  - axis_space_llmtopic_{method}.png       (LLM 유형으로 색칠)")
    print("  - axis_space_rawclustered_{method}.png   (원본 피처 K-means로 색칠, 비교용)")


if __name__ == "__main__":
    main()