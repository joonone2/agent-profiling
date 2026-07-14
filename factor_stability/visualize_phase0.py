# =============================================================================
# visualize_phase0.py
# Phase 0 results: intuitive visualizations.
# Works for BOTH a single K=3 run (run_phase0.py) AND any K from the
# K sweep (run_k_sweep.py) — pass --k to choose which folder to read.
#
# ★ All labels are English-only (verified, no Korean text in any plot)
#   to avoid font glyph errors in matplotlib's default font (DejaVu Sans).
#
# ★ Layout: FA / NMF / SHAP are kept in SEPARATE figures (never mixed in
#   one row).
#
# ★ K-awareness: every plotting function takes the actual number of axes
#   (n_axes) instead of hardcoding range(3), and reads/writes to a
#   K-specific folder so results from different K values never overwrite
#   each other.
#
# ★ Plots included, per K folder:
#   1) axis_correlation_{method}           : axis-pair independence check
#                                             (ALL pairs, not just 0-1)
#   2) axis_correlation_clustered_{method} : same, colored by K-means cluster
#                                             (cluster count == behavioral K)
#   3) axis_space_{method}                 : ALL axis pairs (not just 0-1),
#                                             colored by popularity_bias,
#                                             each panel's title shows r AND
#                                             whether that axis circularly
#                                             depends on popularity_bias as
#                                             an input feature (see
#                                             check_popularity_bias_validity)
#   Methods covered: FA, NMF, SHAP (SHAP axes are importance-based soft
#   groupings, not true loadings — this is noted in the plot title).
#
#   Cross-K plot (saved once, into results/k_sweep/, not per-K folder):
#   4) k_sweep_summary.png : stability and external-validity-significant-axis
#                             count vs K, one line per method. Answers
#                             "what happens as I increase K?"
#
# Usage:
#   python visualize_phase0.py            # uses config.K (single run, e.g. K=3)
#   python visualize_phase0.py --k 5       # reads results/k_sweep/K5/
#   python visualize_phase0.py --all-k     # runs for every value in K_SWEEP_VALUES
#                                           # AND generates the cross-K summary
#
# Required input files (created by run_phase0.py or run_k_sweep.py):
#   - user_axis_scores.csv    : per-user axis scores (NMF/FA/SHAP)
#   - user_feature_table.csv  : per-user feature table (popularity_bias etc.)
#   - axis_interpretation.csv : (optional) used only for the circularity
#                                check described above; plots still work
#                                without it, just skip that diagnostic.
#   - results/k_sweep/k_sweep_summary.csv : (optional) needed only for the
#                                cross-K summary plot.
# =============================================================================

import argparse
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False  # avoid minus-sign glyph issues

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pathlib import Path
from config import (
    RESULTS_DIR, DATA_DIR, K as DEFAULT_K, K_SWEEP_VALUES, K_SWEEP_DIR,
    get_k_results_dir, FNAME_INTERPRET,
)


# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------

def resolve_io_dir(k: int) -> Path:
    """
    Always prefer the k_sweep folder (results/k_sweep/K{k}/), since that's
    where run_k_sweep.py saves every K including K==DEFAULT_K.

    Fallback: if results/k_sweep/K{k}/ doesn't have the data but results/
    (the root, used by the older single-run run_phase0.py) does AND k
    matches DEFAULT_K, use that instead.
    """
    sweep_dir = get_k_results_dir(k)
    if (sweep_dir / "user_axis_scores.csv").exists():
        return sweep_dir
    if k == DEFAULT_K and (RESULTS_DIR / "user_axis_scores.csv").exists():
        return RESULTS_DIR
    return sweep_dir


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_results(io_dir: Path):
    scores = pd.read_csv(io_dir / "user_axis_scores.csv")
    feats  = pd.read_csv(io_dir / "user_feature_table.csv")
    return scores, feats


def infer_n_axes(scores: pd.DataFrame, method: str) -> int:
    """
    Count how many axis columns this method actually has in the scores file
    (e.g. 'NMF_axis0'..'NMF_axis4' for K=5), instead of assuming a fixed K.
    """
    cols = [c for c in scores.columns if c.startswith(f"{method}_axis")]
    return len(cols)


METHOD_COLOR = {"FA": "#4CAF50", "NMF": "#2196F3", "SHAP": "#FB8C00"}

N_CLUSTERS = 3   # only used if you explicitly pass n_clusters=N_CLUSTERS;
                 # default behavior follows behavioral K instead (see below)
CLUSTER_SEED = 42
CLUSTER_PALETTE = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
                    "#00ACC1", "#FDD835", "#6D4C41", "#5E35B1", "#43A047"]

# Above this behavioral K, the K-means-colored plot is skipped: with many
# axes, forcing that many hard clusters onto a continuous mixture becomes
# hard to read and interpret, so it stops being informative.
CLUSTER_MAX_K = 5


def axis_label(method: str, k: int) -> str:
    return f"{method} axis{k}"


def _all_axis_pairs(n_axes: int):
    return [(i, j) for i in range(n_axes) for j in range(i + 1, n_axes)]


# -----------------------------------------------------------------------------
# popularity_bias appropriateness check
#
# popularity_bias is one of the ~19 INPUT features fed into NMF/FA (it is a
# column in user_feature_table.csv that was used to build the axes). So if
# an axis correlates with popularity_bias, that can mean two very different
# things:
#   (a) CIRCULAR  — the axis's loading already weights popularity_bias
#                   heavily as a top input feature, so the axis is *partly
#                   built from* popularity_bias. A high correlation here is
#                   expected almost by construction and is weak evidence.
#   (b) EMERGENT   — popularity_bias is NOT a top feature for that axis, yet
#                   the axis score still correlates with it. This is a much
#                   stronger, non-trivial signal: an axis built from other
#                   features (genres, rating stats) ended up tracking
#                   popularity preference anyway.
# This function checks both, per axis, using axis_interpretation.csv if
# available (falls back gracefully — circularity flag becomes "unknown" —
# if that file is missing, e.g. user only has user_axis_scores.csv).
# -----------------------------------------------------------------------------

def _parse_top_features(raw: str):
    """axis_interpretation.csv stores top_features as a stringified list of
    (feature_name, loading_value) tuples, e.g. "[('Action', 0.82), ...]".
    Parse it back with ast.literal_eval (safe — no eval())."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []


def check_popularity_bias_validity(scores: pd.DataFrame, feats: pd.DataFrame,
                                    method: str, io_dir: Path,
                                    circular_top_n: int = 5):
    """
    For every axis of `method`, compute:
      - r: correlation between the axis score and the popularity_bias feature
      - circular: True if popularity_bias is among that axis's top
        `circular_top_n` loading features (per axis_interpretation.csv).
        None if axis_interpretation.csv isn't available for this K/method.

    Returns a dict {axis_idx: {"r": float, "circular": bool or None}} and
    also prints a readable report.
    """
    n_axes = infer_n_axes(scores, method)
    merged = scores.merge(feats[["user_id", "popularity_bias"]], on="user_id")

    interp_path = io_dir / FNAME_INTERPRET
    interp_df = None
    if interp_path.exists():
        try:
            interp_df = pd.read_csv(interp_path)
        except Exception as e:
            print(f"[visualize] WARNING: could not read {interp_path}: {e}")

    report = {}
    print(f"\n[visualize] popularity_bias appropriateness check — {method} (K={n_axes}):")
    print(f"  (CIRCULAR = popularity_bias is a top-{circular_top_n} input feature for "
          f"that axis -> correlation is expected, weak evidence.")
    print(f"   EMERGENT = popularity_bias is NOT a top input feature, yet the axis "
          f"still correlates with it -> stronger, non-trivial evidence.)")

    for k_ in range(n_axes):
        col = f"{method}_axis{k_}"
        r = np.corrcoef(merged[col], merged["popularity_bias"])[0, 1]

        circular = None
        if interp_df is not None:
            row = interp_df[(interp_df["method"] == method) &
                             (interp_df["axis_idx"] == k_)]
            if len(row) > 0:
                top_feats = _parse_top_features(row.iloc[0]["top_features"])
                top_names = [f for f, _ in top_feats[:circular_top_n]]
                circular = "popularity_bias" in top_names

        if circular is True:
            tag = "CIRCULAR (popularity_bias is a top input feature)"
        elif circular is False:
            tag = "EMERGENT (popularity_bias is NOT a top input feature)"
        else:
            tag = "UNKNOWN (axis_interpretation.csv not found for this K)"

        flag = ""
        if abs(r) < 0.1:
            flag = " — weak/no relationship; color gradient will look flat for this axis"
        print(f"  axis{k_}: r={r:+.3f}  [{tag}]{flag}")

        report[k_] = {"r": r, "circular": circular}

    return report


# -----------------------------------------------------------------------------
# (1) Axis-pair independence check
# -----------------------------------------------------------------------------

def plot_axis_correlation_single(scores: pd.DataFrame, method: str, out_dir: Path):
    """
    For a single method, scatter ALL axis pairs (K-aware: e.g. K=5 -> 10 pairs).
    """
    n_axes = infer_n_axes(scores, method)
    cols = [f"{method}_axis{k}" for k in range(n_axes)]
    color = METHOD_COLOR.get(method, "#757575")
    axis_pairs = _all_axis_pairs(n_axes)

    n_cols = min(len(axis_pairs), 5)
    n_rows = int(np.ceil(len(axis_pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows))
    axes = np.array(axes).reshape(-1)

    overlap_summary = []
    for idx, (i, j) in enumerate(axis_pairs):
        ax = axes[idx]
        x = scores[cols[i]].values
        y = scores[cols[j]].values

        ax.scatter(x, y, alpha=0.12, s=6, color=color)
        r = np.corrcoef(x, y)[0, 1]
        title_color = "red" if abs(r) > 0.5 else "black"

        ax.set_xlabel(f"{method} axis{i}")
        ax.set_ylabel(f"{method} axis{j}")
        ax.set_title(f"r = {r:.3f}" + (" (overlap)" if abs(r) > 0.5 else ""),
                    color=title_color, fontsize=10)
        ax.grid(True, alpha=0.3)
        overlap_summary.append((i, j, r))

    for idx in range(len(axis_pairs), len(axes)):
        axes[idx].axis("off")

    note = ""
    if method == "SHAP":
        note = "\n(SHAP axes are importance-based soft groupings, not true loadings)"
    fig.suptitle(f"{method}: Axis Independence Check (K={n_axes})\n"
                 "(Round cloud = independent / Diagonal cloud = overlapping)" + note,
                 fontsize=13)
    fig.tight_layout()
    path = out_dir / f"axis_correlation_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved: {path}")

    print(f"[visualize] {method} axis correlation summary (K={n_axes}):")
    for i, j, r in overlap_summary:
        flag = " (possible overlap)" if abs(r) > 0.5 else ""
        print(f"  {method} axis{i} vs axis{j}: r={r:.3f}{flag}")


def plot_axis_correlation_clustered(scores: pd.DataFrame, method: str, out_dir: Path,
                                     n_clusters: int = None):
    """
    Same layout as plot_axis_correlation_single, but points are colored by
    K-means cluster (fit once on ALL axes together, so a user has the same
    color in every panel).

    ★ n_clusters=None (default): cluster K matches behavioral K (n_axes).
    """
    n_axes = infer_n_axes(scores, method)
    if n_clusters is None:
        n_clusters = n_axes
    cols = [f"{method}_axis{k}" for k in range(n_axes)]
    X = scores[cols].values

    km = KMeans(n_clusters=n_clusters, random_state=CLUSTER_SEED, n_init=10)
    cluster_labels = km.fit_predict(X)

    axis_pairs = _all_axis_pairs(n_axes)
    n_cols = min(len(axis_pairs), 5)
    n_rows = int(np.ceil(len(axis_pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, (i, j) in enumerate(axis_pairs):
        ax = axes[idx]
        x, y = X[:, i], X[:, j]
        for c in range(n_clusters):
            mask = cluster_labels == c
            ax.scatter(x[mask], y[mask], alpha=0.25, s=8,
                      color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
                      label=f"cluster {c}" if idx == 0 else None)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(f"{method} axis{i}")
        ax.set_ylabel(f"{method} axis{j}")
        ax.set_title(f"r = {r:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    for idx in range(len(axis_pairs), len(axes)):
        axes[idx].axis("off")

    axes[0].legend(loc="best", fontsize=8, markerscale=2)

    fig.suptitle(f"{method}: Axis Pairs Colored by K-means Cluster "
                 f"(behavioral K={n_axes}, cluster K={n_clusters})\n"
                 "(Same user = same color across all panels)", fontsize=12)
    fig.tight_layout()
    path = out_dir / f"axis_correlation_clustered_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved: {path}")

    sizes = pd.Series(cluster_labels).value_counts().sort_index()
    print(f"[visualize] {method} cluster sizes (cluster K={n_clusters}):")
    for c, n in sizes.items():
        print(f"  cluster {c}: {n} users ({n/len(scores):.1%})")

    return cluster_labels


# -----------------------------------------------------------------------------
# (1.5) Cluster in RAW feature space (before NMF/FA), then color axis space
#       with those labels — an independent check requested separately from
#       plot_axis_correlation_clustered (which clusters AFTER compression,
#       inside the axis space itself).
#
# Why this is a stronger check than clustering the axis space directly:
#   Clustering the compressed axis space and then confirming it looks
#   clustered is somewhat circular (NMF/FA already optimized for a
#   K-dimensional summary). Clustering the RAW, uncompressed features
#   first — with zero knowledge of NMF/FA — and then checking whether
#   that independent grouping still separates cleanly in axis space is a
#   more independent test of whether the compression preserved real
#   structure in the original data.
#
# Missing-value handling: raw genre averages contain NaN for genres a user
# never rated. We use listwise deletion here (drop any user with a single
# NaN in any of the 19 raw features) rather than mean imputation.
# Verified on real K=3 data: this drops the sample from 6040 -> 3108 users
# (~48.5% removed). The remaining users are NOT a random subsample — they
# skew toward people who happened to watch many different genres — so
# this raw-space clustering should be read as a check on that subpopulation,
# not the full user base. Mean imputation was considered and rejected: it
# would inject artificial homogeneity into exactly the columns (rare
# genres) most affected by missingness, the same problem this project
# has avoided since the very first feature-table design.
# -----------------------------------------------------------------------------

def cluster_raw_feature_space(feats: pd.DataFrame, n_clusters: int,
                               seed: int = CLUSTER_SEED):
    """
    K-means on the ORIGINAL (pre-NMF/FA) feature table.
    Uses listwise deletion (drop any user with a missing value in any raw
    feature) rather than imputation — see caveat above. Returns a
    DataFrame with columns [user_id, raw_cluster] covering only the
    surviving (complete-case) users.
    """
    feature_cols = [c for c in feats.columns if c != "user_id"]
    n_total = len(feats)

    complete = feats.dropna(subset=feature_cols)
    n_kept = len(complete)
    print(f"[visualize] raw-feature-space clustering: listwise deletion kept "
          f"{n_kept}/{n_total} users ({n_kept/n_total:.1%}) with no missing "
          f"values across all {len(feature_cols)} raw features.")

    X_scaled = StandardScaler().fit_transform(complete[feature_cols].values)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    raw_cluster = km.fit_predict(X_scaled)

    out = pd.DataFrame({"user_id": complete["user_id"].values, "raw_cluster": raw_cluster})
    sizes = out["raw_cluster"].value_counts().sort_index()
    for c, n in sizes.items():
        print(f"  raw_cluster {c}: {n} users ({n/len(out):.1%} of complete-case subsample)")
    return out


def plot_axis_space_raw_clustered(scores: pd.DataFrame, feats: pd.DataFrame,
                                   method: str, out_dir: Path,
                                   raw_cluster_df: pd.DataFrame):
    """
    Colors the axis-space scatter (all axis pairs) using cluster labels
    computed from the RAW, pre-compression feature table (see
    cluster_raw_feature_space). Same panel layout as
    plot_axis_correlation_clustered, but the coloring answers a different
    question: "does a grouping found in the ORIGINAL data still separate
    cleanly after compression into behavioral axes?" — a check of whether
    NMF/FA preserved real structure, rather than whether the axes
    themselves cluster into groups.
    """
    n_axes = infer_n_axes(scores, method)
    n_clusters = raw_cluster_df["raw_cluster"].nunique()
    cols = [f"{method}_axis{k}" for k in range(n_axes)]

    merged = scores.merge(raw_cluster_df, on="user_id")  # inner join: complete-case users only
    n_shown = len(merged)
    n_total = len(scores)
    X = merged[cols].values
    cluster_labels = merged["raw_cluster"].values

    axis_pairs = _all_axis_pairs(n_axes)
    n_cols = min(len(axis_pairs), 5)
    n_rows = int(np.ceil(len(axis_pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, (i, j) in enumerate(axis_pairs):
        ax = axes[idx]
        x, y = X[:, i], X[:, j]
        for c in range(n_clusters):
            mask = cluster_labels == c
            ax.scatter(x[mask], y[mask], alpha=0.25, s=8,
                      color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
                      label=f"raw_cluster {c}" if idx == 0 else None)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(f"{method} axis{i}")
        ax.set_ylabel(f"{method} axis{j}")
        ax.set_title(f"r = {r:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    for idx in range(len(axis_pairs), len(axes)):
        axes[idx].axis("off")

    axes[0].legend(loc="best", fontsize=8, markerscale=2)

    fig.suptitle(
        f"{method}: Axis Space Colored by RAW-Feature-Space Cluster "
        f"(K={n_axes}, raw_cluster K={n_clusters})\n"
        f"(Clusters found BEFORE compression, in the original 19 features, "
        f"listwise deletion: n={n_shown}/{n_total} users shown)", fontsize=12)
    fig.tight_layout()
    path = out_dir / f"axis_space_rawclustered_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved: {path}")


# -----------------------------------------------------------------------------
# (2) Users in 2D axis space — NOW ALL PAIRS, with circularity-aware titles
# -----------------------------------------------------------------------------

def plot_axis_space_all_pairs(scores: pd.DataFrame, feats: pd.DataFrame,
                               method: str, out_dir: Path,
                               validity_report: dict):
    """
    For a single method, plot ALL axis pairs (not just axis0 vs axis1),
    colored by popularity_bias. Each panel's title shows the correlation of
    BOTH axes in that panel with popularity_bias, plus a circularity tag
    (C=circular, E=emergent, ?=unknown) from check_popularity_bias_validity.

    A flat/no-gradient panel is expected and OK if neither axis correlates
    with popularity_bias — that's informative too (this axis pair encodes
    something other than popularity preference).
    """
    n_axes = infer_n_axes(scores, method)
    if n_axes < 2:
        print(f"[visualize] skip {method} axis space: only {n_axes} axis available")
        return

    merged = scores.merge(feats[["user_id", "popularity_bias"]], on="user_id")
    axis_pairs = _all_axis_pairs(n_axes)

    n_cols = min(len(axis_pairs), 5)
    n_rows = int(np.ceil(len(axis_pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.3 * n_rows))
    axes = np.array(axes).reshape(-1)

    def tag(k_):
        c = validity_report.get(k_, {}).get("circular")
        return "C" if c is True else ("E" if c is False else "?")

    for idx, (i, j) in enumerate(axis_pairs):
        ax = axes[idx]
        sc = ax.scatter(
            merged[f"{method}_axis{i}"], merged[f"{method}_axis{j}"],
            c=merged["popularity_bias"], cmap="coolwarm",
            alpha=0.35, s=8, vmin=-0.5, vmax=0.5,
        )
        ri = validity_report.get(i, {}).get("r", float("nan"))
        rj = validity_report.get(j, {}).get("r", float("nan"))
        ax.set_xlabel(f"{method} axis{i} (r={ri:+.2f}, {tag(i)})")
        ax.set_ylabel(f"{method} axis{j} (r={rj:+.2f}, {tag(j)})")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label="popularity_bias", fraction=0.046, pad=0.04)

    for idx in range(len(axis_pairs), len(axes)):
        axes[idx].axis("off")

    note = ""
    if method == "SHAP":
        note = "\n(SHAP axes are importance-based soft groupings, not true loadings)"
    fig.suptitle(
        f"{method}: Users in Axis Space, All Pairs (K={n_axes})\n"
        "(Color = popularity_bias; r/tag in axis labels: "
        "C=circular [popularity_bias is a top input feature for that axis], "
        "E=emergent [it is not, yet still correlates], ?=unknown)" + note,
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / f"axis_space_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved: {path}")


# -----------------------------------------------------------------------------
# (3) Cross-K summary — how do things change as K varies?
# -----------------------------------------------------------------------------

def plot_k_sweep_summary():
    """
    Reads results/k_sweep/k_sweep_summary.csv (written by run_k_sweep.py)
    and plots, per method (NMF/FA/SHAP), how stability and the count of
    significant external-validity axes change as K increases.

    Saved once into results/k_sweep/ (not into a per-K folder, since this
    plot spans all K values).
    """
    summary_path = K_SWEEP_DIR / "k_sweep_summary.csv"
    if not summary_path.exists():
        print(f"[visualize] SKIP k_sweep_summary plot: {summary_path} not found. "
              f"Run run_k_sweep.py first.")
        return

    df = pd.read_csv(summary_path)
    methods = df["method"].unique().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for m in methods:
        sub = df[df["method"] == m].sort_values("k")
        color = METHOD_COLOR.get(m, "#757575")
        ax.errorbar(sub["k"], sub["stability_mean"], yerr=sub["stability_std"],
                    marker="o", label=m, color=color, capsize=3)
    ax.axhline(0.85, color="red", linestyle="--", linewidth=1, label="GO threshold (0.85)")
    ax.set_xlabel("K (number of behavioral axes)")
    ax.set_ylabel("Stability (Tucker congruence / Spearman)")
    ax.set_title("Stability vs K")
    ax.set_xticks(sorted(df["k"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for m in methods:
        sub = df[df["method"] == m].sort_values("k")
        color = METHOD_COLOR.get(m, "#757575")
        ax.plot(sub["k"], sub["valid_significant"], marker="o", label=m, color=color)
    ax.set_xlabel("K (number of behavioral axes)")
    ax.set_ylabel("# axes with significant external validity")
    ax.set_title("External Validity Coverage vs K")
    ax.set_xticks(sorted(df["k"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("How Results Change as K Increases", fontsize=14)
    fig.tight_layout()
    path = K_SWEEP_DIR / "k_sweep_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved: {path}")


# -----------------------------------------------------------------------------
# Label / font sanity check
# -----------------------------------------------------------------------------

def check_no_korean_labels():
    samples = [axis_label("FA", 0), axis_label("NMF", 9), axis_label("SHAP", 0),
               "FA: Axis Independence Check (K=5)",
               "Users in Axis Space, All Pairs"]
    bad = [s for s in samples if re.search(r"[^\x00-\x7F]", s)]
    if bad:
        raise ValueError(f"Non-ASCII characters found in sample labels: {bad}")
    print("[visualize] label check passed: sample labels are pure ASCII.")


# -----------------------------------------------------------------------------
# Per-K runner
# -----------------------------------------------------------------------------

def run_for_k(k: int):
    io_dir = resolve_io_dir(k)
    print(f"\n{'='*60}")
    print(f"VISUALIZE — K = {k}  (reading/writing: {io_dir})")
    print(f"{'='*60}")

    if not (io_dir / "user_axis_scores.csv").exists():
        print(f"[visualize] SKIP K={k}: {io_dir}/user_axis_scores.csv not found. "
              f"Run run_phase0.py (for K={DEFAULT_K}) or run_k_sweep.py first.")
        return

    scores, feats = load_results(io_dir)

    # Raw-feature-space clustering is method-independent (same 19 raw
    # features regardless of FA/NMF/SHAP), but cluster count must match
    # behavioral K, so compute once per K value using any method's n_axes
    # (they're all equal to k for NMF/FA; SHAP's axis count also matches k
    # by construction). Only meaningful for K <= CLUSTER_MAX_K (see note).
    raw_cluster_df = None
    if k <= CLUSTER_MAX_K:
        print(f"\n[visualize][K={k}] === raw feature space clustering ===")
        raw_cluster_df = cluster_raw_feature_space(feats, n_clusters=k)
    else:
        print(f"\n[visualize][K={k}] skip raw-feature-space clustering: "
              f"K={k} > {CLUSTER_MAX_K}")

    saved_files = []
    for method in ["FA", "NMF", "SHAP"]:
        n_axes = infer_n_axes(scores, method)
        if n_axes == 0:
            print(f"\n[visualize][K={k}] === {method} === SKIP: no axis columns found")
            continue

        print(f"\n[visualize][K={k}] === {method} ===")
        plot_axis_correlation_single(scores, method, io_dir)

        # K-means-colored plot only for small K (see CLUSTER_MAX_K note).
        if n_axes <= CLUSTER_MAX_K:
            plot_axis_correlation_clustered(scores, method, io_dir)
            cluster_file = [f"axis_correlation_clustered_{method}.png"]
        else:
            print(f"[visualize] skip clustered plot for {method}: "
                  f"behavioral K={n_axes} > {CLUSTER_MAX_K} (too many hard clusters "
                  f"to be informative)")
            cluster_file = []

        raw_cluster_file = []
        if raw_cluster_df is not None:
            plot_axis_space_raw_clustered(scores, feats, method, io_dir, raw_cluster_df)
            raw_cluster_file = [f"axis_space_rawclustered_{method}.png"]

        validity_report = check_popularity_bias_validity(scores, feats, method, io_dir)
        plot_axis_space_all_pairs(scores, feats, method, io_dir, validity_report)

        saved_files += [
            f"axis_correlation_{method}.png",
        ] + cluster_file + raw_cluster_file + [
            f"axis_space_{method}.png",
        ]

    print(f"\n[visualize][K={k}] done. Files in {io_dir}/:")
    for f in saved_files:
        print(f"  - {f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 0 visualization (K-aware)")
    parser.add_argument("--k", type=int, default=None,
                        help=f"Which K to visualize. Default: config.K ({DEFAULT_K}).")
    parser.add_argument("--all-k", action="store_true",
                        help=f"Visualize every K in K_SWEEP_VALUES ({K_SWEEP_VALUES}) "
                             f"plus the default K ({DEFAULT_K}), and also generate "
                             f"the cross-K summary plot.")
    args = parser.parse_args()

    print("[visualize] checking labels for font-glyph safety...")
    check_no_korean_labels()

    if args.all_k:
        all_ks = sorted(set([DEFAULT_K] + list(K_SWEEP_VALUES)))
        for k in all_ks:
            run_for_k(k)
        print(f"\n{'='*60}")
        print("CROSS-K SUMMARY")
        print(f"{'='*60}")
        plot_k_sweep_summary()
    else:
        k = args.k if args.k is not None else DEFAULT_K
        run_for_k(k)


if __name__ == "__main__":
    main()
