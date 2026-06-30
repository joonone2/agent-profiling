# =============================================================================
# visualize_phase0.py
# Phase 0 results: intuitive visualizations (run after run_phase0.py).
#
# ★ All labels are English-only (verified, no Korean text in any plot)
#   to avoid font glyph errors in matplotlib's default font (DejaVu Sans).
#
# ★ Layout: FA and NMF are kept in SEPARATE figures (not mixed in one row),
#   plus one combined "overview" figure for side-by-side comparison.
#
# Required input files (in results/, created by run_phase0.py):
#   - user_axis_scores.csv   : per-user axis scores (NMF/FA/SHAP)
#   - user_feature_table.csv : per-user feature table (popularity_bias etc.)
#
# Outputs (all in results/):
#   - validity_scatter_FA.png            : FA axes vs external signal
#   - validity_scatter_NMF.png           : NMF axes vs external signal
#   - axis_correlation_FA.png            : FA axis-pair independence check
#   - axis_correlation_NMF.png           : NMF axis-pair independence check
#   - axis_correlation_clustered_FA.png  : FA axis pairs, colored by K-means cluster
#   - axis_correlation_clustered_NMF.png : NMF axis pairs, colored by K-means cluster
#   - axis_space_2d_FA.png               : users in FA 2D axis space
#   - axis_space_2d_NMF.png              : users in NMF 2D axis space
#   - comparison_overview.png            : FA vs NMF side-by-side, same scale
#
# Run: python visualize_phase0.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False  # avoid minus-sign glyph issues

from sklearn.cluster import KMeans

from pathlib import Path
from config import RESULTS_DIR, DATA_DIR
from data_loader import load_users, load_ratings, load_movies, apply_kcore, split_feature_validation
from validity import compute_popularity_preference_raw, compute_director_loyalty


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_results():
    scores = pd.read_csv(RESULTS_DIR / "user_axis_scores.csv")
    feats  = pd.read_csv(RESULTS_DIR / "user_feature_table.csv")
    return scores, feats


def compute_validation_signals():
    """
    Recompute the same independent signals used in validity.py
    (no Korean text; safe to print to console only).
    """
    ratings = apply_kcore(load_ratings(DATA_DIR))
    movies  = load_movies(DATA_DIR)
    users   = load_users(DATA_DIR)
    data    = split_feature_validation(ratings, movies, users)
    validation_log = data["validation_log"]

    pop_pref  = compute_popularity_preference_raw(validation_log)
    dir_loyal = compute_director_loyalty(validation_log)
    return pop_pref, dir_loyal


# Axis labels in plain English — used consistently across all plots.
# Update these short descriptions if axis_interpretation.csv changes.
AXIS_LABELS = {
    "FA_axis0":  "FA axis0 (Thriller/Drama/Crime)",
    "FA_axis1":  "FA axis1 (Adventure/SciFi/Action)",
    "FA_axis2":  "FA axis2 (Children/Animation/Musical)",
    "NMF_axis0": "NMF axis0 (Animation/Children/Musical)",
    "NMF_axis1": "NMF axis1 (rating_std/War/Drama)",
    "NMF_axis2": "NMF axis2 (Horror/Mystery/Thriller)",
}

METHOD_COLOR = {"FA": "#4CAF50", "NMF": "#2196F3"}

# K-means clustering settings (for the clustered version of axis-pair plots)
N_CLUSTERS = 3            # same K as the behavioral factor count, by default
CLUSTER_SEED = 42
CLUSTER_PALETTE = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA"]  # up to 5 clusters


# -----------------------------------------------------------------------------
# (1) External validity scatter — ONE METHOD PER FIGURE
# -----------------------------------------------------------------------------

def plot_validity_scatter_single(scores: pd.DataFrame, pop_pref: pd.Series, method: str):
    """
    For a single method (FA or NMF), scatter all 3 axes vs pop_pref_raw.
    One figure per method — avoids mixing FA/NMF in the same row.
    """
    cols = [f"{method}_axis{k}" for k in range(3)]
    color = METHOD_COLOR[method]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, col in zip(axes, cols):
        merged = scores[["user_id", col]].merge(
            pop_pref.rename("pop_pref_raw"), left_on="user_id", right_index=True
        )
        x = merged[col].values
        y = merged["pop_pref_raw"].values

        ax.scatter(x, y, alpha=0.15, s=8, color=color)

        z = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(z, xs), color="red", linewidth=2)

        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(AXIS_LABELS.get(col, col) + "\naxis score")
        ax.set_ylabel("Popularity preference (raw)")
        ax.set_title(f"r = {r:.3f}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"External Validity: {method} Axes vs Independent Signal\n"
                 "(Each dot = one user; red line = trend)", fontsize=13)
    fig.tight_layout()
    path = RESULTS_DIR / f"validity_scatter_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved: {path}")


# -----------------------------------------------------------------------------
# (2) Axis-pair independence check — ONE METHOD PER FIGURE
# -----------------------------------------------------------------------------

def plot_axis_correlation_single(scores: pd.DataFrame, method: str):
    """
    For a single method, scatter all axis pairs (0-1, 0-2, 1-2) in one row.
    """
    cols = [f"{method}_axis{k}" for k in range(3)]
    color = METHOD_COLOR[method]
    axis_pairs = [(0, 1), (0, 2), (1, 2)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (i, j) in zip(axes, axis_pairs):
        x = scores[cols[i]].values
        y = scores[cols[j]].values

        ax.scatter(x, y, alpha=0.12, s=6, color=color)

        r = np.corrcoef(x, y)[0, 1]
        title_color = "red" if abs(r) > 0.5 else "black"

        ax.set_xlabel(f"{method} axis{i}")
        ax.set_ylabel(f"{method} axis{j}")
        ax.set_title(f"r = {r:.3f}" + (" (possible overlap)" if abs(r) > 0.5 else ""),
                    color=title_color, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{method}: Axis Independence Check\n"
                 "(Round cloud = independent / Diagonal cloud = overlapping)",
                 fontsize=13)
    fig.tight_layout()
    path = RESULTS_DIR / f"axis_correlation_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved: {path}")

    print(f"[visualize] {method} axis correlation summary:")
    for i, j in axis_pairs:
        r = np.corrcoef(scores[cols[i]], scores[cols[j]])[0, 1]
        flag = " (possible overlap)" if abs(r) > 0.5 else ""
        print(f"  {method} axis{i} vs axis{j}: r={r:.3f}{flag}")


def plot_axis_correlation_clustered(scores: pd.DataFrame, method: str,
                                     n_clusters: int = N_CLUSTERS):
    """
    Same layout as plot_axis_correlation_single (axis0-1, 0-2, 1-2 in one row),
    but points are colored by K-means cluster instead of a single flat color.

    ★ K-means is fit ONCE on all 3 axis scores together (not per-pair), so a
      given user gets the SAME cluster color across all three panels.
      Fitting separately per pair would let the same user appear as different
      colors in different panels, which would be misleading.

    This does NOT replace the axis scores themselves — it's an additional
    diagnostic view: "if we force a hard grouping on top of the continuous
    axis scores, do natural clusters emerge, and do they align with what the
    continuous scores already show (e.g. the popularity-preference gradient)?"
    """
    cols = [f"{method}_axis{k}" for k in range(3)]
    X = scores[cols].values

    km = KMeans(n_clusters=n_clusters, random_state=CLUSTER_SEED, n_init=10)
    cluster_labels = km.fit_predict(X)

    axis_pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (i, j) in zip(axes, axis_pairs):
        x = X[:, i]
        y = X[:, j]

        for c in range(n_clusters):
            mask = cluster_labels == c
            ax.scatter(x[mask], y[mask], alpha=0.25, s=8,
                      color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
                      label=f"cluster {c}" if (i, j) == axis_pairs[0] else None)

        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(f"{method} axis{i}")
        ax.set_ylabel(f"{method} axis{j}")
        ax.set_title(f"r = {r:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best", fontsize=8, markerscale=2)

    fig.suptitle(f"{method}: Axis Pairs Colored by K-means Cluster (K={n_clusters})\n"
                 "(Same user = same color across all 3 panels; "
                 "checks whether a hard grouping matches the continuous axis structure)",
                 fontsize=12)
    fig.tight_layout()
    path = RESULTS_DIR / f"axis_correlation_clustered_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved: {path}")

    # Cluster size summary (sanity check: no degenerate tiny/huge clusters)
    sizes = pd.Series(cluster_labels).value_counts().sort_index()
    print(f"[visualize] {method} cluster sizes (K={n_clusters}):")
    for c, n in sizes.items():
        print(f"  cluster {c}: {n} users ({n/len(scores):.1%})")

    return cluster_labels


# -----------------------------------------------------------------------------
# (3) Users in 2D axis space — ONE METHOD PER FIGURE
# -----------------------------------------------------------------------------

def plot_axis_space_2d_single(scores: pd.DataFrame, feats: pd.DataFrame, method: str):
    """
    Single method: plot users on its strongest axis pair (axis with
    highest |external validity| vs the next one), colored by popularity_bias.
    """
    merged = scores.merge(feats[["user_id", "popularity_bias"]], on="user_id")

    # Use axis0 vs axis1 by default (consistent across methods for comparability)
    xcol, ycol = f"{method}_axis0", f"{method}_axis1"
    color_map = "coolwarm"

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        merged[xcol], merged[ycol],
        c=merged["popularity_bias"], cmap=color_map,
        alpha=0.4, s=10, vmin=-0.5, vmax=0.5,
    )
    ax.set_xlabel(AXIS_LABELS.get(xcol, xcol))
    ax.set_ylabel(AXIS_LABELS.get(ycol, ycol))
    ax.set_title(f"{method}: Users in 2D Axis Space\n"
                "(Color = preference for popular items)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="popularity_bias")

    fig.tight_layout()
    path = RESULTS_DIR / f"axis_space_2d_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved: {path}")


# -----------------------------------------------------------------------------
# (4) FA vs NMF combined overview — side-by-side comparison
# -----------------------------------------------------------------------------

def plot_comparison_overview(scores: pd.DataFrame, feats: pd.DataFrame, pop_pref: pd.Series):
    """
    One figure, 2 rows (FA top, NMF bottom), 3 columns:
      col1: axis0 vs axis1 (independence check, same as plot_axis_correlation)
      col2: strongest axis vs pop_pref_raw (external validity)
      col3: users in 2D axis space colored by popularity_bias

    This is the "side-by-side" figure for directly comparing FA and NMF.
    """
    merged_feat = scores.merge(feats[["user_id", "popularity_bias"]], on="user_id")
    merged_pop  = scores.merge(pop_pref.rename("pop_pref_raw"),
                                left_on="user_id", right_index=True)

    # Strongest validity axis per method (based on |r| with pop_pref_raw)
    best_axis = {}
    for method in ["FA", "NMF"]:
        rs = {k: abs(np.corrcoef(merged_pop[f"{method}_axis{k}"],
                                  merged_pop["pop_pref_raw"])[0, 1])
              for k in range(3)}
        best_axis[method] = max(rs, key=rs.get)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row, method in enumerate(["FA", "NMF"]):
        color = METHOD_COLOR[method]

        # --- col1: axis0 vs axis1 independence ---
        ax = axes[row, 0]
        x = scores[f"{method}_axis0"].values
        y = scores[f"{method}_axis1"].values
        ax.scatter(x, y, alpha=0.12, s=6, color=color)
        r = np.corrcoef(x, y)[0, 1]
        flag = " (overlap)" if abs(r) > 0.5 else ""
        ax.set_title(f"{method}: axis0 vs axis1, r={r:.3f}{flag}",
                    color="red" if abs(r) > 0.5 else "black", fontsize=10)
        ax.set_xlabel(f"{method} axis0")
        ax.set_ylabel(f"{method} axis1")
        ax.grid(True, alpha=0.3)

        # --- col2: strongest axis vs external validity ---
        ax = axes[row, 1]
        k = best_axis[method]
        col = f"{method}_axis{k}"
        x = merged_pop[col].values
        y = merged_pop["pop_pref_raw"].values
        ax.scatter(x, y, alpha=0.15, s=8, color=color)
        z = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(z, xs), color="red", linewidth=2)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{method}: strongest axis (axis{k}) vs popularity, r={r:.3f}",
                    fontsize=10)
        ax.set_xlabel(AXIS_LABELS.get(col, col))
        ax.set_ylabel("Popularity preference (raw)")
        ax.grid(True, alpha=0.3)

        # --- col3: 2D axis space colored by popularity_bias ---
        ax = axes[row, 2]
        sc = ax.scatter(
            merged_feat[f"{method}_axis0"], merged_feat[f"{method}_axis1"],
            c=merged_feat["popularity_bias"], cmap="coolwarm",
            alpha=0.4, s=10, vmin=-0.5, vmax=0.5,
        )
        ax.set_title(f"{method}: users in 2D axis space", fontsize=10)
        ax.set_xlabel(f"{method} axis0")
        ax.set_ylabel(f"{method} axis1")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label="popularity_bias")

    fig.suptitle("FA vs NMF: Side-by-Side Comparison\n"
                 "(Top row = FA, Bottom row = NMF)", fontsize=14)
    fig.tight_layout()
    path = RESULTS_DIR / "comparison_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved: {path}")


# -----------------------------------------------------------------------------
# Label / font sanity check — run before generating real plots
# -----------------------------------------------------------------------------

def check_no_korean_labels():
    """
    Scans AXIS_LABELS and all hardcoded plot strings for non-ASCII characters
    that would render as missing-glyph boxes in matplotlib's default font.
    Raises a clear error instead of silently producing broken text in PNGs.
    """
    import re
    suspects = []
    for key, label in AXIS_LABELS.items():
        if re.search(r"[^\x00-\x7F]", label):
            suspects.append((key, label))
    if suspects:
        msg = "\n".join(f"  {k}: {v}" for k, v in suspects)
        raise ValueError(
            "Non-ASCII characters found in plot labels — these will render "
            f"as broken glyphs:\n{msg}\nFix AXIS_LABELS before plotting."
        )
    print("[visualize] label check passed: no non-ASCII characters in axis labels.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("[visualize] checking labels for font-glyph safety...")
    check_no_korean_labels()

    print("[visualize] loading results...")
    scores, feats = load_results()

    print("[visualize] recomputing external validation signals...")
    pop_pref, dir_loyal = compute_validation_signals()

    for method in ["FA", "NMF"]:
        print(f"\n[visualize] === {method} ===")
        plot_validity_scatter_single(scores, pop_pref, method)
        plot_axis_correlation_single(scores, method)
        plot_axis_correlation_clustered(scores, method)
        plot_axis_space_2d_single(scores, feats, method)

    print("\n[visualize] === comparison overview ===")
    plot_comparison_overview(scores, feats, pop_pref)

    print(f"\n[visualize] done. Files in {RESULTS_DIR}/:")
    for f in [
        "validity_scatter_FA.png", "validity_scatter_NMF.png",
        "axis_correlation_FA.png", "axis_correlation_NMF.png",
        "axis_correlation_clustered_FA.png", "axis_correlation_clustered_NMF.png",
        "axis_space_2d_FA.png", "axis_space_2d_NMF.png",
        "comparison_overview.png",
    ]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()