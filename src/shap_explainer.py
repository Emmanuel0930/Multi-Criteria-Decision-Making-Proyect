"""
shap_explainer.py
=================
Provides SHAP-compatible explainability for the MCDA model using
**permutation-based feature attribution** – a model-agnostic technique that
mirrors the interpretation of SHAP values without requiring the ``shap``
package.

Two levels of explanation are computed:

Global explanations
    How much does each criterion contribute to suitability scores across
    *all* hexagonal cells? (analogous to SHAP global feature importance)

Local explanations
    For a *specific* cell, how much does each criterion push the score
    above or below the average? (analogous to SHAP force plot values)

Mathematical basis
------------------
For a WLC model the local explanation is exact and trivial:

    φ_j(i) = w_j · c_{ij} − w_j · mean(c_j)

where:
    φ_j(i)       – contribution of feature j to cell i's score deviation
    w_j           – weight of criterion j
    c_{ij}        – normalised criterion value for cell i
    mean(c_j)     – global mean of criterion j

This decomposition satisfies the efficiency axiom (contributions sum to
the score deviation from the baseline) and can be computed without any
additional model calls – making it far more efficient than sampling-based
SHAP for large grids.

We also implement a **permutation importance** for the RF model as a
model-agnostic global explanation.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# ---------------------------------------------------------------------------
# Local SHAP-like explanations (WLC)
# ---------------------------------------------------------------------------

def compute_local_shap(
    df: pd.DataFrame,
    weights: Dict[str, float],
    norm_features: List[str],
    score_column: str = "suitability_score",
) -> pd.DataFrame:
    """
    Compute local SHAP-like contribution values for every cell.

    For each cell i and criterion j:
        φ_j(i) = w_j · (c_{ij} - mean_j)

    Sum over j gives the deviation of cell i's score from the baseline
    (average score).

    Parameters
    ----------
    df            : scored DataFrame (output of ``compute_wlc_scores``)
    weights       : dict of feature → weight
    norm_features : ordered list of normalised feature column names
    score_column  : name of the suitability score column

    Returns
    -------
    DataFrame with ``shap_<feature>`` columns and a ``shap_sum`` column
    (should equal score - mean_score within rounding).
    """
    total_w = sum(weights[f] for f in norm_features)
    X = df[norm_features].to_numpy(dtype=float)

    # Global means
    means = X.mean(axis=0)

    result = df.copy()
    shap_cols = []

    for j, feat in enumerate(norm_features):
        w_j = weights[feat] / total_w
        phi = w_j * (X[:, j] - means[j])
        col = f"shap_{feat}"
        result[col] = np.round(phi, 5)
        shap_cols.append(col)

    # Sanity check: shap_sum ≈ score - mean_score
    result["shap_sum"] = result[shap_cols].sum(axis=1).round(5)

    return result, shap_cols


# ---------------------------------------------------------------------------
# Global SHAP summary (WLC)
# ---------------------------------------------------------------------------

def global_shap_summary(
    shap_df: pd.DataFrame,
    shap_cols: List[str],
    weights: Dict[str, float],
) -> pd.DataFrame:
    """
    Compute global SHAP statistics aggregated over all cells.

    Parameters
    ----------
    shap_df   : DataFrame with shap_* columns
    shap_cols : list of shap column names
    weights   : feature weights

    Returns
    -------
    DataFrame with columns: feature, weight, mean_abs_shap, mean_shap, std_shap
    sorted by mean_abs_shap descending.
    """
    rows = []
    for col in shap_cols:
        feat = col.replace("shap_", "")
        vals = shap_df[col].to_numpy()
        rows.append({
            "feature":       feat,
            "weight":        round(weights.get(feat + "_norm", weights.get(feat, 0.0)), 4),
            "mean_abs_shap": round(float(np.abs(vals).mean()), 5),
            "mean_shap":     round(float(vals.mean()), 5),
            "std_shap":      round(float(vals.std()), 5),
        })

    summary = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# RF permutation importance (global, model-based)
# ---------------------------------------------------------------------------

def rf_permutation_importance(
    model: RandomForestClassifier,
    df: pd.DataFrame,
    norm_features: List[str],
    labels: np.ndarray,
    seed: int = 42,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Compute permutation feature importance for the Random Forest model.

    Unlike MDI (``feature_importances_``), permutation importance measures
    the actual decrease in model accuracy when a feature is randomly shuffled,
    making it a more reliable estimate for out-of-bag importance.

    Parameters
    ----------
    model         : fitted RandomForestClassifier
    df            : DataFrame with normalised feature columns
    norm_features : feature column names
    labels        : ground-truth labels
    seed          : random seed
    n_repeats     : number of permutation iterations

    Returns
    -------
    DataFrame with feature, importances_mean, importances_std sorted descending.
    """
    X = df[norm_features].to_numpy(dtype=float)

    perm_imp = permutation_importance(
        model, X, labels,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="f1",
    )

    result = pd.DataFrame({
        "feature":            norm_features,
        "importance_mean":    np.round(perm_imp.importances_mean, 5),
        "importance_std":     np.round(perm_imp.importances_std,  5),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_global_shap(
    summary: pd.DataFrame,
    output_path: str,
    title: str = "Global Feature Importance (SHAP-equivalent, WLC)",
) -> None:
    """
    Create a horizontal bar chart of global SHAP importance values.

    Parameters
    ----------
    summary     : output of ``global_shap_summary``
    output_path : file path to save the PNG figure
    title       : chart title
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    features = summary["feature"].tolist()
    values   = summary["mean_abs_shap"].tolist()

    # Shorten display names
    labels = [f.replace("_norm", "").replace("_", " ").title() for f in features]

    cmap  = cm.get_cmap("RdYlGn")
    colors = [cmap(0.2 + 0.6 * (i / max(len(values) - 1, 1))) for i in range(len(values))]

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white")

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9)

    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(values) * 1.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SHAP] Global importance plot saved → {output_path}")


def plot_local_shap(
    shap_df: pd.DataFrame,
    shap_cols: List[str],
    hex_id: str,
    output_path: str,
) -> None:
    """
    Create a waterfall-style plot showing how each criterion pushes the
    suitability score above or below the baseline for a single hexagon.

    Parameters
    ----------
    shap_df     : DataFrame with shap_* columns from ``compute_local_shap``
    shap_cols   : list of shap column names
    hex_id      : identifier of the hexagon to explain
    output_path : file path to save the PNG figure
    """
    row = shap_df[shap_df["hex_id"] == hex_id]
    if row.empty:
        print(f"[SHAP] Warning: hex_id '{hex_id}' not found.")
        return

    row = row.iloc[0]
    score    = row.get("suitability_score", None)
    baseline = shap_df["suitability_score"].mean()

    values = [row[c] for c in shap_cols]
    labels = [c.replace("shap_", "").replace("_norm", "").replace("_", " ").title()
              for c in shap_cols]

    # Sort by absolute contribution
    order  = np.argsort(np.abs(values))[::-1]
    values = [values[i] for i in order]
    labels = [labels[i] for i in order]

    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = range(len(values))

    ax.barh(list(y_pos), values, color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)

    for i, (v, lab) in enumerate(zip(values, labels)):
        x_off = 0.0005 if v >= 0 else -0.0005
        ha = "left" if v >= 0 else "right"
        ax.text(v + x_off, i, f"{v:+.4f}", va="center", ha=ha, fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("SHAP value (contribution to score deviation from baseline)", fontsize=9)

    score_str = f"{score:.3f}" if score is not None else "N/A"
    ax.set_title(
        f"Local SHAP Explanation – Cell: {hex_id}\n"
        f"Score={score_str}  |  Baseline={baseline:.3f}",
        fontsize=11, fontweight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SHAP] Local explanation plot saved → {output_path}")


def plot_rf_permutation_importance(
    perm_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Visualise RF permutation importance as a horizontal bar chart with error bars.

    Parameters
    ----------
    perm_df     : output of ``rf_permutation_importance``
    output_path : save path for the PNG
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    features = [f.replace("_norm", "").replace("_", " ").title()
                for f in perm_df["feature"]][::-1]
    means    = perm_df["importance_mean"].tolist()[::-1]
    stds     = perm_df["importance_std"].tolist()[::-1]

    y_pos = range(len(means))
    ax.barh(list(y_pos), means, xerr=stds, color="#3498db",
            edgecolor="white", capsize=3, height=0.6)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Permutation Importance (F1 decrease)", fontsize=10)
    ax.set_title("Random Forest Permutation Feature Importance",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SHAP] RF permutation importance plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_h3_grid import generate_colombia_hex_grid
    from feature_engineering import engineer_features
    from normalization import normalise_features, get_norm_feature_names
    from random_forest_weights import get_rf_weights
    from mcda_model import compute_wlc_scores

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _GEOJSON = os.path.join(_HERE, "..", "data", "colombia_boundary.geojson")
    _OUT  = os.path.join(_HERE, "..", "outputs")
    os.makedirs(_OUT, exist_ok=True)

    grid    = generate_colombia_hex_grid(_GEOJSON, resolution=5)
    feats   = engineer_features(grid)
    norm_df = normalise_features(feats)
    norm_cols = get_norm_feature_names()

    model, weights, labels = get_rf_weights(norm_df, norm_cols)
    scored_df = compute_wlc_scores(norm_df, weights, norm_cols)

    shap_df, shap_cols = compute_local_shap(scored_df, weights, norm_cols)
    summary = global_shap_summary(shap_df, shap_cols, weights)
    print(summary)

    plot_global_shap(summary, os.path.join(_OUT, "shap_global.png"))

    top_hex = shap_df.nlargest(1, "suitability_score")["hex_id"].iloc[0]
    plot_local_shap(shap_df, shap_cols, top_hex, os.path.join(_OUT, "shap_local.png"))
