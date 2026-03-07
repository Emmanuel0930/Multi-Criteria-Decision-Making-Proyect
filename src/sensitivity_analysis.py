"""
sensitivity_analysis.py
=======================
Performs a **weight sensitivity analysis** to assess the robustness of the
MCDA rankings to small perturbations in criterion weights.

The analysis answers the question:
    "If the weight assigned to a criterion is slightly off, how stable are
     the top-ranked wind farm locations?"

Two analyses are performed:

1. **One-at-a-time (OAT) perturbation**
   Each criterion's weight is varied ±Δ while all other weights are
   rescaled proportionally to maintain unit sum.  The rank correlation
   (Spearman's ρ) between the baseline ranking and the perturbed ranking
   is computed for each perturbation step.

2. **Monte Carlo weight sampling**
   Weights are sampled from a Dirichlet distribution centred on the
   baseline weights.  Spearman rank correlations and rank stability
   statistics are computed across N random scenarios.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perturb_weights(
    weights: Dict[str, float],
    feature: str,
    delta: float,
) -> Dict[str, float]:
    """
    Increase ``feature``'s weight by ``delta``, rescale others proportionally.

    Parameters
    ----------
    weights  : baseline weight dict
    feature  : feature to perturb
    delta    : change in weight (can be negative)

    Returns
    -------
    New weight dict that sums to 1.0
    """
    if feature not in weights:
        raise KeyError(f"Feature '{feature}' not in weights dict.")

    new_w = {k: v for k, v in weights.items()}
    target = new_w[feature] + delta

    # Clip to [0.001, 0.999]
    target = np.clip(target, 0.001, 0.999)
    old    = new_w[feature]
    diff   = target - old

    # Distribute the difference across the remaining features
    others = [k for k in new_w if k != feature]
    others_sum = sum(new_w[k] for k in others)

    if abs(others_sum) < 1e-12:
        return new_w  # cannot redistribute

    scale = 1.0 - diff / others_sum
    for k in others:
        new_w[k] = max(1e-6, new_w[k] * scale)

    new_w[feature] = target

    # Final normalisation
    total = sum(new_w.values())
    return {k: v / total for k, v in new_w.items()}


def _wlc_scores(
    X: np.ndarray,
    weights: Dict[str, float],
    features: List[str],
) -> np.ndarray:
    """Fast WLC computation for sensitivity analysis."""
    w = np.array([weights[f] for f in features])
    w /= w.sum()
    return X @ w


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation between two score arrays."""
    rho, _ = spearmanr(a, b)
    return float(rho)


# ---------------------------------------------------------------------------
# OAT sensitivity analysis
# ---------------------------------------------------------------------------

def oat_sensitivity(
    df: pd.DataFrame,
    weights: Dict[str, float],
    norm_features: List[str],
    delta_range: Tuple[float, float] = (-0.15, 0.15),
    n_steps: int = 13,
    score_column: str = "suitability_score",
) -> pd.DataFrame:
    """
    One-at-a-time (OAT) weight sensitivity analysis.

    For each criterion, vary its weight from ``delta_range[0]`` to
    ``delta_range[1]`` (relative to baseline) and compute the Spearman
    rank correlation of the resulting top-N ranking against the baseline.

    Parameters
    ----------
    df            : scored DataFrame with normalised feature columns
    weights       : baseline MCDA weights
    norm_features : list of feature column names
    delta_range   : (min_delta, max_delta) for weight perturbation
    n_steps       : number of delta steps
    score_column  : name of the baseline score column

    Returns
    -------
    DataFrame with columns:
        feature, delta, perturbed_weight, spearman_rho, rank_change_top10
    """
    X = df[norm_features].to_numpy(dtype=float)
    baseline_scores = df[score_column].to_numpy()
    deltas = np.linspace(delta_range[0], delta_range[1], n_steps)

    rows = []
    for feat in norm_features:
        for d in deltas:
            pw = _perturb_weights(weights, feat, d)
            perturbed_scores = _wlc_scores(X, pw, norm_features)
            rho = _spearman(baseline_scores, perturbed_scores)

            # Count how many of the top-10 remain in the top-10
            top10_base = set(np.argsort(baseline_scores)[-10:])
            top10_pert = set(np.argsort(perturbed_scores)[-10:])
            stable = len(top10_base & top10_pert)

            rows.append({
                "feature":          feat,
                "delta":            round(d, 4),
                "perturbed_weight": round(pw[feat], 4),
                "spearman_rho":     round(rho, 4),
                "top10_stable":     stable,
            })

    result = pd.DataFrame(rows)
    print(f"[Sensitivity] OAT analysis complete: {len(result)} scenarios evaluated.")
    return result


# ---------------------------------------------------------------------------
# Monte Carlo sensitivity analysis
# ---------------------------------------------------------------------------

def monte_carlo_sensitivity(
    df: pd.DataFrame,
    weights: Dict[str, float],
    norm_features: List[str],
    n_samples: int = 500,
    concentration: float = 20.0,
    score_column: str = "suitability_score",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monte Carlo sensitivity analysis using Dirichlet-sampled weights.

    Weights are drawn from a Dirichlet distribution centred on the baseline
    weights.  Higher ``concentration`` → samples stay closer to baseline.

    Parameters
    ----------
    df            : scored DataFrame
    weights       : baseline MCDA weights
    norm_features : list of feature column names
    n_samples     : number of random weight scenarios
    concentration : Dirichlet concentration parameter (higher = less variance)
    score_column  : baseline score column
    seed          : reproducibility

    Returns
    -------
    Tuple of:
        scenarios_df  – per-scenario statistics (n_samples rows)
        stability_df  – per-feature rank stability summary
    """
    rng = np.random.default_rng(seed)
    X = df[norm_features].to_numpy(dtype=float)
    baseline_scores = df[score_column].to_numpy()

    # Dirichlet alpha = concentration * baseline_weights
    alpha = np.array([weights[f] for f in norm_features]) * concentration

    scenarios = []
    for i in range(n_samples):
        w_sample = rng.dirichlet(alpha)
        sample_weights = dict(zip(norm_features, w_sample))
        perturbed_scores = _wlc_scores(X, sample_weights, norm_features)

        rho = _spearman(baseline_scores, perturbed_scores)

        top10_base = set(np.argsort(baseline_scores)[-10:])
        top10_samp = set(np.argsort(perturbed_scores)[-10:])
        stable = len(top10_base & top10_samp)

        row = {"scenario": i, "spearman_rho": rho, "top10_stable": stable}
        row.update({f: round(w_sample[j], 5) for j, f in enumerate(norm_features)})
        scenarios.append(row)

    scenarios_df = pd.DataFrame(scenarios)

    print(f"[Sensitivity] MC analysis: {n_samples} scenarios.")
    print(f"  Spearman rho:   mean={scenarios_df['spearman_rho'].mean():.4f}  "
          f"min={scenarios_df['spearman_rho'].min():.4f}  "
          f"max={scenarios_df['spearman_rho'].max():.4f}")
    print(f"  Top-10 stable:  mean={scenarios_df['top10_stable'].mean():.1f}/10")

    # Stability per feature: coefficient of variation of the scenario weights
    stab_rows = []
    for feat in norm_features:
        vals = scenarios_df[feat].to_numpy()
        stab_rows.append({
            "feature":    feat,
            "weight_mean": round(vals.mean(), 4),
            "weight_std":  round(vals.std(), 4),
            "cv":          round(vals.std() / (vals.mean() + 1e-12), 4),
        })
    stability_df = pd.DataFrame(stab_rows)

    return scenarios_df, stability_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_oat_sensitivity(
    oat_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot OAT sensitivity: one curve per feature showing how Spearman ρ
    changes as its weight is varied.

    Parameters
    ----------
    oat_df      : output of ``oat_sensitivity``
    output_path : save path for the PNG
    """
    features = oat_df["feature"].unique()
    n_feat   = len(features)
    colors   = plt.cm.tab10(np.linspace(0, 1, n_feat))

    fig, ax = plt.subplots(figsize=(10, 5))

    for feat, col in zip(features, colors):
        subset = oat_df[oat_df["feature"] == feat].sort_values("delta")
        label  = feat.replace("_norm", "").replace("_", " ").title()
        ax.plot(subset["delta"], subset["spearman_rho"],
                marker="o", markersize=4, linewidth=1.8,
                label=label, color=col)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0.0, color="grey",  linewidth=0.8, linestyle=":")
    ax.set_xlabel("Weight perturbation (Δ)", fontsize=10)
    ax.set_ylabel("Spearman rank correlation with baseline", fontsize=10)
    ax.set_title("OAT Weight Sensitivity Analysis\n"
                 "(how much rankings change when one weight is perturbed)",
                 fontsize=11, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Sensitivity] OAT plot saved → {output_path}")


def plot_mc_sensitivity(
    scenarios_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot Monte Carlo sensitivity: histogram of Spearman ρ across all
    random weight scenarios.

    Parameters
    ----------
    scenarios_df : output of ``monte_carlo_sensitivity`` (first element)
    output_path  : save path for the PNG
    """
    rho_vals = scenarios_df["spearman_rho"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of Spearman rho
    axes[0].hist(rho_vals, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].axvline(rho_vals.mean(), color="red", linewidth=1.5,
                    linestyle="--", label=f"Mean = {rho_vals.mean():.3f}")
    axes[0].set_xlabel("Spearman ρ (rank correlation with baseline)", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)
    axes[0].set_title("MC Weight Sensitivity\n(distribution of rank correlations)",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Top-10 stability histogram
    stab_vals = scenarios_df["top10_stable"].to_numpy()
    axes[1].hist(stab_vals, bins=range(0, 12), color="#27ae60",
                 edgecolor="white", alpha=0.85, align="left")
    axes[1].set_xlabel("# of original top-10 cells retained", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    axes[1].set_title("Top-10 Rank Stability\nacross MC weight scenarios",
                      fontsize=11, fontweight="bold")
    axes[1].set_xticks(range(0, 11))
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Sensitivity] MC plot saved → {output_path}")


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

    grid      = generate_colombia_hex_grid(_GEOJSON, resolution=5)
    feats     = engineer_features(grid)
    norm_df   = normalise_features(feats)
    norm_cols = get_norm_feature_names()
    model, weights, labels = get_rf_weights(norm_df, norm_cols)
    scored_df = compute_wlc_scores(norm_df, weights, norm_cols)

    oat   = oat_sensitivity(scored_df, weights, norm_cols)
    mc, _ = monte_carlo_sensitivity(scored_df, weights, norm_cols, n_samples=200)

    plot_oat_sensitivity(oat, os.path.join(_OUT, "sensitivity_oat.png"))
    plot_mc_sensitivity(mc,   os.path.join(_OUT, "sensitivity_mc.png"))
