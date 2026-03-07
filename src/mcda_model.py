"""
mcda_model.py
=============
Implements the **Weighted Linear Combination (WLC)** multi-criteria decision
analysis model for wind farm site suitability assessment.

Mathematical formulation
-------------------------
For each hexagonal cell *i*:

    S_i = Σ_j  w_j · c_{ij}

where:
    S_i   – suitability score for cell i
    w_j   – weight of criterion j  (Σ w_j = 1)
    c_{ij}– normalised criterion value [0, 1] for cell i and criterion j

The WLC model is equivalent to a dot product between the normalised feature
matrix and the weight vector, making it extremely efficient even for large
national-scale grids.

Reference
---------
Malczewski, J. (1999). GIS and Multicriteria Decision Analysis.
John Wiley & Sons.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# WLC computation
# ---------------------------------------------------------------------------

def compute_wlc_scores(
    df: pd.DataFrame,
    weights: Dict[str, float],
    norm_features: Optional[List[str]] = None,
    score_column: str = "suitability_score",
) -> pd.DataFrame:
    """
    Apply the Weighted Linear Combination model to compute suitability scores.

    Parameters
    ----------
    df            : DataFrame with normalised feature columns
    weights       : dict mapping feature column name → weight (should sum to 1)
    norm_features : subset of features to include; defaults to all keys in weights
    score_column  : name of the output suitability score column

    Returns
    -------
    Copy of ``df`` with an additional ``suitability_score`` column (and
    intermediate per-criterion weighted contribution columns).
    """
    if norm_features is None:
        norm_features = list(weights.keys())

    # Validate that all features are present
    missing = [f for f in norm_features if f not in df.columns]
    if missing:
        raise ValueError(f"Features missing from DataFrame: {missing}")

    # Normalise weights to sum to 1 (defensive)
    total_w = sum(weights[f] for f in norm_features)
    if abs(total_w - 1.0) > 0.01:
        print(f"[MCDA] Warning: weights sum to {total_w:.4f}. Re-normalising.")
    w_vector = np.array([weights[f] / total_w for f in norm_features])

    # Feature matrix  (n_cells × n_features)
    X = df[norm_features].to_numpy(dtype=float)

    # Weighted linear combination  (dot product)
    scores = X @ w_vector  # shape (n_cells,)
    scores = np.clip(scores, 0.0, 1.0)

    result = df.copy()
    result[score_column] = np.round(scores, 4)

    # Also store per-criterion weighted contributions for transparency
    for feat, w in zip(norm_features, w_vector):
        result[f"contrib_{feat}"] = np.round(df[feat].to_numpy() * w, 5)

    print(f"[MCDA] Suitability scores computed for {len(result):,} cells.")
    print(f"[MCDA] Score statistics:  "
          f"min={scores.min():.3f}  max={scores.max():.3f}  "
          f"mean={scores.mean():.3f}  std={scores.std():.3f}")

    return result


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_locations(
    df: pd.DataFrame,
    score_column: str = "suitability_score",
    top_n: Optional[int] = None,
    exclude_protected: bool = True,
    protected_col: str = "protected_area_norm",
    protected_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Rank all hexagonal cells by suitability score and return a sorted DataFrame.

    Parameters
    ----------
    df                   : DataFrame with suitability scores
    score_column         : name of the score column
    top_n                : if given, return only the top N cells
    exclude_protected    : remove cells in protected areas before ranking
    protected_col        : column used for protected-area filtering
    protected_threshold  : cells with protected_col > threshold are excluded

    Returns
    -------
    DataFrame sorted descending by suitability, with a ``rank`` column added.
    """
    working = df.copy()

    if exclude_protected and protected_col in df.columns:
        n_before = len(working)
        working = working[working[protected_col] <= protected_threshold]
        n_excluded = n_before - len(working)
        if n_excluded > 0:
            print(f"[Ranking] Excluded {n_excluded:,} heavily protected-area cells "
                  f"(protected_area_norm > {protected_threshold}).")
        else:
            print(f"[Ranking] No cells excluded by protected-area filter.")

    ranked = (
        working
        .sort_values(score_column, ascending=False)
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", ranked.index + 1)

    if top_n is not None:
        ranked = ranked.head(top_n)

    print(f"[Ranking] Top {len(ranked)} locations identified.")
    return ranked


def summarise_top_locations(
    ranked_df: pd.DataFrame,
    top_n: int = 10,
    score_column: str = "suitability_score",
) -> pd.DataFrame:
    """
    Print a human-readable summary of the top N locations.

    Parameters
    ----------
    ranked_df    : output of ``rank_locations``
    top_n        : number of top locations to summarise
    score_column : name of the score column

    Returns
    -------
    Subset DataFrame for the top N locations with key display columns.
    """
    display_cols = ["rank", "hex_id", "lon", "lat", score_column,
                    "wind_speed", "slope", "dist_to_grid", "conflict_risk"]
    display_cols = [c for c in display_cols if c in ranked_df.columns]

    top = ranked_df.head(top_n)[display_cols].copy()

    print(f"\n{'='*70}")
    print(f"  TOP {top_n} WIND FARM CANDIDATE LOCATIONS IN COLOMBIA")
    print(f"{'='*70}")
    print(top.to_string(index=False))
    print(f"{'='*70}\n")

    return top


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_h3_grid import generate_colombia_hex_grid
    from feature_engineering import engineer_features
    from normalization import normalise_features, get_norm_feature_names
    from random_forest_weights import get_rf_weights

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _GEOJSON = os.path.join(_HERE, "..", "data", "colombia_boundary.geojson")

    grid = generate_colombia_hex_grid(_GEOJSON, resolution=5)
    feats = engineer_features(grid)
    norm_df = normalise_features(feats)
    norm_cols = get_norm_feature_names()

    model, weights, labels = get_rf_weights(norm_df, norm_cols)

    scored_df = compute_wlc_scores(norm_df, weights, norm_cols)
    ranked_df = rank_locations(scored_df)
    summarise_top_locations(ranked_df, top_n=5)
