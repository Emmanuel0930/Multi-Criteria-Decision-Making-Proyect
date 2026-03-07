"""
random_forest_weights.py
========================
Trains a Random Forest classifier on synthetic labels to derive data-driven
MCDA criterion weights.

Workflow
--------
1. Generate synthetic "suitable / not-suitable" labels using a rule-based
   heuristic that mirrors expert knowledge about wind farm site selection.
2. Train a ``RandomForestClassifier`` on the normalised feature matrix.
3. Extract ``feature_importances_`` (mean decrease in impurity, MDI) as
   the basis for MCDA weights.
4. Return a weight vector normalised to sum to 1.

Why Random Forest for weighting?
---------------------------------
In classical MCDA, weights are elicited from domain experts through
AHP or similar methods – a time-consuming, subjective process.  By
training an RF on labelled (or heuristically labelled) sites we obtain
weights that reflect the *actual statistical importance* of each criterion
for discriminating suitable from unsuitable locations.  SHAP values later
add explainability at the individual-cell level.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report


# ---------------------------------------------------------------------------
# Synthetic label generation
# ---------------------------------------------------------------------------

def generate_synthetic_labels(
    df: pd.DataFrame,
    norm_features: List[str],
    suitability_threshold: float = 0.55,
    seed: int = 42,
) -> np.ndarray:
    """
    Create binary suitability labels using a heuristic weighted combination.

    A location is labelled *suitable (1)* if its weighted average of key
    criteria exceeds ``suitability_threshold``.

    These rules are based on published guidelines from IFC / IRENA for
    onshore wind energy siting:

    * Wind speed is the dominant criterion (weight 0.40)
    * Protected areas and conflict risk are hard knock-outs (high weight, negative)
    * Slope and distance to grid also strongly penalise suitability

    Parameters
    ----------
    df                    : DataFrame with normalised feature columns
    norm_features         : list of ``<feat>_norm`` column names
    suitability_threshold : decision boundary in [0, 1]
    seed                  : for any stochastic elements

    Returns
    -------
    1-D integer array of shape (n,) with values 0 (not suitable) or 1 (suitable).
    """
    # Expert-knowledge prior weights (must correspond to FEATURE_DIRECTION ordering)
    expert_weights = {
        "wind_speed_norm":     0.35,
        "slope_norm":          0.15,
        "dist_to_grid_norm":   0.15,
        "dist_to_roads_norm":  0.10,
        "land_use_norm":       0.10,
        "protected_area_norm": 0.10,
        "conflict_risk_norm":  0.05,
    }

    # Compute weighted sum
    score = np.zeros(len(df))
    total_w = 0.0
    for feat in norm_features:
        w = expert_weights.get(feat, 1.0 / len(norm_features))
        score += w * df[feat].to_numpy(dtype=float)
        total_w += w

    score /= (total_w + 1e-12)

    # Hard exclusion rules:
    # 1. Protected areas > 0.7 → automatically not suitable
    if "protected_area_norm" in df.columns:
        protected_mask = df["protected_area_norm"].to_numpy() > 0.7
        score[protected_mask] = 0.0

    # 2. Slope > 0.8 (raw slope > 24°) → not suitable
    if "slope_norm" in df.columns:
        steep_mask = df["slope_norm"].to_numpy() < 0.2  # flipped: low norm = steep
        score[steep_mask] *= 0.5

    # 3. Conflict risk norm < 0.2 (high conflict) → not suitable
    if "conflict_risk_norm" in df.columns:
        conflict_mask = df["conflict_risk_norm"].to_numpy() < 0.2
        score[conflict_mask] = 0.0

    # Adaptive threshold: ensure at least 8% of cells are labelled suitable
    # (prevents degenerate all-negative labels on coarser grids)
    actual_threshold = suitability_threshold
    for candidate in [suitability_threshold, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
        trial = (score >= candidate).astype(int)
        if trial.mean() >= 0.08:
            actual_threshold = candidate
            break

    if actual_threshold != suitability_threshold:
        print(f"[Labels] Threshold auto-adjusted: {suitability_threshold} → {actual_threshold} "
              f"(to ensure sufficient positive class size)")

    labels = (score >= actual_threshold).astype(int)

    n_suitable = labels.sum()
    n_total = len(labels)
    print(f"[Labels] Suitable: {n_suitable:,}  ({100*n_suitable/n_total:.1f}%)  "
          f"Not suitable: {n_total - n_suitable:,}  ({100*(1-n_suitable/n_total):.1f}%)")

    return labels


# ---------------------------------------------------------------------------
# Random Forest training
# ---------------------------------------------------------------------------

def train_random_forest(
    df: pd.DataFrame,
    norm_features: List[str],
    labels: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 10,
    seed: int = 42,
    cv_folds: int = 5,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Train a Random Forest classifier and extract feature-importance weights.

    Parameters
    ----------
    df            : DataFrame with normalised feature columns
    norm_features : list of column names to use as input features
    labels        : binary target array (0/1) from ``generate_synthetic_labels``
    n_estimators  : number of trees in the forest
    max_depth     : maximum tree depth (prevents overfitting)
    seed          : random seed for reproducibility
    cv_folds      : number of stratified cross-validation folds

    Returns
    -------
    Tuple of:
        model   – fitted ``RandomForestClassifier``
        weights – dict mapping feature name → normalised importance weight
    """
    X = df[norm_features].to_numpy(dtype=float)
    y = labels

    print(f"[RF] Training RandomForest (n_estimators={n_estimators}, "
          f"max_depth={max_depth}, cv={cv_folds})...")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        class_weight="balanced",   # handle class imbalance
        random_state=seed,
        n_jobs=-1,
    )

    # Cross-validation to assess model quality
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
    print(f"[RF] Cross-val F1 scores: {np.round(cv_scores, 3)}  "
          f"mean={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final fit on full dataset
    clf.fit(X, y)

    print("[RF] Classification report (train set):")
    print(classification_report(y, clf.predict(X), target_names=["Not suitable", "Suitable"]))

    # Extract feature importances (MDI – Mean Decrease in Impurity)
    raw_importances = clf.feature_importances_

    # Normalise to sum to 1.0
    total = raw_importances.sum()
    weights = {
        feat: float(imp / total)
        for feat, imp in zip(norm_features, raw_importances)
    }

    print("[RF] Feature importances (MCDA weights):")
    for feat, w in sorted(weights.items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 40)
        print(f"  {feat:28s}  {w:.4f}  {bar}")

    return clf, weights


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def get_rf_weights(
    df: pd.DataFrame,
    norm_features: List[str],
    suitability_threshold: float = 0.55,
    n_estimators: int = 200,
    seed: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, float], np.ndarray]:
    """
    End-to-end wrapper: generate labels → train RF → return weights.

    Parameters
    ----------
    df                    : DataFrame with normalised features
    norm_features         : normalised feature column names
    suitability_threshold : heuristic label threshold
    n_estimators          : RF hyperparameter
    seed                  : reproducibility seed

    Returns
    -------
    Tuple of (model, weights_dict, labels_array)
    """
    labels = generate_synthetic_labels(df, norm_features, suitability_threshold, seed)
    model, weights = train_random_forest(df, norm_features, labels,
                                         n_estimators=n_estimators, seed=seed)
    return model, weights, labels


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_h3_grid import generate_colombia_hex_grid
    from feature_engineering import engineer_features
    from normalization import normalise_features, get_norm_feature_names

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _GEOJSON = os.path.join(_HERE, "..", "data", "colombia_boundary.geojson")

    grid = generate_colombia_hex_grid(_GEOJSON, resolution=5)
    feats = engineer_features(grid)
    norm_df = normalise_features(feats)
    norm_cols = get_norm_feature_names()

    model, weights, labels = get_rf_weights(norm_df, norm_cols)
    print("\nFinal MCDA weights:", weights)
