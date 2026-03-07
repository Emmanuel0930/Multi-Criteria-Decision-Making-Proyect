"""
normalization.py
================
Normalises raw feature values to a **suitability scale in [0, 1]**, where:

    1.0  →  highly suitable for a wind farm
    0.0  →  completely unsuitable

Each criterion is transformed according to its physical interpretation:

  +-------------------+----------+---------------------------------------------+
  | Feature           | Direction| Rationale                                   |
  +===================+==========+=============================================+
  | wind_speed        | positive | Higher wind → more energy potential         |
  +-------------------+----------+---------------------------------------------+
  | slope             | negative | Steeper terrain → harder construction       |
  +-------------------+----------+---------------------------------------------+
  | dist_to_grid      | negative | Farther from grid → higher connection cost  |
  +-------------------+----------+---------------------------------------------+
  | dist_to_roads     | negative | Farther from roads → harder logistics       |
  +-------------------+----------+---------------------------------------------+
  | land_use          | positive | Already encoded as suitability in [0,1]     |
  +-------------------+----------+---------------------------------------------+
  | protected_area    | negative | Protected → not available for development   |
  +-------------------+----------+---------------------------------------------+
  | conflict_risk     | negative | High risk → unsafe for infrastructure       |
  +-------------------+----------+---------------------------------------------+

Two normalisation methods are available:

  * ``minmax``  – linear rescaling to [0, 1]  (default)
  * ``sigmoid`` – S-shaped transformation centred on the median
"""

from __future__ import annotations

from typing import Dict, Literal, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration: which features are positive vs negative
# ---------------------------------------------------------------------------

#: Maps feature name → True if *higher raw value* means *more suitable*
FEATURE_DIRECTION: Dict[str, bool] = {
    "wind_speed":     True,   # higher is better
    "slope":          False,  # lower is better
    "dist_to_grid":   False,  # closer is better
    "dist_to_roads":  False,  # closer is better
    "land_use":       True,   # already a suitability score
    "protected_area": False,  # being protected is bad for development
    "conflict_risk":  False,  # high risk is bad
}

#: Optional hard thresholds – values beyond these are clipped before normalising
FEATURE_CLIP: Dict[str, tuple] = {
    "wind_speed":    (0.0, 12.0),   # m/s – above 12 turbine specs kick in
    "slope":         (0.0, 30.0),   # degrees – above 30° construction infeasible
    "dist_to_grid":  (0.0, 250.0),  # km
    "dist_to_roads": (0.0, 150.0),  # km
    "land_use":      (0.0,   1.0),
    "protected_area":(0.0,   1.0),
    "conflict_risk": (0.0,   1.0),
}

NormMethod = Literal["minmax", "sigmoid"]


# ---------------------------------------------------------------------------
# Core normalisation helpers
# ---------------------------------------------------------------------------

def _minmax_normalise(
    values: np.ndarray,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> np.ndarray:
    """
    Linear min-max normalisation to [0, 1].

    Parameters
    ----------
    values : raw 1-D array
    lo, hi : optional explicit bounds; defaults to array min/max

    Returns
    -------
    Array in [0, 1]
    """
    lo = values.min() if lo is None else lo
    hi = values.max() if hi is None else hi
    denom = hi - lo
    if denom < 1e-12:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / denom, 0.0, 1.0)


def _sigmoid_normalise(
    values: np.ndarray,
    centre: Optional[float] = None,
    scale: Optional[float] = None,
) -> np.ndarray:
    """
    Logistic (sigmoid) normalisation:  1 / (1 + exp(-(x - centre) / scale))

    This compresses extreme values and yields a smooth S-curve.

    Parameters
    ----------
    values  : raw 1-D array
    centre  : inflection point (default: median)
    scale   : steepness parameter (default: IQR / 2)

    Returns
    -------
    Array in (0, 1)
    """
    if centre is None:
        centre = float(np.median(values))
    if scale is None:
        q75, q25 = np.percentile(values, [75, 25])
        scale = max((q75 - q25) / 2.0, 1e-12)
    z = (values - centre) / scale
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Main normalisation function
# ---------------------------------------------------------------------------

def normalise_features(
    df: pd.DataFrame,
    features: Optional[list] = None,
    method: NormMethod = "minmax",
    feature_direction: Optional[Dict[str, bool]] = None,
    feature_clip: Optional[Dict[str, tuple]] = None,
) -> pd.DataFrame:
    """
    Normalise raw spatial features to a [0, 1] suitability scale.

    Parameters
    ----------
    df                : DataFrame with raw feature columns
    features          : list of feature columns to normalise;
                        defaults to all keys in FEATURE_DIRECTION
    method            : "minmax" (default) or "sigmoid"
    feature_direction : override dict for positive/negative features
    feature_clip      : override dict for hard clip bounds

    Returns
    -------
    Copy of ``df`` with additional columns ``<feature>_norm`` for each
    normalised criterion, plus a convenience column ``features_normalised``
    set to True.
    """
    if features is None:
        features = list(FEATURE_DIRECTION.keys())

    direction = feature_direction or FEATURE_DIRECTION
    clip_bounds = feature_clip or FEATURE_CLIP

    result = df.copy()

    print(f"[Normalisation] Method: '{method}' | Features: {features}")

    for feat in features:
        if feat not in df.columns:
            print(f"  [!] Feature '{feat}' not found in DataFrame – skipping.")
            continue

        raw = df[feat].to_numpy(dtype=float)

        # 1. Apply hard clip
        if feat in clip_bounds:
            lo_c, hi_c = clip_bounds[feat]
            raw = np.clip(raw, lo_c, hi_c)

        # 2. Normalise
        if method == "minmax":
            norm = _minmax_normalise(raw)
        elif method == "sigmoid":
            norm = _sigmoid_normalise(raw)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'minmax' or 'sigmoid'.")

        # 3. Flip if the feature is negative (higher raw = less suitable)
        if not direction.get(feat, True):
            norm = 1.0 - norm

        result[f"{feat}_norm"] = np.round(norm, 4)
        print(f"  {feat:20s} → min={norm.min():.3f}  max={norm.max():.3f}  "
              f"mean={norm.mean():.3f}  (direction={'positive' if direction.get(feat, True) else 'negative'})")

    print("[Normalisation] Done.")
    return result


def get_norm_feature_names(features: Optional[list] = None) -> list:
    """
    Return the normalised column names for the given feature list.

    Parameters
    ----------
    features : raw feature names; defaults to all in FEATURE_DIRECTION

    Returns
    -------
    List of ``<feature>_norm`` strings
    """
    if features is None:
        features = list(FEATURE_DIRECTION.keys())
    return [f"{f}_norm" for f in features]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_h3_grid import generate_colombia_hex_grid
    from feature_engineering import engineer_features

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _GEOJSON = os.path.join(_HERE, "..", "data", "colombia_boundary.geojson")

    grid = generate_colombia_hex_grid(_GEOJSON, resolution=5)
    features_df = engineer_features(grid)
    norm_df = normalise_features(features_df)

    norm_cols = get_norm_feature_names()
    print("\nNormalised feature statistics:")
    print(norm_df[norm_cols].describe().round(3).to_string())
