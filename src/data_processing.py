"""
data_processing.py
==================
Handles all data input/output operations for the wind farm suitability system:

  * Loading and validating the Colombia boundary GeoJSON
  * Exporting results to CSV, GeoJSON and GeoPackage-like formats
  * Generating summary reports
  * Utility functions for data validation and logging
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# GeoJSON loading
# ---------------------------------------------------------------------------

def load_boundary(geojson_path: str) -> dict:
    """
    Load and validate a GeoJSON boundary file.

    Parameters
    ----------
    geojson_path : path to the GeoJSON file

    Returns
    -------
    Parsed GeoJSON dict

    Raises
    ------
    FileNotFoundError : if the file does not exist
    ValueError        : if the file is not valid GeoJSON
    """
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(
            f"Boundary file not found: {geojson_path}\n"
            f"Please place colombia_boundary.geojson in the data/ directory."
        )

    with open(geojson_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data.get("type") not in ("FeatureCollection", "Feature", "Polygon", "MultiPolygon"):
        raise ValueError(f"Invalid GeoJSON type: {data.get('type')}")

    n_features = len(data.get("features", [data])) if "features" in data else 1
    print(f"[Data] Loaded boundary: {geojson_path}  ({n_features} feature(s))")
    return data


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def save_results_csv(
    df: pd.DataFrame,
    output_path: str,
    columns: Optional[List[str]] = None,
) -> None:
    """
    Export the results DataFrame to CSV.

    Parameters
    ----------
    df          : results DataFrame
    output_path : destination CSV path
    columns     : subset of columns to export; None exports all non-geometry cols
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Drop non-serialisable geometry column if present
    export_df = df.copy()
    if "vertices" in export_df.columns:
        export_df = export_df.drop(columns=["vertices"])

    if columns is not None:
        export_df = export_df[[c for c in columns if c in export_df.columns]]

    export_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"[Data] Results saved to CSV: {output_path}  ({len(export_df):,} rows)")


def save_results_geojson(
    df: pd.DataFrame,
    output_path: str,
    score_column: str = "suitability_score",
) -> None:
    """
    Export results as a GeoJSON FeatureCollection.

    Each hexagon becomes a GeoJSON Polygon Feature with all feature values
    and the suitability score in the properties.

    Parameters
    ----------
    df          : DataFrame with ``vertices`` column
    output_path : destination GeoJSON path
    score_column: name of the suitability score column
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    features = []
    # Columns to include in properties (exclude geometry)
    skip_cols = {"vertices"}
    prop_cols = [c for c in df.columns if c not in skip_cols]

    for _, row in df.iterrows():
        props = {}
        for col in prop_cols:
            val = row[col]
            if isinstance(val, float) and np.isnan(val):
                props[col] = None
            elif isinstance(val, (np.integer,)):
                props[col] = int(val)
            elif isinstance(val, (np.floating,)):
                props[col] = round(float(val), 4)
            else:
                props[col] = val

        if "vertices" in row.index and row["vertices"] is not None:
            coords = [[v[0], v[1]] for v in row["vertices"]]
        else:
            cx, cy, sz = row["lon"], row["lat"], 0.02
            coords = [
                [cx - sz, cy], [cx, cy + sz], [cx + sz, cy],
                [cx, cy - sz], [cx - sz, cy],
            ]

        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    geojson = {"type": "FeatureCollection", "features": features}

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(geojson, fh, separators=(",", ":"))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[Data] GeoJSON saved: {output_path}  ({len(features):,} features, {size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_summary_report(
    df: pd.DataFrame,
    weights: Dict[str, float],
    norm_features: List[str],
    output_path: str,
    score_column: str = "suitability_score",
    top_n: int = 20,
) -> None:
    """
    Write a plain-text summary report of the analysis results.

    Parameters
    ----------
    df            : scored and ranked DataFrame
    weights       : MCDA weight dictionary
    norm_features : normalised feature column names
    output_path   : path for the .txt report file
    score_column  : name of the suitability score column
    top_n         : number of top sites to list in the report
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scores = df[score_column].dropna().to_numpy()

    lines = [
        "=" * 70,
        "  WIND FARM SUITABILITY ANALYSIS — COLOMBIA",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "ANALYSIS OVERVIEW",
        "-" * 40,
        f"  Total hexagonal cells analysed : {len(df):,}",
        f"  Grid resolution                : custom (~44 km hexagons)",
        f"  MCDA method                    : Weighted Linear Combination (WLC)",
        f"  Criterion weighting            : Random Forest feature importance",
        "",
        "SUITABILITY SCORE STATISTICS",
        "-" * 40,
        f"  Mean   : {scores.mean():.4f}",
        f"  Median : {np.median(scores):.4f}",
        f"  Std    : {scores.std():.4f}",
        f"  Min    : {scores.min():.4f}",
        f"  Max    : {scores.max():.4f}",
        f"  75th pct : {np.percentile(scores, 75):.4f}",
        f"  90th pct : {np.percentile(scores, 90):.4f}",
        "",
        f"  Cells with score > 0.6  : {(scores > 0.6).sum():,}  "
          f"({100*(scores > 0.6).mean():.1f}%)",
        f"  Cells with score > 0.7  : {(scores > 0.7).sum():,}  "
          f"({100*(scores > 0.7).mean():.1f}%)",
        "",
        "MCDA CRITERION WEIGHTS (from Random Forest)",
        "-" * 40,
    ]

    total_w = sum(weights.values())
    for feat, w in sorted(weights.items(), key=lambda x: -x[1]):
        label = feat.replace("_norm", "").replace("_", " ").title()
        bar   = "█" * int((w / total_w) * 30)
        lines.append(f"  {label:25s}  {w/total_w:.4f}  {bar}")

    lines += [
        "",
        f"TOP {top_n} CANDIDATE LOCATIONS",
        "-" * 40,
    ]

    display_cols = ["rank", "hex_id", "lon", "lat", score_column,
                    "wind_speed", "slope", "dist_to_grid", "conflict_risk"]
    display_cols = [c for c in display_cols if c in df.columns]
    top_df = df.head(top_n)[display_cols]

    lines.append(top_df.to_string(index=False))
    lines += [
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ]

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[Data] Summary report saved → {output_path}")


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def validate_feature_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
) -> bool:
    """
    Validate that a DataFrame contains all required columns and has no
    fully-null feature columns.

    Parameters
    ----------
    df            : DataFrame to validate
    required_cols : list of required column names

    Returns
    -------
    True if valid; raises ValueError otherwise.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    null_cols = [c for c in required_cols if df[c].isna().all()]
    if null_cols:
        raise ValueError(f"Columns are entirely null: {null_cols}")

    print(f"[Validation] DataFrame validated: {len(df):,} rows, "
          f"{len(df.columns)} columns. All required columns present.")
    return True
