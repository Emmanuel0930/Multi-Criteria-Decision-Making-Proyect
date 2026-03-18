"""
main.py
=======
Master pipeline for the Colombia Wind Farm Suitability Assessment.

This script orchestrates the full MCDA workflow in sequential steps:

    Step 1 – Generate H3 hexagonal grid covering Colombia
    Step 2 – Engineer spatial features for each hexagon
    Step 3 – Normalise all criteria to a [0, 1] suitability scale
    Step 4 – Train Random Forest classifier and extract criterion weights
    Step 5 – Apply Weighted Linear Combination MCDA model
    Step 6 – Rank candidate locations
    Step 7 – Compute SHAP-like explanations
    Step 8 – Run sensitivity analysis
    Step 9 – Generate visualisations and export results

Usage
-----
    python main.py                          # default resolution 5
    python main.py --resolution 6           # finer grid
    python main.py --resolution 4 --seed 0  # coarser grid, different seed

Configuration
-------------
All tunable parameters are collected in the CONFIG dict at the top of
this file, so the pipeline can be reproduced and adapted without
editing individual module files.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup – allow running from project root or any sub-directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR      = os.path.join(_PROJECT_ROOT, "src")
_DATA_DIR     = os.path.join(_PROJECT_ROOT, "data")
_OUT_DIR      = os.path.join(_PROJECT_ROOT, "outputs")

sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from data_processing      import (load_boundary, save_results_csv,
                                   save_results_geojson, generate_summary_report,
                                   validate_feature_dataframe)
from generate_h3_grid     import generate_colombia_hex_grid
from feature_engineering  import engineer_features
from normalization        import normalise_features, get_norm_feature_names, FEATURE_DIRECTION
from random_forest_weights import get_rf_weights
from mcda_model           import compute_wlc_scores, rank_locations, summarise_top_locations
from shap_explainer       import (compute_local_shap, global_shap_summary,
                                   rf_permutation_importance,
                                   plot_global_shap, plot_local_shap,
                                   plot_rf_permutation_importance)
from sensitivity_analysis import (oat_sensitivity, monte_carlo_sensitivity,
                                   plot_oat_sensitivity, plot_mc_sensitivity)
from visualization        import (create_interactive_map, create_static_map,
                                   plot_score_distribution, plot_feature_correlation)

# ---------------------------------------------------------------------------
# Default pipeline configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Grid
    "resolution":          6,      # H3-like resolution (4=coarse … 7=fine; 55=~600 hex, 65=~800 hex)

    # Feature engineering
    "feature_seed":        42,

    # Normalisation
    "norm_method":         "minmax",  # "minmax" or "sigmoid"

    # Random Forest
    "rf_n_estimators":     200,
    "rf_seed":             42,
    "label_threshold":     0.55,    # heuristic label boundary

    # MCDA
    "score_column":        "suitability_score",
    "top_n":               20,

    # Sensitivity analysis
    "oat_delta_range":     (-0.15, 0.15),
    "oat_n_steps":         13,
    "mc_n_samples":        300,
    "mc_concentration":    20.0,

    # Paths (relative to project root)
    "geojson_path":  os.path.join(_DATA_DIR, "colombia_boundary.geojson"),
    "municipios_path": os.path.join(_DATA_DIR, "DIVIPOLA_CentrosPoblados.csv"),
    "output_dir":    _OUT_DIR,
}


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the full MCDA wind farm suitability pipeline.

    Parameters
    ----------
    config : configuration dictionary (see DEFAULT_CONFIG)

    Returns
    -------
    Dict with references to all major output objects:
        hex_grid, features_df, norm_df, model, weights, scored_df,
        ranked_df, shap_df, oat_df, mc_df
    """
    os.makedirs(config["output_dir"], exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1 – H3 hexagonal grid
    # ------------------------------------------------------------------
    _banner("STEP 1 – Generating H3 hexagonal grid")
    _ = load_boundary(config["geojson_path"])   # validate file exists
    hex_grid = generate_colombia_hex_grid(
        config["geojson_path"],
        resolution=config["resolution"],
        municipios_path=config["municipios_path"],
    )

    # ------------------------------------------------------------------
    # Step 2 – Feature engineering
    # ------------------------------------------------------------------
    _banner("STEP 2 – Engineering spatial features")
    features_df = engineer_features(hex_grid, seed=config["feature_seed"])

    raw_feature_cols = list(FEATURE_DIRECTION.keys())
    validate_feature_dataframe(features_df, ["hex_id", "lon", "lat"] + raw_feature_cols)

    # ------------------------------------------------------------------
    # Step 3 – Normalisation
    # ------------------------------------------------------------------
    _banner("STEP 3 – Normalising criteria to [0, 1] suitability scale")
    norm_df = normalise_features(
        features_df,
        method=config["norm_method"],
    )
    norm_cols = get_norm_feature_names()

    # ------------------------------------------------------------------
    # Step 4 – Random Forest weighting
    # ------------------------------------------------------------------
    _banner("STEP 4 – Training Random Forest to derive criterion weights")
    model, weights, labels = get_rf_weights(
        norm_df,
        norm_cols,
        suitability_threshold=config["label_threshold"],
        n_estimators=config["rf_n_estimators"],
        seed=config["rf_seed"],
    )

    # ------------------------------------------------------------------
    # Step 5 – WLC MCDA
    # ------------------------------------------------------------------
    _banner("STEP 5 – Computing WLC suitability scores")
    scored_df = compute_wlc_scores(norm_df, weights, norm_cols,
                                    score_column=config["score_column"])

    # ------------------------------------------------------------------
    # Step 6 – Ranking
    # ------------------------------------------------------------------
    _banner("STEP 6 – Ranking candidate locations")
    ranked_df = rank_locations(scored_df, score_column=config["score_column"],
                                top_n=None,
                                protected_threshold=0.7)
    summarise_top_locations(ranked_df, top_n=config["top_n"],
                             score_column=config["score_column"])

    # ------------------------------------------------------------------
    # Step 7 – SHAP explainability
    # ------------------------------------------------------------------
    _banner("STEP 7 – Computing SHAP-like explanations")

    shap_df, shap_cols = compute_local_shap(ranked_df, weights, norm_cols,
                                             score_column=config["score_column"])
    shap_summary = global_shap_summary(shap_df, shap_cols, weights)
    print("\nGlobal SHAP summary:")
    print(shap_summary.to_string(index=False))

    # RF permutation importance
    perm_imp = rf_permutation_importance(model, norm_df, norm_cols, labels,
                                          seed=config["rf_seed"], n_repeats=5)
    print("\nPermutation importance (RF):")
    print(perm_imp.to_string(index=False))

    # Plot global SHAP
    plot_global_shap(
        shap_summary,
        os.path.join(config["output_dir"], "shap_global_importance.png"),
    )

    # Plot local SHAP for top-1 and top-10 cells
    top1_id = shap_df.nlargest(1, config["score_column"])["hex_id"].iloc[0]
    plot_local_shap(
        shap_df, shap_cols, top1_id,
        os.path.join(config["output_dir"], "shap_local_top1.png"),
    )

    # Try to find an "interesting" mid-ranked cell for contrast
    mid_rank = len(shap_df) // 2
    mid_id = shap_df.iloc[mid_rank]["hex_id"]
    plot_local_shap(
        shap_df, shap_cols, mid_id,
        os.path.join(config["output_dir"], "shap_local_midrank.png"),
    )

    # RF permutation importance chart
    plot_rf_permutation_importance(
        perm_imp,
        os.path.join(config["output_dir"], "rf_permutation_importance.png"),
    )

    # ------------------------------------------------------------------
    # Step 8 – Sensitivity analysis
    # ------------------------------------------------------------------
    _banner("STEP 8 – Sensitivity analysis")

    oat_df = oat_sensitivity(
        scored_df, weights, norm_cols,
        delta_range=config["oat_delta_range"],
        n_steps=config["oat_n_steps"],
    )
    mc_df, mc_stability = monte_carlo_sensitivity(
        scored_df, weights, norm_cols,
        n_samples=config["mc_n_samples"],
        concentration=config["mc_concentration"],
    )

    plot_oat_sensitivity(oat_df,
                          os.path.join(config["output_dir"], "sensitivity_oat.png"))
    plot_mc_sensitivity(mc_df,
                         os.path.join(config["output_dir"], "sensitivity_mc.png"))

    print("\nMC weight stability per feature:")
    print(mc_stability.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 9 – Visualisation and export
    # ------------------------------------------------------------------
    _banner("STEP 9 – Generating visualisations and exporting results")

    # Interactive HTML map – show ALL cells coloured by score, highlight top-N
    create_interactive_map(
        scored_df,                 # ← full grid, not just ranked survivors
        os.path.join(config["output_dir"], "map_interactive.html"),
        score_column=config["score_column"],
        top_n_highlight=10,
    )

    # Static PNG map
    create_static_map(
        ranked_df,
        os.path.join(config["output_dir"], "map_static.png"),
        score_column=config["score_column"],
    )

    # Distribution plot
    plot_score_distribution(
        ranked_df,
        os.path.join(config["output_dir"], "score_distribution.png"),
        score_column=config["score_column"],
    )

    # Feature correlation heatmap
    plot_feature_correlation(
        norm_df, norm_cols,
        os.path.join(config["output_dir"], "feature_correlation.png"),
        score_column=config["score_column"],
    )

    # ------------------------------------------------------------------
    # Data exports
    # ------------------------------------------------------------------
    save_results_csv(
        ranked_df,
        os.path.join(config["output_dir"], "suitability_scores.csv"),
    )
    save_results_geojson(
        ranked_df,
        os.path.join(config["output_dir"], "suitability_scores.geojson"),
        score_column=config["score_column"],
    )

    # OAT sensitivity results
    oat_df.to_csv(
        os.path.join(config["output_dir"], "sensitivity_oat.csv"),
        index=False, float_format="%.4f",
    )

    # Summary report
    generate_summary_report(
        ranked_df, weights, norm_cols,
        os.path.join(config["output_dir"], "summary_report.txt"),
        score_column=config["score_column"],
        top_n=config["top_n"],
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    _banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")

    print("\nOutput files:")
    for fname in sorted(os.listdir(config["output_dir"])):
        fpath = os.path.join(config["output_dir"], fname)
        size  = os.path.getsize(fpath) / 1024
        print(f"  {fname:45s}  {size:7.0f} KB")

    return {
        "hex_grid":    hex_grid,
        "features_df": features_df,
        "norm_df":     norm_df,
        "model":       model,
        "weights":     weights,
        "labels":      labels,
        "scored_df":   scored_df,
        "ranked_df":   ranked_df,
        "shap_df":     shap_df,
        "oat_df":      oat_df,
        "mc_df":       mc_df,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colombia Wind Farm Suitability – MCDA Pipeline"
    )
    parser.add_argument(
        "--resolution", type=int, default=6,
        choices=[4, 5, 55, 65, 6, 75, 7],
        help="Grid resolution: 4=~88km, 5=~44km, 55=~27km (~600 hex), 65=~23km (~800 hex), 6=~22km, 7=~11km"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed"
    )
    parser.add_argument(
        "--norm-method", choices=["minmax", "sigmoid"], default="minmax",
        dest="norm_method",
        help="Feature normalisation method"
    )
    parser.add_argument(
        "--mc-samples", type=int, default=300,
        dest="mc_n_samples",
        help="Monte Carlo sensitivity sample count"
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        dest="top_n",
        help="Number of top locations to display in ranking"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update({
        "resolution":    args.resolution,
        "feature_seed":  args.seed,
        "rf_seed":       args.seed,
        "norm_method":   args.norm_method,
        "mc_n_samples":  args.mc_n_samples,
        "top_n":         args.top_n,
    })

    results = run_pipeline(config)


