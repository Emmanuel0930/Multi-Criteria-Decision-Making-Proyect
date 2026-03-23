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

import os
import sys
import time
from typing import Any, Dict, List, Tuple

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
from ahp_model            import (build_pairwise_matrix, compute_ahp_weights,
                                    compute_ahp_scores, print_ahp_report)
from visualization        import (create_interactive_map,
                                   plot_score_distribution, plot_feature_correlation)

# ===========================================================================
# CONFIGURACIÓN GENERAL
# ===========================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    # Algoritmo MCDA: "wlc" o "ahp"
    "algorithm":      "ahp",

    # Grid
    "resolution":     6,          # 4=~88km  5=~44km  55=~27km  6=~22km  7=~11km  8=~5km

    # Feature engineering
    "feature_seed":   42,

    # Normalización
    "norm_method":    "minmax",   # "minmax" | "sigmoid"

    # Random Forest (solo se usa con algorithm="wlc")
    "rf_n_estimators": 200,
    "rf_seed":         42,
    "label_threshold": 0.55,

    # Ranking
    "score_column":   "suitability_score",
    "top_n":          20,
    "protected_threshold": 0.7,   # celdas con área protegida > umbral se excluyen

    # Rutas
    "geojson_path":        os.path.join(_DATA_DIR, "colombia_boundary.geojson"),
    "output_dir":          _OUT_DIR,
}


# ===========================================================================
# CONFIGURACIÓN AHP
# ===========================================================================
# Criterios en el mismo orden que get_norm_feature_names()
# (wind_speed_norm, slope_norm, dist_to_grid_norm, dist_to_roads_norm,
#  land_use_norm, protected_area_norm, conflict_risk_norm)
#
# CÓMO EDITAR:
#   1. Modifica los valores en AHP_COMPARISONS usando la escala de Saaty:
#      1=igual  3=moderado  5=fuerte  7=muy fuerte  9=extremo
#      Los valores intermedios 2,4,6,8 son aceptables.
#   2. El par (A, B) = v significa "A es v veces más importante que B".
#      El recíproco (B, A) = 1/v se calcula automáticamente.
#   3. Ejecuta el pipeline y revisa el CR en consola.
#      Si CR >= 0.10 los juicios son inconsistentes — revisa los valores.
#
# Fuente de referencia para los valores iniciales:
#   Tegou et al. (2010) - Environmental management framework for wind farm
#   siting. Energy Policy, 38(5), 2341-2350.
#   Aydin et al. (2013) - GIS-based site selection methodology for hybrid
#   renewable energy systems. Energy Conversion and Management, 70, 90-106.
# ---------------------------------------------------------------------------

AHP_CRITERIA: List[str] = [
    "wind_speed_norm",       # C1 – Velocidad del viento
    "slope_norm",            # C2 – Pendiente (normalizada inversa)
    "dist_to_grid_norm",     # C3 – Distancia a red eléctrica (inversa)
    "dist_to_roads_norm",    # C4 – Distancia a carreteras (inversa)
    "land_use_norm",         # C5 – Aptitud del uso del suelo
    "protected_area_norm",   # C6 – Área protegida (inversa)
    "conflict_risk_norm",    # C7 – Riesgo de conflicto (inverso)
]

AHP_COMPARISONS: Dict[Tuple[str, str], float] = {
    # Viento es el criterio dominante para energía eólica
    ("wind_speed_norm",     "slope_norm"):           5,   # viento >> pendiente
    ("wind_speed_norm",     "dist_to_grid_norm"):    3,   # viento > distancia red
    ("wind_speed_norm",     "dist_to_roads_norm"):   4,   # viento > distancia vías
    ("wind_speed_norm",     "land_use_norm"):         4,   # viento > uso suelo
    ("wind_speed_norm",     "protected_area_norm"):   7,   # viento >> área protegida
    ("wind_speed_norm",     "conflict_risk_norm"):    6,   # viento >> conflicto

    # Área protegida es una restricción importante (en negativo)
    ("slope_norm",          "dist_to_grid_norm"):    0.5, # pendiente < distancia red
    ("slope_norm",          "dist_to_roads_norm"):   0.5, # pendiente < distancia vías
    ("slope_norm",          "land_use_norm"):         0.5, # pendiente < uso suelo
    ("slope_norm",          "protected_area_norm"):   3,   # pendiente > área protegida
    ("slope_norm",          "conflict_risk_norm"):    2,   # pendiente > conflicto

    ("dist_to_grid_norm",   "dist_to_roads_norm"):   2,   # red > vías
    ("dist_to_grid_norm",   "land_use_norm"):         1,   # red = uso suelo
    ("dist_to_grid_norm",   "protected_area_norm"):   4,   # red > área protegida
    ("dist_to_grid_norm",   "conflict_risk_norm"):    3,   # red > conflicto

    ("dist_to_roads_norm",  "land_use_norm"):         0.5, # vías < uso suelo
    ("dist_to_roads_norm",  "protected_area_norm"):   3,   # vías > área protegida
    ("dist_to_roads_norm",  "conflict_risk_norm"):    2,   # vías > conflicto

    ("land_use_norm",       "protected_area_norm"):   4,   # uso suelo > área protegida
    ("land_use_norm",       "conflict_risk_norm"):    3,   # uso suelo > conflicto

    ("protected_area_norm", "conflict_risk_norm"):    0.5, # área protegida < conflicto
}


# ===========================================================================
# PASOS DEL PIPELINE
# ===========================================================================

def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta el pipeline MCDA completo según el algoritmo seleccionado.

    Returns
    -------
    Dict con: hex_grid, features_df, norm_df, weights, scored_df, ranked_df
    """
    os.makedirs(config["output_dir"], exist_ok=True)
    t0 = time.time()
    algorithm = config["algorithm"].lower()

    print(f"\n  Algoritmo seleccionado: {algorithm.upper()}")

    # ------------------------------------------------------------------
    # Paso 1 – Grilla H3
    # ------------------------------------------------------------------
    _banner("PASO 1 – Generando grilla hexagonal H3")
    load_boundary(config["geojson_path"])
    hex_grid = generate_colombia_hex_grid(
        config["geojson_path"],
        resolution=config["resolution"],
    )

    # ------------------------------------------------------------------
    # Paso 2 – Features espaciales
    # ------------------------------------------------------------------
    _banner("PASO 2 – Ingeniería de características espaciales")
    features_df = engineer_features(hex_grid, seed=config["feature_seed"])
    raw_cols = list(FEATURE_DIRECTION.keys())
    validate_feature_dataframe(features_df, ["hex_id", "lon", "lat"] + raw_cols)

    # ------------------------------------------------------------------
    # Paso 3 – Normalización
    # ------------------------------------------------------------------
    _banner("PASO 3 – Normalizando criterios a escala [0, 1]")
    norm_df = normalise_features(features_df, method=config["norm_method"])
    norm_cols = get_norm_feature_names()

    # ------------------------------------------------------------------
    # Paso 4 – Pesos según algoritmo
    # ------------------------------------------------------------------
    model  = None
    labels = None

    if algorithm == "wlc":
        _banner("PASO 4 – WLC: entrenando Random Forest para derivar pesos")
        model, weights, labels = get_rf_weights(
            norm_df,
            norm_cols,
            suitability_threshold=config["label_threshold"],
            n_estimators=config["rf_n_estimators"],
            seed=config["rf_seed"],
        )
        print("\n[WLC] Pesos derivados del Random Forest (MDI):")
        for col, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {col:30s}  {w:.4f}  {'█' * int(w * 40)}")

    elif algorithm == "ahp":
        _banner("PASO 4 – AHP: calculando pesos desde matriz de comparación")

        # Construir la matriz y calcular pesos
        pairwise_matrix = build_pairwise_matrix(AHP_CRITERIA, AHP_COMPARISONS)
        weights, lambda_max, cr, is_consistent = compute_ahp_weights(
            pairwise_matrix, AHP_CRITERIA
        )

        # Reporte detallado
        print_ahp_report(weights, lambda_max, cr, is_consistent,
                         pairwise_matrix, AHP_CRITERIA)

        if not is_consistent:
            print("\n  ⚠  CR >= 0.10 — los juicios son inconsistentes.")
            print("     Revisa AHP_COMPARISONS en main.py y ajusta los valores.")
            print("     El pipeline continúa pero los pesos pueden no ser fiables.")

        # Asegurarse de que todos los norm_cols tienen peso (0 si no están en AHP_CRITERIA)
        for col in norm_cols:
            if col not in weights:
                weights[col] = 0.0

    else:
        raise ValueError(
            f"Algoritmo desconocido: '{algorithm}'. "
            f"Opciones válidas: 'wlc', 'ahp'."
        )

    # ------------------------------------------------------------------
    # Paso 5 – Puntuaciones MCDA
    # ------------------------------------------------------------------
    _banner("PASO 5 – Calculando puntuaciones de aptitud")

    if algorithm == "wlc":
        scored_df = compute_wlc_scores(
            norm_df, weights, norm_cols,
            score_column=config["score_column"],
        )
    else:  # ahp
        scored_df = compute_ahp_scores(
            norm_df, weights, norm_cols,
            score_column=config["score_column"],
        )

    # ------------------------------------------------------------------
    # Paso 6 – Ranking
    # ------------------------------------------------------------------
    _banner("PASO 6 – Ranking de localidades candidatas")
    ranked_df = rank_locations(
        scored_df,
        score_column=config["score_column"],
        top_n=None,
        protected_threshold=config["protected_threshold"],
    )
    summarise_top_locations(ranked_df, top_n=config["top_n"],
                             score_column=config["score_column"])

    # ------------------------------------------------------------------
    # Paso 7 – Visualización y exportación
    # ------------------------------------------------------------------
    _banner("PASO 7 – Generando visualizaciones y exportando resultados")

    # Mapa interactivo HTML
    create_interactive_map(
        scored_df,
        os.path.join(config["output_dir"], "map_interactive.html"),
        score_column=config["score_column"],
        top_n_highlight=10,
    )

    # Distribución de puntuaciones
    plot_score_distribution(
        ranked_df,
        os.path.join(config["output_dir"], "score_distribution.png"),
        score_column=config["score_column"],
    )

    # Correlación de features
    plot_feature_correlation(
        norm_df, norm_cols,
        os.path.join(config["output_dir"], "feature_correlation.png"),
        score_column=config["score_column"],
    )

    # Exportar CSV y GeoJSON
    save_results_csv(
        ranked_df,
        os.path.join(config["output_dir"], "suitability_scores.csv"),
    )
    save_results_geojson(
        ranked_df,
        os.path.join(config["output_dir"], "suitability_scores.geojson"),
        score_column=config["score_column"],
    )

    # Reporte textual
    generate_summary_report(
        ranked_df, weights, norm_cols,
        os.path.join(config["output_dir"], "summary_report.txt"),
        score_column=config["score_column"],
        top_n=config["top_n"],
    )

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    _banner(f"PIPELINE COMPLETO  ({elapsed:.1f}s)  [{algorithm.upper()}]")

    print("\nArchivos generados:")
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
    }


# ===========================================================================
# MENÚ INTERACTIVO
# ===========================================================================

def _show_menu() -> Dict[str, Any]:
    """
    Muestra el menú de selección de algoritmo y retorna la configuración
    elegida por el usuario.
    """
    print("\n" + "=" * 50)
    print("  EVALUACIÓN DE APTITUD — PARQUES EÓLICOS")
    print("  Colombia · MCDA")
    print("=" * 50)
    print("\n  Selecciona el algoritmo a utilizar:\n")
    print("    1. WLC  (Weighted Linear Combination)")
    print("       Pesos derivados automáticamente por Random Forest\n")
    print("    2. AHP  (Analytic Hierarchy Process)")
    print("       Pesos fijos basados en literatura eólica\n")
    print("=" * 50)

    while True:
        opcion = input("\n  Ingresa 1 o 2: ").strip()
        if opcion == "1":
            algorithm = "wlc"
            print("\n  ✓ Seleccionado: WLC + Random Forest")
            break
        elif opcion == "2":
            algorithm = "ahp"
            print("\n  ✓ Seleccionado: AHP")
            break
        else:
            print("  Opción inválida. Por favor ingresa 1 o 2.")

    config = dict(DEFAULT_CONFIG)
    config["algorithm"] = algorithm
    return config


if __name__ == "__main__":
    config = _show_menu()
    run_pipeline(config)
