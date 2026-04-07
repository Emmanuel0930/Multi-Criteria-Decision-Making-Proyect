"""
ahp_model.py
============
Implementación del Analytic Hierarchy Process (AHP) para el modelo MCDA
de aptitud para parques eólicos en Colombia.

Referencia metodológica
-----------------------
Saaty, T.L. (1980). The Analytic Hierarchy Process.
McGraw-Hill, New York.

La escala de Saaty para comparaciones por pares:
    1  = igual importancia
    3  = moderadamente más importante
    5  = fuertemente más importante
    7  = muy fuertemente más importante
    9  = extremadamente más importante
    2,4,6,8 = valores intermedios

Ratio de Consistencia (CR)
--------------------------
    CR = CI / RI
    CI = (lambda_max - n) / (n - 1)
    RI = índice aleatorio tabulado por Saaty según n criterios

    CR < 0.10  → matriz consistente (aceptable)
    CR >= 0.10 → inconsistente, revisar juicios
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Índices aleatorios de Saaty (RI) para n = 1..15
# ---------------------------------------------------------------------------
_RI = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}


# ---------------------------------------------------------------------------
# Funciones matemáticas del AHP
# ---------------------------------------------------------------------------

def compute_ahp_weights(
    pairwise_matrix: np.ndarray,
    criteria: List[str],
) -> Tuple[Dict[str, float], float, float, bool]:
    """
    Calcula los pesos AHP a partir de una matriz de comparación por pares
    y devuelve el Ratio de Consistencia (CR).

    Método: eigenvector principal (método exacto de Saaty).

    Parameters
    ----------
    pairwise_matrix : array (n, n) con las comparaciones por pares.
                      Debe ser recíproca: M[i,j] = 1 / M[j,i].
                      La diagonal debe ser 1.
    criteria        : lista de nombres de criterios (len == n)

    Returns
    -------
    weights    : dict {criterio: peso_normalizado}  (suman 1.0)
    lambda_max : eigenvalor principal
    cr         : Consistency Ratio (debe ser < 0.10)
    is_consistent : True si CR < 0.10
    """
    M = np.array(pairwise_matrix, dtype=np.float64)
    n = M.shape[0]
    assert M.shape == (n, n), "La matriz debe ser cuadrada."
    assert len(criteria) == n, f"Se esperaban {n} criterios, se dieron {len(criteria)}."

    # 1. Normalizar columnas
    col_sums = M.sum(axis=0)
    M_norm = M / col_sums

    # 2. Vector de pesos = media de filas
    weights_vec = M_norm.mean(axis=1)

    # 3. Lambda máximo
    weighted_sum = M @ weights_vec
    lambda_max = (weighted_sum / weights_vec).mean()

    # 4. Consistency Index
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0

    # 5. Consistency Ratio
    ri = _RI.get(n, 1.59)
    cr = ci / ri if ri > 0 else 0.0

    is_consistent = cr < 0.10

    weights = {crit: float(w) for crit, w in zip(criteria, weights_vec)}

    return weights, float(lambda_max), float(cr), is_consistent


def build_pairwise_matrix(
    criteria: List[str],
    comparisons: Dict[Tuple[str, str], float],
) -> np.ndarray:
    """
    Construye la matriz de comparación por pares a partir de un dict
    de juicios explícitos. Los recíprocos se calculan automáticamente.

    Parameters
    ----------
    criteria    : lista de nombres de criterios
    comparisons : dict {(criterio_i, criterio_j): valor_saaty}
                  Solo necesitas la mitad superior de la matriz.
                  Ej: {("wind_speed", "slope"): 3} significa que
                  wind_speed es moderadamente más importante que slope.

    Returns
    -------
    Matriz (n, n) simétrica recíproca con diagonal = 1
    """
    n = len(criteria)
    idx = {c: i for i, c in enumerate(criteria)}
    M = np.ones((n, n), dtype=np.float64)

    for (ci, cj), val in comparisons.items():
        i, j = idx[ci], idx[cj]
        M[i, j] = val
        M[j, i] = 1.0 / val

    return M


def compute_ahp_scores(
    norm_df: pd.DataFrame,
    weights: Dict[str, float],
    norm_cols: List[str],
    score_column: str = "suitability_score",
) -> pd.DataFrame:
    """
    Aplica los pesos AHP mediante combinación lineal ponderada (WLC)
    sobre las columnas normalizadas.

    Idéntico matemáticamente al WLC del modelo RF, pero con pesos
    derivados de AHP en lugar de importancia de Random Forest.

    Parameters
    ----------
    norm_df      : DataFrame con columnas normalizadas [0, 1]
    weights      : dict {col_norm: peso}  (deben sumar 1.0)
    norm_cols    : lista ordenada de columnas normalizadas a usar
    score_column : nombre de la columna resultado

    Returns
    -------
    DataFrame con columna score_column añadida
    """
    df = norm_df.copy()

    # Construir vector de pesos alineado con norm_cols
    w_vector = np.array([weights.get(col, 0.0) for col in norm_cols])

    # Renormalizar por si los pesos no suman exactamente 1
    total = w_vector.sum()
    if total > 0:
        w_vector = w_vector / total

    X = df[norm_cols].fillna(0).to_numpy()
    df[score_column] = (X @ w_vector).round(6)

    print(f"[AHP] Puntuaciones calculadas para {len(df):,} celdas.")
    stats = df[score_column]
    print(f"[AHP] Score:  min={stats.min():.3f}  max={stats.max():.3f}  "
          f"mean={stats.mean():.3f}  std={stats.std():.3f}")

    return df


def print_ahp_report(
    weights: Dict[str, float],
    lambda_max: float,
    cr: float,
    is_consistent: bool,
    pairwise_matrix: np.ndarray,
    criteria: List[str],
) -> None:
    """Imprime un reporte completo del AHP en consola."""
    n = len(criteria)
    bar = "=" * 70

    print(f"\n{bar}")
    print("  REPORTE AHP — Analytic Hierarchy Process")
    print(bar)

    # Matriz de comparación
    print("\n  Matriz de comparación por pares:")
    header = f"{'':22s}" + "".join(f"{c:>12s}" for c in criteria)
    print(f"  {header}")
    for i, ci in enumerate(criteria):
        row = f"  {ci:22s}" + "".join(f"{pairwise_matrix[i,j]:>12.3f}" for j in range(n))
        print(row)

    # Pesos
    print("\n  Pesos normalizados (eigenvector principal):")
    for crit, w in sorted(weights.items(), key=lambda x: -x[1]):
        bar_len = int(w * 40)
        print(f"  {crit:30s}  {w:.4f}  {'█' * bar_len}")

    # Consistencia
    print(f"\n  λ_max        = {lambda_max:.4f}")
    print(f"  n criterios  = {n}")
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    print(f"  CI           = {ci:.4f}")
    ri = _RI.get(n, 1.59)
    print(f"  RI (n={n})     = {ri:.4f}")
    print(f"  CR           = {cr:.4f}  {'✓ CONSISTENTE (CR < 0.10)' if is_consistent else '✗ INCONSISTENTE — revisar juicios'}")
    print(bar)


