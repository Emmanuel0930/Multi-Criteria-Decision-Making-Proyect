"""
bwm_promethee_model.py
======================
Modelo híbrido MCDA: Best Worst Method (BWM) + PROMETHEE II.

Diseñado para escalar desde miles hasta millones de alternativas (hexágonos H3)
mediante procesamiento por bloques (chunking) que mantiene el pico de RAM
bajo un umbral configurable.

Dependencias: numpy, pandas, scipy
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: estimación y configuración de chunk_size
# ─────────────────────────────────────────────────────────────────────────────

def recommended_chunk_size(
    n_alternatives: int,
    n_criteria: int,
    ram_gb: float = 8.0,
    safety_factor: float = 0.5,
) -> int:
    """
    Calcula el chunk_size óptimo para no superar un umbral de RAM.

    La matriz de diferencias D tiene forma (chunk_size, n_alternatives) por
    criterio y dtype float64 (8 bytes). El factor de seguridad (0.5 por defecto)
    reserva la mitad de la RAM para el resto del proceso.

    Parameters
    ----------
    n_alternatives : int   Número total de alternativas (hexágonos).
    n_criteria     : int   Número de criterios.
    ram_gb         : float RAM disponible en GB.
    safety_factor  : float Fracción de la RAM a usar (0–1).

    Returns
    -------
    int  chunk_size recomendado (mínimo 100, máximo n_alternatives).
    """
    bytes_available = ram_gb * (1024 ** 3) * safety_factor
    # Cada fila del chunk necesita n_alternatives × n_criteria × 8 bytes
    bytes_per_row = n_alternatives * n_criteria * 8
    chunk = int(bytes_available / bytes_per_row)
    chunk = max(100, min(chunk, n_alternatives))
    return chunk


# ─────────────────────────────────────────────────────────────────────────────
# 1. BWM — Best Worst Method
# ─────────────────────────────────────────────────────────────────────────────

def compute_bwm_weights(
    best_to_others: Dict[str, float],
    others_to_worst: Dict[str, float],
    criteria: List[str],
) -> Dict[str, float]:
    """
    Calcula los pesos de los criterios usando el Best Worst Method (BWM)
    con formulación lineal (Rezaei, 2016).

    Minimiza ξ (máxima violación de consistencia) sujeto a:
        |w_B - a_Bj · w_j| ≤ ξ   ∀j
        |w_j - a_jW · w_W| ≤ ξ   ∀j
        Σ w_j = 1,  w_j ≥ 0

    Parameters
    ----------
    best_to_others : Dict[str, float]
        Preferencia del mejor criterio sobre cada uno (escala 1–9).
        El mejor criterio tiene valor 1 sobre sí mismo.
    others_to_worst : Dict[str, float]
        Preferencia de cada criterio sobre el peor (escala 1–9).
        El peor criterio tiene valor 1 sobre sí mismo.
    criteria : List[str]
        Lista de criterios en el orden deseado.

    Returns
    -------
    Dict[str, float]  Pesos normalizados (suman exactamente 1).
    """
    n = len(criteria)
    idx = {c: i for i, c in enumerate(criteria)}

    best  = min(best_to_others,  key=best_to_others.get)
    worst = min(others_to_worst, key=others_to_worst.get)

    a_B = np.array([best_to_others[c]  for c in criteria], dtype=float)
    a_W = np.array([others_to_worst[c] for c in criteria], dtype=float)

    # Variables: [w_0 … w_{n-1}, ξ]
    c_obj      = np.zeros(n + 1); c_obj[-1] = 1.0
    A_ub_rows  = []
    b_ub_rows  = []
    b_idx, w_idx = idx[best], idx[worst]

    for j in range(n):
        for sign in (1, -1):
            # Restricción BWM-B: sign*(w_B - a_Bj*w_j) ≤ ξ
            r1 = np.zeros(n + 1)
            r1[b_idx] =  sign
            r1[j]     = -sign * a_B[j]
            r1[-1]    = -1.0
            A_ub_rows.append(r1); b_ub_rows.append(0.0)

            # Restricción BWM-W: sign*(w_j - a_jW*w_W) ≤ ξ
            r2 = np.zeros(n + 1)
            r2[j]     =  sign
            r2[w_idx] = -sign * a_W[j]
            r2[-1]    = -1.0
            A_ub_rows.append(r2); b_ub_rows.append(0.0)

    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    bounds = [(0.0, None)] * n + [(0.0, None)]

    result = linprog(
        c_obj,
        A_ub=np.array(A_ub_rows), b_ub=np.array(b_ub_rows),
        A_eq=A_eq, b_eq=np.array([1.0]),
        bounds=bounds, method="highs",
    )
    if not result.success:
        raise RuntimeError(f"[BWM] Optimización fallida: {result.message}")

    raw = result.x[:n]
    return {c: float(w) for c, w in zip(criteria, raw / raw.sum())}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Funciones de preferencia PROMETHEE
# ─────────────────────────────────────────────────────────────────────────────

def _pf_usual(d: np.ndarray, **_)    -> np.ndarray:
    return (d > 0).astype(float)

def _pf_ushape(d: np.ndarray, q: float = 0.1, **_) -> np.ndarray:
    return (d > q).astype(float)

def _pf_vshape(d: np.ndarray, p: float = 0.5, **_) -> np.ndarray:
    return np.where(d <= 0, 0.0, np.where(d >= p, 1.0, d / p))

def _pf_linear(d: np.ndarray, q: float = 0.1, p: float = 0.5, **_) -> np.ndarray:
    return np.where(d <= q, 0.0, np.where(d >= p, 1.0, (d - q) / (p - q)))

def _pf_gaussian(d: np.ndarray, s: float = 0.3, **_) -> np.ndarray:
    return np.where(d <= 0, 0.0, 1.0 - np.exp(-(d ** 2) / (2 * s ** 2)))

_PREFERENCE_FUNCTIONS = {
    "usual":    _pf_usual,
    "u-shape":  _pf_ushape,
    "ushape":   _pf_ushape,
    "v-shape":  _pf_vshape,
    "vshape":   _pf_vshape,
    "linear":   _pf_linear,
    "gaussian": _pf_gaussian,
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMETHEE II con chunking
# ─────────────────────────────────────────────────────────────────────────────

def compute_promethee_scores(
    norm_df: pd.DataFrame,
    weights: Dict[str, float],
    criteria: List[str],
    preference_functions: Dict[str, str],
    pf_params: Optional[Dict[str, Dict]] = None,
    chunk_size: Optional[int] = None,
    ram_gb: float = 8.0,
    score_column: str = "promethee_score",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calcula el flujo neto PROMETHEE II con chunking escalable.

    Estrategia de chunking
    ----------------------
    En lugar de construir la matriz completa D[n × n] — que para 200k
    alternativas requeriría ~320 GB — se procesan bloques de filas:

        Para cada chunk de alternativas i = [start, end):
            D_chunk[chunk_size × n] = M[start:end, k] - M[:, k]ᵀ
            φ+[start:end] += w_k · P(D_chunk).sum(axis=1)   (cuánto domina)
            φ-            += w_k · P(D_chunk).sum(axis=0)   (cuánto es dominado)

    El resultado es matemáticamente idéntico al cálculo completo.
    El pico de RAM es O(chunk_size × n) en lugar de O(n²).

    Fórmula de memoria por chunk (por criterio):
        RAM ≈ chunk_size × n × 8 bytes (float64)
    Ejemplo: chunk=5000, n=200k → 5000 × 200000 × 8 = 800 MB

    Parameters
    ----------
    norm_df            : DataFrame con columnas normalizadas en [0, 1]. NaN→0.
    weights            : Pesos por criterio (suman 1).
    criteria           : Columnas a usar como criterios.
    preference_functions : Tipo de función por criterio.
    pf_params          : Parámetros adicionales (q, p, s).
    chunk_size         : Filas por bloque. Si None, se calcula automáticamente.
    ram_gb             : RAM disponible en GB (para cálculo automático).
    score_column       : Nombre de la columna de salida.
    verbose            : Imprimir progreso por chunk.

    Returns
    -------
    pd.DataFrame  norm_df con columna score_column (flujo neto φ).
    """
    pf_params = pf_params or {}
    M = norm_df[criteria].fillna(0.0).to_numpy(dtype=float)
    n, k = M.shape
    w = np.array([weights[c] for c in criteria], dtype=float)

    # Determinar chunk_size
    if chunk_size is None:
        chunk_size = recommended_chunk_size(n, k, ram_gb)
        if verbose:
            mem_mb = chunk_size * n * 8 / (1024 ** 2)
            print(f"[Chunking] chunk_size automático = {chunk_size:,} "
                  f"(~{mem_mb:.0f} MB pico por criterio, RAM={ram_gb} GB)")

    n_chunks = math.ceil(n / chunk_size)

    # Validar funciones de preferencia antes de iterar
    pf_funcs = []
    for crit in criteria:
        name = preference_functions.get(crit, "usual").lower()
        fn = _PREFERENCE_FUNCTIONS.get(name)
        if fn is None:
            raise ValueError(
                f"Función desconocida '{name}' para '{crit}'. "
                f"Opciones: {list(_PREFERENCE_FUNCTIONS.keys())}"
            )
        pf_funcs.append((fn, pf_params.get(crit, {})))

    phi_plus  = np.zeros(n)
    phi_minus = np.zeros(n)

    if verbose:
        print(f"[PROMETHEE] {n:,} alternativas · {k} criterios · "
              f"{n_chunks} chunks de hasta {chunk_size:,} filas")

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, n)
        M_chunk = M[start:end]          # (chunk_size_real × k)
        c_size  = end - start

        # Acumular contribución de cada criterio en este chunk
        # D[i_local, j] = M_chunk[i_local, k] - M[j, k]
        for ci, (fn, params) in enumerate(pf_funcs):
            col_chunk = M_chunk[:, ci]  # (c_size,)
            col_all   = M[:, ci]        # (n,)

            # Diferencias: shape (c_size, n)
            D = col_chunk[:, None] - col_all[None, :]

            P = fn(D, **params)         # (c_size, n)

            phi_plus[start:end] += w[ci] * P.sum(axis=1)
            phi_minus           += w[ci] * P.sum(axis=0)

        if verbose:
            pct = (chunk_idx + 1) / n_chunks * 100
            print(f"  chunk {chunk_idx + 1:>4}/{n_chunks}  "
                  f"filas {start:>8,}–{end - 1:>8,}  [{pct:5.1f}%]",
                  end="\r" if chunk_idx < n_chunks - 1 else "\n")

    # Normalizar por (n-1) para obtener flujos medios
    if n > 1:
        phi_plus  /= (n - 1)
        phi_minus /= (n - 1)

    phi_net = phi_plus - phi_minus

    result = norm_df.copy()
    result[score_column] = phi_net
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline completo: BWM + PROMETHEE II
# ─────────────────────────────────────────────────────────────────────────────

def compute_bwm_promethee(
    norm_df: pd.DataFrame,
    criteria: List[str],
    best_to_others: Dict[str, float],
    others_to_worst: Dict[str, float],
    preference_functions: Dict[str, str],
    pf_params: Optional[Dict[str, Dict]] = None,
    chunk_size: Optional[int] = None,
    ram_gb: float = 8.0,
    score_column: str = "suitability_score",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo escalable: Best Worst Method → PROMETHEE II.

    Equivalente funcional de compute_ahp_scores (AHP + WLC), diseñado
    para 200k–2M hexágonos H3 con uso de RAM controlado por chunking.

    Guía rápida de chunk_size según RAM disponible
    -----------------------------------------------
    RAM disponible   chunk_size sugerido   Pico RAM por criterio
    8  GB            2 000                 ~300 MB
    16 GB            5 000                 ~750 MB
    32 GB            10 000               ~1.5 GB
    servidor/HPC     25 000               ~3.7 GB

    Deja chunk_size=None para selección automática basada en ram_gb.

    Parameters
    ----------
    norm_df              : DataFrame con alternativas y columnas en [0, 1].
    criteria             : Criterios (deben existir como columnas).
    best_to_others       : Comparaciones BWM del mejor criterio.
    others_to_worst      : Comparaciones BWM de cada criterio al peor.
    preference_functions : Tipo de función PROMETHEE por criterio.
    pf_params            : Parámetros adicionales de preferencia.
    chunk_size           : Filas por bloque (None = automático).
    ram_gb               : RAM disponible para cálculo automático de chunk.
    score_column         : Nombre de la columna de score final.
    verbose              : Mostrar progreso y resumen.

    Returns
    -------
    pd.DataFrame  norm_df con columna score_column (flujo neto φ de PROMETHEE II).
    """
    missing = [c for c in criteria if c not in norm_df.columns]
    if missing:
        raise ValueError(f"Criterios no encontrados en el DataFrame: {missing}")

    # ── Paso 1: Pesos BWM ──────────────────────────────────────────────────
    weights = compute_bwm_weights(best_to_others, others_to_worst, criteria)

    if verbose:
        print("\n[BWM] Pesos calculados:")
        for c in criteria:
            print(f"  {c:<20}: {weights[c]:.4f}")
        print(f"  {'Suma total':<20}: {sum(weights.values()):.6f}\n")

    # ── Paso 2: PROMETHEE II con chunking ──────────────────────────────────
    result_df = compute_promethee_scores(
        norm_df=norm_df,
        weights=weights,
        criteria=criteria,
        preference_functions=preference_functions,
        pf_params=pf_params,
        chunk_size=chunk_size,
        ram_gb=ram_gb,
        score_column=score_column,
        verbose=verbose,
    )

    if verbose:
        scores = result_df[score_column]
        print(f"\n[PROMETHEE II] Score stats ('{score_column}'):")
        print(f"  min  : {scores.min():.6f}")
        print(f"  max  : {scores.max():.6f}")
        print(f"  mean : {scores.mean():.6f}")
        print(f"  std  : {scores.std():.6f}")
        print(f"  n    : {len(scores):,}\n")

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Ejemplo de uso
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    N_HEX = 200_000          # Simula tu escenario real
    RAM_GB = 16.0            # Ajusta a tu servidor

    print(f"Generando {N_HEX:,} hexágonos simulados...")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "hex_id":   [f"hex_{i:07d}" for i in range(N_HEX)],
        "wind":     rng.random(N_HEX),
        "slope":    rng.random(N_HEX),
        "distance": rng.random(N_HEX),
        "grid":     rng.random(N_HEX),
    }).set_index("hex_id")

    # Consultar chunk_size recomendado antes de ejecutar
    cs = recommended_chunk_size(N_HEX, n_criteria=4, ram_gb=RAM_GB)
    print(f"chunk_size recomendado para {RAM_GB} GB RAM: {cs:,}\n")

    t0 = time.perf_counter()
    df_result = compute_bwm_promethee(
        norm_df=df,
        criteria=["wind", "slope", "distance", "grid"],
        best_to_others={"wind": 1, "slope": 5, "distance": 7, "grid": 3},
        others_to_worst={"wind": 7, "slope": 3, "distance": 1, "grid": 5},
        preference_functions={
            "wind":     "usual",
            "slope":    "v-shape",
            "distance": "linear",
            "grid":     "gaussian",
        },
        pf_params={
            "slope":    {"p": 0.3},
            "distance": {"q": 0.05, "p": 0.4},
            "grid":     {"s": 0.25},
        },
        chunk_size=None,     # Se calcula automáticamente
        ram_gb=RAM_GB,
        score_column="suitability_score",
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nTiempo total: {elapsed:.1f}s")
    print("\nTop 10 hexágonos más aptos:")
    print(
        df_result["suitability_score"]
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )