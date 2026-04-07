"""
feature_engineering.py
=======================
Computes (or realistically simulates) spatial features for each H3 hexagon.

In a production system these features would be extracted from:
  - ERA5 / MERRA-2 reanalysis data     → wind speed
  - SRTM / ALOS DEM                    → terrain slope
  - OpenStreetMap / UPME network data  → distance to transmission lines / roads
  - IGAC / IDEAM land-use maps         → land use category
  - RUNAP registry                     → protected area indicator
  - UNOCHA / SIPRI conflict databases  → conflict risk indicator

For this research prototype we generate *spatially correlated synthetic data*
that preserves realistic geographic patterns:

  • Wind speed   – higher near the Caribbean coast (north) and Andean passes
  • Slope        – higher in the Andes cordilleras (centre strip)
  • Grid access  – better near the main urban corridor (Bogotá–Medellín–Cali)
  • Road access  – follows similar pattern to grid
  • Land use     – mixed; highlands less suitable
  • Protected    – national parks mainly in Amazonia and Pacific coast
  • Conflict     – historically higher in south / border regions

MEMORY OPTIMISATION (v2)
------------------------
The original implementation materialised an (n_hexagons × n_basis) float64
matrix all at once.  At resolution 7 (216 k hexagons, ~10 k basis points) that
requires ~17 GB of RAM and raises ArrayMemoryError.

Two changes fix this:

1. **n_basis cap** – The smoothness of the Gaussian field does not improve
   beyond ~500–800 basis points.  We cap n_basis at MAX_BASIS_POINTS (default
   600) regardless of grid size.  This keeps the matrix under ~1 GB even at
   the highest resolutions.

2. **Chunked dot-product** – The RBF matrix is computed and accumulated in
   row-chunks of CHUNK_SIZE rows (default 20 000).  Each chunk is at most
   20 000 × 600 × 8 bytes ≈ 96 MB, so peak RAM stays well below 200 MB for
   the gaussian field computation.

Both changes are backward-compatible: results at lower resolutions are
numerically identical (same seeds, same anchor points).
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tuneable memory-budget constants
# ---------------------------------------------------------------------------

#: Hard cap on the number of RBF anchor points.
#: 600 is sufficient for smooth continental-scale fields; raising it beyond
#: ~1000 gives diminishing returns and uses more memory.
MAX_BASIS_POINTS: int = 600

#: Number of hexagon rows processed per chunk in _gaussian_field.
#: 20 000 rows × 600 basis × 8 bytes = ~96 MB peak – safe on any modern laptop.
CHUNK_SIZE: int = 20_000


# ---------------------------------------------------------------------------
# Spatial covariance helper (Gaussian RBF kernel) – memory-safe version
# ---------------------------------------------------------------------------

def _gaussian_field(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    length_scale: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a spatially smooth random field via a simplified Gaussian process.

    Computes a weighted sum of Radial Basis Functions anchored at random
    locations, producing a smooth geographically-continuous surface without
    needing scipy or specialised GP libraries.

    Memory usage
    ------------
    Peak RAM ≈ CHUNK_SIZE × min(n // 20, MAX_BASIS_POINTS) × 8 bytes.
    With defaults (20 000 rows, 600 basis): ~96 MB per call.

    Parameters
    ----------
    lons, lats    : 1-D coordinate arrays (degrees)
    length_scale  : RBF bandwidth in degrees (~111 km per degree)
    seed          : random seed for reproducibility

    Returns
    -------
    Normalised field values in [0, 1], shape (n,)
    """
    rng = np.random.default_rng(seed)
    n = len(lons)

    # ── KEY FIX 1: cap n_basis so the RBF matrix never explodes ─────────────
    n_basis = min(max(50, n // 20), MAX_BASIS_POINTS)

    # Random anchor locations within the data extent
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    ax = rng.uniform(lon_min, lon_max, n_basis)
    ay = rng.uniform(lat_min, lat_max, n_basis)
    weights = rng.standard_normal(n_basis)

    inv_2ls2 = 1.0 / (2.0 * length_scale ** 2)

    # ── KEY FIX 2: chunked dot-product – never materialise the full matrix ───
    field = np.empty(n, dtype=np.float64)
    n_chunks = math.ceil(n / CHUNK_SIZE)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, n)

        lo_chunk = lons[start:end]
        la_chunk = lats[start:end]

        # shape: (chunk_size, n_basis)  ← at most CHUNK_SIZE × MAX_BASIS_POINTS
        d2 = (
            (lo_chunk[:, None] - ax[None, :]) ** 2
            + (la_chunk[:, None] - ay[None, :]) ** 2
        )
        rbf = np.exp(-d2 * inv_2ls2)           # (chunk_size, n_basis)
        field[start:end] = rbf @ weights        # (chunk_size,)

    # Normalise to [0, 1]
    f_min, f_max = field.min(), field.max()
    field = (field - f_min) / (f_max - f_min + 1e-12)
    return field


# ---------------------------------------------------------------------------
# Feature simulation functions
# ---------------------------------------------------------------------------

def _simulate_wind_speed(
    lons: np.ndarray,
    lats: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate average annual wind speed (m/s).

    Physical reasoning:
    - The Caribbean coast (lat > 9°) and La Guajira peninsula benefit from
      strong trade winds (avg 6–9 m/s).
    - The Andean highlands have orographic acceleration in passes (~5–7 m/s).
    - The Amazonian lowlands are calm (~2–3 m/s).

    Returns values roughly in [2, 9] m/s.
    """
    base = _gaussian_field(lons, lats, length_scale=3.0, seed=1)

    caribbean_boost = np.clip((lats - 9.0) / 3.5, 0, 1) * 0.35
    andean_mask  = np.exp(-((lons + 74.5) ** 2) / (1.5 ** 2))
    andean_boost = andean_mask * 0.25

    field = base * 0.4 + caribbean_boost + andean_boost
    field += rng.normal(0, 0.03, len(lons))
    field = np.clip(field, 0, 1)

    wind_ms = 1.5 + field * 8.0
    return wind_ms.round(2)


def _simulate_slope(
    lons: np.ndarray,
    lats: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate terrain slope in degrees.

    Andes run roughly as a diagonal band; lowlands are flat.
    High slope (>15°) is unfavourable for wind farm construction.

    Returns values in [0, 45] degrees.
    """
    w_ridge = np.exp(-((lons + 77.0) ** 2) / (0.8 ** 2))
    c_ridge = np.exp(-((lons + 75.5) ** 2) / (0.9 ** 2))
    e_ridge = np.exp(-((lons + 73.5) ** 2) / (1.0 ** 2))
    andean  = (w_ridge + c_ridge + e_ridge) / 3.0

    noise = _gaussian_field(lons, lats, length_scale=1.0, seed=7) * 0.3
    field = andean * 0.7 + noise
    field += rng.normal(0, 0.02, len(lons))
    field = np.clip(field, 0, 1)

    slope_deg = field * 45.0
    return slope_deg.round(2)


def _simulate_distance_to_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate distance to the nearest high-voltage transmission line (km).

    Returns values in [0, 300] km.
    """
    urban_centres = [
        (-74.1, 4.7),   # Bogotá
        (-75.6, 6.2),   # Medellín
        (-76.5, 3.4),   # Cali
        (-75.0, 10.4),  # Barranquilla
        (-73.1, 7.1),   # Bucaramanga
        (-76.2, 1.2),   # Pasto
    ]
    dists_deg = np.full(len(lons), np.inf)
    for (cx, cy) in urban_centres:
        d = np.sqrt((lons - cx) ** 2 + (lats - cy) ** 2)
        dists_deg = np.minimum(dists_deg, d)

    dist_km  = dists_deg * 111.0
    dist_km += rng.normal(0, 5.0, len(lons))
    dist_km  = np.clip(dist_km, 0, 300)
    return dist_km.round(1)


def _simulate_distance_to_roads(
    lons: np.ndarray,
    lats: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate distance to the nearest primary road (km).

    Returns values in [0, 200] km.
    """
    urban_centres = [
        (-74.1, 4.7), (-75.6, 6.2), (-76.5, 3.4),
        (-75.0, 10.4), (-73.1, 7.1), (-72.5, 11.5),
        (-71.5, 1.0),
    ]
    dists_deg = np.full(len(lons), np.inf)
    for (cx, cy) in urban_centres:
        d = np.sqrt((lons - cx) ** 2 + (lats - cy) ** 2)
        dists_deg = np.minimum(dists_deg, d)

    andes_dist   = np.abs(lons + 75.0)
    combined_deg = dists_deg * 0.7 + andes_dist * 0.3

    dist_km  = combined_deg * 111.0
    dist_km += rng.normal(0, 3.0, len(lons))
    dist_km  = np.clip(dist_km, 0, 200)
    return dist_km.round(1)


def _simulate_land_use(
    lons: np.ndarray,
    lats: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate land use suitability score in [0, 1].

        1.0 → open grassland / shrubland (La Guajira, Llanos)
        0.7 → agriculture / pasture
        0.4 → secondary forest
        0.1 → dense tropical forest (Amazonia, Chocó)
        0.0 → urban / waterbody
    """
    guajira = (
        np.clip((lats - 9.0) / 3.0, 0, 1)
        * np.clip((lons + 72.0) / 3.0, 0, 1)
    )
    llanos = (
        np.clip((lons + 73.0) / 3.0, 0, 1)
        * np.clip((lats - 2.0) / 3.0, 0, 1)
    )
    amazon_penalty = (
        np.clip((lons + 73.0) / 2.0, 0, 1)
        * np.clip(-(lats - 1.0) / 3.0, 0, 1)
    )
    choco_penalty = np.clip(-(lons + 77.0) / 1.5, 0, 1) * 0.8

    score = (
        guajira * 0.4
        + llanos * 0.35
        + _gaussian_field(lons, lats, length_scale=2.0, seed=99) * 0.25
        - amazon_penalty * 0.5
        - choco_penalty * 0.4
    )
    score += rng.normal(0, 0.05, len(lons))
    return np.clip(score, 0, 1).round(3)


def _simulate_protected_area(
    lons: np.ndarray,
    lats: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate protected-area indicator [0, 1].  1 = fully protected.

    Major Colombian PAs modelled:
      Sierra Nevada de Santa Marta, Chiribiquete, Chocó hotspot,
      Serranía de la Macarena.
    """
    snm      = np.exp(-(((lons + 74.0) ** 2 + (lats - 11.0) ** 2)) / (0.5 ** 2))
    chiri    = np.exp(-(((lons + 72.5) ** 2 + (lats -  0.5) ** 2)) / (1.5 ** 2))
    choco    = np.clip(-(lons + 77.5) / 1.0, 0, 1) * 0.7
    macarena = np.exp(-(((lons + 73.8) ** 2 + (lats -  2.9) ** 2)) / (0.4 ** 2))

    pa_score  = np.clip(snm + chiri * 0.8 + choco + macarena, 0, 1)
    noise     = _gaussian_field(lons, lats, length_scale=0.8, seed=55) * 0.15
    pa_score += noise
    pa_score += rng.normal(0, 0.02, len(lons))
    return np.clip(pa_score, 0, 1).round(3)


def _simulate_conflict_risk(
    lons: np.ndarray,
    lats: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate conflict risk indicator [0, 1].  1 = very high risk.

    Historically higher in:
      Pacific coast / Chocó, Catatumbo, southern departments, Urabá / Córdoba.
    """
    pacific_risk = np.clip(-(lons + 77.0) / 2.0, 0, 1) * 0.75
    catatumbo    = np.exp(-(((lons + 72.5) ** 2 + (lats - 8.5) ** 2)) / (0.6 ** 2)) * 0.9
    south_risk   = np.clip(-(lats - 1.0) / 5.0, 0, 1) * 0.6
    uraba        = np.exp(-(((lons + 76.3) ** 2 + (lats - 7.8) ** 2)) / (0.8 ** 2)) * 0.7

    risk  = np.clip(pacific_risk + catatumbo + south_risk * 0.5 + uraba, 0, 1)
    noise = _gaussian_field(lons, lats, length_scale=1.5, seed=33) * 0.1
    risk += noise + rng.normal(0, 0.03, len(lons))
    return np.clip(risk, 0, 1).round(3)


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------

def engineer_features(
    hex_grid: pd.DataFrame,
    seed: int = 42,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute all spatial features for every hexagon in the grid.

    This version is memory-safe at any H3 resolution by processing the
    Gaussian RBF kernel in chunks and capping the number of basis points.

    Parameters
    ----------
    hex_grid   : DataFrame from ``generate_h3_grid.generate_colombia_hex_grid``
                 Must contain columns ``hex_id``, ``lon``, ``lat``.
    seed       : master random seed for reproducibility.
    chunk_size : override CHUNK_SIZE for the RBF computation (advanced use).

    Returns
    -------
    DataFrame with all original columns plus:

    =====================  ====================================================
    Column                 Description
    =====================  ====================================================
    wind_speed             Average annual wind speed (m/s)
    slope                  Terrain slope (degrees)
    dist_to_grid           Distance to nearest transmission line (km)
    dist_to_roads          Distance to nearest primary road (km)
    land_use               Land-use suitability score [0-1]
    protected_area         Protected-area intensity [0-1]; 1 = fully protected
    conflict_risk          Conflict risk score [0-1]; 1 = very high risk
    =====================  ====================================================
    """
    # Allow caller to override chunk size at runtime
    if chunk_size is not None:
        global CHUNK_SIZE
        CHUNK_SIZE = chunk_size

    n = len(hex_grid)
    print(f"[Features] Computing spatial features for {n:,} hexagons "
          f"(n_basis cap={MAX_BASIS_POINTS}, chunk_size={CHUNK_SIZE})...")

    rng  = np.random.default_rng(seed)
    lons = hex_grid["lon"].to_numpy()
    lats = hex_grid["lat"].to_numpy()

    df = hex_grid.copy()

    df["wind_speed"]     = _simulate_wind_speed(lons, lats, rng)
    df["slope"]          = _simulate_slope(lons, lats, rng)
    df["dist_to_grid"]   = _simulate_distance_to_grid(lons, lats, rng)
    df["dist_to_roads"]  = _simulate_distance_to_roads(lons, lats, rng)
    df["land_use"]       = _simulate_land_use(lons, lats, rng)
    df["protected_area"] = _simulate_protected_area(lons, lats, rng)
    df["conflict_risk"]  = _simulate_conflict_risk(lons, lats, rng)

    print(f"[Features] Feature matrix shape: {df.shape}")
    print("[Features] Summary statistics:")
    feature_cols = [
        "wind_speed", "slope", "dist_to_grid", "dist_to_roads",
        "land_use", "protected_area", "conflict_risk",
    ]
    print(df[feature_cols].describe().round(2).to_string())
    return df


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_h3_grid import generate_colombia_hex_grid

    _HERE   = os.path.dirname(os.path.abspath(__file__))
    _GEOJSON = os.path.join(_HERE, "..", "data", "colombia_boundary.geojson")

    for res in [5, 6, 7]:
        print(f"\n{'='*60}")
        print(f"  Testing resolution {res}")
        print(f"{'='*60}")
        grid = generate_colombia_hex_grid(_GEOJSON, resolution=res)
        feat = engineer_features(grid)
        print(feat[["wind_speed", "slope", "conflict_risk"]].describe().round(2))