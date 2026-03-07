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
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Spatial covariance helper (Gaussian RBF kernel)
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

    We sample n_basis random anchor points and compute a weighted sum of radial
    basis functions, which gives a smooth, geographically continuous surface
    without needing scipy or specialised GP libraries.

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
    n_basis = max(50, n // 20)  # number of anchor points

    # Random anchor locations within the data extent
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    ax = rng.uniform(lon_min, lon_max, n_basis)
    ay = rng.uniform(lat_min, lat_max, n_basis)
    weights = rng.standard_normal(n_basis)

    # Sum of RBF contributions  (vectorised)
    # shape: (n, n_basis)
    d2 = (lons[:, None] - ax[None, :]) ** 2 + (lats[:, None] - ay[None, :]) ** 2
    rbf = np.exp(-d2 / (2.0 * length_scale ** 2))
    field = rbf @ weights  # shape (n,)

    # Normalise to [0, 1]
    field = (field - field.min()) / (field.max() - field.min() + 1e-12)
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
    # Base: spatial smooth field
    base = _gaussian_field(lons, lats, length_scale=3.0, seed=1)

    # Caribbean boost: lat gradient above 9°N
    caribbean_boost = np.clip((lats - 9.0) / 3.5, 0, 1) * 0.35

    # Andean boost: Colombia's main cordillera runs roughly lon ≈ -76 to -73
    andean_mask = np.exp(-((lons + 74.5) ** 2) / (1.5 ** 2))
    andean_boost = andean_mask * 0.25

    field = base * 0.4 + caribbean_boost + andean_boost
    # Add small noise
    field += rng.normal(0, 0.03, len(lons))
    field = np.clip(field, 0, 1)

    # Scale to m/s: range [1.5, 9.5]
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
    # Andean cordillera: three parallel ridges
    # Western: lon ≈ -77, Central: lon ≈ -75.5, Eastern: lon ≈ -73.5
    w_ridge = np.exp(-((lons + 77.0) ** 2) / (0.8 ** 2))
    c_ridge = np.exp(-((lons + 75.5) ** 2) / (0.9 ** 2))
    e_ridge = np.exp(-((lons + 73.5) ** 2) / (1.0 ** 2))

    andean = (w_ridge + c_ridge + e_ridge) / 3.0

    # Add spatial noise
    noise = _gaussian_field(lons, lats, length_scale=1.0, seed=7) * 0.3
    field = andean * 0.7 + noise
    field += rng.normal(0, 0.02, len(lons))
    field = np.clip(field, 0, 1)

    # Scale to degrees
    slope_deg = field * 45.0
    return slope_deg.round(2)


def _simulate_distance_to_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate distance to the nearest high-voltage transmission line (km).

    The main Colombian grid connects Bogotá (4.7°N, -74.1°W),
    Medellín (6.2°N, -75.6°W) and Cali (3.4°N, -76.5°W).

    Returns values in [0, 300] km.
    """
    # Main urban spine acts as proxy for grid presence
    urban_centres = [
        (-74.1, 4.7),   # Bogotá
        (-75.6, 6.2),   # Medellín
        (-76.5, 3.4),   # Cali
        (-75.0, 10.4),  # Barranquilla
        (-73.1, 7.1),   # Bucaramanga
        (-76.2, 1.2),   # Pasto (southern)
    ]
    # Min distance to any urban centre (approx. in degrees, 1° ≈ 111 km)
    dists_deg = np.full(len(lons), np.inf)
    for (cx, cy) in urban_centres:
        d = np.sqrt((lons - cx) ** 2 + (lats - cy) ** 2)
        dists_deg = np.minimum(dists_deg, d)

    dist_km = dists_deg * 111.0  # rough degree-to-km conversion
    # Add noise
    dist_km += rng.normal(0, 5.0, len(lons))
    dist_km = np.clip(dist_km, 0, 300)
    return dist_km.round(1)


def _simulate_distance_to_roads(
    lons: np.ndarray,
    lats: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate distance to the nearest primary road (km).

    Road network largely follows the same urban corridor but with wider spread.

    Returns values in [0, 200] km.
    """
    urban_centres = [
        (-74.1, 4.7), (-75.6, 6.2), (-76.5, 3.4),
        (-75.0, 10.4), (-73.1, 7.1), (-72.5, 11.5),  # La Guajira road
        (-71.5, 1.0),   # Leticia (Amazonia) – isolated
    ]
    dists_deg = np.full(len(lons), np.inf)
    for (cx, cy) in urban_centres:
        d = np.sqrt((lons - cx) ** 2 + (lats - cy) ** 2)
        dists_deg = np.minimum(dists_deg, d)

    # Roads also follow the Andes corridor (add a linear component)
    andes_dist = np.abs(lons + 75.0)  # distance from central Andes
    combined_deg = dists_deg * 0.7 + andes_dist * 0.3

    dist_km = combined_deg * 111.0
    dist_km += rng.normal(0, 3.0, len(lons))
    dist_km = np.clip(dist_km, 0, 200)
    return dist_km.round(1)


def _simulate_land_use(
    lons: np.ndarray,
    lats: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate land use category as a suitability score in [0, 1].

    Categories (simplified):
        1.0 → open grassland / shrubland (La Guajira, Llanos)
        0.7 → agriculture / pasture
        0.4 → secondary forest
        0.1 → dense tropical forest (Amazonia, Chocó)
        0.0 → urban / waterbody

    Returns continuous score in [0, 1].
    """
    # La Guajira / Caribbean: highly suitable open land
    guajira = np.clip((lats - 9.0) / 3.0, 0, 1) * np.clip((lons + 72.0) / 3.0, 0, 1)

    # Llanos Orientales (eastern plains): lon > -73, lat 2-8
    llanos = (
        np.clip((lons + 73.0) / 3.0, 0, 1) *
        np.clip((lats - 2.0) / 3.0, 0, 1)
    )

    # Amazonia (south-east): very low suitability
    amazon_penalty = (
        np.clip((lons + 73.0) / 2.0, 0, 1) *
        np.clip(-(lats - 1.0) / 3.0, 0, 1)
    )

    # Chocó (Pacific coast): very low suitability (dense rainforest)
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
    Simulate a binary/continuous protected-area indicator.

    0 = not protected (suitable), 1 = fully protected (not suitable).

    Colombia has major PAs in:
      - Sierra Nevada de Santa Marta
      - Serranía de la Macarena / Chiribiquete (Amazonia)
      - Los Flamencos / Tayrona (Caribbean coast)
      - Serranía del Baudó (Chocó)
    """
    # Sierra Nevada de Santa Marta ≈ (-74.0, 11.0)
    snm = np.exp(-(((lons + 74.0) ** 2 + (lats - 11.0) ** 2)) / (0.5 ** 2))

    # Chiribiquete ≈ (-72.5, 0.5) – huge Amazonian tepui park
    chiri = np.exp(-(((lons + 72.5) ** 2 + (lats - 0.5) ** 2)) / (1.5 ** 2))

    # Chocó biodiversity hotspot
    choco = np.clip(-(lons + 77.5) / 1.0, 0, 1) * 0.7

    # Serranía de la Macarena ≈ (-73.8, 2.9)
    macarena = np.exp(-(((lons + 73.8) ** 2 + (lats - 2.9) ** 2)) / (0.4 ** 2))

    pa_score = np.clip(snm + chiri * 0.8 + choco + macarena, 0, 1)
    # Add random small PAs
    noise = _gaussian_field(lons, lats, length_scale=0.8, seed=55) * 0.15
    pa_score += noise
    pa_score += rng.normal(0, 0.02, len(lons))
    return np.clip(pa_score, 0, 1).round(3)


def _simulate_conflict_risk(
    lons: np.ndarray,
    lats: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate conflict risk indicator in [0, 1].

    0 = safe, 1 = high risk.

    Historically higher in:
      - Pacific coast / Chocó (lon < -77)
      - Catatumbo region (lon ≈ -72.5, lat ≈ 8.5)
      - Southern departments: Nariño, Putumayo, Caquetá
      - Córdoba / Urabá (lon ≈ -76, lat ≈ 8)
    """
    # Chocó / Pacific
    pacific_risk = np.clip(-(lons + 77.0) / 2.0, 0, 1) * 0.75

    # Catatumbo
    catatumbo = np.exp(-(((lons + 72.5) ** 2 + (lats - 8.5) ** 2)) / (0.6 ** 2)) * 0.9

    # Southern border region (lat < 1°)
    south_risk = np.clip(-(lats - 1.0) / 5.0, 0, 1) * 0.6

    # Urabá / Córdoba
    uraba = np.exp(-(((lons + 76.3) ** 2 + (lats - 7.8) ** 2)) / (0.8 ** 2)) * 0.7

    risk = np.clip(pacific_risk + catatumbo + south_risk * 0.5 + uraba, 0, 1)
    noise = _gaussian_field(lons, lats, length_scale=1.5, seed=33) * 0.1
    risk += noise + rng.normal(0, 0.03, len(lons))
    return np.clip(risk, 0, 1).round(3)


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------

def engineer_features(
    hex_grid: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute all spatial features for every hexagon in the grid.

    Parameters
    ----------
    hex_grid : DataFrame produced by ``generate_h3_grid.generate_colombia_hex_grid``
               Must contain columns ``hex_id``, ``lon``, ``lat``.
    seed     : master random seed for reproducibility.

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
    print("[Features] Computing spatial features for each hexagon...")

    rng = np.random.default_rng(seed)
    lons = hex_grid["lon"].to_numpy()
    lats = hex_grid["lat"].to_numpy()

    df = hex_grid.copy()

    df["wind_speed"]    = _simulate_wind_speed(lons, lats, rng)
    df["slope"]         = _simulate_slope(lons, lats, rng)
    df["dist_to_grid"]  = _simulate_distance_to_grid(lons, lats, rng)
    df["dist_to_roads"] = _simulate_distance_to_roads(lons, lats, rng)
    df["land_use"]      = _simulate_land_use(lons, lats, rng)
    df["protected_area"]= _simulate_protected_area(lons, lats, rng)
    df["conflict_risk"] = _simulate_conflict_risk(lons, lats, rng)

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
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_h3_grid import generate_colombia_hex_grid

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _GEOJSON = os.path.join(_HERE, "..", "data", "colombia_boundary.geojson")

    grid = generate_colombia_hex_grid(_GEOJSON, resolution=5)
    features = engineer_features(grid)
    print(features.head())
