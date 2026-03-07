"""
generate_h3_grid.py
===================
Genera una grilla hexagonal que cubre Colombia, compatible con cualquier
GeoJSON de entrada:

  * Un solo Feature con geometría Polygon o MultiPolygon  (frontera nacional)
  * FeatureCollection con múltiples Features              (departamentos)

En ambos casos se extrae la colección completa de anillos y se usa un test
punto-dentro-de-polígono (ray-casting) vectorizado con NumPy para determinar
qué hexágonos caen dentro de Colombia.

Referencia matemática:
  https://www.redblobgames.com/grids/hexagons/
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Carga del GeoJSON  →  lista de anillos
# ---------------------------------------------------------------------------

def _extract_rings(geojson_path: str) -> List[List[Tuple[float, float]]]:
    """
    Lee un GeoJSON (Polygon, MultiPolygon, o FeatureCollection de cualquier
    combinación) y devuelve TODOS los anillos exteriores.

    Ignora el archipiélago de San Andrés (lon < -79.5).
    """
    with open(geojson_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data["type"] == "FeatureCollection":
        geometries = [f["geometry"] for f in data["features"] if f.get("geometry")]
    elif data["type"] == "Feature":
        geometries = [data["geometry"]]
    else:
        geometries = [data]

    rings: List[List[Tuple[float, float]]] = []
    for geom in geometries:
        if geom is None:
            continue
        if geom["type"] == "Polygon":
            polys = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue

        for poly in polys:
            outer = poly[0]
            lons = [c[0] for c in outer]
            if min(lons) < -79.5:   # ignorar San Andrés
                continue
            rings.append([(c[0], c[1]) for c in outer])

    print(f"[H3-Grid] GeoJSON cargado: {len(rings)} polígono(s) válidos.")
    return rings


def _bbox_from_rings(rings):
    all_lons = [lon for ring in rings for lon, lat in ring]
    all_lats = [lat for ring in rings for lon, lat in ring]
    return min(all_lons), max(all_lons), min(all_lats), max(all_lats)


# ---------------------------------------------------------------------------
# Test punto-dentro-de-polígono  (vectorizado NumPy)
# ---------------------------------------------------------------------------

def _points_in_any_ring(
    lons: np.ndarray,
    lats: np.ndarray,
    rings: List[List[Tuple[float, float]]],
) -> np.ndarray:
    """
    Ray-casting vectorizado: devuelve bool array (N,) indicando si cada
    punto está dentro de al menos uno de los anillos dados.
    """
    result = np.zeros(len(lons), dtype=bool)

    for ring in rings:
        arr   = np.array(ring, dtype=np.float64)  # (V, 2)
        rx    = arr[:, 0]
        ry    = arr[:, 1]
        rxj   = np.roll(rx, 1)
        ryj   = np.roll(ry, 1)

        px = lons[:, None]   # (N, 1)
        py = lats[:, None]   # (N, 1)

        cond1  = (ry > py) != (ryj > py)          # segmento cruza altura del punto
        denom  = ry - ryj
        safe_d = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        x_cross = rxj + (py - ryj) / safe_d * (rx - rxj)
        cond2  = px < x_cross                      # cruce a la derecha

        crossings = np.sum(cond1 & cond2, axis=1)
        result |= (crossings % 2) == 1

    return result


# ---------------------------------------------------------------------------
# Geometría de hexágonos flat-top
# ---------------------------------------------------------------------------

def _hex_centers_flat_top(lon_min, lon_max, lat_min, lat_max, hex_size):
    h_step = 1.5 * hex_size
    v_step = math.sqrt(3) * hex_size

    centers = []
    col_idx = 0
    lon = lon_min
    while lon <= lon_max + hex_size:
        lat_offset = (v_step / 2.0) if (col_idx % 2 == 1) else 0.0
        lat = lat_min + lat_offset
        while lat <= lat_max + hex_size:
            centers.append((round(lon, 6), round(lat, 6)))
            lat += v_step
        lon += h_step
        col_idx += 1

    return np.array(centers, dtype=np.float64)


def _hex_vertices_flat_top(cx, cy, size):
    verts = [
        (round(cx + size * math.cos(math.radians(60.0 * i)), 6),
         round(cy + size * math.sin(math.radians(60.0 * i)), 6))
        for i in range(6)
    ]
    verts.append(verts[0])
    return verts


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def generate_colombia_hex_grid(
    geojson_path: str,
    resolution: int = 5,
) -> pd.DataFrame:
    """
    Genera una grilla hexagonal sobre Colombia.

    Acepta cualquier GeoJSON válido:
      - Frontera nacional (Polygon / MultiPolygon único)
      - FeatureCollection de departamentos (33 features)
      - Cualquier combinación

    Resoluciones disponibles:
        4 → ~88 km   5 → ~44 km   6 → ~22 km   7 → ~11 km   8 → ~5 km

    Parameters
    ----------
    geojson_path : ruta al archivo GeoJSON
    resolution   : resolución (4–8)

    Returns
    -------
    DataFrame con: hex_id, lon, lat, hex_size, dept, vertices
    """
    res_map = {3: 1.60, 4: 0.80, 5: 0.40, 55: 0.24, 65: 0.21, 6: 0.20, 7: 0.10, 8: 0.05}
    if resolution not in res_map:
        raise ValueError(f"resolution debe ser uno de {list(res_map.keys())}")

    hex_size = res_map[resolution]

    # 1. Cargar polígonos
    rings = _extract_rings(geojson_path)
    if not rings:
        raise ValueError(
            "No se encontraron polígonos válidos. "
            "Asegúrate de que el GeoJSON tenga geometrías Polygon o MultiPolygon."
        )

    # 2. Bounding box con padding
    lon_min, lon_max, lat_min, lat_max = _bbox_from_rings(rings)
    pad = hex_size * 1.5
    print(
        f"[H3-Grid] Resolución={resolution} | hex={hex_size}° | "
        f"BBox lon[{lon_min:.2f},{lon_max:.2f}] lat[{lat_min:.2f},{lat_max:.2f}]"
    )

    # 3. Generar candidatos en el bbox
    centers = _hex_centers_flat_top(
        lon_min - pad, lon_max + pad,
        lat_min - pad, lat_max + pad,
        hex_size
    )
    print(f"[H3-Grid] Candidatos en bbox: {len(centers):,}")

    # 4. Filtrar por punto-en-polígono (vectorizado)
    lons = centers[:, 0]
    lats = centers[:, 1]
    mask = _points_in_any_ring(lons, lats, rings)

    # 5. Construir DataFrame
    valid = centers[mask]
    rows = [
        {
            "hex_id":   f"H{i:05d}",
            "lon":      float(lon),
            "lat":      float(lat),
            "hex_size": hex_size,
            "vertices": _hex_vertices_flat_top(float(lon), float(lat), hex_size),
        }
        for i, (lon, lat) in enumerate(valid)
    ]

    df = pd.DataFrame(rows).reset_index(drop=True)

    # 6. Asignar departamento a cada hexágono
    df = _assign_departments(df, geojson_path)

    print(f"[H3-Grid] ✓ {len(df):,} hexágonos generados sobre Colombia.")
    return df


# ---------------------------------------------------------------------------
# Asignación de departamento
# ---------------------------------------------------------------------------

def _assign_departments(df: pd.DataFrame, geojson_path: str) -> pd.DataFrame:
    """Añade columna 'dept' con el nombre del departamento de cada hexágono."""
    with open(geojson_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data["type"] != "FeatureCollection" or len(data["features"]) <= 1:
        df["dept"] = "Colombia"
        return df

    dept_rings: List[Tuple[str, List]] = []
    for feat in data["features"]:
        geom  = feat.get("geometry")
        if not geom:
            continue
        props = feat.get("properties", {})
        name  = (props.get("DPTO_CNMBR") or props.get("NOMBRE_DPT")
                 or props.get("name") or props.get("NAME") or "Desconocido")
        name  = str(name).title()

        polys = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"] if geom["type"] == "MultiPolygon"
                 else [])
        drings = []
        for poly in polys:
            outer = poly[0]
            if min(c[0] for c in outer) < -79.5:
                continue
            drings.append([(c[0], c[1]) for c in outer])

        if drings:
            dept_rings.append((name, drings))

    lons   = df["lon"].to_numpy()
    lats   = df["lat"].to_numpy()
    labels = np.full(len(df), "Otro", dtype=object)

    for dept_name, drings in dept_rings:
        inside = _points_in_any_ring(lons, lats, drings)
        labels[inside] = dept_name

    df["dept"] = labels
    return df


# ---------------------------------------------------------------------------
# Export GeoJSON
# ---------------------------------------------------------------------------

def hexgrid_to_geojson(df: pd.DataFrame) -> Dict[str, Any]:
    features = []
    skip = {"vertices"}
    prop_cols = [c for c in df.columns if c not in skip]

    for _, row in df.iterrows():
        coords = [[v[0], v[1]] for v in row["vertices"]]
        props = {}
        for col in prop_cols:
            v = row[col]
            if isinstance(v, np.integer):  v = int(v)
            elif isinstance(v, np.floating): v = round(float(v), 5)
            props[col] = v
        features.append({
            "type": "Feature", "id": row["hex_id"],
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Retrocompatibilidad: _load_colombia_polygon (usado por otros módulos)
# ---------------------------------------------------------------------------

def _load_colombia_polygon(geojson_path: str) -> List[Tuple[float, float]]:
    """
    Retrocompatibilidad: devuelve el anillo más grande como lista plana.
    Para uso interno en scripts legacy.
    """
    rings = _extract_rings(geojson_path)
    return max(rings, key=len)


# ---------------------------------------------------------------------------
# Auto-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "colombia_boundary.geojson"
    )
    grid = generate_colombia_hex_grid(path, resolution=5)
    print(grid[["hex_id", "lon", "lat", "dept"]].head(10).to_string())
    print(f"\nTotal: {len(grid)} hexágonos | Departamentos: {grid['dept'].nunique()}")
    print(grid["dept"].value_counts().head(10))
