"""
generate_h3_grid.py
===================
Genera una grilla H3 real (Uber H3) sobre Colombia y asigna a cada
hexágono el municipio y código DIVIPOLA oficial (DANE).

Dependencias
------------
    pip install h3 geopandas shapely requests

Uso rápido
----------
    from generate_h3_grid import generate_colombia_hex_grid

    df = generate_colombia_hex_grid(
        geojson_path="data/colombia_boundary.geojson",
        resolution=6,
    )
    print(df.head())

Resoluciones H3 recomendadas para Colombia
------------------------------------------
    5  ->  ~252 km2  por celda  (~500 hexagonos)
    6  ->  ~36  km2  por celda  (~3 500 hexagonos)
    7  ->  ~5   km2  por celda  (~25 000 hexagonos)
    8  ->  ~0.7 km2  por celda  (~180 000 hexagonos)  <- limite practico sin RAM extra

Referencias
-----------
    https://h3geo.org/docs/api/indexing
    https://www.dane.gov.co/
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports opcionales con mensajes claros si faltan
# ---------------------------------------------------------------------------
try:
    import h3
except ImportError:
    raise ImportError(
        "La libreria H3 no esta instalada.\n"
        "Instalala con:  pip install h3"
    )

try:
    import geopandas as gpd
    from shapely.geometry import Point, shape, mapping
    from shapely.validation import make_valid
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    logging.warning(
        "GeoPandas / Shapely no encontrados. "
        "La asignacion DIVIPOLA usara ray-casting basico. "
        "Para rendimiento optimo: pip install geopandas shapely"
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[H3-Grid] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parametros CSV DIVIPOLA
# Columnas esperadas del CSV oficial del DANE
# ---------------------------------------------------------------------------
_CSV_COL_DEPT_CODE  = "Código_Departamento"
_CSV_COL_DEPT_NAME  = "Nombre_Departamento"
_CSV_COL_MPIO_CODE  = "Código_Municipio"
_CSV_COL_MPIO_NAME  = "Nombre_Municipio"
_CSV_COL_LON        = "Longitud"
_CSV_COL_LAT        = "Latitud"


# ===========================================================================
# 1. Extraccion de geometrias del GeoJSON de frontera
# ===========================================================================

def _load_boundary_geometries(geojson_path: str) -> List[Dict]:
    """
    Carga el GeoJSON de frontera y devuelve lista de geometrias.
    Acepta FeatureCollection, Feature, Polygon, MultiPolygon.
    Ignora San Andres y Providencia (lon < -79.5).
    """
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON no encontrado: {geojson_path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    geom_type = data.get("type", "")
    raw_geoms: List[Dict] = []

    if geom_type == "FeatureCollection":
        for feat in data.get("features", []):
            g = feat.get("geometry")
            if g:
                raw_geoms.append(g)
    elif geom_type == "Feature":
        g = data.get("geometry")
        if g:
            raw_geoms.append(g)
    elif geom_type in ("Polygon", "MultiPolygon"):
        raw_geoms.append(data)
    else:
        raise ValueError(f"Tipo GeoJSON no soportado: {geom_type}")

    def _is_continental(geom: Dict) -> bool:
        coords = geom.get("coordinates", [])
        if geom["type"] == "Polygon":
            ring = coords[0] if coords else []
        elif geom["type"] == "MultiPolygon":
            ring = coords[0][0] if coords and coords[0] else []
        else:
            return True
        lons = [c[0] for c in ring]
        return not lons or min(lons) > -79.5

    filtered = [g for g in raw_geoms if g and _is_continental(g)]
    log.info(f"GeoJSON cargado: {len(filtered)} geometria(s) continental(es).")
    return filtered


def _validate_and_fix_geometry(geom_dict: Dict) -> Optional[Dict]:
    """Valida y repara una geometria GeoJSON usando Shapely si disponible."""
    if not HAS_GEOPANDAS:
        return geom_dict
    try:
        shp = shape(geom_dict)
        if not shp.is_valid:
            shp = make_valid(shp)
        if shp.is_empty:
            return None
        return mapping(shp)
    except Exception as e:
        log.warning(f"Geometria invalida ignorada: {e}")
        return None


# ===========================================================================
# 2. Polyfill H3  ->  conjunto de indices
# ===========================================================================

def _detect_h3_version() -> Tuple[int, int]:
    """Detecta version mayor.menor de h3."""
    return tuple(int(x) for x in h3.__version__.split(".")[:2])


def _geojson_geom_to_h3_cells(geom: Dict, resolution: int) -> set:
    """
    Convierte una geometria GeoJSON (Polygon o MultiPolygon) a un conjunto
    de indices H3. Soporta H3 v3.x y v4.x automaticamente.
    """
    cells: set = set()
    use_new_api = _detect_h3_version() >= (4, 0)

    def _polyfill_polygon(coords_lonlat: List) -> set:
        outer = coords_lonlat[0]
        holes = coords_lonlat[1:] if len(coords_lonlat) > 1 else []

        if use_new_api:
            # H3 >= 4.x: geo_to_cells acepta GeoJSON dict (lon, lat)
            poly_geojson = {"type": "Polygon", "coordinates": [outer] + holes}
            try:
                return set(h3.geo_to_cells(poly_geojson, resolution))
            except Exception as e:
                log.debug(f"geo_to_cells fallo: {e}")
                return set()
        else:
            # H3 3.x: polyfill_geojson o polyfill con (lat, lon)
            poly_geojson = {"type": "Polygon", "coordinates": [outer] + holes}
            try:
                if hasattr(h3, "polyfill_geojson"):
                    return h3.polyfill_geojson(poly_geojson, resolution)
                else:
                    # API antigua: espera {"outer": [(lat,lon),...]}
                    outer_ll = [(lat, lon) for lon, lat in outer]
                    holes_ll = [[(lat, lon) for lon, lat in h] for h in holes]
                    return h3.polyfill({"outer": outer_ll, "holes": holes_ll}, resolution)
            except Exception as e:
                log.debug(f"polyfill fallo: {e}")
                return set()

    if geom["type"] == "Polygon":
        cells |= _polyfill_polygon(geom["coordinates"])
    elif geom["type"] == "MultiPolygon":
        for poly_coords in geom["coordinates"]:
            cells |= _polyfill_polygon(poly_coords)

    return cells


# ===========================================================================
# 3. Construccion del DataFrame
# ===========================================================================

def _cells_to_dataframe(cells: set, resolution: int) -> pd.DataFrame:
    """
    Convierte un conjunto de indices H3 a DataFrame con centroides y vertices.
    """
    if not cells:
        return pd.DataFrame(columns=["hex_id", "lat", "lon", "resolution", "vertices"])

    use_new_api = _detect_h3_version() >= (4, 0)
    rows = []

    for cell in cells:
        if use_new_api:
            lat, lon = h3.cell_to_latlng(cell)
            boundary = h3.cell_to_boundary(cell)   # [(lat, lon), ...]
        else:
            lat, lon = h3.h3_to_geo(cell)
            boundary = h3.h3_to_geo_boundary(cell)  # [(lat, lon), ...]

        # Vertices en formato GeoJSON estandar: (lon, lat)
        vertices = [(v[1], v[0]) for v in boundary]
        vertices.append(vertices[0])  # cerrar anillo

        rows.append({
            "hex_id":     cell,
            "lat":        round(lat, 6),
            "lon":        round(lon, 6),
            "resolution": resolution,
            "vertices":   vertices,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["lat", "lon"], ascending=[False, True]).reset_index(drop=True)
    return df


# ===========================================================================
# 4. Asignacion DIVIPOLA
# ===========================================================================

def _load_divipola_csv(csv_path: str) -> pd.DataFrame:
    """
    Carga el CSV DIVIPOLA del DANE y lo normaliza.

    Formato esperado (separador ; o tabulador, coma decimal):
        Código_Departamento | Nombre_Departamento | Código_Municipio |
        Nombre_Municipio    | Longitud            | Latitud

    Returns
    -------
    DataFrame con columnas: divipola_code, municipality, department, lon, lat
    Solo filas de tipo municipio (Código_Municipio unico por municipio).
    """
    df = None
    last_error = None

    # Intentar archivo con cabecera (formato estandar DANE)
    for enc in ["utf-8-sig", "latin-1"]:
        for sep in [";", "\t", ","]:
            try:
                candidate = pd.read_csv(csv_path, sep=sep, encoding=enc, dtype=str)
                if len(candidate.columns) >= 6:
                    candidate.columns = candidate.columns.str.strip()
                    required = [_CSV_COL_MPIO_CODE, _CSV_COL_MPIO_NAME,
                                _CSV_COL_DEPT_CODE, _CSV_COL_DEPT_NAME,
                                _CSV_COL_LON, _CSV_COL_LAT]
                    missing = [c for c in required if c not in candidate.columns]
                    if not missing:
                        df = candidate
                        break
            except Exception as e:
                last_error = e
                continue
        if df is not None:
            break

    if df is not None:
        # Normalizar coordenadas: coma decimal -> punto
        df[_CSV_COL_LON] = df[_CSV_COL_LON].str.replace(",", ".", regex=False).astype(float)
        df[_CSV_COL_LAT] = df[_CSV_COL_LAT].str.replace(",", ".", regex=False).astype(float)

        # Renombrar a esquema estandar
        result = pd.DataFrame({
            "divipola_code": df[_CSV_COL_MPIO_CODE].str.strip(),
            "municipality":  df[_CSV_COL_MPIO_NAME].str.strip().str.title(),
            "department":    df[_CSV_COL_DEPT_NAME].str.strip().str.title(),
            "lon":           df[_CSV_COL_LON],
            "lat":           df[_CSV_COL_LAT],
        })
    else:
        # Fallback para CSV sin cabecera, p. ej. DIVIPOLA_CentrosPoblados.csv
        fallback = None
        for enc in ["utf-8-sig", "latin-1"]:
            try:
                candidate = pd.read_csv(csv_path, sep=";", encoding=enc, dtype=str, header=None)
                if len(candidate.columns) >= 9:
                    fallback = candidate
                    break
            except Exception as e:
                last_error = e
                continue

        if fallback is None:
            raise ValueError(
                f"No se pudo interpretar el CSV DIVIPOLA: {csv_path}\n"
                f"Ultimo error: {last_error}"
            )

        # Estructura esperada sin cabecera:
        # 0:cod_dpto, 1:depto, 2:cod_mpio, 3:municipio, 4:cod_centro,
        # 5:nombre_centro, 6:categoria, 7:longitud, 8:latitud
        fallback = fallback.rename(columns={
            0: "cod_dpto",
            1: "department",
            2: "divipola_code",
            3: "municipality",
            7: "lon",
            8: "lat",
        })

        result = pd.DataFrame({
            "divipola_code": fallback["divipola_code"].astype(str).str.strip(),
            "municipality":  fallback["municipality"].astype(str).str.strip().str.title(),
            "department":    fallback["department"].astype(str).str.strip().str.title(),
            "lon":           fallback["lon"].astype(str).str.replace(",", ".", regex=False).astype(float),
            "lat":           fallback["lat"].astype(str).str.replace(",", ".", regex=False).astype(float),
        })

    # Quedarse con un punto representativo por municipio (el primero)
    result = (result
              .dropna(subset=["lon", "lat"])
              .query("divipola_code != ''")
              .drop_duplicates(subset=["divipola_code"])
              .reset_index(drop=True))

    log.info(f"DIVIPOLA CSV: {len(result):,} municipios cargados.")
    return result


def _assign_divipola_from_csv(
    df: pd.DataFrame,
    muni_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Asigna DIVIPOLA a cada hexagono por proximidad al municipio mas cercano.

    Usa distancia euclidea sobre lon/lat (valida para distancias cortas
    dentro de Colombia). Para cada centroide H3 se busca el municipio
    cuyo punto representativo (del CSV) sea el mas cercano.

    Complejidad: O(N * M) con broadcasting NumPy — rapido hasta ~200k hexagonos.

    Parameters
    ----------
    df      : DataFrame H3 con columnas lon, lat
    muni_df : DataFrame DIVIPOLA con columnas divipola_code, municipality,
              department, lon, lat

    Returns
    -------
    df con columnas divipola_code, municipality, department añadidas
    """
    log.info(f"DIVIPOLA: asignando por proximidad ({len(df):,} hexagonos x {len(muni_df):,} municipios)...")

    # Arrays NumPy para operacion vectorizada
    hex_lon = df["lon"].to_numpy()[:, None]   # (N, 1)
    hex_lat = df["lat"].to_numpy()[:, None]   # (N, 1)
    mun_lon = muni_df["lon"].to_numpy()[None, :]  # (1, M)
    mun_lat = muni_df["lat"].to_numpy()[None, :]  # (1, M)

    # Distancia euclidea al cuadrado (no necesitamos sqrt para encontrar el minimo)
    dist2 = (hex_lon - mun_lon) ** 2 + (hex_lat - mun_lat) ** 2  # (N, M)
    nearest_idx = dist2.argmin(axis=1)  # (N,) indice del municipio mas cercano

    df = df.copy()
    df["divipola_code"] = muni_df["divipola_code"].iloc[nearest_idx].values
    df["municipality"]  = muni_df["municipality"].iloc[nearest_idx].values
    df["department"]    = muni_df["department"].iloc[nearest_idx].values

    log.info(
        f"DIVIPOLA: asignacion completa | "
        f"{df['municipality'].nunique():,} municipios unicos | "
        f"{df['department'].nunique():,} departamentos unicos"
    )
    return df


def _normalize_municipios_gdf(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Normaliza GeoDataFrame de municipios a columnas estandar:
    divipola_code, municipality, department, geometry.

    Soporta esquemas IGAC, DANE y GeoJSON generico.
    """
    col_map: Dict[str, List[str]] = {
        "divipola_code": ["MPIO_CCDGO", "DIVIPOLA", "COD_DANE", "CODIGO", "id", "OBJECTID"],
        "municipality":  ["MPIO_CNMBR", "MUNICIPIO", "NOMBRE_MPIO", "name", "NAME"],
        "department":    ["DPTO_CNMBR", "DEPARTAMENTO", "NOMBRE_DPTO", "department"],
    }

    result = gdf[["geometry"]].copy()
    for target, candidates in col_map.items():
        for c in candidates:
            if c in gdf.columns:
                result[target] = gdf[c].astype(str).str.strip()
                break
        else:
            result[target] = "Unknown"

    if HAS_GEOPANDAS:
        result["geometry"] = result["geometry"].apply(
            lambda g: make_valid(g) if g and not g.is_valid else g
        )
    result = result[result["geometry"].notna() & ~result["geometry"].is_empty]
    return result.reset_index(drop=True)


def _assign_divipola_geopandas(
    df: pd.DataFrame,
    municipios_gdf: "gpd.GeoDataFrame",
) -> pd.DataFrame:
    """Spatial join vectorizado con GeoPandas (rapido)."""
    centroids = gpd.GeoDataFrame(
        df[["hex_id"]].copy(),
        geometry=[Point(row.lon, row.lat) for row in df.itertuples()],
        crs="EPSG:4326",
    )

    muni = (municipios_gdf.to_crs("EPSG:4326")
            if municipios_gdf.crs else municipios_gdf.set_crs("EPSG:4326"))

    joined = gpd.sjoin(
        centroids,
        muni[["divipola_code", "municipality", "department", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.drop_duplicates(subset=["hex_id"], keep="first")

    result = df.merge(
        joined[["hex_id", "divipola_code", "municipality", "department"]],
        on="hex_id",
        how="left",
    )
    for col in ["divipola_code", "municipality", "department"]:
        result[col] = result[col].fillna("Unknown")
    return result


def _assign_divipola_fallback(df: pd.DataFrame, municipios_features: List[Dict]) -> pd.DataFrame:
    """Ray-casting basico sin GeoPandas (mas lento)."""
    def pip(px, py, poly_coords):
        ring = poly_coords[0]
        inside = False
        j = len(ring) - 1
        for i in range(len(ring)):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    codes, munis, depts = [], [], []
    for row in df.itertuples():
        found_code, found_muni, found_dept = "Unknown", "Unknown", "Unknown"
        for feat in municipios_features:
            geom = feat.get("geometry", {})
            coords_list = (
                [geom["coordinates"]] if geom["type"] == "Polygon"
                else geom["coordinates"] if geom["type"] == "MultiPolygon"
                else []
            )
            for coords in coords_list:
                if pip(row.lon, row.lat, coords):
                    props = feat.get("properties", {})
                    found_code = str(props.get("MPIO_CCDGO") or props.get("DIVIPOLA") or "Unknown")
                    found_muni = str(props.get("MPIO_CNMBR") or props.get("MUNICIPIO") or "Unknown")
                    found_dept = str(props.get("DPTO_CNMBR") or props.get("DEPARTAMENTO") or "Unknown")
                    break
        codes.append(found_code)
        munis.append(found_muni)
        depts.append(found_dept)

    df = df.copy()
    df["divipola_code"] = codes
    df["municipality"]  = munis
    df["department"]    = depts
    return df


def assign_divipola(
    df: pd.DataFrame,
    municipios_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Asigna divipola_code, municipality y department a cada hexagono.

    Estrategia (en orden de prioridad):
      1. CSV DIVIPOLA (si municipios_path apunta a un .csv)  ← RECOMENDADO
      2. GeoJSON de municipios (si municipios_path apunta a un .geojson)
      3. Asignar 'Unknown' con instrucciones claras

    Parameters
    ----------
    df              : DataFrame con columnas lon, lat, hex_id
    municipios_path : ruta al CSV o GeoJSON de municipios con DIVIPOLA.
                      Si es None, asigna 'Unknown' y muestra instrucciones.
    cache_dir       : no usado con CSV, reservado para compatibilidad futura

    Returns
    -------
    df con columnas: divipola_code, municipality, department
    """
    log.info("DIVIPOLA: iniciando asignacion...")

    # ------------------------------------------------------------------
    # Caso 1: CSV proporcionado (fuente preferida)
    # ------------------------------------------------------------------
    if municipios_path and Path(municipios_path).exists():
        ext = Path(municipios_path).suffix.lower()

        if ext == ".csv":
            muni_df = _load_divipola_csv(municipios_path)
            return _assign_divipola_from_csv(df, muni_df)

        elif ext in (".geojson", ".json", ".shp") and HAS_GEOPANDAS:
            log.info(f"DIVIPOLA: usando GeoJSON/SHP -> {municipios_path}")
            muni_gdf = gpd.read_file(municipios_path)
            muni_normalized = _normalize_municipios_gdf(muni_gdf)
            return _assign_divipola_geopandas(df, muni_normalized)

        else:
            log.warning(f"DIVIPOLA: formato no reconocido o GeoPandas no disponible -> {municipios_path}")

    # ------------------------------------------------------------------
    # Caso 2: No hay archivo -> Unknown con instrucciones
    # ------------------------------------------------------------------
    log.warning(
        "DIVIPOLA: no se proporcionó archivo de municipios.\n"
        "  Para asignar DIVIPOLA pasa el CSV del DANE:\n"
        "    generate_colombia_hex_grid(\n"
        "        geojson_path='...',\n"
        "        municipios_path='ruta/al/DIVIPOLA.csv'\n"
        "    )\n"
        "  El CSV debe tener columnas:\n"
        "    Código_Municipio, Nombre_Municipio, Nombre_Departamento, "
        "Longitud, Latitud"
    )
    df = df.copy()
    df["divipola_code"] = "Unknown"
    df["municipality"]  = "Unknown"
    df["department"]    = "Unknown"
    return df


# ===========================================================================
# 5. Funcion principal publica
# ===========================================================================

def generate_colombia_hex_grid(
    geojson_path: str,
    resolution: int = 6,
    municipios_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    compact: bool = False,
    return_geometry: bool = True,
) -> pd.DataFrame:
    """
    Genera una grilla H3 real cubriendo Colombia y asigna DIVIPOLA.

    Acepta cualquier GeoJSON de frontera:
      - Frontera nacional unica (Polygon / MultiPolygon)
      - FeatureCollection de departamentos (33 features del DANE)

    Resoluciones H3 recomendadas:
        5  -> ~252 km2  ~500   hexagonos   (analisis regional)
        6  -> ~36  km2  ~3 500 hexagonos   (analisis departamental)  <- default
        7  -> ~5   km2  ~25 000 hexagonos  (analisis municipal)
        8  -> ~0.7 km2  ~180 000 hexagonos (analisis detallado - lento)

    Parameters
    ----------
    geojson_path    : ruta al GeoJSON de frontera colombiana
    resolution      : resolucion H3 (0-15, recomendado 5-8 para Colombia)
    municipios_path : ruta opcional al GeoJSON de municipios con DIVIPOLA
    cache_dir       : directorio para cachear el GeoJSON de municipios
    compact         : si True, aplica h3.compact_cells() para reducir celdas
    return_geometry : si False, omite la columna 'vertices' (mas rapido)

    Returns
    -------
    pandas.DataFrame con columnas:
        hex_id         - indice H3 string (ej. "8a6a2c51b5fffff")
        lat            - latitud del centroide
        lon            - longitud del centroide
        resolution     - resolucion H3 usada
        vertices       - lista de (lon, lat) del hexagono (si return_geometry=True)
        divipola_code  - codigo DIVIPOLA del municipio
        municipality   - nombre del municipio
        department     - nombre del departamento
    """
    log.info(f"Iniciando generacion H3 | resolucion={resolution} | archivo={geojson_path}")

    # 1. Cargar y validar geometrias
    geometries = _load_boundary_geometries(geojson_path)
    if not geometries:
        raise ValueError("No se encontraron geometrias validas en el GeoJSON.")

    geometries = [_validate_and_fix_geometry(g) for g in geometries]
    geometries = [g for g in geometries if g is not None]
    if not geometries:
        raise ValueError("Todas las geometrias son invalidas o vacias.")

    # 2. Polyfill H3
    log.info(f"Aplicando H3 polyfill sobre {len(geometries)} geometria(s)...")
    all_cells: set = set()
    for geom in geometries:
        cells = _geojson_geom_to_h3_cells(geom, resolution)
        all_cells |= cells

    if not all_cells:
        raise RuntimeError(
            f"H3 polyfill no genero hexagonos. "
            f"Verifica que la resolucion {resolution} sea adecuada para el area."
        )

    log.info(f"Polyfill completo: {len(all_cells):,} hexagonos H3 unicos.")

    # 3. Compactar (opcional)
    if compact:
        if _detect_h3_version() >= (4, 0):
            all_cells = set(h3.compact_cells(all_cells))
        else:
            all_cells = set(h3.compact(all_cells))
        log.info(f"Compactado: {len(all_cells):,} celdas tras compact_cells().")

    # 4. Construir DataFrame base
    df = _cells_to_dataframe(all_cells, resolution)

    if not return_geometry and "vertices" in df.columns:
        df = df.drop(columns=["vertices"])

    # 5. Asignar DIVIPOLA
    df = assign_divipola(df, municipios_path=municipios_path, cache_dir=cache_dir)

    # 6. Resumen
    log.info(
        f"Grid H3 generado: {len(df):,} hexagonos | "
        f"resolucion={resolution} | "
        f"municipios={df['municipality'].nunique()} | "
        f"departamentos={df['department'].nunique()}"
    )
    return df


# ===========================================================================
# 6. Exportar a GeoJSON
# ===========================================================================

def hexgrid_to_geojson(df: pd.DataFrame) -> Dict[str, Any]:
    """Convierte el DataFrame H3 a GeoJSON FeatureCollection."""
    features = []
    prop_cols = [c for c in df.columns if c != "vertices"]

    for _, row in df.iterrows():
        coords = [[v[0], v[1]] for v in row.get("vertices", [])]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        props: Dict[str, Any] = {}
        for col in prop_cols:
            v = row[col]
            if isinstance(v, (np.integer,)):    v = int(v)
            elif isinstance(v, (np.floating,)): v = round(float(v), 6)
            elif pd.isna(v):                    v = None
            props[col] = v

        features.append({
            "type": "Feature",
            "id": row["hex_id"],
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    return {"type": "FeatureCollection", "features": features}


# ===========================================================================
# 7. Retrocompatibilidad con modulos legacy
# ===========================================================================

def _load_colombia_polygon(geojson_path: str):
    """
    Retrocompatibilidad: devuelve el anillo mas largo como lista de (lon, lat).
    Usado por modulos legacy.
    """
    geoms = _load_boundary_geometries(geojson_path)
    best_ring = []
    for g in geoms:
        if g["type"] == "Polygon":
            ring = g["coordinates"][0]
        elif g["type"] == "MultiPolygon":
            ring = max(g["coordinates"], key=lambda p: len(p[0]))[0]
        else:
            continue
        if len(ring) > len(best_ring):
            best_ring = ring
    return [(c[0], c[1]) for c in best_ring]


# ===========================================================================
# Auto-test
# ===========================================================================
if __name__ == "__main__":
    import sys

    geo = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(os.path.dirname(__file__), "..", "data", "colombia_boundary.geojson")
    )
    res = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    print(f"\nProbando con resolucion H3 = {res}")
    print(f"Archivo: {geo}\n")

    grid = generate_colombia_hex_grid(geo, resolution=res)

    print("\n-- Primeras 5 filas --")
    print(grid[["hex_id", "lat", "lon", "divipola_code", "municipality", "department"]].head().to_string())
    print(f"\n-- Resumen --")
    print(f"Total hexagonos : {len(grid):,}")
    print(f"Municipios       : {grid['municipality'].nunique()}")
    print(f"Departamentos    : {grid['department'].nunique()}")
    print(f"Sin DIVIPOLA     : {(grid['municipality'] == 'Unknown').sum()}")
    print("\n-- Top 10 departamentos por hexagonos --")
    print(grid["department"].value_counts().head(10).to_string())