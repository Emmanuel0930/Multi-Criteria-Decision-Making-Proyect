"""
visualization.py
================
Crea todos los outputs visuales del modelo MCDA de aptitud eólica.

Arquitectura unificada
-----------------------
Combina dos mejoras complementarias:

1. **Multi-LOD rendering** (del optimizado):
   - LOD0 zoom<=5 : ~2 000 puntos simples, submuestreo 0.5°
   - LOD1 zoom 6-7: ~5 000 circleMarkers, submuestreo 0.15°
   - LOD3 zoom>=10: polígonos H3 reales, viewport-culled
   - Debounce de 120 ms en renderViewport para evitar re-renders en cadena
   - HUD de zoom/nivel en el mapa
   - _spatial_sample(): selecciona el hexágono de mayor score por celda

2. **UI Prototipo ** (del nuevo):
   - Topbar oscuro: logo, versión, subtítulo EAFIT
   - Panel izquierdo: ponderación de criterios por categorías A/B/C
   - Panel derecho: análisis del hexágono seleccionado con:
       · ID del hexágono (IDX-H3-XXXXX)
       · Índice Global (score × 10)
       · SHAP Values (barras + / - de explicabilidad)
       · Grilla de detalles (municipio, viento, pendiente, etc.)
   - Al hacer click en cualquier LOD se puebla el panel derecho

3. **Gráfica de distribución** – histograma de scores (matplotlib).
4. **Heatmap de correlación** – correlación de Pearson (matplotlib).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math


# ---------------------------------------------------------------------------
# Constante de rendimiento
# ---------------------------------------------------------------------------
MAX_FULL_HEXAGONS: int = 5_000


# ---------------------------------------------------------------------------
# Helpers de color
# ---------------------------------------------------------------------------

def _score_to_hex_colour(score: float, cmap_name: str = "RdYlGn") -> str:
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(float(np.clip(score, 0, 1)))
    return mcolors.to_hex(rgba)


def _build_colour_scale_html(n_steps: int = 6) -> str:
    cmap = cm.get_cmap("RdYlGn")
    stops = []
    for i in range(n_steps + 1):
        v = i / n_steps
        colour = mcolors.to_hex(cmap(v))
        stops.append(
            f'<span style="background:{colour};flex:1;display:inline-block;'
            f'height:16px;" title="{v:.1f}"></span>'
        )
    labels = "".join(
        f'<span style="flex:1;font-size:10px;text-align:center;">{i/n_steps:.1f}</span>'
        for i in range(n_steps + 1)
    )
    return (
        '<div style="background:white;padding:8px;border-radius:4px;'
        'box-shadow:0 1px 4px rgba(0,0,0,0.3);min-width:200px;">'
        "<b style='font-size:12px;'>Suitability Score</b><br>"
        f'<div style="display:flex;margin-top:4px;">{"".join(stops)}</div>'
        f'<div style="display:flex;margin-top:2px;">{labels}</div>'
        "<div style='font-size:10px;margin-top:4px;color:#555;'>★ Top 10 candidatos</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# GeoJSON builder (exportación, no se usa en el mapa interactivo)
# ---------------------------------------------------------------------------

def _simplify_ring(coords: list, tol: float = 0.002) -> list:
    """
    Douglas-Peucker simplificado sobre un anillo de coordenadas.
    """
    if len(coords) <= 4:
        return coords

    def _perp_dist(pt, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if dx == dy == 0:
            return ((pt[0] - a[0]) ** 2 + (pt[1] - a[1]) ** 2) ** 0.5
        t = ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        return ((pt[0] - a[0] - t * dx) ** 2 + (pt[1] - a[1] - t * dy) ** 2) ** 0.5

    def _dp(pts, tol):
        if len(pts) < 3:
            return pts
        dmax, idx = 0.0, 0
        for i in range(1, len(pts) - 1):
            d = _perp_dist(pts[i], pts[0], pts[-1])
            if d > dmax:
                dmax, idx = d, i
        if dmax > tol:
            return _dp(pts[:idx + 1], tol)[:-1] + _dp(pts[idx:], tol)
        return [pts[0], pts[-1]]

    simplified = _dp(coords, tol)
    if simplified[0] != simplified[-1]:
        simplified.append(simplified[0])
    return simplified


def df_to_geojson(
    df: pd.DataFrame,
    score_column: str = "suitability_score",
    feature_cols: Optional[List[str]] = None,
    simplify: bool = False,
) -> dict:
    """
    Convierte el DataFrame de hexágonos a GeoJSON FeatureCollection.
    """
    if feature_cols is None:
        feature_cols = [
            "wind_speed", "slope", "dist_to_grid", "dist_to_roads",
            "land_use", "protected_area", "conflict_risk", score_column,
        ]
    if "rank" in df.columns:
        feature_cols = ["rank"] + feature_cols

    features = []
    for _, row in df.iterrows():
        props = {
            col: (None if pd.isna(row[col]) else round(float(row[col]), 3))
            for col in feature_cols
            if col in row.index
        }
        props["hex_id"] = row["hex_id"]

        for meta in ("municipality", "department", "divipola_code"):
            if meta in row.index:
                val = row[meta]
                props[meta] = str(val) if not pd.isna(val) else "—"

        score = float(row.get(score_column, 0) or 0)
        props["colour"] = _score_to_hex_colour(score)
        props["opacity"] = round(0.4 + 0.5 * score, 2)

        if "vertices" in row.index and row["vertices"] is not None:
            coords = [[round(v[0], 5), round(v[1], 5)] for v in row["vertices"]]
        else:
            cx, cy, sz = row["lon"], row["lat"], 0.02
            coords = [
                [cx - sz, cy], [cx, cy + sz], [cx + sz, cy],
                [cx, cy - sz], [cx - sz, cy],
            ]

        if simplify:
            coords = _simplify_ring(coords, tol=0.002)

        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Serialización multi-LOD (del optimizado)
# ---------------------------------------------------------------------------

def _spatial_sample(
    df: pd.DataFrame,
    score_column: str,
    cell_deg: float,
) -> pd.DataFrame:
    """
    Submuestreo espacial: divide Colombia en una cuadrícula de celdas de
    `cell_deg` grados y selecciona el hexágono con mayor score en cada celda.
    Garantiza que el mapa nunca tenga más de (cols × rows) objetos en pantalla.
    """
    df = df.copy()
    df["_gx"] = (df["lon"] / cell_deg).astype(int)
    df["_gy"] = (df["lat"] / cell_deg).astype(int)
    idx = df.groupby(["_gx", "_gy"])[score_column].idxmax()
    return df.loc[idx].drop(columns=["_gx", "_gy"]).reset_index(drop=True)


def _build_lod_payload(
    df: pd.DataFrame,
    score_column: str,
) -> tuple:
    """
    Construye 4 niveles de detalle pre-muestreados espacialmente en Python.

    LOD0 zoom<=5  : puntos [lat,lon,score]              submuestreo 0.5 deg  ~2k
    LOD1 zoom 6-7 : circulos [lat,lon,score,rank,...]   submuestreo 0.15 deg ~5k
    LOD3 zoom>=10 : poligonos reales [verts,score,...]  viewport-culled en JS

    El submuestreo es espacial (mejor score por celda), no aleatorio,
    asi la informacion relevante siempre se preserva.
    """
    df = df.sort_values(by=score_column, ascending=False).reset_index(drop=True)

    munis = list(df.get("municipality", pd.Series(["—"] * len(df))).fillna("—").unique())
    depts = list(df.get("department",   pd.Series(["—"] * len(df))).fillna("—").unique())
    divis = list(df.get("divipola_code", pd.Series(["—"] * len(df))).fillna("—").astype(str).unique())
    muni_idx_map = {m: i for i, m in enumerate(munis)}
    dept_idx_map = {d: i for i, d in enumerate(depts)}
    divi_idx_map = {d: i for i, d in enumerate(divis)}

    has_vertices = "vertices" in df.columns

    def _meta(r):
        def _f(col, default=0.0):
            v = r.get(col, default)
            try:
                fv = float(v)
                return round(fv, 3) if not math.isnan(fv) else default
            except (TypeError, ValueError):
                return default
        rank_val = int(r["rank"]) if "rank" in r.index and not pd.isna(r.get("rank")) else 0
        muni_str = str(r.get("municipality", "—")) if "municipality" in r.index else "—"
        dept_str = str(r.get("department",   "—")) if "department"   in r.index else "—"
        divi_str = str(r.get("divipola_code", "—")) if "divipola_code" in r.index else "—"
        return (_f(score_column), _f("wind_speed"), _f("slope"),
                _f("dist_to_grid"), _f("dist_to_roads"), _f("land_use"),
                _f("protected_area"), _f("conflict_risk"),
          rank_val, muni_idx_map.get(muni_str, 0), dept_idx_map.get(dept_str, 0),
          divi_idx_map.get(divi_str, 0))

    # LOD0: puntos [lat, lon, score] submuestreados 0.5 deg
    s0 = _spatial_sample(df, score_column, 0.5)
    lod0 = [[round(float(r["lat"]), 3), round(float(r["lon"]), 3),
             round(float(r[score_column]), 3)] for _, r in s0.iterrows()]

    # LOD1: circulos 0.15 deg [lat,lon,score,rank,mi,di,dpi,ws,sl,dg,dr,lu,pa,cr]
    s1 = _spatial_sample(df, score_column, 0.15)
    lod1 = []
    for _, r in s1.iterrows():
        sc, ws, sl, dg, dr, lu, pa, cr, rk, mi, di, dpi = _meta(r)
        lod1.append([round(float(r["lat"]), 3), round(float(r["lon"]), 3),
             sc, rk, mi, di, dpi, ws, sl, dg, dr, lu, pa, cr])

    # LOD3: poligonos reales (viewport-culled en JS)
    lod3 = []
    for _, r in df.iterrows():
        sc, ws, sl, dg, dr, lu, pa, cr, rk, mi, di, dpi = _meta(r)
        if has_vertices and r["vertices"] is not None:
            raw  = r["vertices"]
            ring = raw[:-1] if len(raw) == 7 and raw[0] == raw[-1] else raw[:6]
            verts = [[round(v[1], 4), round(v[0], 4)] for v in ring]
        else:
            lat_c, lon_c = float(r["lat"]), float(r["lon"])
            r_deg = 0.05
            rx = r_deg / max(math.cos(math.radians(lat_c)), 0.01)
            verts = [[round(lat_c + r_deg * math.sin(math.radians(60 * k + 30)), 4),
                      round(lon_c + rx    * math.cos(math.radians(60 * k + 30)), 4)]
                     for k in range(6)]
        lod3.append([verts, sc, ws, sl, dg, dr, lu, pa, cr, rk, mi, di, dpi])

    sep = (",", ":")
    print(f"[Map] LOD sizes -> LOD0:{len(lod0):,}  LOD1:{len(lod1):,}  LOD3:{len(lod3):,}")
    return (
        json.dumps(lod0, separators=sep),
        json.dumps(lod1, separators=sep),
        json.dumps(lod3, separators=sep),
        json.dumps(munis, separators=sep),
        json.dumps(depts, separators=sep),
        json.dumps(divis, separators=sep),
    )


def _build_top_n_js(df: pd.DataFrame, score_column: str, top_n: int) -> str:
    """Serializa los Top-N como array JS para los marcadores de estrella."""
    top = df.nlargest(top_n, score_column)
    items = []
    for _, r in top.iterrows():
        items.append({
            "lat":   round(float(r["lat"]), 5),
            "lon":   round(float(r["lon"]), 5),
            "score": round(float(r.get(score_column, 0)), 4),
            "rank":  int(r["rank"]) if "rank" in r.index else 0,
            "ws":    round(float(r.get("wind_speed", 0)), 1),
            "muni":  str(r.get("municipality", "—")),
            "dept":  str(r.get("department",   "—")),
        })
    return json.dumps(items, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Mapa interactivo — LOD + UI Prototipo 
# ---------------------------------------------------------------------------

def create_interactive_map(
    df: pd.DataFrame,
    output_path: str,
    score_column: str = "suitability_score",
    top_n_highlight: int = 10,
    centre_lat: float = 4.711,
    centre_lon: float = -74.0721,
    zoom: int = 6,
) -> None:
    """
    Genera un mapa HTML interactivo con:
      - Multi-LOD rendering (4 niveles de detalle segun zoom)
      - UI Prototipo: topbar, panel de criterios, panel de analisis multicriterio

    Parameters
    ----------
    df               : DataFrame scored y rankeado (debe tener lat, lon)
    output_path      : ruta del archivo .html a generar
    score_column     : columna de score de aptitud
    top_n_highlight  : numero de top celdas a marcar con estrella
    centre_lat/lon   : centro inicial del mapa
    zoom             : zoom inicial (6 = pais completo, 10 = poligonos reales)
    """
    try:
        import folium
        from folium.plugins import MiniMap, Fullscreen
    except ImportError:
        raise ImportError("Instala Folium:  pip install folium")

    n = len(df)
    print(f"[Map] Construyendo mapa multi-LOD | {n:,} hexagonos | zoom={zoom}")

    # 1. Serializar los 4 niveles de detalle
    lod0_js, lod1_js, lod3_js, muni_table_js, dept_table_js, divi_table_js = \
        _build_lod_payload(df, score_column)
    top_js = _build_top_n_js(df, score_column, top_n_highlight)

    # 2. Mapa base Folium
    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer("OpenStreetMap",       name="OpenStreetMap", show=False).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Carto Dark",    show=False).add_to(m)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    Fullscreen(position="topright").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # 3. UI Prototipo + motor LOD
    dss_ui = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', sans-serif;
    background: #0d1117;
    color: #e2e8f0;
    overflow: hidden;
  }}

  #dss-shell {{
    position: fixed;
    inset: 0;
    display: flex;
    flex-direction: column;
    z-index: 10000;
    pointer-events: none;
  }}

  /* Topbar */
  #dss-topbar {{
    display: flex;
    align-items: center;
    background: #0d1117;
    border-bottom: 1px solid #1e2a3a;
    padding: 0 20px;
    height: 52px;
    flex-shrink: 0;
    pointer-events: all;
    z-index: 10001;
  }}
  #dss-topbar .logo {{
    display: flex; align-items: center; gap: 10px;
    font-size: 15px; font-weight: 700; color: #e2e8f0; letter-spacing: -0.3px;
  }}
  #dss-topbar .logo .icon {{
    width: 32px; height: 32px; background: #2563eb; border-radius: 6px;
    display: flex; align-items: center; justify-content: center; font-size: 14px;
  }}
  #dss-topbar .logo .accent {{ color: #60a5fa; font-style: italic; }}
  #dss-topbar .separator {{ width: 1px; height: 28px; background: #1e2a3a; margin: 0 20px; }}
  #dss-topbar .subtitle {{ font-size: 11px; font-weight: 500; color: #64748b; letter-spacing: 1.5px; text-transform: uppercase; }}
  #dss-topbar .spacer {{ flex: 1; }}
  #dss-topbar .model-controls {{ display:flex; align-items:center; gap:8px; }}
  #top-model-select {{
    height: 30px; min-width: 178px;
    border: 1px solid #1e2a3a; border-radius: 6px;
    background: #0f1724; color: #cbd5e1;
    font-size: 11px; padding: 0 8px;
  }}
  #top-load-btn {{
    height: 30px; border: 1px solid #1e40af; border-radius: 6px;
    background: #13253d; color: #93c5fd;
    font-size: 11px; font-weight: 600; padding: 0 10px;
    cursor: pointer;
  }}
  #top-load-btn:hover {{ background: #1a3456; }}
  #top-load-btn:disabled {{ opacity: 0.6; cursor: wait; }}
  #top-model-status {{ font-size: 10px; color: #64748b; min-width: 86px; text-align: right; }}
  #dss-topbar .share-btn {{
    margin-left: 14px; width: 30px; height: 30px;
    border: 1px solid #1e2a3a; border-radius: 6px;
    background: transparent; color: #64748b; cursor: pointer;
    display: flex; align-items: center; justify-content: center; font-size: 13px;
    transition: all 0.15s;
  }}
  #dss-topbar .share-btn:hover {{ background: #1e2a3a; color: #94a3b8; }}

  /* Body */
  #dss-body {{ display: flex; flex: 1; overflow: hidden; pointer-events: none; }}

  /* Left panel */
  #dss-left {{
    width: 260px; flex-shrink: 0;
    background: #0d1117; border-right: 1px solid #1e2a3a;
    overflow-y: auto; padding: 18px 16px; pointer-events: all;
  }}
  #dss-left::-webkit-scrollbar {{ width: 4px; }}
  #dss-left::-webkit-scrollbar-thumb {{ background: #1e2a3a; border-radius: 2px; }}

  .panel-section-title {{
    font-size: 10px; font-weight: 600; letter-spacing: 1.8px;
    text-transform: uppercase; color: #3b82f6; margin-bottom: 14px;
  }}
  .crit-category {{ margin-bottom: 20px; }}
  .crit-category-header {{
    display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
    font-size: 11px; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.8px;
  }}
  .crit-row {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 7px 0; border-bottom: 1px solid #141d29;
  }}
  .crit-row:last-child {{ border-bottom: none; }}
  .crit-label {{ font-size: 12px; color: #cbd5e1; font-weight: 400; }}
  .crit-badge {{
    font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 4px;
    background: #1e3a5f; color: #60a5fa; min-width: 44px; text-align: center;
  }}
  .crit-badge.excluded {{ background: #3b1212; color: #f87171; }}
  .crit-badge.green    {{ background: #14301f; color: #4ade80; }}
  .crit-badge.orange   {{ background: #2d1e0a; color: #fb923c; }}

  /* Map area */
  #dss-map-area {{ flex: 1; position: relative; pointer-events: all; }}

  /* LOD HUD */
  #lod-hud {{
    position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
    background: rgba(13,17,23,0.85); border: 1px solid #1e2a3a;
    padding: 4px 14px; border-radius: 12px;
    font-size: 11px; font-family: 'Inter', sans-serif; color: #64748b;
    z-index: 500; pointer-events: none;
  }}

  /* Legend */
  #dss-legend {{
    position: absolute; bottom: 36px; right: 10px; z-index: 500;
    background: rgba(13,17,23,0.92); border: 1px solid #1e2a3a;
    border-radius: 8px; padding: 10px 14px; min-width: 210px; pointer-events: none;
  }}
  #dss-legend .leg-title {{
    font-size: 10px; font-weight: 600; letter-spacing: 1.4px;
    text-transform: uppercase; color: #64748b; margin-bottom: 8px;
  }}
  #dss-legend .leg-bar {{
    display: flex; height: 12px; border-radius: 3px; overflow: hidden; margin-bottom: 5px;
  }}
  #dss-legend .leg-labels {{ display: flex; justify-content: space-between; font-size: 10px; color: #475569; }}
  #dss-legend .leg-note {{ font-size: 10px; color: #475569; margin-top: 6px; }}

  /* Right panel */
  #dss-right {{
    width: 270px; flex-shrink: 0;
    background: #0d1117; border-left: 1px solid #1e2a3a;
    padding: 18px 16px; pointer-events: all;
    overflow-y: auto; display: flex; flex-direction: column;
  }}
  #dss-right::-webkit-scrollbar {{ width: 4px; }}
  #dss-right::-webkit-scrollbar-thumb {{ background: #1e2a3a; border-radius: 2px; }}

  .rp-section-label {{
    font-size: 10px; font-weight: 600; letter-spacing: 1.8px;
    text-transform: uppercase; color: #3b82f6; margin-bottom: 4px;
  }}
  #rp-hex-id {{
    font-size: 22px; font-weight: 700; color: #e2e8f0;
    margin-bottom: 18px; letter-spacing: -0.5px; word-break: break-all;
  }}
  #rp-close {{
    position: absolute; top: 66px; right: 12px;
    width: 24px; height: 24px; background: #1e2a3a; border: none;
    border-radius: 50%; color: #64748b; cursor: pointer;
    font-size: 14px; line-height: 24px; text-align: center;
    pointer-events: all; display: none;
  }}
  #rp-close:hover {{ background: #263548; color: #94a3b8; }}

  #rp-score-card {{
    background: #141d29; border: 1px solid #1e2a3a; border-radius: 10px;
    padding: 18px; margin-bottom: 18px; text-align: center;
  }}
  #rp-score-card .sc-label {{
    font-size: 10px; font-weight: 600; letter-spacing: 1.6px;
    text-transform: uppercase; color: #64748b; margin-bottom: 6px;
  }}
  #rp-score-card .sc-value {{
    font-size: 52px; font-weight: 700; line-height: 1;
    background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }}
  #rp-score-card .sc-sub {{ font-size: 11px; color: #475569; margin-top: 6px; }}

  .shap-section-header {{
    display: flex; align-items: center; gap: 7px; margin-bottom: 12px;
    font-size: 11px; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.8px;
  }}
  .shap-row {{ display: flex; align-items: center; margin-bottom: 10px; gap: 8px; }}
  .shap-label {{ font-size: 12px; color: #94a3b8; width: 130px; flex-shrink: 0; }}
  .shap-value {{ font-size: 12px; font-weight: 600; width: 42px; text-align: right; flex-shrink: 0; }}
  .shap-value.pos {{ color: #4ade80; }}
  .shap-value.neg {{ color: #f87171; }}
  .shap-bar-wrap {{ flex: 1; height: 4px; background: #1e2a3a; border-radius: 2px; position: relative; overflow: visible; }}
  .shap-bar {{ height: 4px; border-radius: 2px; position: absolute; top: 0; transition: width 0.4s ease; }}
  .shap-bar.pos {{ background: #4ade80; left: 0; }}
  .shap-bar.neg {{ background: #f87171; right: 0; }}

  .detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px; }}
  .detail-cell {{ background: #141d29; border: 1px solid #1e2a3a; border-radius: 7px; padding: 9px 11px; }}
  .detail-cell .dc-label {{ font-size: 10px; color: #475569; margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.6px; }}
  .detail-cell .dc-value {{ font-size: 13px; font-weight: 600; color: #e2e8f0; }}

  .cmp-section {{ margin-top: 14px; }}
  .cmp-head {{
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 8px;
  }}
  .cmp-title {{
    font-size: 10px; font-weight: 600; letter-spacing: 1.3px;
    text-transform: uppercase; color: #64748b;
  }}
  .cmp-actions {{ display: flex; gap: 6px; }}
  .cmp-btn {{
    border: 1px solid #1e2a3a; background: #141d29; color: #94a3b8;
    border-radius: 6px; font-size: 11px; font-weight: 600;
    padding: 5px 8px; cursor: pointer;
  }}
  .cmp-btn:hover {{ background: #1b2735; color: #cbd5e1; }}
  .cmp-btn.primary {{ border-color: #1e40af; color: #93c5fd; }}
  .cmp-input, .cmp-select {{
    width: 100%;
    border: 1px solid #1e2a3a;
    background: #0f1724;
    color: #cbd5e1;
    border-radius: 6px;
    font-size: 11px;
    padding: 6px 8px;
    margin-bottom: 6px;
  }}
  .cmp-row {{ display: flex; gap: 6px; margin-bottom: 6px; }}
  .cmp-row .cmp-btn {{ flex: 1; }}
  .cmp-hint {{ font-size: 10px; color: #64748b; margin-bottom: 8px; }}
  .cmp-toggle {{ display: flex; align-items: center; gap: 6px; margin: 4px 0 8px; color: #94a3b8; font-size: 11px; }}
  .cmp-toggle input {{ accent-color: #3b82f6; }}
  .cmp-table {{ width: 100%; border-collapse: collapse; margin-top: 6px; }}
  .cmp-table th {{
    font-size: 10px; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.7px; font-weight: 600; text-align: left;
    padding: 6px 4px; border-bottom: 1px solid #1e2a3a;
  }}
  .cmp-table td {{
    font-size: 11px; color: #cbd5e1; padding: 6px 4px;
    border-bottom: 1px solid #182233;
  }}
  .cmp-score {{ font-weight: 700; color: #67e8f9; }}
  .cmp-best {{ color: #4ade80; font-weight: 700; }}
  .cmp-empty {{ font-size: 11px; color: #475569; margin-top: 6px; }}
  .cmp-group-chip {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}

  #rp-placeholder {{
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 12px; color: #2d3748; text-align: center; padding: 30px 20px;
  }}
  #rp-placeholder svg {{ opacity: 0.3; }}
  #rp-placeholder p {{ font-size: 12px; line-height: 1.5; }}

  .rp-divider {{ border: none; border-top: 1px solid #1e2a3a; margin: 16px 0; }}
  .leaflet-container {{ background: #0d1117; }}
</style>

<!-- DSS Shell -->
<div id="dss-shell">

  <div id="dss-topbar">
    <div class="logo">
      <div class="icon">🗺️</div>
      Prototipo <span class="accent">&nbsp;MCDM</span>
    </div>
    <div class="separator"></div>
    <div class="spacer"></div>
    <div class="model-controls">
      <select id="top-model-select" title="Seleccionar metodo">
        <option value="ahp">AHP + WLC</option>
        <option value="wlc">WLC (Random Forest)</option>
        <option value="bwm">BWM + PROMETHEE II</option>
      </select>
      <button id="top-load-btn" type="button">Cargar</button>
      <span id="top-model-status"></span>
    </div>
  </div>

  <div id="dss-body">

    <!-- Left panel: Criterion weights -->
    <div id="dss-left">
      <div class="panel-section-title">Ponderacion de Criterios (Fijo)</div>

      <div class="crit-category">
        <div class="crit-category-header">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path d="M12 2a10 10 0 0 1 0 20A10 10 0 0 1 12 2z"/><path d="M12 6v6l4 2"/>
          </svg>
          A. Meteorologicos (45%)
        </div>
        <div class="crit-row"><span class="crit-label">Velocidad Viento (Avg)</span><span class="crit-badge">25%</span></div>
        <div class="crit-row"><span class="crit-label">Densidad del Aire</span><span class="crit-badge">10%</span></div>
        <div class="crit-row"><span class="crit-label">Indice de Turbulencia</span><span class="crit-badge">10%</span></div>
      </div>

      <div class="crit-category">
        <div class="crit-category-header">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.07 4.93A10 10 0 0 1 21 12a10 10 0 0 1-1.93 5.07"/>
            <path d="M4.93 4.93A10 10 0 0 0 3 12a10 10 0 0 0 1.93 5.07"/>
          </svg>
          B. Tecnicos y Suelo (35%)
        </div>
        <div class="crit-row"><span class="crit-label">Cercania a Red Electrica</span><span class="crit-badge">15%</span></div>
        <div class="crit-row"><span class="crit-label">Pendiente Maxima</span><span class="crit-badge">10%</span></div>
        <div class="crit-row"><span class="crit-label">Accesibilidad Vial</span><span class="crit-badge">5%</span></div>
        <div class="crit-row"><span class="crit-label">Capacidad Portante Suelo</span><span class="crit-badge">5%</span></div>
      </div>

      <div class="crit-category">
        <div class="crit-category-header">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
          </svg>
          C. Socio-Ambientales (20%)
        </div>
        <div class="crit-row"><span class="crit-label">Zonas Protegidas</span><span class="crit-badge excluded">EXCLUIDO</span></div>
        <div class="crit-row"><span class="crit-label">Riesgo de Conflicto</span><span class="crit-badge orange">10%</span></div>
        <div class="crit-row"><span class="crit-label">Uso del Suelo</span><span class="crit-badge green">10%</span></div>
      </div>
    </div>

    <!-- Map area -->
    <div id="dss-map-area">
      <div id="lod-hud">zoom 6 - vista region</div>
      <div id="dss-legend">
        <div class="leg-title">Puntuación de idoneidad</div>
        <div class="leg-bar" id="leg-gradient"></div>
        <div class="leg-labels">
          <span>0</span><span>2</span><span>4</span><span>6</span><span>8</span><span>10</span>
        </div>
        <div class="leg-note">* Top-{top_n_highlight} candidatos</div>
      </div>
    </div>

    <!-- Right panel: Multicriteria analysis -->
    <div id="dss-right">
      <div id="rp-placeholder">
        <svg width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
          <circle cx="12" cy="10" r="3"/>
        </svg>
        <p>Haz clic en un hexagono del mapa para ver su analisis multicriterio</p>
      </div>

      <div id="rp-content" style="display:none;">
        <div class="rp-section-label">Analisis Multicriterio</div>
        <div id="rp-hex-id">-</div>
        <button id="rp-close">x</button>

        <div id="rp-score-card">
          <div class="sc-label">Puntuación</div>
          <div class="sc-value" id="rp-score-value">-</div>
          <div class="sc-sub" id="rp-score-sub">-</div>
        </div>

        <hr class="rp-divider">
        <div class="detail-grid" id="rp-detail-grid"></div>

        <div class="cmp-section">
          <div class="cmp-head">
            <div class="cmp-title">Grupos de Cuadriculas</div>
          </div>

          <input id="grp-name" class="cmp-input" type="text" placeholder="Nombre del grupo (ej: Norte Viento Alto)">
          <div class="cmp-row">
            <button id="grp-create" class="cmp-btn primary">Crear grupo</button>
            <button id="grp-leave" class="cmp-btn">Dejar grupo</button>
          </div>

          <select id="grp-active" class="cmp-select"></select>
          <div class="cmp-row">
            <button id="grp-add-current" class="cmp-btn primary">Agregar seleccion actual</button>
            <button id="grp-remove-current" class="cmp-btn">Quitar actual</button>
          </div>
          <label class="cmp-toggle">
            <input id="grp-auto-add" type="checkbox" checked>
            Auto agregar al grupo activo al hacer clic
          </label>
          <div class="cmp-row">
            <button id="grp-clear" class="cmp-btn">Vaciar grupo</button>
            <button id="grp-delete" class="cmp-btn">Eliminar grupo</button>
          </div>

          <div class="cmp-hint">Compara grupos por score promedio y mejor celda.</div>
          <div id="grp-empty" class="cmp-empty">Crea al menos un grupo y agrega cuadriculas para comparar.</div>
          <table id="grp-table" class="cmp-table" style="display:none;">
            <thead>
              <tr>
                <th>Grupo</th>
                <th>#</th>
                <th>Prom</th>
                <th>Mejor</th>
                <th>Estado</th>
              </tr>
            </thead>
            <tbody id="grp-body"></tbody>
          </table>
        </div>
      </div>
    </div>

  </div>
</div>

<!-- Legend gradient builder -->
<script>
(function() {{
  var bar = document.getElementById('leg-gradient');
  var palette = ["#a50026","#d73027","#f46d43","#fdae61","#fee08b",
                 "#ffffbf","#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"];
  bar.innerHTML = palette.map(function(c) {{
    return '<span style="background:' + c + ';flex:1;display:inline-block;height:12px;"></span>';
  }}).join('');
}})();
</script>

<!-- Reposition Folium map inside dss-map-area -->
<script>
(function waitForMap() {{
  var mapArea = document.getElementById('dss-map-area');
  if (!mapArea) {{ setTimeout(waitForMap, 100); return; }}
  var foliumMap = document.querySelector('.folium-map') || document.querySelector('[id^="map_"]');
  if (!foliumMap) {{ setTimeout(waitForMap, 100); return; }}
  foliumMap.style.position = 'absolute';
  foliumMap.style.inset = '0';
  foliumMap.style.width = '100%';
  foliumMap.style.height = '100%';
  foliumMap.style.zIndex = '1';
  mapArea.appendChild(foliumMap);
}})();
</script>

<!-- LOD renderer + DSS right-panel logic -->
<script>
// Data from Python
// LOD0 [lat,lon,score]              zoom<=5  submuestreo 0.5 deg  ~2k puntos
// LOD1 [lat,lon,score,rank,mi,di,…] zoom 6-7 submuestreo 0.15 deg ~5k circulos
// LOD3 [verts,score,…]              zoom>=10 poligonos reales viewport-culled
var LOD0       = {lod0_js};
var LOD1       = {lod1_js};
var LOD3       = {lod3_js};
var MUNI_TABLE = {muni_table_js};
var DEPT_TABLE = {dept_table_js};
var DIVI_TABLE = {divi_table_js};
var TOP_N      = {top_js};

// Color palette RdYlGn
var PALETTE = [
  "#a50026","#d73027","#f46d43","#fdae61","#fee08b",
  "#ffffbf","#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"
];
function scoreToColor(s) {{
  return PALETTE[Math.min(10, Math.floor(Math.max(0, s) * 10.999))];
}}

// Renderer and state
var canvasRenderer = L.canvas({{ padding: 0.5 }});
var activeLayers  = [];
var renderTimer   = null;
var selectedLayer = null;
var currentHex    = null;
var groups       = {{}};
var groupOrder   = [];
var activeGroup  = '';
var GROUP_COLORS = ['#38bdf8', '#a78bfa', '#f59e0b', '#34d399', '#fb7185', '#f97316', '#22d3ee'];

function getLeafletMap() {{
  for (var k in window) {{
    if (k.indexOf('map_') === 0 && window[k] instanceof L.Map) return window[k];
  }}
  return null;
}}

function buildHexId(lat, lon) {{
  return 'IDX-H3-' + Math.abs(Math.round(lat * 10000 + lon * 1000)).toString().padStart(5, '0');
}}

function getGroupColor(name) {{
  var idx = groupOrder.indexOf(name);
  if (idx < 0) return '#64748b';
  return GROUP_COLORS[idx % GROUP_COLORS.length];
}}

function findGroupForHex(hexId) {{
  for (var i = 0; i < groupOrder.length; i++) {{
    var name = groupOrder[i];
    var items = groups[name].items;
    for (var j = 0; j < items.length; j++) {{
      if (items[j].hexId === hexId) return name;
    }}
  }}
  return null;
}}

function renderGroupSelector() {{
  var sel = document.getElementById('grp-active');
  if (!sel) return;
  var html = '<option value=""' + (!activeGroup ? ' selected' : '') + '> </option>';
  if (!groupOrder.length) {{
    html = '<option value="" selected> </option>';
  }} else {{
    for (var i = 0; i < groupOrder.length; i++) {{
      var g = groupOrder[i];
      var selected = (g === activeGroup) ? ' selected' : '';
      html += '<option value="' + g + '"' + selected + '>' + g + ' (' + groups[g].items.length + ')</option>';
    }}
  }}
  if (activeGroup && !groups[activeGroup]) activeGroup = '';
  sel.innerHTML = html;
  sel.value = activeGroup;
}}

function renderGroupCompareTable() {{
  var table = document.getElementById('grp-table');
  var body = document.getElementById('grp-body');
  var empty = document.getElementById('grp-empty');
  if (!table || !body || !empty) return;

  var rows = [];
  for (var i = 0; i < groupOrder.length; i++) {{
    var name = groupOrder[i];
    var items = groups[name].items;
    if (!items.length) continue;
    var sum = 0;
    var best = items[0].score;
    for (var j = 0; j < items.length; j++) {{
      sum += items[j].score;
      if (items[j].score > best) best = items[j].score;
    }}
    rows.push({{
      name: name,
      count: items.length,
      avg: sum / items.length,
      best: best,
      color: getGroupColor(name),
    }});
  }}

  if (!rows.length) {{
    table.style.display = 'none';
    empty.style.display = 'block';
    body.innerHTML = '';
    return;
  }}

  var winner = rows[0].avg;
  for (var k = 1; k < rows.length; k++) if (rows[k].avg > winner) winner = rows[k].avg;

  table.style.display = 'table';
  empty.style.display = 'none';

  var html = '';
  for (var r = 0; r < rows.length; r++) {{
    var row = rows[r];
    var isBest = Math.abs(row.avg - winner) < 1e-9;
    var state = isBest ? '<span class="cmp-best">MEJOR</span>' : '<span>-</span>';
    html +=
      '<tr>' +
        '<td><span class="cmp-group-chip" style="background:' + row.color + '"></span>' + row.name + '</td>' +
        '<td>' + row.count + '</td>' +
        '<td class="cmp-score">' + row.avg.toFixed(4) + '</td>' +
        '<td>' + row.best.toFixed(4) + '</td>' +
        '<td>' + state + '</td>' +
      '</tr>';
  }}
  body.innerHTML = html;
  renderGroupSelector();
}}

function createGroup() {{
  var input = document.getElementById('grp-name');
  if (!input) return;
  var name = (input.value || '').trim();
  if (!name) return;
  if (!groups[name]) {{
    groups[name] = {{ name: name, items: [] }};
    groupOrder.push(name);
  }}
  activeGroup = name;
  input.value = '';
  renderGroupCompareTable();
  renderViewport();
}}

function leaveActiveGroup() {{
  activeGroup = '';
  renderGroupSelector();
  renderViewport();
}}

function addCurrentToActiveGroup() {{
  if (!currentHex || !activeGroup || !groups[activeGroup]) return;
  var items = groups[activeGroup].items;
  for (var i = 0; i < items.length; i++) {{
    if (items[i].hexId === currentHex.hexId) return;
  }}
  items.push(currentHex);
  renderGroupCompareTable();
  renderViewport();
}}

function removeCurrentFromActiveGroup() {{
  if (!currentHex || !activeGroup || !groups[activeGroup]) return;
  var items = groups[activeGroup].items;
  groups[activeGroup].items = items.filter(function(it) {{ return it.hexId !== currentHex.hexId; }});
  renderGroupCompareTable();
  renderViewport();
}}

function clearActiveGroup() {{
  if (!activeGroup || !groups[activeGroup]) return;
  groups[activeGroup].items = [];
  renderGroupCompareTable();
  renderViewport();
}}

function deleteActiveGroup() {{
  if (!activeGroup || !groups[activeGroup]) return;
  delete groups[activeGroup];
  groupOrder = groupOrder.filter(function(g) {{ return g !== activeGroup; }});
  activeGroup = groupOrder.length ? groupOrder[0] : '';
  renderGroupCompareTable();
  renderViewport();
}}

// DSS Right panel
function showHexAnalysis(score, ws, slope, dg, dr, lu, pa, cr, rank, mi, di, dpi, lat, lon) {{
  document.getElementById('rp-placeholder').style.display = 'none';
  document.getElementById('rp-content').style.display = 'block';

  var hexId = buildHexId(lat, lon);
  currentHex = {{
    hexId: hexId,
    score: score,
    muni: (MUNI_TABLE[mi] || '-'),
    dept: (DEPT_TABLE[di] || '-'),
    divi: (DIVI_TABLE[dpi] || '-'),
    rank: rank,
    lat: lat,
    lon: lon
  }};
  document.getElementById('rp-hex-id').textContent = hexId;

  document.getElementById('rp-score-value').textContent = (score * 10).toFixed(1);
  document.getElementById('rp-score-sub').textContent = rank > 0 ? ('Ranking #' + rank) : ('Escala del 0(no idoneo) al 10(idoneo)');

  document.getElementById('rp-detail-grid').innerHTML =
    '<div class="detail-cell"><div class="dc-label">Municipio</div><div class="dc-value" style="font-size:11px;">' + (MUNI_TABLE[mi]||'-') + '</div></div>' +
    '<div class="detail-cell"><div class="dc-label">Departamento</div><div class="dc-value" style="font-size:11px;">' + (DEPT_TABLE[di]||'-') + '</div></div>' +
    '<div class="detail-cell"><div class="dc-label">Divipola</div><div class="dc-value" style="font-size:11px;">' + (DIVI_TABLE[dpi]||'-') + '</div></div>' +
    '<div class="detail-cell"><div class="dc-label">Viento</div><div class="dc-value">' + ws.toFixed(1) + ' m/s</div></div>' +
    '<div class="detail-cell"><div class="dc-label">Pendiente</div><div class="dc-value">' + slope.toFixed(1) + 'deg</div></div>' +
    '<div class="detail-cell"><div class="dc-label">Dist. Red</div><div class="dc-value">' + dg.toFixed(0) + ' km</div></div>' +
    '<div class="detail-cell"><div class="dc-label">Dist. Vias</div><div class="dc-value">' + dr.toFixed(0) + ' km</div></div>' +
    '<div class="detail-cell"><div class="dc-label">Area Proteg.</div><div class="dc-value">' + (pa*100).toFixed(0) + '%</div></div>' +
    '<div class="detail-cell"><div class="dc-label">Riesgo</div><div class="dc-value">' + cr.toFixed(3) + '</div></div>';

  document.getElementById('rp-close').style.display = 'block';
  var autoAdd = document.getElementById('grp-auto-add');
  if (autoAdd && autoAdd.checked) addCurrentToActiveGroup();
}}

function hideHexAnalysis() {{
  document.getElementById('rp-placeholder').style.display = 'flex';
  document.getElementById('rp-content').style.display = 'none';
  document.getElementById('rp-close').style.display = 'none';
  currentHex = null;
  if (selectedLayer && selectedLayer.setStyle) {{
    selectedLayer.setStyle({{
      color: selectedLayer._baseColor || "rgba(0,0,0,0.1)",
      weight: selectedLayer._baseWeight || 0.6,
    }});
  }}
  selectedLayer = null;
}}

document.getElementById('rp-close').addEventListener('click', hideHexAnalysis);
document.getElementById('grp-create').addEventListener('click', createGroup);
document.getElementById('grp-leave').addEventListener('click', leaveActiveGroup);
document.getElementById('grp-add-current').addEventListener('click', addCurrentToActiveGroup);
document.getElementById('grp-remove-current').addEventListener('click', removeCurrentFromActiveGroup);
document.getElementById('grp-clear').addEventListener('click', clearActiveGroup);
document.getElementById('grp-delete').addEventListener('click', deleteActiveGroup);
document.getElementById('grp-active').addEventListener('change', function(e) {{
  activeGroup = e.target.value || '';
  renderViewport();
}});
document.getElementById('grp-name').addEventListener('keydown', function(e) {{
  if (e.key === 'Enter') createGroup();
}});

// LOD renderers
function renderCircles(map, data, bounds, radius, interactive) {{
  var sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  var n = 0;
  for (var i = 0; i < data.length; i++) {{
    var row = data[i];
    var lat = row[0], lon = row[1], score = row[2];
    var rank = row[3]||0, mi = row[4]||0, di = row[5]||0, dpi = row[6]||0;
    var ws = row[7]||0, slope = row[8]||0, dg = row[9]||0;
    var dr = row[10]||0, lu = row[11]||0, pa = row[12]||0, cr = row[13]||0;
    var hexId = buildHexId(lat, lon);
    var groupName = findGroupForHex(hexId);
    var groupColor = groupName ? getGroupColor(groupName) : null;
    if (lat < sw.lat || lat > ne.lat || lon < sw.lng || lon > ne.lng) continue;
    (function(lat, lon, score, rank, mi, di, dpi, ws, slope, dg, dr, lu, pa, cr, groupColor) {{
      var baseColor = groupColor || (interactive ? "rgba(255,255,255,0.15)" : "none");
      var baseWeight = groupColor ? 1.8 : (interactive ? 0.5 : 0);
      var cm = L.circleMarker([lat, lon], {{
        renderer: canvasRenderer, radius: radius,
        fillColor: scoreToColor(score), fillOpacity: 0.35 + 0.55 * score,
        color: baseColor,
        weight: baseWeight,
        interactive: interactive
      }});
      cm._baseColor = baseColor;
      cm._baseWeight = baseWeight;
      if (interactive) {{
        cm.on('click', function(e) {{
          if (selectedLayer && selectedLayer !== cm && selectedLayer.setStyle) {{
            selectedLayer.setStyle({{
              color: selectedLayer._baseColor || "rgba(255,255,255,0.15)",
              weight: selectedLayer._baseWeight || 0.5,
            }});
          }}
          cm.setStyle({{ color: "#60a5fa", weight: 2 }});
          selectedLayer = cm;
          showHexAnalysis(score, ws, slope, dg, dr, lu, pa, cr, rank, mi, di, dpi, lat, lon);
        }});
      }}
      cm.addTo(map);
      activeLayers.push(cm);
    }})(lat, lon, score, rank, mi, di, dpi, ws, slope, dg, dr, lu, pa, cr, groupColor);
    n++;
  }}
  return n;
}}

function renderLOD0(map, bounds) {{
  var sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  var n = 0;
  for (var i = 0; i < LOD0.length; i++) {{
    var row = LOD0[i];
    var lat = row[0], lon = row[1], score = row[2];
    if (lat < sw.lat || lat > ne.lat || lon < sw.lng || lon > ne.lng) continue;
    var cm = L.circleMarker([lat, lon], {{
      renderer: canvasRenderer, radius: 3,
      fillColor: scoreToColor(score), fillOpacity: 0.35 + 0.55 * score,
      color: "none", weight: 0, interactive: false
    }});
    cm.addTo(map);
    activeLayers.push(cm);
    n++;
  }}
  return n;
}}

function renderLOD3(map, bounds) {{
  var sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  var n = 0;
  for (var i = 0; i < LOD3.length; i++) {{
    (function(row) {{
      var verts = row[0], score = row[1], ws = row[2], slope = row[3];
      var dg = row[4], dr = row[5], lu = row[6], pa = row[7], cr = row[8];
      var rank = row[9], mi = row[10], di = row[11], dpi = row[12];
      var clat = (verts[0][0] + verts[3][0]) / 2;
      var clon = (verts[1][1] + verts[4][1]) / 2;
      var hexId = buildHexId(clat, clon);
      var groupName = findGroupForHex(hexId);
      var groupColor = groupName ? getGroupColor(groupName) : null;
      if (clat < sw.lat || clat > ne.lat || clon < sw.lng || clon > ne.lng) return;
      var baseColor = groupColor || "rgba(255,255,255,0.08)";
      var baseWeight = groupColor ? 2.0 : 0.6;
      var poly = L.polygon(verts, {{
        renderer: canvasRenderer,
        fillColor: scoreToColor(score), fillOpacity: 0.2 + 0.6 * score,
        color: baseColor, weight: baseWeight, interactive: true
      }});
      poly._baseColor = baseColor;
      poly._baseWeight = baseWeight;
      poly.on('click', function(e) {{
        if (selectedLayer && selectedLayer !== poly && selectedLayer.setStyle) {{
          selectedLayer.setStyle({{
            color: selectedLayer._baseColor || "rgba(255,255,255,0.08)",
            weight: selectedLayer._baseWeight || 0.6,
          }});
        }}
        poly.setStyle({{ color: "#60a5fa", weight: 2.5 }});
        selectedLayer = poly;
        showHexAnalysis(score, ws, slope, dg, dr, lu, pa, cr, rank, mi, di, dpi, clat, clon);
      }});
      poly.addTo(map);
      activeLayers.push(poly);
      n++;
    }})(LOD3[i]);
  }}
  return n;
}}

// Main render dispatcher with debounce
function renderViewport() {{
  if (renderTimer) clearTimeout(renderTimer);
  renderTimer = setTimeout(function() {{
    var map = getLeafletMap();
    if (!map) return;
    for (var i = 0; i < activeLayers.length; i++) map.removeLayer(activeLayers[i]);
    activeLayers = [];
    selectedLayer = null;

    var zoom   = map.getZoom();
    var bounds = map.getBounds().pad(0.05);
    var n = 0;

    if      (zoom <= 5) n = renderLOD0(map, bounds);
    else if (zoom <= 7) n = renderCircles(map, LOD1, bounds, zoom <= 6 ? 5 : 7, true);
    else                n = renderLOD3(map, bounds);

    var lodNum = zoom<=5?0:zoom<=7?1:zoom<=9?2:3;
    console.log('[LOD' + lodNum + '] ' + n + ' objetos | zoom=' + zoom);
  }}, 120);
}}

// Top-N star markers
function addTopMarkers() {{
  var map = getLeafletMap();
  if (!map) return;
  for (var i = 0; i < TOP_N.length; i++) {{
    (function(t) {{
      var icon = L.divIcon({{
        html: '<div style="font-size:20px;color:#fbbf24;text-shadow:0 0 6px rgba(251,191,36,0.6);line-height:1;cursor:pointer;">&#9733;</div>',
        iconSize: [24, 24], iconAnchor: [12, 12], className: ''
      }});
      L.marker([t.lat, t.lon], {{ icon: icon }})
        .bindTooltip('#' + t.rank + ' · Score ' + t.score.toFixed(3) + ' · ' + t.muni, {{ direction: 'top', offset: [0, -8] }})
        .addTo(map);
    }})(TOP_N[i]);
  }}
}}

// LOD HUD
function addLodHud() {{
  var map = getLeafletMap();
  if (!map) return;
  var hud = document.getElementById('lod-hud');
  if (!hud) return;
  var descs = {{5:'vista pais',6:'vista region',7:'vista regional',
                8:'vista departamento',9:'vista local',10:'detalle completo'}};
  function update() {{
    var z = map.getZoom();
    var lodNum = z<=5?0:z<=7?1:3;
    hud.textContent = 'LOD' + lodNum + ' · zoom ' + z + ' · ' + (descs[Math.min(z,10)] || 'detalle completo');
  }}
  map.on('zoomend', update);
  update();
}}

// Model selector in top bar (backend /run-model)
function bindTopModelSelector() {{
  var sel = document.getElementById('top-model-select');
  var btn = document.getElementById('top-load-btn');
  var status = document.getElementById('top-model-status');
  if (!sel || !btn || !status) return;

  btn.addEventListener('click', function() {{
    var modelId = sel.value;
    if (!modelId) return;
    btn.disabled = true;
    status.textContent = 'Cargando...';

    fetch('/run-model', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ model: modelId }})
    }})
      .then(function(res) {{
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      }})
      .then(function(data) {{
        LOD0 = data.lod0 || [];
        LOD1 = data.lod1 || [];
        LOD3 = data.lod3 || [];
        MUNI_TABLE = (data.params && data.params.muni_table) || [];
        DEPT_TABLE = (data.params && data.params.dept_table) || [];
        DIVI_TABLE = (data.params && data.params.divi_table) || [];

        groups = {{}};
        groupOrder = [];
        activeGroup = '';
        renderGroupCompareTable();
        hideHexAnalysis();
        renderViewport();

        status.textContent = data.from_cache ? 'Desde cache' : 'Recalculado';
      }})
      .catch(function(err) {{
        console.error('[Model Selector] Error:', err);
        status.textContent = 'Error';
      }})
      .finally(function() {{
        btn.disabled = false;
      }});
  }});
}}

// Init
function init() {{
  var map = getLeafletMap();
  if (!map) {{ setTimeout(init, 150); return; }}
  bindTopModelSelector();
  renderGroupCompareTable();
  renderViewport();
  addTopMarkers();
  addLodHud();
  map.on('moveend zoomend', renderViewport);
  setTimeout(function() {{ map.invalidateSize(); }}, 300);
}}

if (document.readyState === 'complete') {{ init(); }}
else {{ window.addEventListener('load', init); }}
</script>
"""

    m.get_root().html.add_child(folium.Element(dss_ui))

    # 4. Guardar
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[Map] Mapa guardado -> {output_path}  ({size_kb:.0f} KB)")
    if size_kb > 15_000:
        print(f"[Map] AVISO: {size_kb/1024:.1f} MB — considera zoom=7 como resolucion maxima.")


# ---------------------------------------------------------------------------
# Score distribution plot
# ---------------------------------------------------------------------------

def plot_score_distribution(
    df: pd.DataFrame,
    output_path: str,
    score_column: str = "suitability_score",
) -> None:
    """
    Histograma de la distribucion de scores de aptitud.
    """
    scores = df[score_column].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(scores, bins=40, edgecolor="white")

    cmap = cm.get_cmap("RdYlGn")
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    for patch, centre in zip(patches, bin_centres):
        patch.set_facecolor(cmap(centre))

    ax.axvline(scores.mean(), color="#2c3e50", linewidth=1.5,
               linestyle="--", label=f"Media = {scores.mean():.3f}")
    ax.axvline(np.percentile(scores, 75), color="#8e44ad", linewidth=1.2,
               linestyle=":", label=f"Pct 75 = {np.percentile(scores, 75):.3f}")

    ax.set_xlabel("Suitability Score", fontsize=10)
    ax.set_ylabel("Numero de hexagonos", fontsize=10)
    ax.set_title("Distribucion de aptitud para parques eolicos — Colombia",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Distribucion de scores -> {output_path}")


# ---------------------------------------------------------------------------
# Feature correlation heatmap
# ---------------------------------------------------------------------------

def plot_feature_correlation(
    df: pd.DataFrame,
    norm_features: List[str],
    output_path: str,
    score_column: str = "suitability_score",
) -> None:
    """
    Heatmap de correlacion de Pearson entre features normalizadas y score.
    """
    cols = norm_features + ([score_column] if score_column in df.columns else [])
    corr = df[cols].corr()
    labels = [c.replace("_norm", "").replace("_", "\n").title() for c in cols]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            colour = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=colour)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="Pearson r")
    ax.set_title("Correlacion de features (criterios normalizados + score)",
                 fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Heatmap de correlacion -> {output_path}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_h3_grid      import generate_colombia_hex_grid
    from feature_engineering   import engineer_features
    from normalization         import normalise_features, get_norm_feature_names
    from random_forest_weights import get_rf_weights
    from mcda_model            import compute_wlc_scores, rank_locations

    _HERE    = os.path.dirname(os.path.abspath(__file__))
    _GEOJSON = os.path.join(_HERE, "..", "data", "colombia_boundary.geojson")
    _OUT     = os.path.join(_HERE, "..", "outputs")
    os.makedirs(_OUT, exist_ok=True)

    grid      = generate_colombia_hex_grid(_GEOJSON, resolution=4)
    feats     = engineer_features(grid)
    norm_df   = normalise_features(feats)
    norm_cols = get_norm_feature_names()
    model, weights, labels = get_rf_weights(norm_df, norm_cols)
    scored_df = compute_wlc_scores(norm_df, weights, norm_cols)
    ranked_df = rank_locations(scored_df)

    create_interactive_map(
        ranked_df,
        os.path.join(_OUT, "map_interactive.html"),
        zoom=6,
    )
    plot_score_distribution(
        scored_df,
        os.path.join(_OUT, "score_distribution.png"),
    )
