"""
visualization.py
================
Crea todos los outputs visuales del modelo MCDA de aptitud eólica:

1. **Mapa interactivo HTML** – arquitectura híbrida:
   - Folium genera el mapa base con tiles CartoDB/OSM y controles estándar.
   - Los hexágonos se renderizan con Leaflet L.Canvas directamente inyectado
     en el HTML, evitando el cuello de botella de ``folium.GeoJson`` que
     embebe un objeto JS gigante (la línea ``geo_json_XXX_add({...})``) que
     congela el navegador con resoluciones altas.

   Estrategia de rendimiento
   -------------------------
   • **L.Canvas renderer**: Leaflet dibuja polígonos sobre un <canvas> en
     lugar de crear un nodo SVG por hexágono. Para 26 000+ hexágonos la
     diferencia es de minutos a < 2 segundos de carga.
   • **Datos compactos**: los hexágonos se serializan como un array JS de
     arrays numéricos ``[lon_c, lat_c, score, ws, slope, dg, dr, lu, pa, cr,
     rank, muni_idx, dept_idx]`` + tablas de strings separadas.
     Esto reduce el payload ~70% respecto a GeoJSON completo.
   • **Viewport culling**: en cada evento ``moveend`` / ``zoomend`` solo se
     añaden al mapa los hexágonos cuyos centroides están dentro del bounds
     visible + 20% de margen. Los fuera de vista se eliminan del DOM.
   • **Vértices reconstruidos en JS**: los 6 vértices del hexágono se
     calculan en el cliente a partir del centroide y el radio, evitando
     transmitir 6 coordenadas por celda.
   • **Popup bajo demanda**: el popup HTML se construye solo al hacer click,
     no hay 26 000 tooltips pre-renderizados en el DOM.

2. **Gráfica de distribución** – histograma de scores (matplotlib).
3. **Heatmap de correlación** – correlación de Pearson (matplotlib).
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
#: Por encima de este número de hexágonos se activa la simplificación de
#: coordenadas para mantener el archivo HTML por debajo de ~10 MB.
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
# GeoJSON builder
# ---------------------------------------------------------------------------

def _simplify_ring(coords: list, tol: float = 0.002) -> list:
    """
    Douglas-Peucker simplificado sobre un anillo de coordenadas.
    Para hexágonos regulares con 6 vértices la simplificación no aplica,
    pero sí reduce rings complejos en polígonos irregulares.
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

    Parameters
    ----------
    df            : DataFrame con columnas ``vertices``, ``hex_id``, score y features
    score_column  : nombre de la columna de score
    feature_cols  : columnas extra a incluir en properties
    simplify      : si True aplica simplificación de coordenadas (resoluciones altas)

    Returns
    -------
    GeoJSON dict
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

        # Municipio / departamento si existen
        for meta in ("municipality", "department", "divipola_code"):
            if meta in row.index:
                val = row[meta]
                props[meta] = str(val) if not pd.isna(val) else "—"

        score = float(row.get(score_column, 0) or 0)
        props["colour"] = _score_to_hex_colour(score)
        props["opacity"] = round(0.4 + 0.5 * score, 2)  # más opaco = más apto

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

        # Cerrar el anillo si no está cerrado
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Helpers de serialización compacta
# ---------------------------------------------------------------------------

def _build_compact_payload(
    df: pd.DataFrame,
    score_column: str,
) -> tuple[str, str, str]:
    """
    Serializa los hexágonos como arrays JS compactos usando los vértices
    REALES devueltos por H3, en lugar de recalcularlos en JS.

    Por qué los vértices reales son imprescindibles
    ------------------------------------------------
    Los hexágonos H3 NO son regulares en coordenadas geográficas: su forma
    varía ligeramente según la latitud y el índice de la celda. Si se
    reconstruyen desde el centroide con una fórmula trigonométrica aproximada,
    los bordes entre celdas vecinas no coinciden, produciendo solapamientos
    y huecos visibles (el bug "hexágonos encima de hexágonos").

    Formato de cada fila en HEX_DATA:
        [
          [[lat0,lon0],[lat1,lon1],...,[lat5,lon5]],  ← 6 vértices reales de H3
          score, wind_speed, slope, dist_grid, dist_roads,
          land_use, protected_area, conflict_risk,
          rank, muni_idx, dept_idx
        ]

    Los vértices se almacenan como [lat,lon] (orden Leaflet) con 4 decimales
    (~11 m de precisión), suficiente para visualización.

    Tamaño típico: ~180 bytes/hexágono vs ~800 bytes en GeoJSON completo
    (ahorro ~77 %).
    """
    df = df.sort_values(by=score_column, ascending=False)
    munis = list(
        df.get("municipality", pd.Series(["—"] * len(df))).fillna("—").unique()
    )
    depts = list(
        df.get("department", pd.Series(["—"] * len(df))).fillna("—").unique()
    )
    muni_idx_map = {m: i for i, m in enumerate(munis)}
    dept_idx_map = {d: i for i, d in enumerate(depts)}

    has_vertices = "vertices" in df.columns

    rows = []
    for _, r in df.iterrows():
        def _f(col, default=0.0):
            v = r.get(col, default)
            try:
                fv = float(v)
                return round(fv, 4) if not math.isnan(fv) else default
            except (TypeError, ValueError):
                return default

        # ── Vértices reales de H3 en formato [[lat,lon], ...] ──────────────
        # La columna 'vertices' de generate_h3_grid tiene formato [(lon,lat),...]
        # Leaflet necesita [lat,lon], así que invertimos aquí en Python.
        if has_vertices and r["vertices"] is not None:
            raw = r["vertices"]
            # Excluir el último punto si cierra el anillo (== primero)
            ring = raw[:-1] if len(raw) == 7 and raw[0] == raw[-1] else raw[:6]
            verts = [[round(v[1], 4), round(v[0], 4)] for v in ring]  # [lat, lon]
        else:
            # Fallback: aproximación desde centroide si no hay vértices
            # (no debería ocurrir con generate_h3_grid normal)
            lon_c, lat_c = float(r["lon"]), float(r["lat"])
            r_deg = 0.10
            rx = r_deg / max(math.cos(math.radians(lat_c)), 0.01)
            verts = [
                [round(lat_c + r_deg * math.sin(math.radians(60 * k + 30)), 4),
                 round(lon_c + rx  * math.cos(math.radians(60 * k + 30)), 4)]
                for k in range(6)
            ]

        rank_val = int(r["rank"]) if "rank" in r.index and not pd.isna(r.get("rank")) else 0
        muni_str = str(r.get("municipality", "—")) if "municipality" in r.index else "—"
        dept_str = str(r.get("department",   "—")) if "department"   in r.index else "—"

        rows.append([
            verts,                          # índice 0: [[lat,lon]×6]
            _f(score_column),               # 1
            _f("wind_speed"),               # 2
            _f("slope"),                    # 3
            _f("dist_to_grid"),             # 4
            _f("dist_to_roads"),            # 5
            _f("land_use"),                 # 6
            _f("protected_area"),           # 7
            _f("conflict_risk"),            # 8
            rank_val,                       # 9
            muni_idx_map.get(muni_str, 0),  # 10
            dept_idx_map.get(dept_str, 0),  # 11
        ])

    hex_data_js   = json.dumps(rows,  separators=(",", ":"))
    muni_table_js = json.dumps(munis, separators=(",", ":"))
    dept_table_js = json.dumps(depts, separators=(",", ":"))
    return hex_data_js, muni_table_js, dept_table_js


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
# Mapa interactivo — arquitectura híbrida Folium + Leaflet Canvas
# ---------------------------------------------------------------------------

def create_interactive_map(
    df: pd.DataFrame,
    output_path: str,
    score_column: str = "suitability_score",
    top_n_highlight: int = 10,
    centre_lat: float = 4.711,
    centre_lon: float = -74.0721,
    zoom: int = 14,
) -> None:
    """
    Genera un mapa HTML interactivo de alto rendimiento.

    Arquitectura
    ------------
    Folium genera el esqueleto HTML con el mapa base Leaflet (tiles CartoDB/OSM)
    y la infraestructura de controles. Los hexágonos NO se pasan por
    ``folium.GeoJson`` — esa es la línea que congela el navegador.

    En su lugar, Python inyecta directamente en el HTML:

    1. **HEX_DATA**: array JS compacto con los datos de cada hexágono
       (13 números por celda en lugar de un objeto GeoJSON completo).
    2. **Motor de renderizado Leaflet L.Canvas**: dibuja los hexágonos
       como polígonos sobre un elemento <canvas>, que el navegador puede
       pintar en < 1 segundo para 200 000+ polígonos frente a los varios
       minutos del renderer SVG.
    3. **Viewport culling**: en cada ``moveend`` / ``zoomend`` se eliminan
       las capas fuera del bounding box visible y se añaden solo las nuevas
       que entran al viewport.
    4. **Radio calculado en JS**: los vértices del hexágono se calculan en
       el cliente a partir del centroide + radio estimado, evitando
       transmitir 6 × 2 coordenadas por celda.
    5. **Popup bajo demanda**: el HTML del popup se construye solo al hacer
       click, sin pre-renderizar miles de nodos DOM.

    Parameters
    ----------
    df               : DataFrame scored y rankeado (debe tener ``lat``, ``lon``)
    output_path      : ruta del archivo .html a generar
    score_column     : columna de score de aptitud
    top_n_highlight  : número de top celdas a marcar con estrella
    centre_lat/lon   : centro inicial (default: Bogotá 4.711, -74.072)
    zoom             : zoom inicial (6 = país completo, 8 = regional)
    """
    try:
        import folium
        from folium.plugins import MiniMap, Fullscreen
    except ImportError:
        raise ImportError("Instala Folium:  pip install folium")

    n = len(df)
    print(f"[Map] Construyendo mapa híbrido | {n:,} hexágonos | zoom={zoom}")

    # ------------------------------------------------------------------
    # 1. Serializar datos compactos en Python
    # ------------------------------------------------------------------
    hex_data_js, muni_table_js, dept_table_js = _build_compact_payload(df, score_column)
    top_js = _build_top_n_js(df, score_column, top_n_highlight)

    # ------------------------------------------------------------------
    # 2. Mapa base Folium — solo tiles y controles
    # ------------------------------------------------------------------
    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer("OpenStreetMap",      name="OpenStreetMap", show=False).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Carto Dark",    show=False).add_to(m)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    Fullscreen(position="topright").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Leyenda de color
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:40px;right:15px;z-index:9999;pointer-events:none;">'
        f'{_build_colour_scale_html()}</div>'
    ))

    # ------------------------------------------------------------------
    # 3. Inyectar motor de renderizado Canvas directamente en el HTML
    #    IMPORTANTE: esto se añade DESPUÉS del HTML de Folium para que
    #    el objeto `map_XXXX` de Leaflet ya exista cuando se ejecute.
    # ------------------------------------------------------------------
    canvas_script = f"""
<script>
// ── Datos generados por Python ────────────────────────────────────────────────
const HEX_DATA   = {hex_data_js};
const MUNI_TABLE = {muni_table_js};
const DEPT_TABLE = {dept_table_js};
const TOP_N      = {top_js};

// ── Paleta ────────────────────────────────────────────────────────────────────
const PALETTE = [
  "#a50026","#d73027","#f46d43","#fdae61","#fee08b",
  "#ffffbf",
  "#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"
];

function scoreToColor(s) {{
  const i = Math.min(10, Math.floor(Math.max(0, s) * 10.999));
  return PALETTE[i];
}}

// ── Canvas renderer ───────────────────────────────────────────────────────────
const canvasRenderer = L.canvas({{ padding: 0.3 }});

// ── Estado ────────────────────────────────────────────────────────────────────
let activePolygons = [];
let activePopup    = null;

function getLeafletMap() {{
  for (const k of Object.keys(window)) {{
    if (k.startsWith("map_") && window[k] instanceof L.Map) return window[k];
  }}
  return null;
}}

// ── Render optimizado por zoom ────────────────────────────────────────────────
function renderViewport() {{
  const map = getLeafletMap();
  if (!map) return;

  const zoom = map.getZoom();
  const isLowZoom = zoom <= 2;

  const bounds = map.getBounds().pad(0.15);
  const minLat = bounds.getSouth(), maxLat = bounds.getNorth();
  const minLon = bounds.getWest(),  maxLon = bounds.getEast();

  // Limpiar mapa
  for (const p of activePolygons) map.removeLayer(p);
  activePolygons = [];

  // Control por zoom (OPTIMIZACIÓN CLAVE)
  let dataToRender = HEX_DATA;

  if (zoom <= 2) {{
    dataToRender = HEX_DATA.slice(0, 3000);
  }} else if (zoom <= 4) {{
    dataToRender = HEX_DATA.slice(0, 8000);
  }}

  for (const row of dataToRender) {{

    const [verts, score, ws, slope, dg, dr, lu, pa, cr, rank, muniIdx, deptIdx] = row;

    const clat = (verts[0][0] + verts[3][0]) / 2;
    const clon = (verts[1][1] + verts[4][1]) / 2;

    if (clat < minLat || clat > maxLat || clon < minLon || clon > maxLon) continue;

    const color   = scoreToColor(score);
    const opacity = 0.2 + 0.55 * score;

    const poly = L.polygon(verts, {{
      renderer: canvasRenderer,
      fillColor: color,
      fillOpacity: opacity,
      color: "rgba(0,0,0,0.05)",
      weight: 0.4,
      interactive: !isLowZoom,
    }});

    poly.on("click", function(e) {{
      if (activePopup) activePopup.remove();

      const rankStr = rank > 0
        ? `<b style="color:#c0392b;font-size:14px;">★ Ranking #${{rank}}</b><br>`
        : "";

      const html =
        `<div style="font-family:sans-serif;font-size:12px;min-width:210px;line-height:1.8">
          ${{rankStr}}
          <b>Score aptitud:</b> ${{score.toFixed(4)}}<br>
          <b>Municipio:</b> ${{MUNI_TABLE[muniIdx] || "—"}}<br>
          <b>Departamento:</b> ${{DEPT_TABLE[deptIdx] || "—"}}<br>
          <b>Viento:</b> ${{ws.toFixed(1)}} m/s<br>
          <b>Pendiente:</b> ${{slope.toFixed(1)}}°<br>
          <b>Dist. red eléctrica:</b> ${{dg.toFixed(0)}} km<br>
          <b>Dist. vías:</b> ${{dr.toFixed(0)}} km<br>
          <b>Uso suelo:</b> ${{lu.toFixed(3)}}<br>
          <b>Área protegida:</b> ${{pa.toFixed(3)}}<br>
          <b>Riesgo conflicto:</b> ${{cr.toFixed(3)}}<br>
        </div>`;

      activePopup = L.popup({{ maxWidth: 280, closeButton: true }})
        .setLatLng(e.latlng)
        .setContent(html)
        .openOn(map);
    }});

    poly.addTo(map);
    activePolygons.push(poly);
  }}

  console.log(`[HexGrid] ${{activePolygons.length}} hexágonos en viewport | zoom=${{zoom}}`);
}}

// ── Marcadores Top-N ──────────────────────────────────────────────────────────
function addTopMarkers() {{
  const map = getLeafletMap();
  if (!map) return;

  for (const t of TOP_N) {{
    const icon = L.divIcon({{
      html: `<div style="font-size:22px;color:#c0392b;text-shadow:0 0 4px #fff;
                         line-height:1;cursor:pointer;">★</div>`,
      iconSize: [26, 26], iconAnchor: [13, 13], className: ""
    }});

    const html =
      `<div style="font-family:sans-serif;font-size:13px;min-width:180px;line-height:1.7">
        <b style="font-size:15px;">★ #${{t.rank}}</b><br>
        <b>Score:</b> ${{t.score.toFixed(4)}}<br>
        <b>Viento:</b> ${{t.ws}} m/s<br>
        <b>Municipio:</b> ${{t.muni}}<br>
        <b>Departamento:</b> ${{t.dept}}
      </div>`;

    L.marker([t.lat, t.lon], {{ icon }})
      .bindPopup(html, {{ maxWidth: 240 }})
      .bindTooltip(`#${{t.rank}} · Score ${{t.score.toFixed(3)}}`)
      .addTo(map);
  }}
}}

// ── Inicialización ────────────────────────────────────────────────────────────
function init() {{
  const map = getLeafletMap();
  if (!map) {{ setTimeout(init, 150); return; }}

  renderViewport();
  addTopMarkers();

  map.on("moveend zoomend", renderViewport);
}}

if (document.readyState === "complete") {{ init(); }}
else {{ window.addEventListener("load", init); }}
</script>
"""

    # Inyectar el script al final del body del HTML de Folium
    m.get_root().html.add_child(folium.Element(canvas_script))

    # ------------------------------------------------------------------
    # 4. Guardar
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[Map] Mapa guardado → {output_path}  ({size_kb:.0f} KB)")
    if size_kb > 15_000:
        print(f"[Map] ⚠  {size_kb/1024:.1f} MB — considera zoom=7 como resolución máxima "
              f"o activa simplificación de vértices.")


# ---------------------------------------------------------------------------
# Score distribution plot
# ---------------------------------------------------------------------------

def plot_score_distribution(
    df: pd.DataFrame,
    output_path: str,
    score_column: str = "suitability_score",
) -> None:
    """
    Histograma de la distribución de scores de aptitud.
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
    ax.set_ylabel("Número de hexágonos", fontsize=10)
    ax.set_title("Distribución de aptitud para parques eólicos — Colombia",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Distribución de scores → {output_path}")


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
    Heatmap de correlación de Pearson entre features normalizadas y score.
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
    ax.set_title("Correlación de features (criterios normalizados + score)",
                 fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Heatmap de correlación → {output_path}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_h3_grid    import generate_colombia_hex_grid
    from feature_engineering import engineer_features
    from normalization       import normalise_features, get_norm_feature_names
    from random_forest_weights import get_rf_weights
    from mcda_model          import compute_wlc_scores, rank_locations

    _HERE   = os.path.dirname(os.path.abspath(__file__))
    _GEOJSON = os.path.join(_HERE, "..", "data", "colombia_boundary.geojson")
    _OUT    = os.path.join(_HERE, "..", "outputs")
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
        zoom=14,
    )
    plot_score_distribution(
        scored_df,
        os.path.join(_OUT, "score_distribution.png"),
    ) 
