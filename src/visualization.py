"""
visualization.py
================
Creates all visual outputs for the wind farm suitability assessment:

1. **Interactive HTML map** – a Leaflet.js-powered choropleth map rendered
   as a single self-contained HTML file (no Folium dependency needed).
   Hexagons are coloured by suitability score with a popup showing
   all feature values on click.

2. **Static overview map** – matplotlib figure showing suitability as a
   scatter plot on a geographic axis (useful for publication figures).

3. **Distribution plots** – score distribution histogram and feature
   correlation heatmap.

The HTML map uses only browser-native JavaScript and the CDN-hosted
Leaflet library, so it opens in any modern browser without installation.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _score_to_hex_colour(score: float, cmap_name: str = "RdYlGn") -> str:
    """
    Map a suitability score in [0, 1] to an HTML hex colour string.

    Parameters
    ----------
    score     : float in [0, 1]
    cmap_name : matplotlib colormap name

    Returns
    -------
    '#RRGGBB' string
    """
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(float(np.clip(score, 0, 1)))
    return mcolors.to_hex(rgba)


def _build_colour_scale_html(n_steps: int = 6) -> str:
    """Build an HTML legend gradient bar for the interactive map."""
    cmap = cm.get_cmap("RdYlGn")
    stops = []
    for i in range(n_steps + 1):
        v = i / n_steps
        colour = mcolors.to_hex(cmap(v))
        stops.append(f'<span style="background:{colour};flex:1;display:inline-block;height:16px;" title="{v:.1f}"></span>')

    labels = "".join(
        f'<span style="flex:1;font-size:10px;text-align:center;">{i/n_steps:.1f}</span>'
        for i in range(n_steps + 1)
    )

    return f"""
    <div style="background:white;padding:8px;border-radius:4px;
                box-shadow:0 1px 4px rgba(0,0,0,0.3);min-width:200px;">
      <b style="font-size:12px;">Suitability Score</b><br>
      <div style="display:flex;margin-top:4px;">{"".join(stops)}</div>
      <div style="display:flex;margin-top:2px;">{labels}</div>
      <div style="font-size:10px;margin-top:4px;color:#555;">
        ● Top 10 sites marked with ★
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# GeoJSON builder from hex grid DataFrame
# ---------------------------------------------------------------------------

def df_to_geojson(
    df: pd.DataFrame,
    score_column: str = "suitability_score",
    feature_cols: Optional[List[str]] = None,
) -> dict:
    """
    Convert the scored hex-grid DataFrame to a GeoJSON FeatureCollection.

    Parameters
    ----------
    df            : DataFrame with ``vertices``, ``hex_id``, score and feature columns
    score_column  : suitability score column name
    feature_cols  : list of extra columns to include in feature properties

    Returns
    -------
    GeoJSON dict
    """
    if feature_cols is None:
        feature_cols = [
            "wind_speed", "slope", "dist_to_grid", "dist_to_roads",
            "land_use", "protected_area", "conflict_risk", score_column,
        ]
    # Add rank if present
    if "rank" in df.columns:
        feature_cols = ["rank"] + feature_cols

    features = []
    for _, row in df.iterrows():
        props = {col: (None if pd.isna(row[col]) else round(float(row[col]), 3))
                 for col in feature_cols if col in row.index}
        props["hex_id"] = row["hex_id"]
        props["colour"] = _score_to_hex_colour(row.get(score_column, 0))

        if "vertices" in row.index and row["vertices"] is not None:
            coords = [[v[0], v[1]] for v in row["vertices"]]
        else:
            # Fall back to a tiny marker polygon
            cx, cy, sz = row["lon"], row["lat"], 0.02
            coords = [
                [cx - sz, cy], [cx, cy + sz], [cx + sz, cy],
                [cx, cy - sz], [cx - sz, cy],
            ]

        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Interactive HTML map
# ---------------------------------------------------------------------------

def create_interactive_map(
    df: pd.DataFrame,
    output_path: str,
    score_column: str = "suitability_score",
    top_n_highlight: int = 10,
    centre_lat: float = 4.5,
    centre_lon: float = -74.0,
    zoom: int = 6,
) -> None:
    """
    Generate a fully self-contained interactive HTML map using pure
    inline Canvas + JavaScript — zero external dependencies, works in
    any sandboxed iframe or offline environment.

    The map supports:
      - Pan (click + drag) and zoom (scroll wheel / buttons)
      - Hexagons coloured by suitability score (RdYlGn palette)
      - Click-to-inspect popup with all feature values
      - Top-N star markers for best candidate sites
      - Colour-scale legend

    Parameters
    ----------
    df                : scored hex-grid DataFrame (must have ``vertices``)
    output_path       : path to write the .html file
    score_column      : name of the suitability score column
    top_n_highlight   : number of top cells to mark with stars
    centre_lat/lon    : initial map centre (unused — auto-fitted)
    zoom              : unused — view auto-fitted to data extent
    """
    print("[Map] Building self-contained Canvas map (no external deps)...")

    def _safe_str(value: object, default: str = "Unknown") -> str:
      if pd.isna(value):
        return default
      text = str(value).strip()
      return text if text else default

    # ------------------------------------------------------------------
    # Build a compact data payload for JavaScript
    # ------------------------------------------------------------------
    top_ids = set(df.nlargest(top_n_highlight, score_column)["hex_id"].tolist())

    cells = []
    for _, row in df.iterrows():
        if "vertices" in row.index and row["vertices"] is not None:
            verts = [[round(v[0], 5), round(v[1], 5)] for v in row["vertices"]]
        else:
            cx, cy, sz = row["lon"], row["lat"], 0.2
            verts = [[cx-sz,cy],[cx,cy+sz],[cx+sz,cy],[cx,cy-sz],[cx-sz,cy]]

        score = float(row.get(score_column, 0) or 0)
        cells.append({
            "id":    row["hex_id"],
            "v":     verts,          # vertices [[lon,lat],...]
            "s":     round(score, 4),
            "top":   row["hex_id"] in top_ids,
            "rank":  int(row["rank"]) if "rank" in row.index and not pd.isna(row.get("rank")) else None,
          "dp":    _safe_str(row.get("divipola_code", "Unknown")),
          "mu":    _safe_str(row.get("municipality", "Unknown")),
          "de":    _safe_str(row.get("department", "Unknown")),
            "ws":    round(float(row.get("wind_speed",   0) or 0), 2),
            "sl":    round(float(row.get("slope",        0) or 0), 2),
            "dg":    round(float(row.get("dist_to_grid", 0) or 0), 1),
            "dr":    round(float(row.get("dist_to_roads",0) or 0), 1),
            "lu":    round(float(row.get("land_use",     0) or 0), 3),
            "pa":    round(float(row.get("protected_area",0) or 0), 3),
            "cr":    round(float(row.get("conflict_risk", 0) or 0), 3),
        })

    import json as _json
    cells_json = _json.dumps(cells, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Colombia Wind Farm Suitability</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#1a2a3a;font-family:'Segoe UI',sans-serif;overflow:hidden;height:100vh}}
  #cvs{{display:block;cursor:grab}}
  #cvs.dragging{{cursor:grabbing}}
  #ui{{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}}
  #title{{position:absolute;top:10px;left:50%;transform:translateX(-50%);
    background:rgba(255,255,255,0.93);padding:7px 18px;border-radius:20px;
    font-size:14px;font-weight:700;color:#1a2a3a;white-space:nowrap;
    box-shadow:0 2px 8px rgba(0,0,0,0.4);pointer-events:none}}
  #info{{position:absolute;top:10px;left:12px;background:rgba(255,255,255,0.92);
    padding:9px 13px;border-radius:8px;font-size:11px;color:#222;
    box-shadow:0 1px 5px rgba(0,0,0,0.3);line-height:1.6;pointer-events:none}}
  #legend{{position:absolute;bottom:20px;right:12px;background:rgba(255,255,255,0.92);
    padding:10px 14px;border-radius:8px;font-size:11px;color:#222;
    box-shadow:0 1px 5px rgba(0,0,0,0.3);pointer-events:none}}
  #legend .bar{{display:flex;height:14px;border-radius:3px;overflow:hidden;margin:5px 0 2px}}
  #legend .bar span{{flex:1}}
  #legend .ticks{{display:flex;justify-content:space-between;font-size:10px;color:#555}}
  #popup{{position:absolute;display:none;background:white;border-radius:8px;
    padding:11px 14px;font-size:12px;line-height:1.7;color:#222;
    box-shadow:0 3px 12px rgba(0,0,0,0.35);min-width:200px;max-width:240px;
    pointer-events:auto;border-left:4px solid #27ae60}}
  #popup.bad{{border-left-color:#e74c3c}}
  #popup.mid{{border-left-color:#f39c12}}
  #popup h4{{margin:0 0 6px;font-size:13px;color:#1a2a3a}}
  #popup .close{{float:right;cursor:pointer;font-size:15px;color:#aaa;
    line-height:1;margin-left:8px}}
  #popup hr{{border:none;border-top:1px solid #eee;margin:6px 0}}
  #zoom-btns{{position:absolute;bottom:20px;left:12px;pointer-events:auto}}
  #zoom-btns button{{display:block;width:32px;height:32px;margin-bottom:4px;
    border:none;border-radius:6px;background:rgba(255,255,255,0.92);
    font-size:18px;cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,0.3);
    color:#333;line-height:32px;text-align:center}}
  #zoom-btns button:hover{{background:white}}
</style>
</head>
<body>
<canvas id="cvs"></canvas>
<div id="ui">
  <div id="title">🌬️ Wind Farm Suitability Assessment — Colombia</div>
  <div id="info">
    <b>MCDA Wind Farm Suitability</b><br>
    H3 hexagonal grid · WLC model<br>
    RF-derived weights · {len(df):,} cells<br>
    <hr style="border:none;border-top:1px solid #ccc;margin:4px 0">
    🖱 Scroll to zoom · Drag to pan<br>
    Click a cell for details<br>
    ⭐ = Top {top_n_highlight} candidates
  </div>
  <div id="legend">
    <b>Suitability Score</b>
    <div class="bar" id="lgbar"></div>
    <div class="ticks"><span>0.0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1.0</span></div>
  </div>
  <div id="popup">
    <span class="close" id="popclose">✕</span>
    <h4 id="pop-title">Cell</h4>
    <div id="pop-body"></div>
  </div>
  <div id="zoom-btns">
    <button id="zin">+</button>
    <button id="zout">−</button>
  </div>
</div>

<script>
// ── Data ────────────────────────────────────────────────────────────────────
var CELLS = {cells_json};

// ── Colour scale (RdYlGn 9-class) ───────────────────────────────────────────
var RAMP = [
  [165,0,38],[215,48,39],[244,109,67],[253,174,97],[255,255,191],
  [217,239,139],[166,217,106],[102,189,99],[26,152,80]
];
function scoreColor(s){{
  var t = Math.max(0,Math.min(0.9999,s)) * (RAMP.length-1);
  var i = Math.floor(t), f = t-i;
  var a=RAMP[i], b=RAMP[i+1]||RAMP[i];
  var r=Math.round(a[0]+(b[0]-a[0])*f);
  var g=Math.round(a[1]+(b[1]-a[1])*f);
  var bl=Math.round(a[2]+(b[2]-a[2])*f);
  return 'rgb('+r+','+g+','+bl+')';
}}

// Build legend gradient
(function(){{
  var bar = document.getElementById('lgbar');
  for(var i=0;i<=20;i++){{
    var sp=document.createElement('span');
    sp.style.background=scoreColor(i/20);
    bar.appendChild(sp);
  }}
}})();

// ── Canvas setup ─────────────────────────────────────────────────────────────
var cvs = document.getElementById('cvs');
var ctx = cvs.getContext('2d');

function resize(){{
  cvs.width  = window.innerWidth;
  cvs.height = window.innerHeight;
  draw();
}}
window.addEventListener('resize', resize);

// ── Viewport state ───────────────────────────────────────────────────────────
// Compute data extent
var lonMin=Infinity,lonMax=-Infinity,latMin=Infinity,latMax=-Infinity;
CELLS.forEach(function(c){{
  c.v.forEach(function(v){{
    if(v[0]<lonMin)lonMin=v[0]; if(v[0]>lonMax)lonMax=v[0];
    if(v[1]<latMin)latMin=v[1]; if(v[1]>latMax)latMax=v[1];
  }});
}});

// Transform: lon/lat → canvas px
var view = {{ ox:0, oy:0, scale:1 }};  // ox/oy = pan offset, scale = zoom

function fitView(){{
  var W=cvs.width, H=cvs.height;
  var dLon=lonMax-lonMin, dLat=latMax-latMin;
  var pad=0.06;
  var sx = W*(1-2*pad)/dLon;
  var sy = H*(1-2*pad)/dLat;
  view.scale = Math.min(sx,sy);
  view.ox = W/2 - (lonMin+lonMax)/2*view.scale;
  view.oy = H/2 + (latMin+latMax)/2*view.scale; // y flipped
}}

function lon2x(lon){{ return lon*view.scale + view.ox; }}
function lat2y(lat){{ return -lat*view.scale + view.oy; }}
function x2lon(x){{  return (x - view.ox) / view.scale; }}
function y2lat(y){{  return -(y - view.oy) / view.scale; }}

// ── Draw ─────────────────────────────────────────────────────────────────────
var hovIdx = -1;

function draw(){{
  ctx.clearRect(0,0,cvs.width,cvs.height);
  // Ocean background
  ctx.fillStyle='#cde4f0';
  ctx.fillRect(0,0,cvs.width,cvs.height);

  CELLS.forEach(function(c,i){{
    var verts = c.v;
    ctx.beginPath();
    ctx.moveTo(lon2x(verts[0][0]), lat2y(verts[0][1]));
    for(var j=1;j<verts.length;j++)
      ctx.lineTo(lon2x(verts[j][0]), lat2y(verts[j][1]));
    ctx.closePath();

    ctx.fillStyle = scoreColor(c.s);
    ctx.fill();

    if(i===hovIdx){{
      ctx.strokeStyle='#fff';
      ctx.lineWidth=2;
    }} else {{
      ctx.strokeStyle='rgba(80,80,80,0.35)';
      ctx.lineWidth=0.5;
    }}
    ctx.stroke();

    // Star for top cells
    if(c.top){{
      var cx=(lon2x(verts[0][0])+lon2x(verts[3][0]))/2;
      var cy=(lat2y(verts[0][1])+lat2y(verts[3][1]))/2;
      // use centroid of first & opposite vertex as rough centre
      var xs=verts.map(function(v){{return lon2x(v[0]);}});
      var ys=verts.map(function(v){{return lat2y(v[1]);}});
      cx=xs.reduce(function(a,b){{return a+b;}})/xs.length;
      cy=ys.reduce(function(a,b){{return a+b;}})/ys.length;
      ctx.font = Math.max(8, view.scale*0.18)+'px serif';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillStyle='#fff';
      ctx.fillText('★',cx+0.5,cy+0.5);
      ctx.fillStyle='#1a2a3a';
      ctx.fillText('★',cx,cy);
    }}
  }});
}}

// ── Hit-test: is point (px,py) inside polygon? ───────────────────────────────
function ptInPoly(px,py,verts){{
  var inside=false, n=verts.length, j=n-1;
  for(var i=0;i<n;i++){{
    var xi=lon2x(verts[i][0]), yi=lat2y(verts[i][1]);
    var xj=lon2x(verts[j][0]), yj=lat2y(verts[j][1]);
    if(((yi>py)!==(yj>py))&&(px<(xj-xi)*(py-yi)/(yj-yi)+xi))
      inside=!inside;
    j=i;
  }}
  return inside;
}}

function cellAtPx(px,py){{
  for(var i=CELLS.length-1;i>=0;i--)
    if(ptInPoly(px,py,CELLS[i].v)) return i;
  return -1;
}}

// ── Popup ────────────────────────────────────────────────────────────────────
var popup = document.getElementById('popup');
document.getElementById('popclose').addEventListener('click',function(){{
  popup.style.display='none';
}});

function showPopup(idx, px, py){{
  var c = CELLS[idx];
  var sc = c.s;
  popup.className = sc>0.6?'':'mid';
  if(sc<0.4) popup.className='bad';
  document.getElementById('pop-title').textContent =
    (c.rank ? '#'+c.rank+' — ' : '') + c.id;
  var col = sc>0.6?'#27ae60':sc>0.4?'#e67e22':'#e74c3c';
  document.getElementById('pop-body').innerHTML =
    '<b>DIVIPOLA:</b> '+c.dp+'<br>'+
    '<b>Municipio, Departamento:</b> '+c.mu+', '+c.de+'<hr>'+
    '<b>Suitability:</b> <span style="color:'+col+';font-weight:700">'+sc.toFixed(3)+'</span><hr>'+
    '<b>Wind speed:</b> '+c.ws+' m/s<br>'+
    '<b>Slope:</b> '+c.sl+'°<br>'+
    '<b>Dist. to grid:</b> '+c.dg+' km<br>'+
    '<b>Dist. to roads:</b> '+c.dr+' km<br>'+
    '<b>Land use score:</b> '+c.lu+'<br>'+
    '<b>Protected area:</b> '+c.pa+'<br>'+
    '<b>Conflict risk:</b> '+c.cr;

  // Position popup near click, keep within viewport
  var W=cvs.width, H=cvs.height;
  var pw=244, ph=220;
  var left=px+12, top=py-ph/2;
  if(left+pw>W-10) left=px-pw-12;
  if(top<10) top=10;
  if(top+ph>H-10) top=H-ph-10;
  popup.style.left=left+'px';
  popup.style.top=top+'px';
  popup.style.display='block';
}}

// ── Interaction ───────────────────────────────────────────────────────────────
// Pan
var drag={{active:false,sx:0,sy:0,sox:0,soy:0}};
cvs.addEventListener('mousedown',function(e){{
  drag={{active:true,sx:e.clientX,sy:e.clientY,sox:view.ox,soy:view.oy}};
  cvs.classList.add('dragging');
}});
window.addEventListener('mousemove',function(e){{
  if(drag.active){{
    view.ox=drag.sox+(e.clientX-drag.sx);
    view.oy=drag.soy+(e.clientY-drag.sy);
    draw();
  }} else {{
    var idx=cellAtPx(e.clientX,e.clientY);
    if(idx!==hovIdx){{ hovIdx=idx; draw(); }}
  }}
}});
window.addEventListener('mouseup',function(e){{
  if(drag.active && Math.abs(e.clientX-drag.sx)<4 && Math.abs(e.clientY-drag.sy)<4){{
    var idx=cellAtPx(e.clientX,e.clientY);
    if(idx>=0) showPopup(idx,e.clientX,e.clientY);
    else popup.style.display='none';
  }}
  drag.active=false;
  cvs.classList.remove('dragging');
}});

// Zoom
function zoomAt(px,py,factor){{
  var lon=x2lon(px), lat=y2lat(py);
  view.scale *= factor;
  view.ox = px - lon*view.scale;
  view.oy = py + lat*view.scale;
  draw();
}}
cvs.addEventListener('wheel',function(e){{
  e.preventDefault();
  zoomAt(e.clientX,e.clientY, e.deltaY<0?1.15:1/1.15);
}},{{passive:false}});
document.getElementById('zin').addEventListener('click',function(){{
  zoomAt(cvs.width/2,cvs.height/2,1.3);
}});
document.getElementById('zout').addEventListener('click',function(){{
  zoomAt(cvs.width/2,cvs.height/2,1/1.3);
}});

// Touch pan/zoom
var touches={{}};
cvs.addEventListener('touchstart',function(e){{
  e.preventDefault();
  Array.from(e.changedTouches).forEach(function(t){{
    touches[t.identifier]={{x:t.clientX,y:t.clientY}};
  }});
  drag={{active:Object.keys(touches).length===1,
    sx:e.touches[0].clientX,sy:e.touches[0].clientY,
    sox:view.ox,soy:view.oy}};
}},{{passive:false}});
cvs.addEventListener('touchmove',function(e){{
  e.preventDefault();
  if(Object.keys(touches).length===1){{
    view.ox=drag.sox+(e.touches[0].clientX-drag.sx);
    view.oy=drag.soy+(e.touches[0].clientY-drag.sy);
    draw();
  }}
}},{{passive:false}});
cvs.addEventListener('touchend',function(e){{
  Array.from(e.changedTouches).forEach(function(t){{delete touches[t.identifier];}});
}});

// ── Init ─────────────────────────────────────────────────────────────────────
resize();
fitView();
draw();
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"[Map] Self-contained Canvas map saved → {output_path}  "
          f"({os.path.getsize(output_path) / 1024:.0f} KB)")



# ---------------------------------------------------------------------------
# Score distribution plot
# ---------------------------------------------------------------------------

def plot_score_distribution(
    df: pd.DataFrame,
    output_path: str,
    score_column: str = "suitability_score",
) -> None:
    """
    Histogram of suitability score distribution across all hexagons.

    Parameters
    ----------
    df          : scored DataFrame
    output_path : save path for PNG
    score_column: score column name
    """
    scores = df[score_column].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(scores, bins=40, edgecolor="white")

    # Colour bars by score
    cmap = cm.get_cmap("RdYlGn")
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    for patch, centre in zip(patches, bin_centres):
        patch.set_facecolor(cmap(centre))

    ax.axvline(scores.mean(), color="#2c3e50", linewidth=1.5,
               linestyle="--", label=f"Mean = {scores.mean():.3f}")
    ax.axvline(np.percentile(scores, 75), color="#8e44ad", linewidth=1.2,
               linestyle=":", label=f"75th pct = {np.percentile(scores, 75):.3f}")

    ax.set_xlabel("Suitability Score", fontsize=10)
    ax.set_ylabel("Number of Hexagons", fontsize=10)
    ax.set_title("Distribution of Wind Farm Suitability Scores — Colombia",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Score distribution saved → {output_path}")


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
    Pearson correlation heatmap between normalised features and suitability score.

    Parameters
    ----------
    df            : scored DataFrame with normalised feature columns
    norm_features : list of normalised feature column names
    output_path   : save path for PNG
    score_column  : score column to include in the heatmap
    """
    cols = norm_features + ([score_column] if score_column in df.columns else [])
    corr = df[cols].corr()

    # Short labels
    labels = [c.replace("_norm", "").replace("_", "\n").title() for c in cols]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            colour = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=colour)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="Pearson r")
    ax.set_title("Feature Correlation Matrix (normalised criteria + suitability score)",
                 fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Correlation heatmap saved → {output_path}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_h3_grid import generate_colombia_hex_grid
    from feature_engineering import engineer_features
    from normalization import normalise_features, get_norm_feature_names
    from random_forest_weights import get_rf_weights
    from mcda_model import compute_wlc_scores, rank_locations

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _GEOJSON = os.path.join(_HERE, "..", "data", "colombia_boundary.geojson")
    _OUT  = os.path.join(_HERE, "..", "outputs")
    os.makedirs(_OUT, exist_ok=True)

    grid      = generate_colombia_hex_grid(_GEOJSON, resolution=5)
    feats     = engineer_features(grid)
    norm_df   = normalise_features(feats)
    norm_cols = get_norm_feature_names()
    model, weights, labels = get_rf_weights(norm_df, norm_cols)
    scored_df = compute_wlc_scores(norm_df, weights, norm_cols)
    ranked_df = rank_locations(scored_df)

    plot_score_distribution(scored_df, os.path.join(_OUT, "score_distribution.png"))
    create_interactive_map(ranked_df, os.path.join(_OUT, "map_interactive.html"))