/**
 * app.js
 * ======
 * Lógica del frontend MCDA Eólico.
 *
 * Responsabilidades:
 *  1. Comunicación con el backend FastAPI (/models, /run-model)
 *  2. Renderizado multi-LOD con Leaflet (LOD0/LOD1/LOD3)
 *  3. Panel derecho: análisis multicriterio del hexágono seleccionado
 *  4. UI: loader, status chip, leyenda, HUD de zoom
 */

"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// Configuración
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE    = "http://localhost:8000";    // URL del backend FastAPI
const CENTRE_LAT  = 4.711;
const CENTRE_LON  = -74.072;
const INIT_ZOOM   = 6;

// Paleta RdYlGn idéntica a la de visualization.py
const PALETTE = [
  "#a50026","#d73027","#f46d43","#fdae61","#fee08b",
  "#ffffbf","#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"
];

// Descriptores de zoom para el HUD
const ZOOM_DESCS = {
  5: "vista país", 6: "vista región", 7: "vista regional",
  8: "vista departamento", 9: "vista local", 10: "detalle completo"
};


// ─────────────────────────────────────────────────────────────────────────────
// Estado global
// ─────────────────────────────────────────────────────────────────────────────

let map           = null;         // instancia Leaflet
let canvasRenderer = null;
let activeLayers  = [];           // capas activas en el mapa
let renderTimer   = null;         // debounce timer
let selectedLayer = null;         // capa actualmente seleccionada
let currentModel  = null;         // datos del modelo cargado { lod0, lod1, lod3, ... }
let muniTable     = [];
let deptTable     = [];
let diviTable     = [];


// ─────────────────────────────────────────────────────────────────────────────
// Utilidades de color
// ─────────────────────────────────────────────────────────────────────────────

function scoreToColor(s) {
  return PALETTE[Math.min(10, Math.floor(Math.max(0, s) * 10.999))];
}

function buildLegendGradient() {
  const bar = document.getElementById("leg-gradient");
  if (!bar) return;
  bar.innerHTML = PALETTE.map(c =>
    `<span style="background:${c};flex:1;display:inline-block;height:12px;"></span>`
  ).join("");
}


// ─────────────────────────────────────────────────────────────────────────────
// UI helpers
// ─────────────────────────────────────────────────────────────────────────────

function setStatus(type, text) {
  const chip = document.getElementById("status-chip");
  chip.className = `status-chip ${type}`;
  chip.textContent = text;
  chip.style.display = type ? "inline-block" : "none";
}

function setLoading(on) {
  document.getElementById("load-btn").disabled = on;
  document.getElementById("model-select").disabled = on;
  if (on) setStatus("loading", "⏳ Cargando…");
}

function showPlaceholder(show) {
  document.getElementById("map-placeholder").style.display = show ? "flex" : "none";
}

function showLegend(show) {
  document.getElementById("legend").style.display  = show ? "block" : "none";
  document.getElementById("lod-hud").style.display = show ? "block" : "none";
}

function buildHexId(lat, lon) {
  return "IDX-H3-" + Math.abs(Math.round(lat * 10000 + lon * 1000))
    .toString().padStart(5, "0");
}

function updateLodHud() {
  if (!map) return;
  const z = map.getZoom();
  const lodNum = z <= 5 ? 0 : z <= 7 ? 1 : 3;
  const desc = ZOOM_DESCS[Math.min(z, 10)] || "detalle completo";
  document.getElementById("lod-hud").textContent =
    `LOD${lodNum} · zoom ${z} · ${desc}`;
}


// ─────────────────────────────────────────────────────────────────────────────
// Panel derecho
// ─────────────────────────────────────────────────────────────────────────────

function showHexAnalysis(score, ws, slope, dg, dr, lu, pa, cr, rank, mi, di, dpi, lat, lon) {
  document.getElementById("rp-placeholder").style.display = "none";
  document.getElementById("rp-content").style.display     = "block";
  document.getElementById("rp-close").style.display       = "block";

  const hexId = buildHexId(lat, lon);
  document.getElementById("rp-hex-id").textContent = hexId;

  document.getElementById("rp-score-value").textContent = (score * 10).toFixed(1);
  document.getElementById("rp-score-sub").textContent   =
    rank > 0 ? `Ranking #${rank}` : "Escala 0 (no idóneo) → 10 (idóneo)";

  document.getElementById("rp-detail-grid").innerHTML = `
    <div class="detail-cell"><div class="dc-label">Municipio</div>
      <div class="dc-value" style="font-size:11px;">${muniTable[mi] || "—"}</div></div>
    <div class="detail-cell"><div class="dc-label">Departamento</div>
      <div class="dc-value" style="font-size:11px;">${deptTable[di] || "—"}</div></div>
    <div class="detail-cell"><div class="dc-label">Divipola</div>
      <div class="dc-value" style="font-size:11px;">${diviTable[dpi] || "—"}</div></div>
    <div class="detail-cell"><div class="dc-label">Viento</div>
      <div class="dc-value">${ws.toFixed(1)} m/s</div></div>
    <div class="detail-cell"><div class="dc-label">Pendiente</div>
      <div class="dc-value">${slope.toFixed(1)}°</div></div>
    <div class="detail-cell"><div class="dc-label">Dist. Red</div>
      <div class="dc-value">${dg.toFixed(0)} km</div></div>
    <div class="detail-cell"><div class="dc-label">Dist. Vías</div>
      <div class="dc-value">${dr.toFixed(0)} km</div></div>
    <div class="detail-cell"><div class="dc-label">Área Proteg.</div>
      <div class="dc-value">${(pa * 100).toFixed(0)}%</div></div>
    <div class="detail-cell"><div class="dc-label">Riesgo</div>
      <div class="dc-value">${cr.toFixed(3)}</div></div>
  `;
}

function hideHexAnalysis() {
  document.getElementById("rp-placeholder").style.display = "flex";
  document.getElementById("rp-content").style.display     = "none";
  document.getElementById("rp-close").style.display       = "none";

  if (selectedLayer && selectedLayer.setStyle) {
    selectedLayer.setStyle({
      color:  selectedLayer._baseColor  || "rgba(255,255,255,0.08)",
      weight: selectedLayer._baseWeight || 0.6,
    });
  }
  selectedLayer = null;
}

document.getElementById("rp-close").addEventListener("click", hideHexAnalysis);


// ─────────────────────────────────────────────────────────────────────────────
// LOD renderers
// ─────────────────────────────────────────────────────────────────────────────

function renderLOD0(data, bounds) {
  const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  let n = 0;
  for (const row of data) {
    const [lat, lon, score] = row;
    if (lat < sw.lat || lat > ne.lat || lon < sw.lng || lon > ne.lng) continue;
    const cm = L.circleMarker([lat, lon], {
      renderer: canvasRenderer, radius: 3,
      fillColor: scoreToColor(score), fillOpacity: 0.35 + 0.55 * score,
      color: "none", weight: 0, interactive: false,
    });
    cm.addTo(map);
    activeLayers.push(cm);
    n++;
  }
  return n;
}

function renderLOD1(data, bounds, radius) {
  const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  let n = 0;
  for (const row of data) {
    const [lat, lon, score, rank, mi, di, dpi, ws, slope, dg, dr, lu, pa, cr] = row;
    if (lat < sw.lat || lat > ne.lat || lon < sw.lng || lon > ne.lng) continue;

    const baseColor  = "rgba(255,255,255,0.15)";
    const baseWeight = 0.5;
    const cm = L.circleMarker([lat, lon], {
      renderer: canvasRenderer, radius,
      fillColor: scoreToColor(score), fillOpacity: 0.35 + 0.55 * score,
      color: baseColor, weight: baseWeight, interactive: true,
    });
    cm._baseColor  = baseColor;
    cm._baseWeight = baseWeight;

    cm.on("click", () => {
      if (selectedLayer && selectedLayer !== cm && selectedLayer.setStyle) {
        selectedLayer.setStyle({ color: selectedLayer._baseColor, weight: selectedLayer._baseWeight });
      }
      cm.setStyle({ color: "#60a5fa", weight: 2 });
      selectedLayer = cm;
      showHexAnalysis(score, ws, slope, dg, dr, lu, pa, cr, rank, mi, di, dpi, lat, lon);
    });

    cm.addTo(map);
    activeLayers.push(cm);
    n++;
  }
  return n;
}

function renderLOD3(data, bounds) {
  const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  let n = 0;
  for (const row of data) {
    const [verts, score, ws, slope, dg, dr, lu, pa, cr, rank, mi, di, dpi] = row;
    const clat = (verts[0][0] + verts[3][0]) / 2;
    const clon = (verts[1][1] + verts[4][1]) / 2;
    if (clat < sw.lat || clat > ne.lat || clon < sw.lng || clon > ne.lng) continue;

    const baseColor  = "rgba(255,255,255,0.08)";
    const baseWeight = 0.6;
    const poly = L.polygon(verts, {
      renderer: canvasRenderer,
      fillColor: scoreToColor(score), fillOpacity: 0.2 + 0.6 * score,
      color: baseColor, weight: baseWeight, interactive: true,
    });
    poly._baseColor  = baseColor;
    poly._baseWeight = baseWeight;

    poly.on("click", () => {
      if (selectedLayer && selectedLayer !== poly && selectedLayer.setStyle) {
        selectedLayer.setStyle({ color: selectedLayer._baseColor, weight: selectedLayer._baseWeight });
      }
      poly.setStyle({ color: "#60a5fa", weight: 2.5 });
      selectedLayer = poly;
      showHexAnalysis(score, ws, slope, dg, dr, lu, pa, cr, rank, mi, di, dpi, clat, clon);
    });

    poly.addTo(map);
    activeLayers.push(poly);
    n++;
  }
  return n;
}


// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher LOD principal (con debounce 120 ms)
// ─────────────────────────────────────────────────────────────────────────────

function renderViewport() {
  if (!currentModel) return;
  if (renderTimer) clearTimeout(renderTimer);

  renderTimer = setTimeout(() => {
    // Limpiar capas anteriores
    activeLayers.forEach(l => map.removeLayer(l));
    activeLayers  = [];
    selectedLayer = null;

    const zoom   = map.getZoom();
    const bounds = map.getBounds().pad(0.05);
    let   n      = 0;

    if      (zoom <= 5) n = renderLOD0(currentModel.lod0, bounds);
    else if (zoom <= 7) n = renderLOD1(currentModel.lod1, bounds, zoom <= 6 ? 5 : 7);
    else                n = renderLOD3(currentModel.lod3, bounds);

    const lodNum = zoom <= 5 ? 0 : zoom <= 7 ? 1 : 3;
    console.log(`[LOD${lodNum}] ${n} objetos | zoom=${zoom}`);
    updateLodHud();
  }, 120);
}


// ─────────────────────────────────────────────────────────────────────────────
// Inicialización del mapa
// ─────────────────────────────────────────────────────────────────────────────

function initMap() {
  canvasRenderer = L.canvas({ padding: 0.5 });

  map = L.map("map", {
    center:     [CENTRE_LAT, CENTRE_LON],
    zoom:       INIT_ZOOM,
    preferCanvas: true,
  });

    // ───── Capas base ─────
  const darkMap = L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    { attribution: '© CartoDB', maxZoom: 19 }
  );

  const lightMap = L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    { attribution: '© CartoDB', maxZoom: 19 }
  );

  const streetsMap = L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { attribution: '© OpenStreetMap contributors', maxZoom: 19 }
  );

  // Añadir capa inicial
  darkMap.addTo(map);

  // Control de capas (menú derecha)
  const baseMaps = {
    "Oscuro": darkMap,
    "Claro": lightMap,
    "Calles": streetsMap
  };

  L.control.layers(baseMaps, null, {
    position: "topright",
    collapsed: false // opcional: que esté abierto por defecto
  }).addTo(map);
  map.on("moveend zoomend", renderViewport);

  buildLegendGradient();
}


// ─────────────────────────────────────────────────────────────────────────────
// Comunicación con la API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Consulta /models al inicio y actualiza el <select> con el estado de caché.
 * Si un modelo ya tiene caché, muestra "✓" en el label.
 */
async function fetchModels() {
  try {
    const res    = await fetch(`${API_BASE}/models`);
    if (!res.ok) return;
    const models = await res.json();
    const sel    = document.getElementById("model-select");

    models.forEach(m => {
      const opt = sel.querySelector(`option[value="${m.id}"]`);
      if (opt && m.cached) opt.textContent += " ✓";
    });
  } catch (e) {
    console.warn("[API] /models no disponible:", e.message);
  }
}

/**
 * Llama a POST /run-model con el modelo seleccionado.
 * Maneja el estado de carga y errores.
 */
async function loadModel() {
  const modelId = document.getElementById("model-select").value;
  if (!modelId) return;

  setLoading(true);
  hideHexAnalysis();
  showPlaceholder(false);
  showLegend(false);

  try {
    const res = await fetch(`${API_BASE}/run-model`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ model: modelId }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    // Guardar datos en estado global
    currentModel = {
      lod0: data.lod0,
      lod1: data.lod1,
      lod3: data.lod3,
    };

    // Las tablas de municipios/departamentos vienen dentro de params si el
    // pipeline las incluye; si no, se usan arrays vacíos.
    muniTable = data.params?.muni_table || [];
    deptTable = data.params?.dept_table || [];
    diviTable = data.params?.divi_table || [];

    showLegend(true);
    renderViewport();

    const fromCache = data.from_cache;
    setStatus(
      fromCache ? "cache" : "fresh",
      fromCache ? "✓ Desde caché" : "✓ Calculado y guardado"
    );

    // Actualizar label del select con ✓ si fue calculado en vivo
    if (!fromCache) {
      const opt = document.getElementById("model-select")
        .querySelector(`option[value="${modelId}"]`);
      if (opt && !opt.textContent.includes("✓")) opt.textContent += " ✓";
    }

  } catch (err) {
    console.error("[API] Error:", err);
    setStatus("error", `✗ Error: ${err.message}`);
    showPlaceholder(true);
  } finally {
    setLoading(false);
    document.getElementById("model-select").disabled = false;
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// Event listeners de UI
// ─────────────────────────────────────────────────────────────────────────────

document.getElementById("model-select").addEventListener("change", function () {
  document.getElementById("load-btn").disabled = !this.value;
  setStatus("", "");
});

document.getElementById("load-btn").addEventListener("click", loadModel);


// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

window.addEventListener("load", () => {
  initMap();
  fetchModels();     // decorar opciones del select con ✓ si ya tienen caché
});
