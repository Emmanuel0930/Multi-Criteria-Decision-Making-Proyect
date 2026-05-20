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

const API_BASE    = window.location.origin;      // Backend same-origin (/app, /models, /run-model)
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
let currentHex    = null;         // hex seleccionado para acciones de grupo
let currentModel  = null;         // datos del modelo cargado { lod0, lod1, lod3, ... }
let muniTable     = [];
let deptTable     = [];
let diviTable     = [];
let groups        = {};
let groupOrder    = [];
let activeGroup   = "";

const GROUP_COLORS = ["#38bdf8", "#a78bfa", "#f59e0b", "#34d399", "#fb7185", "#f97316", "#22d3ee"];

const CRITERIA_MODEL = {
  wind_speed: {
    title: "Velocidad del viento",
    category: "Meteorológicos",
    weight: "25%",
    description: "Mide el potencial energético del recurso eólico. A mayor velocidad media del viento, mayor probabilidad de obtener una producción rentable y estable.",
  },
  air_density: {
    title: "Densidad del aire",
    category: "Meteorológicos",
    weight: "10%",
    description: "Ajusta el rendimiento esperado de las turbinas según la densidad atmosférica. Valores más altos suelen aumentar la energía extraíble en el mismo punto.",
  },
  turbulence_index: {
    title: "Índice de turbulencia",
    category: "Meteorológicos",
    weight: "10%",
    description: "Representa la variabilidad del flujo de aire. Una turbulencia alta incrementa cargas mecánicas y reduce la vida útil de los equipos, por lo que impacta negativamente la idoneidad.",
  },
  grid_proximity: {
    title: "Cercanía a la red eléctrica",
    category: "Técnicos y suelo",
    weight: "15%",
    description: "Evalúa cuánto cuesta conectar el proyecto al sistema eléctrico. Menores distancias reducen infraestructura adicional y disminuyen el costo total del parque.",
  },
  max_slope: {
    title: "Pendiente máxima",
    category: "Técnicos y suelo",
    weight: "10%",
    description: "Cuantifica la inclinación del terreno. Pendientes elevadas complican la construcción, el montaje y el mantenimiento, además de aumentar el riesgo geotécnico.",
  },
  road_accessibility: {
    title: "Accesibilidad vial",
    category: "Técnicos y suelo",
    weight: "5%",
    description: "Refleja la facilidad de acceso por carretera para transportar componentes, maquinaria y personal. Una mejor conectividad vial acelera la ejecución y reduce costos logísticos.",
  },
  bearing_capacity: {
    title: "Capacidad portante",
    category: "Técnicos y suelo",
    weight: "5%",
    description: "Indica la resistencia del suelo para soportar cimentaciones y cargas estructurales. Una mayor capacidad portante mejora la viabilidad técnica del emplazamiento.",
  },
  protected_areas: {
    title: "Zonas protegidas",
    category: "Socio-ambientales",
    weight: "Excluido",
    description: "Se usa como restricción ambiental. La presencia de áreas protegidas limita o excluye la instalación de aerogeneradores para minimizar impactos ecológicos y regulatorios.",
  },
  conflict_risk: {
    title: "Riesgo de conflicto",
    category: "Socio-ambientales",
    weight: "10%",
    description: "Captura la sensibilidad social y territorial del área. Un mayor riesgo de conflicto puede retrasar permisos, aumentar oposición local y comprometer la continuidad del proyecto.",
  },
  land_use: {
    title: "Uso del suelo",
    category: "Socio-ambientales",
    weight: "10%",
    description: "Evalúa la compatibilidad entre el uso actual del terreno y la implantación eólica. Suelos con usos más compatibles facilitan la aceptación y reducen conflictos de ocupación.",
  },
};


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

function toggleCriterionDescription(node, criterion) {
  if (!node || !criterion) return;

  const descId = `criterion-desc-${node.dataset.criterionId}`;
  let desc = document.getElementById(descId);

  if (!desc) {
    desc = document.createElement("div");
    desc.id = descId;
    desc.className = "criterion-inline-desc";
    desc.textContent = `${criterion.description}`;
    node.insertAdjacentElement("afterend", desc);
  }

  const isOpen = desc.classList.contains("show");
  desc.classList.toggle("show", !isOpen);
  node.classList.toggle("is-open", !isOpen);
  node.setAttribute("aria-expanded", (!isOpen).toString());
}

function bindCriterionDescriptions() {
  const nodes = document.querySelectorAll(".criterion[data-criterion-id]");
  nodes.forEach(node => {
    const criterionId = node.dataset.criterionId;
    const criterion = CRITERIA_MODEL[criterionId];
    if (!criterion) return;

    node.setAttribute("title", `Mostrar descripción: ${criterion.title}`);
    node.setAttribute("aria-label", `${criterion.title}. Presiona para expandir o contraer su descripción.`);
    node.setAttribute("aria-expanded", "false");

    node.addEventListener("click", () => toggleCriterionDescription(node, criterion));
    node.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggleCriterionDescription(node, criterion);
      }
    });
  });
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

function getGroupColor(name) {
  const idx = groupOrder.indexOf(name);
  if (idx < 0) return "#64748b";
  return GROUP_COLORS[idx % GROUP_COLORS.length];
}

function findGroupForHex(hexId) {
  for (const name of groupOrder) {
    const items = groups[name].items;
    for (const item of items) {
      if (item.hexId === hexId) return name;
    }
  }
  return null;
}

function renderGroupSelector() {
  const sel = document.getElementById("grp-active");
  if (!sel) return;

  if (activeGroup && !groups[activeGroup]) activeGroup = "";

  let html = "";
  if (!groupOrder.length) {
    html = '<option value="" selected>Sin grupos</option>';
    sel.innerHTML = html;
    return;
  }

  for (const name of groupOrder) {
    const selected = name === activeGroup ? " selected" : "";
    html += `<option value="${name}"${selected}>${name} (${groups[name].items.length})</option>`;
  }

  sel.innerHTML = html;
  sel.value = activeGroup;
  if (!activeGroup) sel.selectedIndex = -1;
}

function renderGroupCompareTable() {
  const table = document.getElementById("grp-table");
  const body = document.getElementById("grp-body");
  const empty = document.getElementById("grp-empty");
  if (!table || !body || !empty) return;

  const rows = [];
  for (const name of groupOrder) {
    const items = groups[name].items;
    if (!items.length) continue;
    const sum = items.reduce((acc, it) => acc + it.score, 0);
    const best = items.reduce((mx, it) => Math.max(mx, it.score), items[0].score);
    rows.push({
      name,
      count: items.length,
      avg: sum / items.length,
      best,
      color: getGroupColor(name),
    });
  }

  if (!rows.length) {
    table.style.display = "none";
    empty.style.display = "block";
    body.innerHTML = "";
    renderGroupSelector();
    return;
  }

  const winner = rows.reduce((mx, r) => Math.max(mx, r.avg), rows[0].avg);

  table.style.display = "table";
  empty.style.display = "none";
  body.innerHTML = rows.map(r => {
    const state = Math.abs(r.avg - winner) < 1e-9 ? '<span class="cmp-best">MEJOR</span>' : "-";
    return `<tr>
      <td><span class="cmp-group-chip" style="background:${r.color}"></span>${r.name}</td>
      <td>${r.count}</td>
      <td class="cmp-score">${r.avg.toFixed(4)}</td>
      <td>${r.best.toFixed(4)}</td>
      <td>${state}</td>
    </tr>`;
  }).join("");

  renderGroupSelector();
}

function createGroup() {
  const input = document.getElementById("grp-name");
  if (!input) return;
  const name = (input.value || "").trim();
  if (!name) return;

  if (!groups[name]) {
    groups[name] = { name, items: [] };
    groupOrder.push(name);
  }
  activeGroup = name;
  input.value = "";
  renderGroupCompareTable();
  renderViewport();
  setStatus("fresh", `Grupo activo: ${name}`);
}

function leaveActiveGroup() {
  activeGroup = "";
  renderGroupSelector();
  renderViewport();
}

function addCurrentToActiveGroup() {
  if (!currentHex || !activeGroup || !groups[activeGroup]) return;
  const items = groups[activeGroup].items;
  if (items.some(it => it.hexId === currentHex.hexId)) return;
  items.push(currentHex);
  renderGroupCompareTable();
  renderViewport();
}

function removeCurrentFromActiveGroup() {
  if (!currentHex || !activeGroup || !groups[activeGroup]) return;
  groups[activeGroup].items = groups[activeGroup].items.filter(it => it.hexId !== currentHex.hexId);
  renderGroupCompareTable();
  renderViewport();
}

function clearActiveGroup() {
  if (!activeGroup || !groups[activeGroup]) return;
  groups[activeGroup].items = [];
  renderGroupCompareTable();
  renderViewport();
}

function deleteActiveGroup() {
  if (!activeGroup || !groups[activeGroup]) return;
  delete groups[activeGroup];
  groupOrder = groupOrder.filter(name => name !== activeGroup);
  activeGroup = "";
  renderGroupCompareTable();
  renderViewport();
}


// ─────────────────────────────────────────────────────────────────────────────
// Panel derecho
// ─────────────────────────────────────────────────────────────────────────────

function showHexAnalysis(score, ws, slope, dg, dr, lu, pa, cr, rank, mi, di, dpi, lat, lon) {
  document.getElementById("rp-placeholder").style.display = "none";
  document.getElementById("rp-content").style.display     = "block";
  document.getElementById("rp-close").style.display       = "block";

  const hexId = buildHexId(lat, lon);
  currentHex = {
    hexId,
    score,
    muni: muniTable[mi] || "—",
    dept: deptTable[di] || "—",
    rank,
    lat,
    lon,
  };
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

  const autoAdd = document.getElementById("grp-auto-add");
  if (autoAdd && autoAdd.checked) addCurrentToActiveGroup();
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
  currentHex = null;
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

    const hexId = buildHexId(lat, lon);
    const groupName = findGroupForHex(hexId);
    const groupColor = groupName ? getGroupColor(groupName) : null;

    const baseColor  = groupColor || "rgba(255,255,255,0.15)";
    const baseWeight = groupColor ? 1.8 : 0.5;
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

    const hexId = buildHexId(clat, clon);
    const groupName = findGroupForHex(hexId);
    const groupColor = groupName ? getGroupColor(groupName) : null;

    const baseColor  = groupColor || "rgba(255,255,255,0.08)";
    const baseWeight = groupColor ? 2.0 : 0.6;
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

    groups = {};
    groupOrder = [];
    activeGroup = "";
    renderGroupCompareTable();

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

const grpCreate = document.getElementById("grp-create");
if (grpCreate) grpCreate.addEventListener("click", createGroup);

const grpLeave = document.getElementById("grp-leave");
if (grpLeave) grpLeave.addEventListener("click", leaveActiveGroup);

const grpAdd = document.getElementById("grp-add-current");
if (grpAdd) grpAdd.addEventListener("click", addCurrentToActiveGroup);

const grpRemove = document.getElementById("grp-remove-current");
if (grpRemove) grpRemove.addEventListener("click", removeCurrentFromActiveGroup);

const grpClear = document.getElementById("grp-clear");
if (grpClear) grpClear.addEventListener("click", clearActiveGroup);

const grpDelete = document.getElementById("grp-delete");
if (grpDelete) grpDelete.addEventListener("click", deleteActiveGroup);

const grpActive = document.getElementById("grp-active");
if (grpActive) {
  grpActive.addEventListener("change", function () {
    activeGroup = this.value || "";
    renderViewport();
  });
}

const grpName = document.getElementById("grp-name");
if (grpName) {
  grpName.addEventListener("keydown", function (e) {
    if (e.key === "Enter") createGroup();
  });
}


// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

window.addEventListener("load", () => {
  initMap();
  bindCriterionDescriptions();
  renderGroupCompareTable();
  fetchModels();     // decorar opciones del select con ✓ si ya tienen caché
});
