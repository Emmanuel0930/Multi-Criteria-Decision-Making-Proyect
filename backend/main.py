"""
backend/main.py
===============
API FastAPI para el sistema MCDA de aptitud eólica.

Endpoints
---------
GET  /models       → lista de modelos disponibles
POST /run-model    → carga o ejecuta un modelo y retorna LODs
GET  /health       → health check

Uso
---
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import importlib
import sys
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Asegurar que el raíz del proyecto esté en el path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Añadir la carpeta `src` al path para que los módulos del proyecto sean importables
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

import cache_manager  # tu módulo cache_manager.py

# Importar helpers de visualización para construir LODs
try:
    from visualization import _build_lod_payload
except ImportError:
    _build_lod_payload = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: convertir registros scored/ranked → LODs
# ─────────────────────────────────────────────────────────────────────────────

def _records_to_lods(scored_records: list, ranked_records: list, score_column: str = "suitability_score") -> tuple:
    """
    Convierte listas de registros (scored y ranked) a LODs (lod0, lod1, lod3).
    
    Si _build_lod_payload no está disponible, retorna LODs simples.
    Retorna: (lod0, lod1, lod3, params_dict_con_tablas)
    """
    if not scored_records:
        return [], [], [], {}
    
    try:
        import pandas as pd
        scored_df = pd.DataFrame(scored_records)
        
        if _build_lod_payload is not None:
            # Usar la lógica completa de multi-LOD
            lod0_js, lod1_js, lod3_js, muni_table_js, dept_table_js, divi_table_js = _build_lod_payload(scored_df, score_column)
            import json
            lod0 = json.loads(lod0_js) if lod0_js else []
            lod1 = json.loads(lod1_js) if lod1_js else []
            lod3 = json.loads(lod3_js) if lod3_js else []
            
            # Decodificar las tablas de lookup
            muni_table = json.loads(muni_table_js) if muni_table_js else {}
            dept_table = json.loads(dept_table_js) if dept_table_js else {}
            divi_table = json.loads(divi_table_js) if divi_table_js else {}
            
            params = {
                "muni_table": muni_table,
                "dept_table": dept_table,
                "divi_table": divi_table,
            }
            return lod0, lod1, lod3, params
        else:
            # Fallback simple: LOD0 = primeros 100 puntos, LOD1 = scored, LOD3 = ranked
            lod0 = [{"lat": float(r["lat"]), "lon": float(r["lon"]), "score": float(r.get(score_column, 0))} 
                    for r in scored_records[:100]]
            lod1 = scored_records[:1000]
            lod3 = ranked_records or scored_records
            return lod0, lod1, lod3, {}
    except Exception as e:
        print(f"[API] Aviso: error al construir LODs: {e}")
        # Fallback básico
        lod1 = scored_records or []
        return [], lod1, ranked_records or lod1, {}


# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

# Ruta donde están los JSON cacheados
cache_manager.CACHE_DIR = ROOT / "outputs"

# Modelos disponibles: clave → metadatos
MODEL_REGISTRY: dict[str, dict] = {
    "ahp": {
        "label":       "AHP + WLC",
        "description": "Analytic Hierarchy Process con Weighted Linear Combination",
        "module":      "mcda_model",          # módulo Python a importar
        "function":    "run_ahp_pipeline",    # función a llamar si no hay caché
    },
    "wlc": {
        "label":       "WLC (Random Forest)",
        "description": "Weighted Linear Combination con pesos de Random Forest",
        "module":      "mcda_model",
        "function":    "run_wlc_pipeline",
    },
    "bwm": {
        "label":       "BWM + PROMETHEE II",
        "description": "Best Worst Method con ranking PROMETHEE II",
        "module":      "bwm_promethee_model",
        "function":    "run_bwm_pipeline",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# App FastAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MCDA Eólico API",
    description="Backend para evaluación de aptitud eólica en Colombia con hexágonos H3",
    version="1.0.0",
)


@app.middleware("http")
async def disable_frontend_cache(request: Request, call_next):
    """Evita caché agresiva del navegador en el frontend durante desarrollo."""
    response = await call_next(request)
    if request.url.path.startswith("/app"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# CORS: permite que el frontend (cualquier origen en desarrollo) consuma la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # En producción, reemplazar con el dominio exacto
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Servir el frontend como archivos estáticos en /app
FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# Raíz: redirige al frontend si existe, o muestra enlaces útiles
@app.get("/", include_in_schema=False)
def index():
    if FRONTEND_DIR.exists():
        return RedirectResponse(url="/app")
    return {
        "service": "MCDA Eólico API",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "run-model (POST)": "/run-model",
            "docs": "/docs",
        },
    }


# Favicon: sirve el favicon del frontend si existe, sino devuelve 204
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    icon_path = FRONTEND_DIR / "favicon.ico"
    if icon_path.exists():
        return FileResponse(str(icon_path))
    return FileResponse(str(ROOT / "outputs" / "favicon.ico")) if (ROOT / "outputs" / "favicon.ico").exists() else ("", 204)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas Pydantic
# ─────────────────────────────────────────────────────────────────────────────

class RunModelRequest(BaseModel):
    model: Literal["ahp", "wlc", "bwm"]
    force_rerun: bool = False        # True → ignora caché y recalcula


class ModelInfo(BaseModel):
    id:          str
    label:       str
    description: str
    cached:      bool                # True si ya existe el JSON en disco


class RunModelResponse(BaseModel):
    model:      str
    from_cache: bool
    lod0:       list
    lod1:       list
    lod3:       list
    params:     dict


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
def health():
    """Health check básico."""
    return {"status": "ok", "cache_dir": str(cache_manager.CACHE_DIR)}


@app.get("/models", response_model=list[ModelInfo], tags=["Modelos"])
def list_models():
    """
    Retorna la lista de modelos disponibles con su estado de caché.
    """
    result = []
    for model_id, meta in MODEL_REGISTRY.items():
        cache_path = cache_manager.get_cache_path(model_id)
        result.append(ModelInfo(
            id=model_id,
            label=meta["label"],
            description=meta["description"],
            cached=cache_path.exists(),
        ))
    return result


@app.post("/run-model", response_model=RunModelResponse, tags=["Modelos"])
def run_model(req: RunModelRequest):
    """
    Carga o ejecuta un modelo MCDA.

    Flujo:
      1. Verificar si existe caché JSON para el modelo.
      2a. Si existe (y force_rerun=False) → retornar desde caché.
      2b. Si no existe → importar el módulo del modelo, ejecutar pipeline,
          guardar caché y retornar.
    """
    model_id = req.model
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_id}' no registrado.")

    meta = MODEL_REGISTRY[model_id]
    from_cache = False

    def _run_pipeline(config: dict) -> dict:
        """Importa dinámicamente el módulo y ejecuta la función de pipeline."""
        # Intento de import dinámico según MODEL_REGISTRY
        try:
            mod = importlib.import_module(meta["module"])
            fn = getattr(mod, meta["function"], None)
        except Exception:
            mod = None
            fn = None

        # Si la función existe, usarla directamente
        if fn is not None:
            return fn(config)

        # Si no existe la función esperada, usar el pipeline central `main.run_pipeline`
        try:
            import main as master_main
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=(f"No se encontró la función de pipeline en '{meta['module']}' y "
                        f"no fue posible importar 'main.run_pipeline': {e}"))

        # Construir configuración básica basada en DEFAULT_CONFIG
        cfg = dict(master_main.DEFAULT_CONFIG)
        # Ajustar algoritmo solicitado
        cfg["algorithm"] = config.get("model") or config.get("algorithm") or meta.get("label", "")
        # Pasar force_rerun si fue solicitado
        if "force_rerun" in config:
            cfg["force_rerun"] = bool(config["force_rerun"])

        # Ejecutar pipeline maestro
        result = master_main.run_pipeline(cfg)

        # Normalizar salida: convertir DataFrames a registros JSON serializables
        scored = result.get("scored_df") or result.get("scored_records")
        ranked = result.get("ranked_df") or result.get("ranked_records")

        # Si el pipeline devolvió registros ya serializables, úsalos; si devolvió DataFrames, conviértelos
        if hasattr(scored, "to_dict"):
            scored_records = scored.to_dict(orient="records")
        else:
            scored_records = scored or []

        if hasattr(ranked, "to_dict"):
            ranked_records = ranked.to_dict(orient="records")
        else:
            ranked_records = ranked or []

        # Construir LODs desde los registros
        lod0, lod1, lod3, params_extra = _records_to_lods(scored_records, ranked_records, cfg.get("score_column", "suitability_score"))

        return {
            "lod0": lod0,
            "lod1": lod1,
            "lod3": lod3,
            "scored_records": scored_records,  # guardar también en caché
            "ranked_records": ranked_records,  # guardar también en caché
            "params": {"algorithm": cfg["algorithm"], **params_extra},
        }

    # Verificar caché antes de ejecutar
    if not req.force_rerun:
        cached = cache_manager.load_cache(model_id)
        if cached is not None:
            from_cache = True
            data = cached
        else:
            data = _run_pipeline({"model": model_id})
            cache_manager.save_cache(model_id, data)
    else:
        print(f"[API] force_rerun=True — recalculando '{model_id}'")
        data = _run_pipeline({"model": model_id})
        cache_manager.save_cache(model_id, data)

    # Si el caché tiene scored_records pero no lod0/lod1/lod3, construir LODs ahora
    if "scored_records" in data and "lod0" not in data:
        print(f"[API] Construyendo LODs desde registros en caché...")
        lod0, lod1, lod3, params_extra = _records_to_lods(
            data.get("scored_records", []),
            data.get("ranked_records", []),
            "suitability_score"
        )
        data["lod0"] = lod0
        data["lod1"] = lod1
        data["lod3"] = lod3
        # Incorporar las tablas de lookup en params
        if "params" not in data:
            data["params"] = {}
        data["params"].update(params_extra)

    # Validar estructura mínima del resultado
    for key in ("lod0", "lod1", "lod3"):
        if key not in data:
            raise HTTPException(
                status_code=500,
                detail=f"El resultado del modelo no contiene la clave requerida '{key}'."
            )

    return RunModelResponse(
        model=model_id,
        from_cache=from_cache,
        lod0=data.get("lod0", []),
        lod1=data.get("lod1", []),
        lod3=data.get("lod3", []),
        params=data.get("params", {}),
    )
