"""
cache_manager.py
================
Sistema de caché basado en archivos JSON para modelos MCDA (AHP, WLC, BWM-PROMETHEE).

Evita recalcular modelos costosos guardando y recuperando resultados en disco.

Solo usa librerías de la librería estándar de Python:
    json, os, pathlib, typing, hashlib, datetime
"""

from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuración global
# ─────────────────────────────────────────────────────────────────────────────

# Directorio raíz de caché. Se puede sobreescribir antes de usar el módulo:
#   import cache_manager; cache_manager.CACHE_DIR = Path("mi_directorio")
CACHE_DIR: Path = Path("outputs")

# Versión del esquema de caché. Incrementar si cambia la estructura del JSON.
CACHE_SCHEMA_VERSION: str = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# 1. get_cache_path
# ─────────────────────────────────────────────────────────────────────────────

def get_cache_path(model_name: str) -> Path:
    """
    Retorna la ruta completa del archivo JSON de caché para un modelo.

    Crea el directorio de caché si no existe.

    Parameters
    ----------
    model_name : str
        Nombre del modelo (ej: "ahp", "wlc", "bwm_promethee").
        Se normaliza a minúsculas y se eliminan espacios.

    Returns
    -------
    Path  Ruta absoluta al archivo JSON (ej: outputs/ahp.json).

    Examples
    --------
    >>> get_cache_path("AHP")
    PosixPath('outputs/ahp.json')
    """
    # Normalizar nombre: minúsculas, sin espacios laterales
    safe_name = model_name.strip().lower().replace(" ", "_")

    # Asegurar que el directorio existe
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return CACHE_DIR / f"{safe_name}.json"


# ─────────────────────────────────────────────────────────────────────────────
# 2. load_cache
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Intenta cargar el resultado de caché para un modelo.

    Returns None en cualquiera de estos casos:
        - El archivo no existe.
        - El archivo está corrupto (JSON inválido).
        - El esquema de versión no coincide.

    Parameters
    ----------
    model_name : str  Nombre del modelo a cargar.

    Returns
    -------
    dict | None  Diccionario con los resultados, o None si no hay caché válida.
    """
    path = get_cache_path(model_name)

    if not path.exists():
        print(f"[Cache] '{model_name}': no existe caché en {path}")
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[Cache] '{model_name}': archivo corrupto — {e}. Se ignorará.")
        return None
    except OSError as e:
        print(f"[Cache] '{model_name}': error de lectura — {e}. Se ignorará.")
        return None

    # Verificar versión de esquema para evitar incompatibilidades
    cached_version = data.get("_meta", {}).get("schema_version")
    if cached_version != CACHE_SCHEMA_VERSION:
        print(
            f"[Cache] '{model_name}': versión de esquema incompatible "
            f"(guardado={cached_version}, actual={CACHE_SCHEMA_VERSION}). "
            f"Se recalculará."
        )
        return None

    ts = data.get("_meta", {}).get("saved_at", "fecha desconocida")
    n  = _count_records(data)
    print(f"[Cache] '{model_name}': caché válida cargada ({n} registros, guardado {ts})")

    # Retornar solo los datos del usuario (sin metadatos internos)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────────────
# 3. save_cache
# ─────────────────────────────────────────────────────────────────────────────

def save_cache(model_name: str, data: Dict[str, Any]) -> None:
    """
    Guarda el resultado de un modelo en un archivo JSON.

    El archivo incluye metadatos internos (prefijo "_") para control de versión
    y trazabilidad. Estos metadatos son transparentes al usuario: load_cache
    los filtra antes de retornar el resultado.

    Parameters
    ----------
    model_name : str          Nombre del modelo.
    data       : Dict[str, Any]  Resultado del modelo a persistir.

    Estructura del JSON generado
    ----------------------------
    {
        "_meta": {
            "schema_version": "1.0",
            "model_name": "ahp",
            "saved_at": "2025-04-10T14:32:01Z",
            "config_hash": "a3f9..."   # hash del config, para auditoría
        },
        "lod0":   [...],
        "lod1":   [...],
        "lod3":   [...],
        "params": { "weights": {...}, "method": "AHP" }
    }
    """
    path = get_cache_path(model_name)

    payload = {
        "_meta": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_name": model_name.strip().lower(),
            "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        **data,
    }

    try:
        # Escritura atómica: escribir en archivo temporal y luego renombrar,
        # para evitar dejar un JSON a medias si el proceso se interrumpe.
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_serializer)
        tmp_path.replace(path)

    except (OSError, TypeError) as e:
        print(f"[Cache] '{model_name}': error al guardar — {e}")
        # Limpiar archivo temporal si quedó en disco
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return

    size_kb = path.stat().st_size / 1024
    n       = _count_records(data)
    print(f"[Cache] '{model_name}': guardado en {path} ({n} registros, {size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. get_or_run
# ─────────────────────────────────────────────────────────────────────────────

def get_or_run(
    model_name: str,
    config: Dict[str, Any],
    run_function: Callable[[Dict[str, Any]], Dict[str, Any]],
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """
    Carga la caché del modelo si existe; si no, ejecuta el modelo y la guarda.

    Este es el punto de entrada principal del módulo. Encapsula el ciclo
    completo: verificar → (calcular) → guardar → retornar.

    Parameters
    ----------
    model_name   : str
        Nombre del modelo (ej: "ahp", "bwm_promethee").
    config       : Dict[str, Any]
        Parámetros de configuración del modelo (pesos, método, etc.).
        Se guarda en el JSON dentro de "params" para trazabilidad.
    run_function : Callable[[dict], dict]
        Función que ejecuta el modelo. Recibe `config` y retorna un dict con:
            {"lod0": [...], "lod1": [...], "lod3": [...], "params": {...}}
    force_rerun  : bool
        Si True, ignora la caché existente y recalcula siempre.
        Útil durante desarrollo o cuando cambian los datos de entrada.

    Returns
    -------
    Dict[str, Any]  Resultado del modelo (desde caché o recién calculado).

    Examples
    --------
    >>> def run_ahp(cfg):
    ...     # Tu lógica de modelo aquí
    ...     return {"lod0": [...], "lod1": [...], "lod3": [...], "params": cfg}
    ...
    >>> result = get_or_run("ahp", config={"method": "AHP"}, run_function=run_ahp)
    """
    if not force_rerun:
        cached = load_cache(model_name)
        if cached is not None:
            return cached
    else:
        print(f"[Cache] '{model_name}': force_rerun=True — se ignorará caché existente.")

    # Ejecutar el modelo
    print(f"[Cache] '{model_name}': ejecutando modelo...")
    try:
        result = run_function(config)
    except Exception as e:
        print(f"[Cache] '{model_name}': error durante la ejecución del modelo — {e}")
        raise

    # Inyectar hash del config para trazabilidad (en _meta, no en el resultado)
    config_hash = _hash_config(config)

    # Enriquecer metadatos sin modificar el dict original del usuario
    result_to_save = {**result}

    # Guardar en disco
    save_cache(model_name, result_to_save)

    # Guardar el hash en el archivo de forma separada (post-save patch)
    _patch_meta(model_name, {"config_hash": config_hash})

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. invalidate_cache  (utilidad extra)
# ─────────────────────────────────────────────────────────────────────────────

def invalidate_cache(model_name: str) -> bool:
    """
    Elimina el archivo de caché de un modelo.

    Útil cuando los datos de entrada cambian y quieres forzar un recálculo
    sin usar force_rerun (por ejemplo, en pipelines automatizados).

    Parameters
    ----------
    model_name : str  Nombre del modelo cuya caché se eliminará.

    Returns
    -------
    bool  True si el archivo existía y fue eliminado, False si no existía.
    """
    path = get_cache_path(model_name)
    if path.exists():
        path.unlink()
        print(f"[Cache] '{model_name}': caché eliminada ({path})")
        return True
    print(f"[Cache] '{model_name}': no había caché que eliminar.")
    return False


def list_cached_models() -> list[str]:
    """
    Retorna los nombres de todos los modelos con caché guardada en CACHE_DIR.

    Returns
    -------
    List[str]  Nombres de modelo (sin extensión .json).
    """
    if not CACHE_DIR.exists():
        return []
    names = [p.stem for p in sorted(CACHE_DIR.glob("*.json"))]
    print(f"[Cache] Modelos en caché: {names or 'ninguno'}")
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Funciones internas (prefijo _)
# ─────────────────────────────────────────────────────────────────────────────

def _json_serializer(obj: Any) -> Any:
    """
    Serializador JSON para tipos no soportados por defecto.
    Convierte numpy arrays, pandas Series/DataFrame, sets y otros iterables.
    """
    # numpy / pandas (importación perezosa para no forzar dependencias)
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    try:
        import pandas as pd
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
    except ImportError:
        pass

    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if hasattr(obj, "__iter__"):
        return list(obj)

    raise TypeError(f"Tipo no serializable: {type(obj)}")


def _count_records(data: Dict[str, Any]) -> int:
    """
    Cuenta el número de registros en lod0/lod1/lod3 (el más largo).
    Útil para el log de confirmación.
    """
    lengths = [
        len(v) for k, v in data.items()
        if k in ("lod0", "lod1", "lod3") and isinstance(v, list)
    ]
    if not lengths:
        lengths = [
            len(v) for k, v in data.items()
            if k in ("scored_records", "ranked_records") and isinstance(v, list)
        ]
    return max(lengths, default=0)


def _hash_config(config: Dict[str, Any]) -> str:
    """
    Genera un hash SHA-256 corto del config para trazabilidad.
    Permite detectar si el caché fue generado con parámetros diferentes.
    """
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def _patch_meta(model_name: str, extra: Dict[str, Any]) -> None:
    """
    Agrega campos extra a la sección _meta del JSON ya guardado.
    Operación de bajo costo: re-lee y re-escribe solo el archivo de metadatos.
    """
    path = get_cache_path(model_name)
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("_meta", {}).update(extra)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_serializer)
    except (OSError, json.JSONDecodeError):
        pass  # No crítico — el hash es solo para auditoría


# ─────────────────────────────────────────────────────────────────────────────
# Ejemplo de uso
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Simular función de modelo AHP ──────────────────────────────────────
    def run_ahp(config: dict) -> dict:
        print("  → Calculando modelo AHP (simulado)...")
        return {
            "lod0": [{"hex": f"hex_{i}", "score": round(i * 0.01, 4)} for i in range(100)],
            "lod1": [{"hex": f"hex_{i}", "score": round(i * 0.02, 4)} for i in range(50)],
            "lod3": [{"hex": f"hex_{i}", "score": round(i * 0.05, 4)} for i in range(20)],
            "params": config,
        }

    config_ahp = {
        "method":  "AHP",
        "weights": {"wind": 0.40, "slope": 0.25, "distance": 0.20, "grid": 0.15},
    }

    # Primera llamada: calcula y guarda
    print("=" * 55)
    print("LLAMADA 1 — sin caché previa")
    print("=" * 55)
    result = get_or_run("ahp", config=config_ahp, run_function=run_ahp)
    print(f"  Registros lod0: {len(result['lod0'])}")

    # Segunda llamada: carga desde caché
    print("\n" + "=" * 55)
    print("LLAMADA 2 — debe cargar desde caché")
    print("=" * 55)
    result2 = get_or_run("ahp", config=config_ahp, run_function=run_ahp)
    print(f"  Registros lod0: {len(result2['lod0'])}")

    # Forzar recálculo
    print("\n" + "=" * 55)
    print("LLAMADA 3 — force_rerun=True")
    print("=" * 55)
    result3 = get_or_run("ahp", config=config_ahp, run_function=run_ahp, force_rerun=True)

    # Listar y limpiar
    print("\n" + "=" * 55)
    list_cached_models()
    invalidate_cache("ahp")
    list_cached_models()
