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
import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Literal

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, status
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

AUTH_DB_PATH = Path(os.getenv("AUTH_DB_PATH", str(ROOT / "data" / "auth.db")))
AUTH_SECRET_PATH = Path(os.getenv("AUTH_SECRET_PATH", str(ROOT / "data" / "auth_secret.key")))
SESSION_COOKIE_NAME = "mcda_session"
SESSION_TTL_SECONDS = int(os.getenv("AUTH_SESSION_TTL_SECONDS", str(60 * 60 * 8)))
AUTH_ALLOW_REGISTRATION = os.getenv("AUTH_ALLOW_REGISTRATION", "true").lower() in {"1", "true", "yes", "on"}
AUTH_PBKDF2_ITERATIONS = 260_000
AUTH_MAX_LOGIN_ATTEMPTS = int(os.getenv("AUTH_MAX_LOGIN_ATTEMPTS", "5"))
AUTH_LOGIN_LOCKOUT_SECONDS = int(os.getenv("AUTH_LOGIN_LOCKOUT_SECONDS", "60"))
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if origin.strip()
]
LOGIN_ATTEMPTS: dict[str, dict[str, int]] = {}


def _load_auth_secret() -> bytes:
    """Carga o crea el secreto local usado para firmar cookies de sesion."""
    secret = os.getenv("AUTH_SECRET_KEY")
    if secret:
        return secret.encode("utf-8")

    AUTH_SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not AUTH_SECRET_PATH.exists():
        AUTH_SECRET_PATH.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    return AUTH_SECRET_PATH.read_text(encoding="utf-8").strip().encode("utf-8")


AUTH_SECRET = _load_auth_secret()


def _get_db() -> sqlite3.Connection:
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_auth_db() -> None:
    with _get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL
            )
            """
        )


def _user_count() -> int:
    with _get_db() as conn:
        row = conn.execute("SELECT COUNT(*) AS total FROM users").fetchone()
    return int(row["total"])


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, AUTH_PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        AUTH_PBKDF2_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_b64.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _sign(payload: bytes) -> str:
    return _b64url(hmac.new(AUTH_SECRET, payload, hashlib.sha256).digest())


def _create_session_token(user: sqlite3.Row) -> str:
    payload = {
        "sub": int(user["id"]),
        "username": user["username"],
        "role": user["role"],
        "exp": int(time.time()) + SESSION_TTL_SECONDS,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded_payload = _b64url(payload_bytes)
    return f"{encoded_payload}.{_sign(encoded_payload.encode('ascii'))}"


def _read_session_token(token: str | None) -> dict | None:
    if not token or "." not in token:
        return None
    encoded_payload, signature = token.rsplit(".", 1)
    if not hmac.compare_digest(_sign(encoded_payload.encode("ascii")), signature):
        return None
    try:
        payload = json.loads(_b64url_decode(encoded_payload))
    except Exception:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload


def _public_user(row: sqlite3.Row | dict) -> dict:
    return {
        "id": int(row["id"]),
        "username": row["username"],
        "role": row["role"],
    }


def get_current_user(mcda_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict:
    payload = _read_session_token(mcda_session)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado.")

    with _get_db() as conn:
        user = conn.execute(
            "SELECT id, username, role, is_active FROM users WHERE id = ?",
            (payload["sub"],),
        ).fetchone()

    if not user or not user["is_active"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario no disponible.")
    return _public_user(user)


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=os.getenv("AUTH_COOKIE_SECURE", "false").lower() in {"1", "true", "yes", "on"},
        samesite="lax",
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")


def _normalize_username(username: str) -> str:
    return username.strip()


def _validate_credentials(username: str, password: str) -> tuple[str, str]:
    username = _normalize_username(username)
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,40}", username):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El usuario debe tener 3-40 caracteres y usar letras, numeros, punto, guion o guion bajo.",
        )
    if len(password) < 8 or len(password) > 128:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La contrasena debe tener entre 8 y 128 caracteres.",
        )
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La contrasena debe incluir al menos una letra y un numero.",
        )
    if username.lower() in password.lower():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La contrasena no debe contener el nombre de usuario.",
        )
    return username, password


def _registration_open() -> bool:
    return AUTH_ALLOW_REGISTRATION or _user_count() == 0


def _login_key(request: Request, username: str) -> str:
    client_host = request.client.host if request.client else "unknown"
    return f"{client_host}:{username.lower()}"


def _check_login_throttle(request: Request, username: str) -> None:
    key = _login_key(request, username)
    attempt = LOGIN_ATTEMPTS.get(key)
    now = int(time.time())

    if not attempt:
        return

    locked_until = int(attempt.get("locked_until", 0))
    if locked_until > now:
        retry_after = locked_until - now
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Demasiados intentos fallidos. Intenta de nuevo en {retry_after} segundos.",
            headers={"Retry-After": str(retry_after)},
        )

    if locked_until:
        LOGIN_ATTEMPTS.pop(key, None)


def _record_failed_login(request: Request, username: str) -> None:
    key = _login_key(request, username)
    now = int(time.time())
    attempt = LOGIN_ATTEMPTS.get(key, {"count": 0, "locked_until": 0})
    attempt["count"] = int(attempt.get("count", 0)) + 1
    attempt["locked_until"] = 0

    if attempt["count"] >= AUTH_MAX_LOGIN_ATTEMPTS:
        attempt["locked_until"] = now + AUTH_LOGIN_LOCKOUT_SECONDS
        attempt["count"] = 0

    LOGIN_ATTEMPTS[key] = attempt


def _clear_failed_login(request: Request, username: str) -> None:
    LOGIN_ATTEMPTS.pop(_login_key(request, username), None)


_init_auth_db()

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
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
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


class AuthCredentials(BaseModel):
    username: str
    password: str


class AuthUser(BaseModel):
    id: int
    username: str
    role: str


class AuthStatus(BaseModel):
    authenticated: bool
    user: AuthUser | None = None
    registration_open: bool
    setup_required: bool
    session_ttl_seconds: int


class AuthResponse(BaseModel):
    user: AuthUser
    message: str


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


@app.get("/auth/status", response_model=AuthStatus, tags=["Autenticacion"])
def auth_status(
    response: Response,
    mcda_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """Retorna el estado de sesion y si esta abierto el registro inicial."""
    payload = _read_session_token(mcda_session)
    user = None

    if payload and "sub" in payload:
        with _get_db() as conn:
            row = conn.execute(
                "SELECT id, username, role, is_active FROM users WHERE id = ?",
                (payload["sub"],),
            ).fetchone()
        if row and row["is_active"]:
            user = _public_user(row)

    if not user and mcda_session:
        _clear_session_cookie(response)

    setup_required = _user_count() == 0

    return AuthStatus(
        authenticated=bool(user),
        user=user,
        registration_open=_registration_open(),
        setup_required=setup_required,
        session_ttl_seconds=SESSION_TTL_SECONDS,
    )


@app.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED, tags=["Autenticacion"])
def register(credentials: AuthCredentials, response: Response):
    """
    Crea un usuario. El primer usuario queda como admin; los siguientes como user.
    Para cerrar registros posteriores, usar AUTH_ALLOW_REGISTRATION=false.
    """
    if not _registration_open():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="El registro esta cerrado. Inicia sesion con un usuario existente.",
        )

    username, password = _validate_credentials(credentials.username, credentials.password)
    role = "admin" if _user_count() == 0 else "user"

    try:
        with _get_db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, _hash_password(password), role, int(time.time())),
            )
            user = conn.execute(
                "SELECT id, username, role FROM users WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ese usuario ya existe.",
        )

    _set_session_cookie(response, _create_session_token(user))
    return AuthResponse(user=_public_user(user), message="Usuario creado correctamente.")


@app.post("/auth/login", response_model=AuthResponse, tags=["Autenticacion"])
def login(credentials: AuthCredentials, request: Request, response: Response):
    """Valida credenciales y abre una sesion con cookie HTTPOnly."""
    username = _normalize_username(credentials.username)
    password = credentials.password or ""
    _check_login_throttle(request, username)

    with _get_db() as conn:
        user = conn.execute(
            "SELECT id, username, password_hash, role, is_active FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if not user or not user["is_active"] or not _verify_password(password, user["password_hash"]):
        _record_failed_login(request, username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contrasena incorrectos.",
        )

    _clear_failed_login(request, username)
    _set_session_cookie(response, _create_session_token(user))
    return AuthResponse(user=_public_user(user), message="Sesion iniciada.")


@app.post("/auth/logout", tags=["Autenticacion"])
def logout(response: Response):
    """Cierra la sesion eliminando la cookie."""
    _clear_session_cookie(response)
    return {"message": "Sesion cerrada."}


@app.get("/auth/me", response_model=AuthUser, tags=["Autenticacion"])
def me(current_user: dict = Depends(get_current_user)):
    """Retorna el usuario autenticado actual."""
    return current_user


@app.get("/models", response_model=list[ModelInfo], tags=["Modelos"])
def list_models(current_user: dict = Depends(get_current_user)):
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
def run_model(req: RunModelRequest, current_user: dict = Depends(get_current_user)):
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
