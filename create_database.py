import os
import sqlite3
import numpy as np
import pandas as pd

from src.generate_h3_grid import generate_colombia_hex_grid

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "hexagon_features.db")
GEOJSON = os.path.join(DATA_DIR, "colombia_boundary.geojson")
MUNICIPIOS = os.path.join(DATA_DIR, "DIVIPOLA_CentrosPoblados.csv")

print("Generando grid...")

hex_grid = generate_colombia_hex_grid(
    GEOJSON,
    resolution=7,
    municipios_path=MUNICIPIOS,
)

print("Hexágonos generados:", len(hex_grid))

np.random.seed(42)

features = pd.DataFrame({
    "hex_id": hex_grid["hex_id"],
    "wind_speed": np.random.uniform(4, 10, len(hex_grid)),
    "slope": np.random.uniform(0, 30, len(hex_grid)),
    "dist_to_grid": np.random.uniform(0, 100, len(hex_grid)),
    "dist_to_roads": np.random.uniform(0, 50, len(hex_grid)),
    "land_use": np.random.uniform(0, 1, len(hex_grid)),
    "protected_area": np.random.uniform(0, 1, len(hex_grid)),
    "conflict_risk": np.random.uniform(0, 1, len(hex_grid))
})

conn = sqlite3.connect(DB_PATH)

features.to_sql(
    "hexagon_features",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Base de datos creada correctamente con hex_id reales")